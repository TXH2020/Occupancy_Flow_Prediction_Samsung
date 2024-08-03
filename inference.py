import tensorflow as tf
# tf.compat.v1.keras.config.disable_interactive_logging()
tf.compat.v1.logging.set_verbosity(tf._logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import numpy as np
import csv
matplotlib.use('Agg')
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
from modules import STrajNet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from tqdm import tqdm
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from modules import SwinTransformerEncoder ,CFGS

# from loss import OGMFlow_loss , OGMFlow_loss2

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2

from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
# from waymo_open_dataset.utils import occupancy_flow_metrics
import occu_metric as occupancy_flow_metrics

from google.protobuf import text_format

import csv 

import pathlib
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid
import zlib

from metrics import OGMFlowMetrics,print_metrics
from tqdm import tqdm
from modules import STrajNet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

layer = tf.keras.layers


gpus = tf.config.list_physical_devices('GPU')[0:1]
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_visible_devices(gpus, 'GPU')


#configuration
config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
config_text = """
num_past_steps: 10
num_future_steps: 80
num_waypoints: 8
cumulative_waypoints: false
normalize_sdc_yaw: true
grid_height_cells: 256
grid_width_cells: 256
sdc_y_in_grid: 192
sdc_x_in_grid: 128
pixels_per_meter: 3.2
agent_points_per_side_length: 48
agent_points_per_side_width: 16
"""
text_format.Parse(config_text, config)

print(config)

# Hyper parameters
NUM_PRED_CHANNELS = 4

from time import time

TEST =True

feature = {
    'centerlines': tf.io.FixedLenFeature([], tf.string),
    'actors': tf.io.FixedLenFeature([], tf.string),
    'occl_actors': tf.io.FixedLenFeature([], tf.string),
    'ogm': tf.io.FixedLenFeature([], tf.string),
    'map_image': tf.io.FixedLenFeature([], tf.string),
    'scenario/id':tf.io.FixedLenFeature([], tf.string),
    'vec_flow':tf.io.FixedLenFeature([], tf.string),
    # 'byc_flow':tf.io.FixedLenFeature([], tf.string)
}
if not TEST:
    feature.update({'gt_flow': tf.io.FixedLenFeature([], tf.string),
                    'origin_flow': tf.io.FixedLenFeature([], tf.string),
                    'gt_obs_ogm': tf.io.FixedLenFeature([], tf.string),
                    'gt_occ_ogm': tf.io.FixedLenFeature([], tf.string),
                    })

def _parse_image_function_test(example_proto):
    features = {
        'centerlines': tf.io.FixedLenFeature([], tf.string),
        'actors': tf.io.FixedLenFeature([], tf.string),
        'occl_actors': tf.io.FixedLenFeature([], tf.string),
        'ogm': tf.io.FixedLenFeature([], tf.string),
        'map_image': tf.io.FixedLenFeature([], tf.string),
        'vec_flow': tf.io.FixedLenFeature([], tf.string),
        'scenario/id': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return {
        'centerlines': tf.reshape(tf.io.decode_raw(parsed_features['centerlines'], tf.float64), [256, 10, 7]),
        'actors': tf.reshape(tf.io.decode_raw(parsed_features['actors'], tf.float64), [48, 11, 8]),
        'occl_actors': tf.reshape(tf.io.decode_raw(parsed_features['occl_actors'], tf.float64), [16, 11, 8]),
        'ogm': tf.reshape(tf.cast(tf.io.decode_raw(parsed_features['ogm'], tf.bool), tf.float32), [512, 512, 11, 2]),
        'map_image': tf.cast(tf.reshape(tf.io.decode_raw(parsed_features['map_image'], tf.int8), [256, 256, 3]), tf.float32) / 255.0,
        'vec_flow': tf.reshape(tf.io.decode_raw(parsed_features['vec_flow'], tf.float32), [512, 512, 2]),
        'scenario/id': parsed_features['scenario/id']
    }


def _get_pred_waypoint_logits(
    model_outputs: tf.Tensor,
    mode_flow_outputs:tf.Tensor=None) -> occupancy_flow_grids.WaypointGrids:
  """Slices model predictions into occupancy and flow grids."""
  pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

  # Slice channels into output predictions.
  for k in range(config.num_waypoints):
    index = k * NUM_PRED_CHANNELS
    if mode_flow_outputs is not None:
        waypoint_channels_flow = mode_flow_outputs[:, :, :, index:index + NUM_PRED_CHANNELS]
    waypoint_channels = model_outputs[:, :, :, index:index + NUM_PRED_CHANNELS]
    pred_observed_occupancy = waypoint_channels[:, :, :, :1]
    pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
    pred_flow = waypoint_channels[:, :, :, 2:]
    if mode_flow_outputs is not None:
        pred_flow = waypoint_channels_flow[:, :, :, 2:]
    pred_waypoint_logits.vehicles.observed_occupancy.append(
        pred_observed_occupancy)
    pred_waypoint_logits.vehicles.occluded_occupancy.append(
        pred_occluded_occupancy)
    pred_waypoint_logits.vehicles.flow.append(pred_flow)

  return pred_waypoint_logits

def _apply_sigmoid_to_occupancy_logits(
    pred_waypoint_logits: occupancy_flow_grids.WaypointGrids
) -> occupancy_flow_grids.WaypointGrids:
  """Converts occupancy logits with probabilities."""
  pred_waypoints = occupancy_flow_grids.WaypointGrids()
  pred_waypoints.vehicles.observed_occupancy = [
      tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.observed_occupancy
  ]
  pred_waypoints.vehicles.occluded_occupancy = [
      tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.occluded_occupancy
  ]
  pred_waypoints.vehicles.flow = pred_waypoint_logits.vehicles.flow
  return pred_waypoints


print('load_model...')


cfg=dict(input_size=(512,512), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
model = STrajNet(cfg, sep_actors=False)  
model.load_weights(args.weight_path)  




def compute_auc(true_labels, predictions):
    """Compute the Area Under the ROC Curve (AUC) from prediction scores."""
    return roc_auc_score(true_labels, predictions)

def compute_iou(true_occupancy, predicted_occupancy):
    """Compute Intersection Over Union (IOU) for occupancy grids."""
    intersection = np.logical_and(true_occupancy, predicted_occupancy).sum()
    union = np.logical_or(true_occupancy, predicted_occupancy).sum()
    return intersection / union if union else 0

def compute_flow_epe(true_flow, predicted_flow):
    """Compute the average End-Point Error (EPE) for flow fields."""
    return np.mean(np.linalg.norm(true_flow - predicted_flow, axis=-1))


def test_step(data, model):
    outputs = model.call(data['ogm'], data['map_image'], training=False, obs=data['actors'], 
                         occ=data['occl_actors'], mapt=data['centerlines'], flow=data['vec_flow'])
    logits = _get_pred_waypoint_logits(outputs)
    pred_waypoints = _apply_sigmoid_to_occupancy_logits(logits)

    # Compute metrics here or return values for later metric computation
    return pred_waypoints

def evaluate_metrics(true_waypoints, pred_waypoints):
    metrics_results = {
        "observed_auc": compute_auc(true_waypoints['observed'], pred_waypoints['observed']),
        "occluded_auc": compute_auc(true_waypoints['occluded'], pred_waypoints['occluded']),
        "observed_iou": compute_iou(true_waypoints['observed'], pred_waypoints['observed']),
        "occluded_iou": compute_iou(true_waypoints['occluded'], pred_waypoints['occluded']),
        "flow_epe": compute_flow_epe(true_waypoints['flow'], pred_waypoints['flow'])
    }
    return metrics_results

def _add_waypoints_to_scenario_prediction(
    pred_waypoints: occupancy_flow_grids.WaypointGrids,
    scenario_prediction: occupancy_flow_submission_pb2.ScenarioPrediction,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Add predictions for all waypoints to scenario_prediction message."""
  for k in range(config.num_waypoints):
    waypoint_message = scenario_prediction.waypoints.add()
    # Observed occupancy.
    obs_occupancy = pred_waypoints.vehicles.observed_occupancy[k].numpy()
    obs_occupancy_quantized = np.round(obs_occupancy * 255).astype(np.uint8)
    obs_occupancy_bytes = zlib.compress(obs_occupancy_quantized.tobytes())
    waypoint_message.observed_vehicles_occupancy = obs_occupancy_bytes
    # Occluded occupancy.
    occ_occupancy = pred_waypoints.vehicles.occluded_occupancy[k].numpy()
    occ_occupancy_quantized = np.round(occ_occupancy * 255).astype(np.uint8)
    occ_occupancy_bytes = zlib.compress(occ_occupancy_quantized.tobytes())
    waypoint_message.occluded_vehicles_occupancy = occ_occupancy_bytes
    # Flow.
    flow = pred_waypoints.vehicles.flow[k].numpy()
    flow_quantized = np.clip(np.round(flow), -128, 127).astype(np.int8)
    flow_bytes = zlib.compress(flow_quantized.tobytes())
    waypoint_message.all_vehicles_flow = flow_bytes




def model_testing(test_shard_path, ids, model):    
    metrics = OGMFlowMetrics(prefix='test', no_warp=True)
    file_name = test_shard_path.split('/')[-1]
    basename = os.path.basename(test_shard_path)
    num = basename[:5]
    print(f'Creating submission for test shard {file_name}...')
    test_dataset = _make_test_dataset(test_shard_path=test_shard_path)
    submission = _make_submission_proto()
    save_folder = args.save_dir 

    protobuf_save_folder = os.path.join(args.save_dir, "protobufs") 
    png_save_folder = os.path.join(args.save_dir, "pngs") 

    os.makedirs(protobuf_save_folder, exist_ok=True)  
    os.makedirs(png_save_folder, exist_ok=True) 

    cnt_sample = 0
    for batch in tqdm(test_dataset):
        pred_waypoints = test_step(batch, metrics)
        print(pred_waypoints)
        if cnt_sample < 5:
            save_path = os.path.join(png_save_folder, f'visualization_{num}_{cnt_sample}.png')
            matplotlib_visualize_flow(pred_waypoints, config, save_path=save_path)
            print("Visualization PNG generated")

        scenario_prediction = submission.scenario_predictions.add()
        sc_id = batch['scenario/id'].numpy()[0]
        if isinstance(sc_id, bytes):
            sc_id=str(sc_id, encoding = "utf-8") 
        scenario_prediction.scenario_id = sc_id

        assert sc_id in ids, (sc_id)

        _add_waypoints_to_scenario_prediction(
            pred_waypoints=pred_waypoints,
            scenario_prediction=scenario_prediction,
            config=config)

        cnt_sample += 1
        
    _save_submission_to_file(submission, test_shard_path, protobuf_save_folder)
    
    results = metrics.get_results()
    print_metrics(results, prefix='test', no_warp=True)

    return cnt_sample
        
def _make_submission_proto(
) -> occupancy_flow_submission_pb2.ChallengeSubmission:
    """Makes a submission proto to store predictions for one shard."""
    submission = occupancy_flow_submission_pb2.ChallengeSubmission()
    submission.account_name = ''
    submission.unique_method_name = ''
    # submission.authors.extend([''])
    submission.authors.extend([''])
    submission.description = ''
    submission.method_link = ''
    return submission

def _save_submission_to_file(submission, test_shard_path, save_folder):

    """Save predictions for one test shard as a binary protobuf."""
    # save_folder = os.path.join(pathlib.Path.home(),
    #                             'occupancy_flow_challenge/testing')
    # save_folder = os.path.join(args.save_dir,
    #                             '/test6')
    save_folder = args.save_dir
     
    os.makedirs(save_folder, exist_ok=True)
    basename = os.path.basename(test_shard_path)
    if 'new.tfrecords' not in basename:
        raise ValueError('Cannot determine file path for saving submission.')
    num = basename[:5]
    submission_basename = 'occupancy_flow_submission.binproto' + '-' + num + '-of-00150'

    submission_shard_file_path = os.path.join(save_folder, submission_basename)
    num_scenario_predictions = len(submission.scenario_predictions)
    print(f'Saving {num_scenario_predictions} scenario predictions to '
        f'{submission_shard_file_path}...\n')
    print(submission)
    f = open(submission_shard_file_path, 'wb')
    f.write(submission.SerializeToString())
    f.close()

def _make_test_dataset(test_shard_path: str) -> tf.data.Dataset:
  """Makes a dataset for one shard in the test set."""
  test_dataset = tf.data.TFRecordDataset(test_shard_path)
  test_dataset = test_dataset.map(_parse_image_function_test)
  test_dataset = test_dataset.batch(1)
  return test_dataset

def id_checking(test=True):
    if eval:
            path = f'{args.ids_dir}/validation_scenario_ids.txt'
            print("validation"+args.ids_dir)
    else:
        path = f'{args.ids_dir}/testing_scenario_ids.txt'
        print("testing"+args.ids_dir)

    with tf.io.gfile.GFile(path) as f:
        test_scenario_ids = f.readlines()
        test_scenario_ids = [id.rstrip() for id in test_scenario_ids]
        print(f'original ids num:{len(test_scenario_ids)}')
        test_scenario_ids = set(test_scenario_ids)
    return test_scenario_ids





def clean_and_validate_flow(flow, max_flow_value=10.0):
    """Cleans flow data by replacing NaNs, removing infinite values, and clipping."""
    # Ensure max_flow_value is a float
    max_flow_value = float(max_flow_value)
    flow = np.nan_to_num(flow, nan=0.0)  # Replace NaNs with 0
    flow = np.clip(flow, -max_flow_value, max_flow_value)  # Clip flow values
    return flow


def clean_and_validate_occupancy(occupancy):
    """Cleans occupancy data by replacing NaNs."""
    occupancy = np.nan_to_num(occupancy, nan=0.0)  # Replace NaNs with 0
    return occupancy


def matplotlib_visualize_flow(pred_waypoints, config, save_path=None):
    num_waypoints = 8  # Assuming you still want a total of 8 images
    num_plots_per_row = 4  # Keeping 4 plots per row as per your last request
    
    # Custom bright color map with an expanded range of attractive colors
    bright_cmap = ListedColormap([
        '#FFFFFF',  # Red
        '#fe7e0f',  # Orange
        '#FFF233',  # Yellow
        '#87c830',  # Green
        '#33C1FF',  # Sky Blue
        '#3357FF',  # Blue
        '#8e3ccb',  # Purple
        '#FF33FB',  # Magenta
        '#33FFF2',  # Cyan
        '#ff598f',
    ])

    # Create the subplots with a white background color
    fig, axes = plt.subplots(nrows=2,  # Set to 2 rows to make a total of 8 images
                             ncols=num_plots_per_row,
                             figsize=(20, 10),  # Adjust the size as needed
                             facecolor='white')  # Set the figure background to white

    if axes.ndim == 1:  # If there's only one row, `axes` will be 1D
        axes = np.expand_dims(axes, 0)

    # Normalize flow for all plots
    max_flow = np.max([np.max(np.abs(pred_waypoints.vehicles.flow[i].numpy().squeeze())) for i in range(num_waypoints)])
    
    for i in range(num_waypoints):
        # Get the current axis
        ax = axes[i // num_plots_per_row, i % num_plots_per_row]
        ax.set_facecolor('white')  # Set individual subplot background to white

        # Plot occupancy using scatter plot
        occupancy = pred_waypoints.vehicles.observed_occupancy[i].numpy().squeeze()
        Y, X = np.nonzero(occupancy)
        categories = occupancy[Y, X]
        scatter = ax.scatter(X, Y, c=categories, cmap=bright_cmap, s=1)

        # Plot flow using quiver plot
        flow = pred_waypoints.vehicles.flow[i].numpy().squeeze()
        Y, X = np.mgrid[0:flow.shape[0], 0:flow.shape[1]]
        U, V = flow[..., 0], flow[..., 1]
        ax.quiver(X, Y, U, V, scale=max_flow*20, units='xy', color='black')

        ax.set_title(f'Waypoint {i + 1}')
        ax.set_xlim([0, flow.shape[1]])
        ax.set_ylim([0, flow.shape[0]])
        ax.invert_yaxis()

        # Remove the axis labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Add white grid lines for better visual separation on white background
        # ax.grid(True, which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.5)

    # Adjust layout for better spacing
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    # Save or show the figure
    if save_path:
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=300)
    else:
        plt.figure(facecolor='white')
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.gca().set_facecolor('white')
        plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--ids_dir', type=str, help='ids.txt downloads from Waymos', default="/raid/STrajNet/waymo_open_dataset_motion_v_1_1_0/occupancy_flow_challenge/")
    parser.add_argument('--save_dir', type=str, help='saving directory', default="/raid/STrajNet/inference/")
    parser.add_argument('--file_dir', type=str, help='Test Dataset directory', default="/raid/STrajNet/preprocessed_data/val")
    parser.add_argument('--weight_path', type=str, help='Model weights directory', default="/raid/STrajNet/train_output/final_model.tf")
    args = parser.parse_args()

    # Assuming this is your model initialization
    cfg = dict(input_size=(512,512), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
    model = STrajNet(cfg, sep_actors=False)

    # Now, use the command line argument for weight path
    print(args.ids_dir)
    model.load_weights(args.weight_path)

    print("This is the main function")
    print(args.file_dir + '/*.tfrecords')
    v_filenames = tf.io.gfile.glob(args.file_dir + '/*.tfrecords')
    print(v_filenames)
    print(f'{len(v_filenames)} found, start loading dataset')
    test_scenario_ids = id_checking(test=TEST)
    cnt = 0
    for filename in v_filenames:
        num = model_testing(test_shard_path=filename, ids=test_scenario_ids, model=model)
        cnt += num
    print(cnt)

    OGMFlowMetrics()