import os
import csv
import zlib
import uuid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import pathlib
import tensorflow as tf
from typing import Callable, Dict
import glob
from modules import STrajNet
import argparse
# Set up matplotlib to use a non-interactive backend
matplotlib.use('Agg')

# Suppress TensorFlow logging
tf.compat.v1.logging.set_verbosity(tf._logging.ERROR)

# Local modules and classes
from modules import SwinTransformerEncoder, CFGS
# from loss import OGMFlow_loss, OGMFlow_loss2
import occu_metric as occupancy_flow_metrics

# Waymo Open Dataset utilities
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
#from waymo_open_dataset.utils import occupancy_flow_metrics

# from waymo_open_dataset.utils import occupancy_flow_metrics

# Metrics module
from metrics import OGMFlowMetrics, print_metrics

# Google protobuf
from google.protobuf import text_format

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

# print(config)


import os
# Hyper parameters
NUM_PRED_CHANNELS = 4

from time import time

TEST =False

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
  # Parse the input tf.Example proto using the dictionary above.
  new_dict = {}
  d =  tf.io.parse_single_example(example_proto, feature)
  new_dict['centerlines'] = tf.cast(tf.reshape(tf.io.decode_raw(d['centerlines'],tf.float64),[256,10,7]),tf.float32)
  new_dict['actors'] = tf.cast(tf.reshape(tf.io.decode_raw(d['actors'],tf.float64),[48,11,8]),tf.float32)
  new_dict['occl_actors'] = tf.cast(tf.reshape(tf.io.decode_raw(d['occl_actors'],tf.float64),[16,11,8]),tf.float32)

  new_dict['gt_flow'] = tf.reshape(tf.io.decode_raw(d['gt_flow'],tf.float32),[8,512,512,2])[:,128:128+256,128:128+256,:]
  new_dict['origin_flow'] = tf.reshape(tf.io.decode_raw(d['origin_flow'],tf.float32),[8,512,512,1])[:,128:128+256,128:128+256,:]

  new_dict['ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(d['ogm'],tf.bool),tf.float32),[512,512,11,2])

  new_dict['gt_obs_ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(d['gt_obs_ogm'],tf.bool),tf.float32),[8,512,512,1])[:,128:128+256,128:128+256,:]
  new_dict['gt_occ_ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(d['gt_occ_ogm'],tf.bool),tf.float32),[8,512,512,1])[:,128:128+256,128:128+256,:]

  new_dict['map_image'] = tf.cast(tf.reshape(tf.io.decode_raw(d['map_image'],tf.int8),[256,256,3]),tf.float32) / 256
  new_dict['vec_flow'] = tf.reshape(tf.io.decode_raw(d['vec_flow'],tf.float32),[512,512,2])
  return new_dict


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
model = STrajNet(cfg,sep_actors=False)

def test_step(data):
    map_img = data['map_image']
    centerlines = data['centerlines']
    actors = data['actors']
    occl_actors = data['occl_actors']

    ogm = data['ogm']
    gt_obs_ogm = data['gt_obs_ogm']
    gt_occ_ogm = data['gt_occ_ogm']
    gt_flow = data['gt_flow']
    origin_flow = data['origin_flow']

    flow = data['vec_flow']
    true_waypoints = _warpped_gt(gt_ogm=gt_obs_ogm,gt_occ=gt_occ_ogm,gt_flow=gt_flow,origin_flow=origin_flow)

    outputs = model.call(ogm,map_img,training=False,obs=actors,occ=occl_actors,mapt=centerlines,flow=flow)
    logits = _get_pred_waypoint_logits(outputs)
    pred_waypoints = _apply_sigmoid_to_occupancy_logits(logits)

    return pred_waypoints,true_waypoints

def _add_waypoints_to_scenario_prediction(
    pred_waypoints: occupancy_flow_grids.WaypointGrids,
    scenario_prediction: occupancy_flow_submission_pb2.ScenarioPrediction,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig
) -> None:
    """Add predictions for all waypoints to the scenario prediction message."""
    for k in range(config.num_waypoints):
        waypoint_message = scenario_prediction.waypoints.add()
        # Convert Tensors to bytes more efficiently and without leaving GPU context.
        for tensor, attribute in [
            (pred_waypoints.vehicles.observed_occupancy[k], 'observed_vehicles_occupancy'),
            (pred_waypoints.vehicles.occluded_occupancy[k], 'occluded_vehicles_occupancy'),
            (pred_waypoints.vehicles.flow[k], 'all_vehicles_flow'),
        ]:
            quantized_tensor = tf.cast(tf.round(tensor * 255), tf.uint8)
            compressed_bytes = tf.io.encode_base64(tf.io.serialize_tensor(quantized_tensor))
            setattr(waypoint_message, attribute, compressed_bytes.numpy())

def generate_true_waypoints(inputs):
    """
    Generate ground truth waypoint grids from the input data.
    Assumes 'inputs' contain all necessary fields and metadata.
    
    Args:
        inputs (dict): A dictionary containing input data for generating ground truth.
    
    Returns:
        occupancy_flow_grids.WaypointGrids: Ground truth waypoints, or None if an error occurs.
    """
    try:
        inputs_with_fields = occupancy_flow_data.add_sdc_fields(inputs)
        timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
            inputs=inputs_with_fields, config=config
        )
        true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
            timestep_grids=timestep_grids, config=config
        )
        return true_waypoints
    except Exception as e:
        print(f"Failed to generate true waypoints: {e}")
        return None

def _warpped_gt(
    gt_ogm: tf.Tensor,
    gt_occ: tf.Tensor,
    gt_flow: tf.Tensor,
    origin_flow: tf.Tensor,) -> occupancy_flow_grids.WaypointGrids:

    true_waypoints = occupancy_flow_grids.WaypointGrids()

    for k in range(8):
        true_waypoints.vehicles.observed_occupancy.append(gt_ogm[:,k])
        true_waypoints.vehicles.occluded_occupancy.append(gt_occ[:,k])
        true_waypoints.vehicles.flow.append(gt_flow[:,k])
        true_waypoints.vehicles.flow_origin_occupancy.append(origin_flow[:,k])
    
    return true_waypoints


def model_testing(test_shard_path, ids,metric_shared_path):
    print(f'Creating submission for test shard {test_shard_path}...')
    test_dataset = _make_test_dataset(test_shard_path=test_shard_path)
    dataset = _make_metrics_dataset(metric_shard_path=metric_shared_path)
    print(test_dataset)
    print('\n')
    print(dataset)
    # submission = _make_submission_proto()
    for x,y in tqdm(zip(test_dataset,dataset)):
        pred_waypoints,true_waypoints = test_step(x)
        #true_waypoints = generate_true_waypoints(y)  # Generate true waypoints
        # Compute metrics between predicted and true waypoints
        metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
        config=config,
        true_waypoints=true_waypoints,
        pred_waypoints=pred_waypoints,
        no_warp=False
        )

        
        print(metrics)

        # Append results to the scenario predictions for submission
    #     scenario_prediction = submission.scenario_predictions.add()
    #     sc_id = batch['scenario/id'].numpy()[0].decode('utf-8')
    #     scenario_prediction.scenario_id = sc_id

    #     _add_waypoints_to_scenario_prediction(
    #         pred_waypoints=pred_waypoints,
    #         scenario_prediction=scenario_prediction,
    #         config=config
    #     )

    # _save_submission_to_file(submission, test_shard_path)

    return len(test_dataset)


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
    f = open(submission_shard_file_path, 'wb')
    f.write(submission.SerializeToString())
    f.close()

def _make_test_dataset(test_shard_path: str) -> tf.data.Dataset:
  """Makes a dataset for one shard in the test set."""
  test_dataset = tf.data.TFRecordDataset(test_shard_path)
#   it=iter(test_dataset)
  test_dataset = test_dataset.map(_parse_image_function_test)
  test_dataset = test_dataset.batch(1)
  return test_dataset

def _make_metrics_dataset(metric_shard_path: str) -> tf.data.Dataset:
  
    filenames = tf.io.matching_files(metric_shard_path)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat()
    dataset = dataset.map(occupancy_flow_data.parse_tf_example)
    dataset = dataset.batch(1)
    return dataset
    # test_dataset = tf.data.TFRecordDataset(test_shard_path)
    # it=iter(test_dataset)
    # test_dataset = test_dataset.map(_parse_image_function_test)
    # test_dataset = test_dataset.batch(1)
    # return test_dataset


def id_checking(test=True):
    # if eval:
    #         path = f'{args.ids_dir}/validation_scenario_ids.txt'
    #         print("validation"+args.ids_dir)
    # else:
    if test:
        path = f'{args.ids_dir}/testing_scenario_ids.txt'
        print("testing"+args.ids_dir)

    with tf.io.gfile.GFile(path) as f:
        test_scenario_ids = f.readlines()
        test_scenario_ids = [id.rstrip() for id in test_scenario_ids]
        print(f'original ids num:{len(test_scenario_ids)}')
        test_scenario_ids = set(test_scenario_ids)
    return test_scenario_ids

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--ids_dir', type=str, help='ids.txt downloads from Waymos', default="/raid/STrajNet/waymo_open_dataset_motion_v_1_1_0/occupancy_flow_challenge/")
    parser.add_argument('--save_dir', type=str, help='saving directory',default="/raid/STrajNet/inference/")
    parser.add_argument('--file_dir', type=str, help='Test Dataset directory',default="/raid/STrajNet/preprocessed_data/val")
    parser.add_argument('--weight_path', type=str, help='Model weights directory',default="/raid/STrajNet/train_output/final_model.tf")
    parser.add_argument("--test_path",type = str, help = "Test Directory",default = "/raid/STrajNet/waymo_open_dataset_motion_v_1_1_0/tf_example/testing")
    print("args hello")
    args = parser.parse_args()
    model.load_weights(args.weight_path)
    m_filesnames = tf.io.gfile.glob(args.test_path + "/*.tfrecord*")
    print("This is the M files")
    # print(m_filesnames)
    print("This is main function")
    print(args.file_dir+'/*.tfrecords')
    v_filenames = tf.io.gfile.glob(args.file_dir+'/*.tfrecords')
    # print(v_filenames)
    print(f'{len(v_filenames)} found, start loading dataset')
    combine_files = zip(m_filesnames,v_filenames)
    test_scenario_ids=None#test_scenario_ids = id_checking(test=TEST)
    cnt = 0
    for x,y in combine_files:
        num = model_testing(test_shard_path = y,ids=test_scenario_ids,metric_shared_path = x)
        cnt += num
    print(cnt)
