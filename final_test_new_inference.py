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
import time as time_lib
from google.protobuf.json_format import MessageToDict
from modules import STrajNet
import glob
import argparse
# Set up matplotlib to use a non-interactive backend
matplotlib.use('Agg')

# Suppress TensorFlow logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Local modules and classes
from modules import SwinTransformerEncoder, CFGS
import occu_metric as occupancy_flow_metrics

# Waymo Open Dataset utilities
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids

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

# Configuration
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

NUM_PRED_CHANNELS = 4

TEST = False

feature = {
    'centerlines': tf.io.FixedLenFeature([], tf.string),
    'actors': tf.io.FixedLenFeature([], tf.string),
    'occl_actors': tf.io.FixedLenFeature([], tf.string),
    'ogm': tf.io.FixedLenFeature([], tf.string),
    'map_image': tf.io.FixedLenFeature([], tf.string),
    'scenario/id': tf.io.FixedLenFeature([], tf.string),
    'vec_flow': tf.io.FixedLenFeature([], tf.string),
}
if not TEST:
    feature.update({'gt_flow': tf.io.FixedLenFeature([], tf.string),
                    'origin_flow': tf.io.FixedLenFeature([], tf.string),
                    'gt_obs_ogm': tf.io.FixedLenFeature([], tf.string),
                    'gt_occ_ogm': tf.io.FixedLenFeature([], tf.string),
                    })

def _parse_image_function_test(example_proto):
    new_dict = {}
    try:
        d = tf.io.parse_single_example(example_proto, feature)
        new_dict['centerlines'] = tf.cast(tf.reshape(tf.io.decode_raw(d['centerlines'], tf.float64), [256, 10, 7]), tf.float32)
        new_dict['actors'] = tf.cast(tf.reshape(tf.io.decode_raw(d['actors'], tf.float64), [48, 11, 8]), tf.float32)
        new_dict['occl_actors'] = tf.cast(tf.reshape(tf.io.decode_raw(d['occl_actors'], tf.float64), [16, 11, 8]), tf.float32)

        new_dict['gt_flow'] = tf.reshape(tf.io.decode_raw(d['gt_flow'], tf.float32), [8, 512, 512, 2])[:, 128:128+256, 128:128+256, :]
        new_dict['origin_flow'] = tf.reshape(tf.io.decode_raw(d['origin_flow'], tf.float32), [8, 512, 512, 1])[:, 128:128+256, 128:128+256, :]

        new_dict['ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(d['ogm'], tf.bool), tf.float32), [512, 512, 11, 2])

        new_dict['gt_obs_ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(d['gt_obs_ogm'], tf.bool), tf.float32), [8, 512, 512, 1])[:, 128:128+256, 128:128+256, :]
        new_dict['gt_occ_ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(d['gt_occ_ogm'], tf.bool), tf.float32), [8, 512, 512, 1])[:, 128:128+256, 128:128+256, :]

        new_dict['map_image'] = tf.cast(tf.reshape(tf.io.decode_raw(d['map_image'], tf.int8), [256, 256, 3]), tf.float32) / 256
        new_dict['vec_flow'] = tf.reshape(tf.io.decode_raw(d['vec_flow'], tf.float32), [512, 512, 2])
        return new_dict
    except tf.errors.DataLossError:
        print("DataLossError encountered while parsing example. Skipping this record.")
        return None

def filter_none(dataset):
    return dataset.filter(lambda x: x is not None)

def _make_test_dataset(test_shard_path: str) -> tf.data.Dataset:
    test_dataset = tf.data.TFRecordDataset(test_shard_path)
    test_dataset = test_dataset.map(_parse_image_function_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = filter_none(test_dataset)
    test_dataset = test_dataset.batch(1)
    return test_dataset

def _make_metrics_dataset(metric_shared_path: str) -> tf.data.Dataset:
    filenames = tf.io.matching_files(metric_shared_path)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat()
    dataset = dataset.map(occupancy_flow_data.parse_tf_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1)
    return dataset

def _get_pred_waypoint_logits(model_outputs: tf.Tensor, mode_flow_outputs: tf.Tensor = None) -> occupancy_flow_grids.WaypointGrids:
    pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

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
        pred_waypoint_logits.vehicles.observed_occupancy.append(pred_observed_occupancy)
        pred_waypoint_logits.vehicles.occluded_occupancy.append(pred_occluded_occupancy)
        pred_waypoint_logits.vehicles.flow.append(pred_flow)

    return pred_waypoint_logits

def _apply_sigmoid_to_occupancy_logits(pred_waypoint_logits: occupancy_flow_grids.WaypointGrids) -> occupancy_flow_grids.WaypointGrids:
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

cfg = dict(input_size=(512, 512), window_size=8, embed_dim=96, depths=[2, 2, 2], num_heads=[3, 6, 12])
# model_path = "Saved_Model"
# model =  tf.keras.models.load_model('Saved_Model')
model = tf.saved_model.load('Saved_Model')


# print(model.signatures)
# model = loaded_model(cfg,actor_only=True,sep_actors=False, fg_msa=True, fg=True)

# def test_step(data):
#     map_img = data['map_image']
#     centerlines = data['centerlines']
#     actors = data['actors']
#     occl_actors = data['occl_actors']

#     ogm = data['ogm']
#     gt_obs_ogm = data['gt_obs_ogm']
#     gt_occ_ogm = data['gt_occ_ogm']
#     gt_flow = data['gt_flow']
#     origin_flow = data['origin_flow']

#     flow = data['vec_flow']
#     true_waypoints = _warpped_gt(gt_ogm=gt_obs_ogm, gt_occ=gt_occ_ogm, gt_flow=gt_flow, origin_flow=origin_flow)
#     outputs = model.call(ogm, map_img, training=False, obs=actors, occ=occl_actors, mapt=centerlines, flow=flow)
#     logits = _get_pred_waypoint_logits(outputs)
#     start_time = time_lib.time()
#     pred_waypoints = _apply_sigmoid_to_occupancy_logits(logits)
#     end_time = time_lib.time()
#     elapsed_time = end_time - start_time

#     return pred_waypoints, true_waypoints, elapsed_time


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
    true_waypoints = _warpped_gt(gt_ogm=gt_obs_ogm, gt_occ=gt_occ_ogm, gt_flow=gt_flow, origin_flow=origin_flow)
    concrete_func = model.signatures["serving_default"]
    inputs = {
        'mapt': centerlines,
        'args_0': ogm,
        'obs': actors,
        'occ': occl_actors,
        'args_1': map_img,
        'flow': flow
    }
    outputs = concrete_func(**inputs)
    # print(outputs)
    outputs = outputs['output_1'] 

    logits = _get_pred_waypoint_logits(outputs)
    start_time = time_lib.time()
    pred_waypoints = _apply_sigmoid_to_occupancy_logits(logits)
    end_time = time_lib.time()
    elapsed_time = end_time - start_time

    return pred_waypoints, true_waypoints, elapsed_time

def generate_true_waypoints(inputs):
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

def _warpped_gt(gt_ogm: tf.Tensor, gt_occ: tf.Tensor, gt_flow: tf.Tensor, origin_flow: tf.Tensor,) -> occupancy_flow_grids.WaypointGrids:
    true_waypoints = occupancy_flow_grids.WaypointGrids()

    for k in range(8):
        true_waypoints.vehicles.observed_occupancy.append(gt_ogm[:, k])
        true_waypoints.vehicles.occluded_occupancy.append(gt_occ[:, k])
        true_waypoints.vehicles.flow.append(gt_flow[:, k])
        true_waypoints.vehicles.flow_origin_occupancy.append(origin_flow[:, k])

    return true_waypoints

def model_testing(test_shard_path, ids, metric_shared_path):
    print(f'Creating submission for test shard {test_shard_path}...')
    test_dataset = _make_test_dataset(test_shard_path=test_shard_path)
    dataset = _make_metrics_dataset(metric_shared_path=metric_shared_path)
    print(test_dataset)
    print('\n')
    print(dataset)

    times = []
    metrics_list = []
    best_metrics = None
    best_time = float('inf')
    best_vehicles_observed_auc = float('-inf')

    dataset_iterator = iter(zip(test_dataset, dataset))

    while True:
        try:
            x, y = next(dataset_iterator)
            pred_waypoints, true_waypoints, elapsed_time = test_step(x)
            times.append(elapsed_time)

            metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
                config=config,
                true_waypoints=true_waypoints,
                pred_waypoints=pred_waypoints,
                no_warp=False
            )

            metrics_dict = MessageToDict(metrics)
            metrics_list.append(metrics_dict)

            if metrics.vehicles_observed_auc > best_vehicles_observed_auc:
                best_vehicles_observed_auc = metrics.vehicles_observed_auc
                best_metrics = metrics

            print(f"Time: {elapsed_time:.4f} seconds, Metrics: {metrics_dict}")

        except StopIteration:
            break
        except tf.errors.DataLossError:
            print("DataLossError encountered. Skipping the corrupted record.")
            continue

    average_time = np.mean(times)
    average_metrics = {key: np.mean([metric[key] for metric in metrics_list]) for key in metrics_list[0]}

    print(f"Best Time: {best_time:.4f} seconds, Best Metrics: {MessageToDict(best_metrics)}")
    print(f"Average Time: {average_time:.4f} seconds, Average Metrics: {average_metrics}")

    return len(times), best_time, best_metrics, average_time, average_metrics


def id_checking(test=True):
    if test:
        path = f'{args.ids_dir}/testing_scenario_ids.txt'
        print("testing" + args.ids_dir)

    with tf.io.gfile.GFile(path) as f:
        test_scenario_ids = f.readlines()
        test_scenario_ids = [id.rstrip() for id in test_scenario_ids]
        print(f'original ids num:{len(test_scenario_ids)}')
        test_scenario_ids = set(test_scenario_ids)
    return test_scenario_ids

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--ids_dir', type=str, help='ids.txt downloads from Waymos', default="/raid/STrajNet/waymo_open_dataset_motion_v_1_1_0/occupancy_flow_challenge/")
    parser.add_argument('--save_dir', type=str, help='saving directory', default="/raid/STrajNet/inference/")
    parser.add_argument('--file_dir', type=str, help='Test Dataset directory', default="/raid/STrajNet/preprocessed_data/val")
    parser.add_argument('--weight_path', type=str, help='Model weights directory', default="/raid/STrajNet/train_output/final_model_new.tf")
    parser.add_argument("--test_path", type=str, help="Test Directory", default="/raid/STrajNet/waymo_open_dataset_motion_v_1_1_0/tf_example/testing")
    args = parser.parse_args()
    # model.summary()
    # tf.keras.models.load_weights(args.weight_path)
    # tf.saved_model.save(model,'saved_model')
    # model.save("./Saved_Model")
    m_filesnames = tf.io.gfile.glob(args.test_path + "/*.tfrecord*")
    
    print(args.file_dir + '/*.tfrecords')
    v_filenames = tf.io.gfile.glob(args.file_dir + '/*.tfrecords')
    print(f'{len(v_filenames)} found, start loading dataset')
    combine_files = zip(m_filesnames, v_filenames)
    test_scenario_ids = None
    cnt = 0
    best_time = float('inf')
    best_metrics = None
    total_time = 0
    all_metrics = []

    for x, y in combine_files:
        num, best_time_current, best_metrics_current, average_time_current, average_metrics_current = model_testing(
            test_shard_path=y, ids=test_scenario_ids, metric_shared_path=x
        )
        cnt += num
        total_time += average_time_current * num
        all_metrics.append(average_metrics_current)

        if best_metrics_current.vehicles_observed_auc > (best_metrics.vehicles_observed_auc if best_metrics else float('-inf')):
            best_time = best_time_current
            best_metrics = best_metrics_current

    overall_average_time = total_time / cnt
    overall_average_metrics = {key: np.mean([metrics[key] for metrics in all_metrics]) for key in all_metrics[0]}

    print(f"Total Count: {cnt}")
    print(f"Best Time: {best_time:.4f} seconds, Best Metrics: {MessageToDict(best_metrics)}")
    print(f"Overall Average Time: {overall_average_time:.4f} seconds, Overall Average Metrics: {overall_average_metrics}")
