import os
import pathlib
from typing import Sequence
import uuid
import zlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from imageio_ffmpeg import get_ffmpeg_exe
import tensorflow as tf
import tensorflow_graphics.image.transformer as tfg_transformer

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils import occupancy_flow_metrics
from waymo_open_dataset.utils import occupancy_flow_renderer
from waymo_open_dataset.utils import occupancy_flow_vis


NUM_PRED_CHANNELS = 4


def _apply_sigmoid_to_occupancy_logits(
    pred_waypoint_logits: occupancy_flow_grids.WaypointGrids,
) -> occupancy_flow_grids.WaypointGrids:
  """Converts occupancy logits to probabilities."""
  pred_waypoints = occupancy_flow_grids.WaypointGrids()
  pred_waypoints.vehicles.observed_occupancy = [
      tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.observed_occupancy
  ]
  pred_waypoints.vehicles.occluded_occupancy = [
      tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.occluded_occupancy
  ]
  pred_waypoints.vehicles.flow = pred_waypoint_logits.vehicles.flow
  return pred_waypoints


def _make_model_inputs(
    timestep_grids: occupancy_flow_grids.TimestepGrids,
    vis_grids: occupancy_flow_grids.VisGrids,
) -> tf.Tensor:
  """Concatenates all occupancy grids over past, current to a single tensor."""
  model_inputs = tf.concat(
      [
          vis_grids.roadgraph,
          timestep_grids.vehicles.past_occupancy,
          timestep_grids.vehicles.current_occupancy,
          tf.clip_by_value(
              timestep_grids.pedestrians.past_occupancy
              + timestep_grids.cyclists.past_occupancy,
              0,
              1,
          ),
          tf.clip_by_value(
              timestep_grids.pedestrians.current_occupancy
              + timestep_grids.cyclists.current_occupancy,
              0,
              1,
          ),
      ],
      axis=-1,
  )
  return model_inputs


def _get_pred_waypoint_logits(
    model_outputs: tf.Tensor,
) -> occupancy_flow_grids.WaypointGrids:
  """Slices model predictions into occupancy and flow grids."""
  pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

  # Slice channels into output predictions.
  for k in range(config.num_waypoints):
    index = k * 4
    waypoint_channels = model_outputs[
        :, :, :, index : index + 4
    ]
    pred_observed_occupancy = waypoint_channels[:, :, :, :1]
    pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
    pred_flow = waypoint_channels[:, :, :, 2:]
    pred_waypoint_logits.vehicles.observed_occupancy.append(
        pred_observed_occupancy
    )
    pred_waypoint_logits.vehicles.occluded_occupancy.append(
        pred_occluded_occupancy
    )
    pred_waypoint_logits.vehicles.flow.append(pred_flow)

  return pred_waypoint_logits


def _make_model_inputs(
    timestep_grids: occupancy_flow_grids.TimestepGrids,
    vis_grids: occupancy_flow_grids.VisGrids,
) -> tf.Tensor:
  """Concatenates all occupancy grids over past, current to a single tensor."""
  model_inputs = tf.concat(
      [
          vis_grids.roadgraph,
          timestep_grids.vehicles.past_occupancy,
          timestep_grids.vehicles.current_occupancy,
          tf.clip_by_value(
              timestep_grids.pedestrians.past_occupancy
              + timestep_grids.cyclists.past_occupancy,
              0,
              1,
          ),
          tf.clip_by_value(
              timestep_grids.pedestrians.current_occupancy
              + timestep_grids.cyclists.current_occupancy,
              0,
              1,
          ),
      ],
      axis=-1,
  )
  return model_inputs


# def _make_model(
#     model_inputs: tf.Tensor,
#     config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
# ) -> tf.keras.Model:
#   """Simple convolutional model."""
#   inputs = tf.keras.Input(tensor=model_inputs)

#   encoder = tf.keras.applications.ResNet50V2(
#       include_top=False, weights=None, input_tensor=inputs
#   )

#   num_output_channels = NUM_PRED_CHANNELS * config.num_waypoints
#   decoder_channels = [32, 64, 128, 256, 512]

#   conv2d_kwargs = {
#       'kernel_size': 3,
#       'strides': 1,
#       'padding': 'same',
#   }

#   x = encoder(inputs)

#   for i in [4, 3, 2, 1, 0]:
#     x = tf.keras.layers.Conv2D(
#         filters=decoder_channels[i],
#         activation='relu',
#         name=f'upconv_{i}_0',
#         **conv2d_kwargs,
#     )(x)
#     x = tf.keras.layers.UpSampling2D(name=f'upsample_{i}')(x)
#     x = tf.keras.layers.Conv2D(
#         filters=decoder_channels[i],
#         activation='relu',
#         name=f'upconv_{i}_1',
#         **conv2d_kwargs,
#     )(x)

#   outputs = tf.keras.layers.Conv2D(
#       filters=num_output_channels,
#       activation=None,
#       name=f'outconv',
#       **conv2d_kwargs,
#   )(x)

#   return tf.keras.Model(
#       inputs=inputs, outputs=outputs, name='occupancy_flow_model'
#   )




from modules import STrajNet
cfg=dict(input_size=(512,512), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
model = STrajNet(cfg,sep_actors=False)


DATASET_FOLDER = './waymo_open_dataset_motion_v_1_1_0'

# TFRecord dataset.
TRAIN_FILES = f'{DATASET_FOLDER}/tf_example/training/training_tfexample.tfrecord*'
TRAIN_FILES = f'{DATASET_FOLDER}/tf_example/training/training_tfexample.tfrecord*'
VAL_FILES = f'{DATASET_FOLDER}/tf_example/validation/validation_tfexample.tfrecord*'
TEST_FILES = f'{DATASET_FOLDER}/tf_example/testing/testing_tfexample.tfrecord*'
WEIGHT_FILES = "/raid/STrajNet/train_output/final_model.tf"
model.load_weights(WEIGHT_FILES)
# Text files containing validation and test scenario IDs for this challenge.
VAL_SCENARIO_IDS_FILE = f'{DATASET_FOLDER}/occupancy_flow_challenge/validation_scenario_ids.txt'
TEST_SCENARIO_IDS_FILE = f'{DATASET_FOLDER}/occupancy_flow_challenge/testing_scenario_ids.txt'

filenames = tf.io.matching_files(TRAIN_FILES)
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.repeat()
dataset = dataset.map(occupancy_flow_data.parse_tf_example)
dataset = dataset.batch(1)
it = iter(dataset)

inputs = next(it)

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


inputs = occupancy_flow_data.add_sdc_fields(inputs)

timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
    inputs=inputs, config=config
)

true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
    timestep_grids=timestep_grids, config=config
)


vis_grids = occupancy_flow_grids.create_ground_truth_vis_grids(
    inputs=inputs, timestep_grids=timestep_grids, config=config
)

def test_step(data):
    map_img = data['map_image']
    centerlines = data['centerlines']
    actors = data['actors']
    occl_actors = data['occl_actors']
    ogm = data['ogm']
    flow = data['vec_flow']

    outputs = model.call(ogm,map_img,training=False,obs=actors,occ=occl_actors,mapt=centerlines,flow=flow)
    logits = _get_pred_waypoint_logits(outputs)
    pred_waypoints = _apply_sigmoid_to_occupancy_logits(logits)

    return pred_waypoints
# [batch_size, grid_height_cells, grid_width_cells, 23]
model_inputs = _make_model_inputs(timestep_grids, vis_grids)
# [batch_size, grid_height_cells, grid_width_cells, 32]
print(model_inputs)

# model = _make_model(model_inputs=model_inputs, config=config)
model_outputs = test_step(model_inputs)
pred_waypoint_logits = _get_pred_waypoint_logits(model_outputs)
pred_waypoints = _apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)



metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
    config=config,
    true_waypoints=true_waypoints,
    pred_waypoints=pred_waypoints,
)


print('Metrics:')
print(metrics)