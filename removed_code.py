
#  final_test new inference file

def _add_waypoints_to_scenario_prediction(pred_waypoints: occupancy_flow_grids.WaypointGrids,
                                          scenario_prediction: occupancy_flow_submission_pb2.ScenarioPrediction,
                                          config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig) -> None:
    for k in range(config.num_waypoints):
        waypoint_message = scenario_prediction.waypoints.add()
        for tensor, attribute in [
            (pred_waypoints.vehicles.observed_occupancy[k], 'observed_vehicles_occupancy'),
            (pred_waypoints.vehicles.occluded_occupancy[k], 'occluded_vehicles_occupancy'),
            (pred_waypoints.vehicles.flow[k], 'all_vehicles_flow'),
        ]:
            quantized_tensor = tf.cast(tf.round(tensor * 255), tf.uint8)
            compressed_bytes = tf.io.encode_base64(tf.io.serialize_tensor(quantized_tensor))
            setattr(waypoint_message, attribute, compressed_bytes.numpy())

#  Line 189           


def _make_submission_proto() -> occupancy_flow_submission_pb2.ChallengeSubmission:
    submission = occupancy_flow_submission_pb2.ChallengeSubmission()
    submission.account_name = ''
    submission.unique_method_name = ''
    submission.authors.extend([''])
    submission.description = ''
    submission.method_link = ''
    return submission

def _save_submission_to_file(submission, test_shard_path, save_folder):
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



# line no 268



# DataProcessing

        
    # test_files = glob(f'{args.file_dir}/testing/*')
    # print(f'Processing testing data...{len(test_files)} found!')
    # print('Starting processing pooling...')
    # with Pool(NUM_POOLS) as p:
    #     p.map(process_test_data, test_files)

    # train_files = glob(f'{args.file_dir}/training/*')
    # print(f'Processing training data...{len(train_files)} found!')
    # print('Starting processing pooling...')
    # with Pool(NUM_POOLS) as p:
    #     p.map(process_training_data, train_files)



# Line 506 at last


# Modules File
