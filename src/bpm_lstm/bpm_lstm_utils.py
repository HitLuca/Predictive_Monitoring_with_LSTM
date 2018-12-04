import csv
from datetime import datetime
import numpy as np
from keras.utils import to_categorical


def _find_dataset_limits(dataset, additional_features):
    max_activity_id = 0
    max_resource_id = 0
    max_time_since_case_start = 0
    max_time_since_last_event = 0
    max_case_length = 0

    for case, features in zip(dataset, additional_features):
        if len(case) > max_case_length:
            max_case_length = len(case)

        for step_case, step_features in zip(case, features):
            (activity_id, resource_id, time_since_case_start) = step_case
            (time_since_last_event, _, _) = step_features
            if activity_id > max_activity_id:
                max_activity_id = activity_id
            if resource_id > max_resource_id:
                max_resource_id = resource_id
            if time_since_case_start > max_time_since_case_start:
                max_time_since_case_start = time_since_case_start
            if time_since_last_event > max_time_since_last_event:
                max_time_since_last_event = time_since_last_event

    return max_activity_id, max_resource_id, max_time_since_case_start, max_time_since_last_event, max_case_length


def _pad_dataset(dataset, additional_features, limits):
    # activity_id, resource_id, time_since_case_start, time_since_last_event, time_since_midnight, day_of_week
    max_activity_id, max_resource_id, max_time_since_case_start, max_time_since_last_event, max_case_length = limits

    padding_element_dataset = (max_activity_id + 1, max_resource_id + 1, 0)
    padding_element_additional_features = (0, 0, 0)

    for i in range(len(dataset)):
        if len(dataset[i]) < max_case_length + 1:
            for j in range(len(dataset[i]), max_case_length + 2):
                dataset[i].append(padding_element_dataset)
                additional_features[i].append(padding_element_additional_features)
    return dataset, additional_features


def _normalize_and_convert_dataset(dataset, additional_features, limits):
    dataset = np.array(dataset, dtype=np.float)
    additional_features = np.array(additional_features, dtype=np.float)

    # activity_id, resource_id, time_since_case_start, time_since_last_event, time_since_midnight, day_of_week
    max_activity_id, max_resource_id, max_time_since_case_start, max_time_since_last_event, max_case_length = limits

    dataset[:, :, 2] = dataset[:, :, 2] / (max_time_since_case_start)
    additional_features[:, :, :2] = additional_features[:, :, :2] / np.array([max_time_since_case_start, 86400])

    return dataset, additional_features


def load_dataset_with_features(log_filepath):
    dataset = []
    additional_features = []

    current_case = []
    current_additional_features = []

    current_case_id = None
    current_case_start_time = None
    last_event_time = None

    log_file = open(log_filepath, 'r')
    csv_reader = csv.reader(log_file, delimiter=',', quotechar='|')
    next(csv_reader, None)

    for row in csv_reader:  # CaseID, ActivityID, Timestamp, ResourceID
        case_id = row[0]
        activity_id = int(row[1])
        timestamp = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S')
        resource_id = int(row[3])

        if case_id != current_case_id:
            if len(current_case) > 0:
                dataset.append(current_case)
                additional_features.append(current_additional_features)
            current_case = []
            current_additional_features = []
            current_case_id = case_id
            current_case_start_time = timestamp
            last_event_time = timestamp

        time_since_case_start = int((timestamp - current_case_start_time).total_seconds())
        time_since_last_event = int((timestamp - last_event_time).total_seconds())
        time_since_midnight = int(
            (timestamp - timestamp.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())
        day_of_week = timestamp.weekday()

        current_case.append(
            (activity_id, resource_id, time_since_case_start))
        current_additional_features.append((time_since_last_event, time_since_midnight, day_of_week))

    limits = _find_dataset_limits(dataset, additional_features)
    max_activity_id, max_resource_id, _, _, _ = limits

    dataset, additional_features = _pad_dataset(dataset, additional_features, limits)
    dataset, additional_features = _normalize_and_convert_dataset(dataset, additional_features, limits)

    return dataset, additional_features, max_activity_id + 1, max_resource_id + 1


def build_train_test_datasets(dataset, additional_features, max_activity_id, max_resource_id):
    # activity_id, resource_id, time_since_case_start
    X = dataset[:, :-1, :]
    X = np.concatenate((to_categorical(X[:, :, 0], max_activity_id + 1),
                        to_categorical(X[:, :, 1], max_resource_id + 1),
                        X[:, :, 2:]), axis=-1)

    # time_since_last_event, time_since_midnight, day_of_week
    X2 = additional_features[:, :-1, :]
    X2 = np.concatenate((X2[:, :, :2], to_categorical(X2[:, :, 2], 7),), axis=-1)

    Y = dataset[:, 1:, :]
    Y = [to_categorical(Y[:, :, 0], max_activity_id + 1),
         to_categorical(Y[:, :, 1], max_resource_id + 1),
         Y[:, :, 2:]]

    return [X, X2], Y
