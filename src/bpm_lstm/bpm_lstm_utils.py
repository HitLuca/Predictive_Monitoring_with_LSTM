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

    dataset[:, :, 2] = dataset[:, :, 2] / max_time_since_case_start
    additional_features[:, :, :2] = additional_features[:, :, :2] / np.array([max_time_since_case_start, 86400])

    return dataset, additional_features


def load_dataset_with_features(log_filepath, shuffle=True):
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

    if shuffle:
        shuffled_indexes = np.random.permutation(range(dataset.shape[0]))
        dataset = dataset[shuffled_indexes]
        additional_features = additional_features[shuffled_indexes]

    return dataset, additional_features, max_activity_id + 1, max_resource_id + 1


def build_train_test_datasets(dataset, additional_features, max_activity_id, max_resource_id, test_split):
    # activity_id, resource_id, time_since_case_start
    dataset_elements = [to_categorical(dataset[..., 0], max_activity_id + 1),
                        to_categorical(dataset[..., 1], max_resource_id + 1),
                        dataset[..., 2:]]

    additional_features = np.concatenate((additional_features[..., :2],
                                          to_categorical(additional_features[..., 2], 7)), axis=-1)

    split_index = int(dataset.shape[0] * test_split)

    X_train = np.concatenate(dataset_elements, axis=-1)[split_index:, :-1]
    X2_train = additional_features[split_index:, :-1]

    Y_train = [dataset_elements[0][split_index:, 1:], dataset_elements[1][split_index:, 1:],
               dataset_elements[2][split_index:, 1:]]

    X_test = np.concatenate(dataset_elements, axis=-1)[:split_index, :-1]
    X2_test = additional_features[:split_index, :-1]

    return ([X_train, X2_train], Y_train), (X_test, X2_test)


def discretize_softmax(sample):
    return np.eye(sample.shape[-1])[np.argmax(sample, 2)]


def save_results(output_filepath, model_scores):
    with open(output_filepath + 'results.csv', 'w') as f:
        print('validation,accuracy,std', file=f)
        print('activity_id,%.4f,%.4f' % (
            float(np.mean(model_scores['validation'][:, 0])), float(np.std(model_scores['validation'][:, 0]))), file=f)
        print('resource_id,%.4f,%.4f' % (
            float(np.mean(model_scores['validation'][:, 1])), float(np.std(model_scores['validation'][:, 1]))), file=f)
        print(',time,mse,std', file=f)
        print('time,%.4f,%.4f' % (
            float(np.mean(model_scores['validation'][:, 2])), float(np.std(model_scores['validation'][:, 2]))), file=f)
        print(file=f)
        print('test,nlevenshtein,std', file=f)
        print('activity_id,%.4f,%.4f' % (
            float(np.mean(model_scores['test'][:, 0])), float(np.std(model_scores['test'][:, 0]))), file=f)
        print('resource_id,%.4f,%.4f' % (
            float(np.mean(model_scores['test'][:, 1])), float(np.std(model_scores['test'][:, 1]))), file=f)
