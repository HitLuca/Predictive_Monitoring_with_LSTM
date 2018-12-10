import numpy as np
from keras.utils import to_categorical


def build_train_test_datasets(dataset, additional_features, max_activity_id, max_resource_id, test_split, delete_last):
    dataset = dataset[:, :-delete_last, :]
    additional_features = additional_features[:, :-delete_last, :]

    # activity_id, resource_id, time_since_case_start
    dataset_elements = [to_categorical(dataset[..., 0], max_activity_id + 1),
                        to_categorical(dataset[..., 1], max_resource_id + 1),
                        dataset[..., 2:]]

    additional_features = np.concatenate((additional_features[..., :2],
                                          to_categorical(additional_features[..., 2], 7)), axis=-1)

    split_index = int(dataset.shape[0] * test_split)

    X_train = np.concatenate(dataset_elements, axis=-1)[split_index:, :-1]
    X2_train = additional_features[split_index:, :-1]

    Y_train = [dataset_elements[0][split_index:, -1], dataset_elements[1][split_index:, -1],
               dataset_elements[2][split_index:, -1]]

    X_test = np.concatenate(dataset_elements, axis=-1)[:split_index, :-1]
    X2_test = additional_features[:split_index, :-1]

    Y_test = [dataset_elements[0][:split_index, -1], dataset_elements[1][:split_index, -1],
              dataset_elements[2][:split_index, -1]]

    return ([X_train, X2_train], Y_train), ([X_test, X2_test], Y_test)


def discretize_softmax(sample):
    return np.expand_dims(np.eye(sample.shape[-1])[np.argmax(sample)], 0)
