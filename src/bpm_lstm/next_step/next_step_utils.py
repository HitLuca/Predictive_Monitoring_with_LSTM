import numpy as np
from keras.utils import to_categorical


def build_train_test_datasets(dataset, additional_features, max_activity_id, max_resource_id, test_split, subsequences_step=2):
    # activity_id, resource_id, time_since_case_start
    dataset_elements = [to_categorical(dataset[..., 0], max_activity_id + 1),
                        to_categorical(dataset[..., 1], max_resource_id + 1),
                        dataset[..., 2:]]

    additional_features = np.concatenate((additional_features[..., :2],
                                          to_categorical(additional_features[..., 2], 7)), axis=-1)

    split_index = int(dataset.shape[0] * test_split)

    X_train_temp = np.concatenate(dataset_elements, axis=-1)[split_index:]
    X2_train_temp = additional_features[split_index:]

    X_test = np.concatenate(dataset_elements, axis=-1)[:split_index]
    X2_test = additional_features[:split_index]

    X_train = None
    X2_train = None
    Y_train = None

    for i in range(1, X_train_temp.shape[1]):
        X_temp = np.zeros(X_train_temp.shape)
        X_temp[:, -i:, :] = X_train_temp[:, :i]

        X2_temp = np.zeros(X2_train_temp.shape)
        X2_temp[:, -i:, :] = X2_train_temp[:, :i]

        Y_temp = [X_train_temp[:, i, :max_activity_id+1], X_train_temp[:, i, max_activity_id+1:max_activity_id+max_resource_id+2], X_train_temp[:, i, -1]]
        if X_train is None:
            X_train = X_temp
            X2_train = X2_temp
            Y_train = Y_temp
        else:
            X_train = np.concatenate([X_train, X_temp])
            Y_train = [np.concatenate([Y_train[0], Y_temp[0]]), np.concatenate([Y_train[1], Y_temp[1]]), np.concatenate([Y_train[2], Y_temp[2]])]
            X2_train = np.concatenate([X2_train, X2_temp])

    return ([X_train, X2_train], Y_train), ([X_test, X2_test])


def discretize_softmax(sample):
    return np.expand_dims(np.eye(sample.shape[-1])[np.argmax(sample)], 0)
