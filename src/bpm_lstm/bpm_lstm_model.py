import os
from datetime import datetime

import distance
import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, LSTM, TimeDistributed, Concatenate
from keras.optimizers import Nadam
from sklearn.model_selection import KFold

from . import bpm_lstm_utils


# noinspection PyAttributeOutsideInit
class BPM_LSTM:
    def __init__(self, log_name, log_filepath, model_name, output_filepath='../outputs', logs_filepath='logs/',
                 validation_split=0.2, test_split=0.1, test_prefix_size=5):
        self._model_name = model_name
        self._log_name = log_name
        self._log_filepath = log_filepath
        self._output_filepath = output_filepath
        self._logs_filepath = logs_filepath
        self._validation_split = validation_split
        self._test_split = test_split
        self._test_prefix_size = test_prefix_size

    def _build_model(self):
        model_input = Input((self._X_train[0].shape[1:]))
        additional_features_input = Input((self._X_train[1].shape[1:]))

        processed = Concatenate()([model_input, additional_features_input])

        processed = TimeDistributed(Dense(32))(processed)
        processed = BatchNormalization()(processed)
        processed = LeakyReLU()(processed)
        processed = Dropout(0.5)(processed)

        processed = LSTM(64, return_sequences=True)(processed)

        processed = TimeDistributed(Dense(32))(processed)
        processed = BatchNormalization()(processed)
        processed = LeakyReLU()(processed)
        processed = Dropout(0.5)(processed)

        activity_id_output = Dense(self._max_activity_id + 1, activation='softmax', name='activity_id')(processed)
        resource_id_output = Dense(self._max_resource_id + 1, activation='softmax', name='resource_id')(processed)
        time_output = Dense(1, activation='relu', name='time')(processed)

        self._model = Model([model_input, additional_features_input],
                            [activity_id_output, resource_id_output, time_output])

        self._model.compile(loss={'activity_id': 'categorical_crossentropy',
                                  'resource_id': 'categorical_crossentropy',
                                  'time': 'mse'},
                            metrics={'activity_id': 'categorical_accuracy',
                                     'resource_id': 'categorical_accuracy',
                                     'time': 'mse'},
                            optimizer='adam')

    def _build_model_old(self):
        model_input = Input((self._X_train[0].shape[1:]))
        additional_features_input = Input((self._X_train[1].shape[1:]))

        processed = Concatenate()([model_input, additional_features_input])

        processed = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        processed = BatchNormalization()(processed)

        activity_id_output = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        activity_id_output = BatchNormalization()(activity_id_output)

        resource_id_output = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        resource_id_output = BatchNormalization()(resource_id_output)

        time_output = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        time_output = BatchNormalization()(time_output)

        activity_id_output = TimeDistributed(Dense(self._max_activity_id + 1, activation='softmax'),
                                             name='activity_id')(
            activity_id_output)
        resource_id_output = TimeDistributed(Dense(self._max_resource_id + 1, activation='softmax'),
                                             name='resource_id')(
            resource_id_output)
        time_output = TimeDistributed(Dense(1, activation='relu'), name='time')(time_output)

        self._model = Model([model_input, additional_features_input],
                            [activity_id_output, resource_id_output, time_output])

        opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

        self._model.compile(loss={'activity_id': 'categorical_crossentropy',
                                  'resource_id': 'categorical_crossentropy',
                                  'time': 'mse'},
                            metrics={'activity_id': 'categorical_accuracy',
                                     'resource_id': 'categorical_accuracy',
                                     'time': 'mse'},
                            optimizer=opt)

    def _train_model(self, checkpoint_name, log_path, train_indexes):
        model_checkpoint = ModelCheckpoint(checkpoint_name, save_best_only=True)
        early_stopping = EarlyStopping(patience=10)
        tensorboard = TensorBoard(log_dir=log_path)

        self._model.fit([self._X_train[0][train_indexes], self._X_train[1][train_indexes]],
                        [self._Y_train[0][train_indexes], self._Y_train[1][train_indexes],
                         self._Y_train[2][train_indexes]],
                        validation_split=self._validation_split,
                        verbose=0,
                        callbacks=[model_checkpoint, early_stopping, tensorboard],
                        epochs=300)

    def _evaluate_model_validation(self, validation_indexes):
        return self._model.evaluate([self._X_train[0][validation_indexes], self._X_train[1][validation_indexes]],
                                    [self._Y_train[0][validation_indexes], self._Y_train[1][validation_indexes],
                                     self._Y_train[2][validation_indexes]],
                                    verbose=0)

    def _evaluate_model_test(self):
        test_scores = [[], []]

        for sample, additional_features in zip(self._X_test, self._X2_test):
            predicted_trace = np.expand_dims(np.zeros(sample.shape), 0)
            predicted_trace[:, :self._test_prefix_size] = sample[:self._test_prefix_size:]

            for i in range(self._test_prefix_size, self._X_train[0].shape[1]):
                activity_id, resource_id, time = self._model.predict(
                    [predicted_trace, np.expand_dims(additional_features, 0)])
                activity_id = bpm_lstm_utils.discretize_softmax(activity_id)
                resource_id = bpm_lstm_utils.discretize_softmax(resource_id)
                model_prediction = np.concatenate([activity_id, resource_id, time], axis=-1)
                predicted_trace[:, i] = model_prediction[:, i]

            sample_activity = np.argmax(sample[:, :self._max_activity_id + 1], 1)
            sample_activity_string = ''.join(str(e) for e in sample_activity.tolist())

            predicted_activity = np.argmax(predicted_trace[:, :, :self._max_activity_id + 1], 2)[0]
            predicted_activity_string = ''.join(str(e) for e in predicted_activity.tolist())

            sample_resource = np.argmax(
                sample[:, self._max_activity_id + 1:self._max_activity_id + self._max_resource_id + 2], 1)
            sample_resource_string = ''.join(str(e) for e in sample_resource.tolist())

            predicted_resource = np.argmax(
                predicted_trace[:, :, self._max_activity_id + 1:self._max_activity_id + self._max_resource_id + 2], 2)[
                0]
            predicted_resource_string = ''.join(str(e) for e in predicted_resource.tolist())

            sample_time = sample[:, -1]
            predicted_time = predicted_trace[:, :, -1][0]

            test_scores[0].append((1 - distance.nlevenshtein(sample_activity_string, predicted_activity_string)))
            test_scores[1].append((1 - distance.nlevenshtein(sample_resource_string, predicted_resource_string)))

        test_scores = np.array(test_scores)
        return np.mean(test_scores, -1)

    def _create_checkpoints_path(self, fold):
        folder_path = '/'.join(
            [self._output_filepath, self._model_name, self._log_name, str(fold), ''])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        checkpoint_filepath = folder_path + 'model_{epoch:03d}-{val_loss:.3f}.h5'
        return checkpoint_filepath

    def _create_logs_path(self, fold):
        folder_path = '/'.join(
            [self._output_filepath, self._model_name, self._log_name, str(fold),
             self._logs_filepath, ''])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path

    def train(self, folds):
        dataset, additional_features, self._max_activity_id, self._max_resource_id = bpm_lstm_utils.load_dataset_with_features(
            self._log_filepath, shuffle=True)

        (self._X_train, self._Y_train), (self._X_test, self._X2_test) = bpm_lstm_utils.build_train_test_datasets(
            dataset,
            additional_features,
            self._max_activity_id,
            self._max_resource_id,
            self._test_split)

        kfold = KFold(n_splits=folds, shuffle=True)
        model_scores = {'validation': [],
                        'test': []}

        fold = 0
        for train_indexes, validation_indexes in kfold.split(self._X_train[0]):
            checkpoint_filepath = self._create_checkpoints_path(fold)
            log_path = self._create_logs_path(fold)

            self._build_model()
            self._train_model(checkpoint_filepath, log_path, train_indexes)
            validation_scores = self._evaluate_model_validation(validation_indexes)
            model_scores['validation'].append(
                (validation_scores[4], validation_scores[5], validation_scores[6]))

            test_scores = self._evaluate_model_test()
            model_scores['test'].append(test_scores)

            fold += 1
        model_scores['validation'] = np.array(model_scores['validation'])
        model_scores['test'] = np.array(model_scores['test'])

        results_filepath = '/'.join([self._output_filepath, self._model_name, self._log_name, ''])
        bpm_lstm_utils.save_results(results_filepath, model_scores)

