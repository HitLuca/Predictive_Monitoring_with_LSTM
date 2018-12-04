from datetime import datetime
import os
import numpy as np

from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, LSTM, TimeDistributed, Concatenate
from keras.optimizers import Nadam
from sklearn.model_selection import KFold

from . import bpm_lstm_utils

# noinspection PyAttributeOutsideInit
class BPM_LSTM:
    def __init__(self, log_name, log_filepath, model_name, output_filepath='../outputs/', logs_filepath='logs/'):
        self._model_name = model_name
        self._log_name = log_name
        self._log_filepath = log_filepath
        self._output_filepath = output_filepath
        self._logs_filepath = logs_filepath
        self._initialization_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    def _build_model(self, max_activity_id, max_resource_id):
        model_input = Input((self._X[0].shape[1:]))
        additional_features_input = Input((self._X[1].shape[1:]))

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

        activity_id_output = Dense(max_activity_id + 1, activation='softmax', name='activity_id')(processed)
        resource_id_output = Dense(max_resource_id + 1, activation='softmax', name='resource_id')(processed)
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

    def _build_model_old(self, max_activity_id, max_resource_id):
        model_input = Input((self._X[0].shape[1:]))
        additional_features_input = Input((self._X[1].shape[1:]))

        processed = Concatenate()([model_input, additional_features_input])

        processed = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        processed = BatchNormalization()(processed)

        activity_id_output = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        activity_id_output = BatchNormalization()(activity_id_output)

        resource_id_output = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        resource_id_output = BatchNormalization()(resource_id_output)

        time_output = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        time_output = BatchNormalization()(time_output)

        activity_id_output = TimeDistributed(Dense(max_activity_id + 1, activation='softmax'), name='activity_id')(activity_id_output)
        resource_id_output = TimeDistributed(Dense(max_resource_id + 1, activation='softmax'), name='resource_id')(resource_id_output)
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

        self._model.fit([self._X[0][train_indexes], self._X[1][train_indexes]],
                        [self._Y[0][train_indexes], self._Y[1][train_indexes], self._Y[2][train_indexes]],
                        validation_split=0.2,
                        verbose=0,
                        callbacks=[model_checkpoint, early_stopping, tensorboard],
                        epochs=300)

    def _evaluate_model(self, test_indexes):
        return self._model.evaluate([self._X[0][test_indexes], self._X[1][test_indexes]],
                                    [self._Y[0][test_indexes], self._Y[1][test_indexes], self._Y[2][test_indexes]],
                                    verbose=0)

    def _create_checkpoints_path(self, fold):
        folder_path = '/'.join([self._output_filepath, self._model_name, self._log_name, self._initialization_datetime, str(fold), ''])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        checkpoint_filepath = folder_path + 'model_{epoch:03d}-{val_loss:.3f}.h5'
        return checkpoint_filepath

    def _create_logs_path(self, fold):
        folder_path = '/'.join([self._output_filepath, self._model_name, self._log_name, self._initialization_datetime, str(fold), self._logs_filepath, ''])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path

    def train(self, folds):
        dataset, additional_features, max_activity_id, max_resource_id = bpm_lstm_utils.load_dataset_with_features(
            self._log_filepath)

        self._X, self._Y = bpm_lstm_utils.build_train_test_datasets(dataset, additional_features, max_activity_id, max_resource_id)

        kfold = KFold(n_splits=folds, shuffle=True)
        cvscores = []

        fold = 0

        for train_indexes, test_indexes in kfold.split(self._X[0]):
            checkpoint_filepath = self._create_checkpoints_path(fold)
            log_path = self._create_logs_path(fold)

            self._build_model_old(max_activity_id, max_resource_id)
            self._train_model(checkpoint_filepath, log_path, train_indexes)
            scores = self._evaluate_model(test_indexes)

            cvscores.append((scores[4] * 100, scores[5] * 100, scores[6]))
            fold += 1

        cvscores = np.array(cvscores)
        print("activity_id accuracy %.2f%% (+/- %.2f%%)" % (np.mean(cvscores[:, 0]), np.std(cvscores[:, 0])))
        print("resource_id accuracy %.2f%% (+/- %.2f%%)" % (np.mean(cvscores[:, 1]), np.std(cvscores[:, 1])))
        print("time mse %.4f" % (np.mean(cvscores[:, 2])))
