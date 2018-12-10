import abc
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


class BPM_LSTM(abc.ABC):
    def __init__(self, log_name, log_filepath, model_name, write_logs, output_filepath,
                 logs_filepath, validation_split, test_split, test_prefix_size):
        self._model_type = NotImplemented
        self._model = NotImplemented
        self._X_train = NotImplemented
        self._Y_train = NotImplemented
        self._results_filepath = NotImplemented

        self._model_name = model_name
        self._log_name = log_name
        self._log_filepath = log_filepath
        self._write_logs = write_logs
        self._output_filepath = output_filepath
        self._logs_filepath = logs_filepath
        self._validation_split = validation_split
        self._test_split = test_split
        self._test_prefix_size = test_prefix_size

    @abc.abstractmethod
    def _evaluate_model_test(self):
        pass

    @abc.abstractmethod
    def train(self, folds):
        pass

    def _create_checkpoints_path(self, fold):
        folder_path = '/'.join(
            [self._output_filepath, self._model_type, self._model_name, self._log_name, str(fold), ''])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        checkpoint_filepath = folder_path + 'model_{epoch:03d}-{val_loss:.3f}.h5'
        return checkpoint_filepath

    def _create_logs_path(self, fold):
        folder_path = '/'.join(
            [self._output_filepath, self._model_type, self._model_name, self._log_name, str(fold),
             self._logs_filepath, ''])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path

    def _train_model(self, checkpoint_name, log_path, train_indexes, validation_indexes):
        callbacks = []
        callbacks.append(ModelCheckpoint(checkpoint_name, save_best_only=True))
        callbacks.append(EarlyStopping(patience=10))
        if self._write_logs:
            callbacks.append(TensorBoard(log_dir=log_path))

        return self._model.fit([self._X_train[0][train_indexes], self._X_train[1][train_indexes]],
                               [self._Y_train[0][train_indexes], self._Y_train[1][train_indexes],
                                self._Y_train[2][train_indexes]],
                               validation_data=(
                                   [self._X_train[0][validation_indexes], self._X_train[1][validation_indexes]],
                                   [self._Y_train[0][validation_indexes], self._Y_train[1][validation_indexes],
                                    self._Y_train[2][validation_indexes]]),
                               verbose=2,
                               callbacks=callbacks,
                               epochs=300)
