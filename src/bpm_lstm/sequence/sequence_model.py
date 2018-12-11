import distance
import numpy as np
from sklearn.model_selection import ShuffleSplit

from bpm_lstm.bpm_lstm_model import BPM_LSTM
from bpm_lstm.bpm_lstm_utils import save_results, load_dataset_with_features
from bpm_lstm.sequence.sequence_models import available_models
from bpm_lstm.sequence.sequence_utils import discretize_softmax, build_train_test_datasets


# noinspection PyAttributeOutsideInit
class BPM_LSTM_SEQUENCE(BPM_LSTM):
    def __init__(self, log_name, log_filepath, write_logs, model_name='new_model_v2', output_filepath='../outputs',
                 logs_filepath='logs/', validation_split=0.2, test_split=0.1, test_prefix_size=5):
        super().__init__(log_name, log_filepath, model_name, write_logs, output_filepath,
                         logs_filepath, validation_split, test_split, test_prefix_size)
        self._model_type = 'sequence'
        self._results_filepath = '/'.join(
            [self._output_filepath, self._model_type, self._model_name, self._log_name, ''])

    def _evaluate_model_test(self):
        test_scores = [[], []]

        for sample, additional_features in zip(self._X_test, self._X2_test):
            predicted_trace = np.expand_dims(np.zeros(sample.shape), 0)
            predicted_trace[:, :self._test_prefix_size] = sample[:self._test_prefix_size:]

            for i in range(self._test_prefix_size, self._X_train[0].shape[1]):
                activity_id, resource_id, time = self._model.predict(
                    [predicted_trace, np.expand_dims(additional_features, 0)])
                activity_id = discretize_softmax(activity_id)
                resource_id = discretize_softmax(resource_id)
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

    def train(self, folds):
        dataset, additional_features, self._max_activity_id, self._max_resource_id = load_dataset_with_features(
            self._log_filepath, shuffle=True)

        (self._X_train, self._Y_train), (self._X_test, self._X2_test) = build_train_test_datasets(
            dataset,
            additional_features,
            self._max_activity_id,
            self._max_resource_id,
            self._test_split)

        kfold = ShuffleSplit(n_splits=folds, test_size=0.2)
        model_scores = {'validation': [],
                        'test': []}

        fold = 0
        for train_indexes, validation_indexes in kfold.split(self._X_train[0]):
            checkpoint_filepath = self._create_checkpoints_path(fold)
            log_path = self._create_logs_path(fold)

            self._model = available_models[self._model_name](self._X_train, self._max_activity_id,
                                                             self._max_resource_id)
            history = self._train_model(checkpoint_filepath, log_path, train_indexes, validation_indexes)
            validation_scores = history.history
            model_scores['validation'].append(
                (validation_scores['val_activity_id_categorical_accuracy'][-1],
                 validation_scores['val_resource_id_categorical_accuracy'][-1],
                 validation_scores['val_time_mean_squared_error'][-1]))

            test_scores = self._evaluate_model_test()
            model_scores['test'].append(test_scores)

            fold += 1
        model_scores['validation'] = np.array(model_scores['validation'])
        model_scores['test'] = np.array(model_scores['test'])

        save_results(self._results_filepath, model_scores)
