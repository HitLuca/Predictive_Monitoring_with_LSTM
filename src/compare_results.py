import csv
from matplotlib import pyplot as plt
import numpy as np

log_names = [
    '10x5_1S',
    '10x5_1W',
    '10x5_3S',
    '10x5_3W',
    '5x5_1W',
    '5x5_1S',
    '5x5_3W',
    '5x5_3S',
    '10x20_1W',
    '10x20_1S',
    '10x20_3W',
    '10x20_3S',
    '10x2_1W',
    # '10x2_1S',
    # '10x2_3W',
    # '10x2_3S',
    # '50x5_1W',
    # '50x5_1S',
    # '50x5_3W',
    # '50x5_3S'
]

# validation, accuracy, std
# activity_id, 0.9108, 0.0016
# resource_id, 0.8538, 0.0010
# , time, mse, std
# time, 0.0114, 0.0006)
#
# test, nlevenshtein, std
# activity_id, 0.7542, 0.0071
# resource_id, 0.8646, 0.0037


def _parse_scores(scores_filepath):
    with open(scores_filepath, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='|')
        next(csv_reader, None)
        line = next(csv_reader, None)
        activity_id_accuracy = float(line[1])
        activity_id_std = float(line[2])
        line = next(csv_reader, None)
        resource_id_accuracy = float(line[1])
        resource_id_std = float(line[2])
        next(csv_reader, None)
        line = next(csv_reader, None)
        time_mse = float(line[1])
        time_std = float(line[2])
        next(csv_reader, None)
        next(csv_reader, None)
        line = next(csv_reader, None)
        activity_id_nlevenshtein = float(line[1])
        activity_id_nlevenshtein_std = float(line[2])
        line = next(csv_reader, None)
        resource_id_nlevenshtein = float(line[1])
        resource_id_nlevenshtein_std = float(line[2])
    return (activity_id_accuracy, activity_id_std, resource_id_accuracy, resource_id_std, time_mse, time_std, activity_id_nlevenshtein, activity_id_nlevenshtein_std, resource_id_nlevenshtein, resource_id_nlevenshtein_std)


def _plot_scores(scores):
    bar_width = 0.4
    axis = np.arange(len(log_names))

    models = scores.keys()

    plt.figure()
    plt.title('activity_id accuracy')
    for i, model in enumerate(models):
        model_scores = np.array(scores[model])
        plt.bar(axis - bar_width / 2 * len(models) / 2 + i * bar_width, model_scores[:, 0], yerr=model_scores[:, 1], width=bar_width, label=model)
        plt.ylim([0.8, 1])
        plt.xticks(axis, log_names, rotation=90)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('resource_id accuracy')
    for i, model in enumerate(models):
        model_scores = np.array(scores[model])
        plt.bar(axis - bar_width / 2 * len(models) / 2 + i * bar_width, model_scores[:, 2], yerr=model_scores[:, 3], width=bar_width, label=model)
        plt.ylim([0.6, 1])
        plt.xticks(axis, log_names, rotation=90)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('time mse')
    for i, model in enumerate(models):
        model_scores = np.array(scores[model])
        plt.bar(axis - bar_width / 2 * len(models) / 2 + i * bar_width, model_scores[:, 4], yerr=model_scores[:, 5], width=bar_width, label=model)
        plt.xticks(axis, log_names, rotation=90)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('activity_id nlevenshtein')
    for i, model in enumerate(models):
        model_scores = np.array(scores[model])
        plt.bar(axis - bar_width / 2 * len(models) / 2 + i * bar_width, model_scores[:, 6], yerr=model_scores[:, 7], width=bar_width, label=model)
        plt.ylim([0.6, 1])
        plt.xticks(axis, log_names, rotation=90)
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('resource_id nlevenshtein')
    for i, model in enumerate(models):
        model_scores = np.array(scores[model])
        plt.bar(axis - bar_width / 2 * len(models) / 2 + i * bar_width, model_scores[:, 8], yerr=model_scores[:, 9], width=bar_width, label=model)
        plt.ylim([0.6, 1])
        plt.xticks(axis, log_names, rotation=90)
    plt.legend()
    plt.show()

def compare_models(models):
    scores = {}

    for model in models:
        scores[model] = []

    for log_name in log_names:
        for model in models:
            results_filepath = '/'.join(['..', 'outputs', model, log_name, 'results.csv'])
            scores[model].append(_parse_scores(results_filepath))
    _plot_scores(scores)

if __name__ == "__main__":
    model_1 = 'old_model'
    model_2 = 'new_model'
    compare_models([model_1, model_2])
