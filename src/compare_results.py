import csv

import numpy as np
from matplotlib import pyplot as plt
import os

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
    '10x2_1S',
    '10x2_3W',
    '10x2_3S',
    '50x5_1W',
    '50x5_1S',
    '50x5_3W',
    '50x5_3S',
    'BPI2017_50k',
]


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
    return activity_id_accuracy, activity_id_std, resource_id_accuracy, resource_id_std, time_mse, time_std, activity_id_nlevenshtein, activity_id_nlevenshtein_std, resource_id_nlevenshtein, resource_id_nlevenshtein_std


def plot_scores_figure(scores, index, title, top=1):
    models = sorted(scores.keys())

    bar_width = 0.2
    axis = np.arange(len(log_names))

    plt.figure()
    plt.title(title)

    y_min = 1
    y_max = 0
    for i, model in enumerate(models):
        model_scores = np.array(scores[model])
        plt.bar(axis - bar_width / 2 * len(models) / 2 + i * bar_width, model_scores[:, index], yerr=model_scores[:, index+1],
                width=bar_width, label=model)
        y_min_temp = np.min(model_scores[:, index])
        y_max_temp = np.max(model_scores[:, index])
        if y_min_temp < y_min:
            y_min = y_min_temp
        if y_max_temp > y_max:
            y_max = y_max_temp

    if top > 0:
        plt.ylim(top=top)
    y_min -= 0.1 * (y_max - y_min)
    plt.ylim(bottom=max(0, y_min))

    plt.xticks(axis, log_names, rotation=90)
    plt.legend()
    plt.show()


def _plot_scores(scores):
    plot_scores_figure(scores, 0, 'activity_id_accuracy')
    plot_scores_figure(scores, 2, 'resource_id accuracy')
    plot_scores_figure(scores, 4, 'time mse', top=0)
    plot_scores_figure(scores, 6, 'activity_id nlevenshtein')
    plot_scores_figure(scores, 8, 'resource_id nlevenshtein')


def compare_models(model_type):
    scores = {}

    models = [f.name for f in os.scandir('/'.join(['..', 'outputs', model_type])) if f.is_dir()]

    for model in models:
        scores[model] = []

    for log_name in log_names:
        for model in models:
            results_filepath = '/'.join(['..', 'outputs', model_type, model, log_name, 'results.csv'])
            scores[model].append(_parse_scores(results_filepath))
    _plot_scores(scores)


if __name__ == "__main__":
    compare_models('sequence')
