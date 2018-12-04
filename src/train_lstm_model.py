
from src.bpm_lstm.bpm_lstm_model import BPM_LSTM

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
    '50x5_3S'
]


def main():
    model_name = 'old_model_act_res_time'
    folds = 3

    for log_name in log_names:
        print(log_name)
        log_filepath = '../data/' + log_name + '.csv'

        lstm = BPM_LSTM(log_name, log_filepath, model_name)

        lstm.train(folds)


if __name__ == "__main__":
    main()
