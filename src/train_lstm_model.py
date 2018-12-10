from bpm_lstm.next_step.next_step_model import BPM_LSTM_NEXT_STEP
from bpm_lstm.sequence.sequence_model import BPM_LSTM_SEQUENCE
from remove_unused_checkpoints import remove_unused_checkpoints

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


model_types = {
    'sequence': BPM_LSTM_SEQUENCE,
    'next_step': BPM_LSTM_NEXT_STEP
}


def main():
    model_type = 'next_step'
    predictive_model = model_types[model_type]
    folds = 3
    write_logs = False

    print('model type:', model_type)
    print('folds:', folds)
    print('write logs:', write_logs)

    for log_name in log_names:
        print(log_name)
        log_filepath = '../data/' + log_name + '.csv'

        lstm = predictive_model(log_name, log_filepath, write_logs)

        lstm.train(folds)

    remove_unused_checkpoints()


if __name__ == "__main__":
    main()
