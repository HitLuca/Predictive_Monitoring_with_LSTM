from keras import Model, Input
from keras.layers import Dropout, LeakyReLU, BatchNormalization, Dense, Concatenate, CuDNNLSTM, Flatten, LSTM, \
    Bidirectional, Multiply, RepeatVector, Activation, Permute, Conv1D, MaxPooling1D


class NextStepModels:
    @staticmethod
    def build_new_model_v1(X_train, max_activity_id, max_resource_id):
        model_input = Input((X_train[0].shape[1:]))
        additional_features_input = Input((X_train[1].shape[1:]))

        processed = Concatenate()([model_input, additional_features_input])

        processed = CuDNNLSTM(32, return_sequences=False)(processed)
        processed = Dropout(0.5)(processed)

        activity_id_output = Dense(max_activity_id + 1, activation='softmax', name='activity_id')(processed)
        resource_id_output = Dense(max_resource_id + 1, activation='softmax', name='resource_id')(processed)
        time_output = Dense(1, activation='relu', name='time')(processed)

        model = Model([model_input, additional_features_input],
                      [activity_id_output, resource_id_output, time_output])

        model.compile(loss={'activity_id': 'categorical_crossentropy',
                            'resource_id': 'categorical_crossentropy',
                            'time': 'mse'},
                      metrics={'activity_id': 'categorical_accuracy',
                               'resource_id': 'categorical_accuracy',
                               'time': 'mse'},
                      optimizer='adam')
        return model

    @staticmethod
    def build_new_model_v2(X_train, max_activity_id, max_resource_id):
        model_input = Input((X_train[0].shape[1:]))
        additional_features_input = Input((X_train[1].shape[1:]))

        processed = Concatenate()([model_input, additional_features_input])

        processed = Conv1D(32, 3, padding='same')(processed)
        processed = LeakyReLU(0.2)(processed)
        processed = MaxPooling1D()(processed)

        processed = Conv1D(32, 3, padding='same')(processed)
        processed = LeakyReLU(0.2)(processed)
        processed = MaxPooling1D()(processed)

        processed = Conv1D(32, 3, padding='same')(processed)
        processed = LeakyReLU(0.2)(processed)
        processed = MaxPooling1D()(processed)

        processed = Flatten()(processed)

        activity_id_output = Dense(max_activity_id + 1, activation='softmax', name='activity_id')(processed)
        resource_id_output = Dense(max_resource_id + 1, activation='softmax', name='resource_id')(processed)
        time_output = Dense(1, activation='relu', name='time')(processed)

        model = Model([model_input, additional_features_input],
                      [activity_id_output, resource_id_output, time_output])

        model.compile(loss={'activity_id': 'categorical_crossentropy',
                            'resource_id': 'categorical_crossentropy',
                            'time': 'mse'},
                      metrics={'activity_id': 'categorical_accuracy',
                               'resource_id': 'categorical_accuracy',
                               'time': 'mse'},
                      optimizer='adam')
        return model

available_models = {
    'new_model_v2': NextStepModels.build_new_model_v2,
}

