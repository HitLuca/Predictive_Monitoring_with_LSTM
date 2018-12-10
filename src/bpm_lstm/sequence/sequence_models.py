from keras import Model, Input
from keras.layers import TimeDistributed, Dropout, LeakyReLU, BatchNormalization, Dense, LSTM, Concatenate
from keras.optimizers import Nadam


class SequenceModels:
    @staticmethod
    def build_old_model(X_train, max_activity_id, max_resource_id):
        model_input = Input((X_train[0].shape[1:]))
        additional_features_input = Input((X_train[1].shape[1:]))

        processed = Concatenate()([model_input, additional_features_input])

        processed = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        processed = BatchNormalization()(processed)

        activity_id_output = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        activity_id_output = BatchNormalization()(activity_id_output)

        resource_id_output = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        resource_id_output = BatchNormalization()(resource_id_output)

        time_output = LSTM(100, return_sequences=True, dropout=0.2)(processed)
        time_output = BatchNormalization()(time_output)

        activity_id_output = TimeDistributed(Dense(max_activity_id + 1, activation='softmax'),
                                             name='activity_id')(
            activity_id_output)
        resource_id_output = TimeDistributed(Dense(max_resource_id + 1, activation='softmax'),
                                             name='resource_id')(
            resource_id_output)
        time_output = TimeDistributed(Dense(1, activation='relu'), name='time')(time_output)

        model = Model([model_input, additional_features_input],
                      [activity_id_output, resource_id_output, time_output])

        opt = Nadam(clipvalue=3)

        model.compile(loss={'activity_id': 'categorical_crossentropy',
                            'resource_id': 'categorical_crossentropy',
                            'time': 'mse'},
                      metrics={'activity_id': 'categorical_accuracy',
                               'resource_id': 'categorical_accuracy',
                               'time': 'mse'},
                      optimizer=opt)
        return model

    @staticmethod
    def build_new_model_v1(X_train, max_activity_id, max_resource_id):
        model_input = Input((X_train[0].shape[1:]))
        additional_features_input = Input((X_train[1].shape[1:]))

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

        processed = TimeDistributed(Dense(64))(processed)
        processed = BatchNormalization()(processed)
        processed = LeakyReLU()(processed)
        processed = Dropout(0.5)(processed)

        processed = LSTM(128, return_sequences=True)(processed)

        processed = TimeDistributed(Dense(64))(processed)
        processed = BatchNormalization()(processed)
        processed = LeakyReLU()(processed)
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
    def build_new_model_v3(X_train, max_activity_id, max_resource_id):
        model_input = Input((X_train[0].shape[1:]))
        additional_features_input = Input((X_train[1].shape[1:]))

        processed = Concatenate()([model_input, additional_features_input])

        processed = TimeDistributed(Dense(64))(processed)
        processed = BatchNormalization()(processed)
        processed = LeakyReLU()(processed)
        processed = Dropout(0.5)(processed)

        processed = LSTM(128, return_sequences=True, recurrent_dropout=0.5)(processed)

        processed = TimeDistributed(Dense(64))(processed)
        processed = BatchNormalization()(processed)
        processed = LeakyReLU()(processed)
        processed = Dropout(0.5)(processed)

        activity_id_output = Dense(max_activity_id + 1, activation='softmax', name='activity_id')(processed)
        resource_id_output = Dense(max_resource_id + 1, activation='softmax', name='resource_id')(processed)
        time_output = Dense(1, activation='sigmoid', name='time')(processed)

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
    'old_model': SequenceModels.build_old_model,
    'new_model_v1': SequenceModels.build_new_model_v1,
    'new_model_v2': SequenceModels.build_new_model_v2,
    'new_model_v3': SequenceModels.build_new_model_v3
}
