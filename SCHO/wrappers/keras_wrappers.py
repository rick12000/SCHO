import random
import numpy as np
import tensorflow as tf


class CNNClassifier:
    def __init__(self,
                 solver='adam',
                 learning_rate=0.0001,
                 drop_out_rate=0.01,
                 layer_1_tuple=(50, 9),
                 dense_layer_1=100,
                 # epochs=10,
                 batch_norm=0,
                 layer_1_pooling=2,
                 layer_2_pooling=2,
                 layer_3_pooling=2,
                 layer_4_pooling=0,
                 layer_2_tuple=None,
                 layer_3_tuple=None,
                 layer_4_tuple=None,
                 dense_layer_2=None,
                 random_state=None
                 ):
        self.solver = solver
        self.learning_rate = learning_rate
        self.drop_out_rate = drop_out_rate
        self.batch_norm = batch_norm
        # self.epochs = epochs
        self.layer_1_tuple = layer_1_tuple  # (number of convolutions, convolution size)
        self.layer_1_pooling = layer_1_pooling
        self.layer_2_tuple = layer_2_tuple
        self.layer_2_pooling = layer_2_pooling
        self.layer_3_tuple = layer_3_tuple
        self.layer_3_pooling = layer_3_pooling
        self.layer_4_tuple = layer_4_tuple
        self.layer_4_pooling = layer_4_pooling
        self.dense_layer_1 = dense_layer_1
        self.dense_layer_2 = dense_layer_2

        self.n_epochs_used = None
        self.trained_model = None
        self.random_state = random_state

    def __str__(self):
        return "CNNClassifier()"

    def __repr__(self):
        return "CNNClassifier()"

    def fit(self, X, y, val_X=None, val_Y=None):
        if self.random_state is not None:
            random.seed(1234)
            np.random.seed(1234)
            tf.random.set_seed(1234)

        model = tf.keras.models.Sequential()
        input_shape = list(X.shape)
        input_shape.remove(max(input_shape))
        try:
            channels = input_shape[2]
        except:
            channels = 1

        model.add(
            tf.keras.layers.Conv2D(self.layer_1_tuple[0], (self.layer_1_tuple[1], self.layer_1_tuple[1]),
                                   activation='relu',
                                   padding='same'
                                   , input_shape=tuple([(set([x for x in (X.shape) if (X.shape).count(x) > 1])).pop(),
                                                        (set([x for x in (X.shape) if (X.shape).count(x) > 1])).pop(),
                                                        channels])))
        if self.batch_norm == 1:
            model.add(tf.keras.layers.BatchNormalization())
        if self.layer_1_pooling is not None:
            if self.layer_1_pooling != 0:
                model.add(tf.keras.layers.MaxPooling2D((self.layer_1_pooling, self.layer_1_pooling)))
                if self.batch_norm == 1:
                    model.add(tf.keras.layers.BatchNormalization())
        if self.layer_2_tuple is not None:
            if self.layer_2_tuple[0] != 0 and self.layer_2_tuple[1] != 0:
                model.add(tf.keras.layers.Conv2D(self.layer_2_tuple[0], (self.layer_2_tuple[1], self.layer_2_tuple[1])
                                                 , activation='relu',
                                                 padding='same'))
                if self.batch_norm == 1:
                    model.add(tf.keras.layers.BatchNormalization())
        if self.layer_2_pooling is not None:
            if self.layer_2_pooling != 0:
                model.add(tf.keras.layers.MaxPooling2D((self.layer_2_pooling, self.layer_2_pooling)))
                if self.batch_norm == 1:
                    model.add(tf.keras.layers.BatchNormalization())
        if self.layer_3_tuple is not None:
            if self.layer_3_tuple[0] != 0 and self.layer_3_tuple[1] != 0:
                model.add(tf.keras.layers.Conv2D(self.layer_3_tuple[0], (self.layer_3_tuple[1], self.layer_3_tuple[1])
                                                 , activation='relu',
                                                 padding='same'))
                if self.batch_norm == 1:
                    model.add(tf.keras.layers.BatchNormalization())
        if self.layer_3_pooling is not None:
            if self.layer_3_pooling != 0:
                model.add(tf.keras.layers.MaxPooling2D((self.layer_3_pooling, self.layer_3_pooling)))
                if self.batch_norm == 1:
                    model.add(tf.keras.layers.BatchNormalization())
        if self.layer_4_tuple is not None:
            if self.layer_4_tuple[0] != 0 and self.layer_4_tuple[1] != 0:
                model.add(tf.keras.layers.Conv2D(self.layer_4_tuple[0], (self.layer_4_tuple[1], self.layer_4_tuple[1])
                                                 , activation='relu',
                                                 padding='same'))
                if self.batch_norm == 1:
                    model.add(tf.keras.layers.BatchNormalization())
        if self.layer_4_pooling is not None:
            if self.layer_4_pooling != 0:
                model.add(tf.keras.layers.MaxPooling2D((self.layer_4_pooling, self.layer_4_pooling)))
                if self.batch_norm == 1:
                    model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.dense_layer_1, activation='relu'
                                        ))
        model.add(tf.keras.layers.Dropout(self.drop_out_rate))  # , seed=self.random_state))
        if self.dense_layer_2 is not None:
            if self.dense_layer_2 != 0:
                model.add(tf.keras.layers.Dense(self.dense_layer_2, activation='relu'
                                                ))
                model.add(tf.keras.layers.Dropout(self.drop_out_rate))  # , seed=self.random_state))

        model.add(tf.keras.layers.Dense(len(np.unique(y)), activation='softmax'))

        if self.solver == 'adam':
            optimizer_config = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.solver == 'sgd':
            optimizer_config = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'], optimizer=optimizer_config)

        if val_X is not None and val_Y is not None:
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3,
                                                        min_delta=0.01)  # min_delta=-0.005
            model.fit(X, y, validation_data=(val_X, val_Y), epochs=10, batch_size=32, verbose=0,
                      callbacks=[callback],
                      shuffle=False)
            self.n_epochs_used = callback.stopped_epoch
        elif val_X is None and val_Y is None:
            model.fit(X, y, epochs=10, batch_size=32, verbose=0,
                      shuffle=False)
        self.trained_model = model

    def predict(self, X):
        y_prob = self.trained_model.predict(X)
        y_classes = y_prob.argmax(axis=-1)
        return np.array(y_classes)

    def predict_proba(self, X):
        y_prob = self.trained_model.predict(X)
        return np.array(y_prob)

    def evaluate(self, X, y):
        test_loss, test_acc = self.trained_model.evaluate(X, y, verbose=0)
        return test_loss, test_acc
