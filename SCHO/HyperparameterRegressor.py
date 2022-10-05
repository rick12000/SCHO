import os

os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from typing import Union

import math
import random
import time
import warnings

import pandas as pd
import numpy as np

random.seed(1234)
np.random.seed(1234)
import tensorflow as tf
from sklearn import datasets as sklearn_datasets
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from tensorflow.keras import datasets as keras_datasets
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler

from SCHO.ConformalPredictor import Conformal
from SCHO.utils.RunEfficiency import TimeLogger
from SCHO.utils.RunEfficiency import ConformalRuntimeOptimizer
from SCHO.utils.NLPhelper import NLPEncoder
from SCHO.utils.DataHandling import Filing
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm


class HyperRegCV:
    def __init__(self, model, random_state=None):
        self.model = model
        self.random_state = random_state

        self.n_epochs_used = None

    @staticmethod
    def get_toy_dataset(dataset_name, return_pre_split_tuple=False):
        if dataset_name == "iris":
            toy_dataset = sklearn_datasets.load_iris()
            X = toy_dataset.data
            Y = toy_dataset.target
        elif dataset_name == "cup":
            toy_dataset = sklearn_datasets.fetch_kddcup99(percent10=False, subset="SA", random_state=10)
            X = toy_dataset.data
            Y = toy_dataset.target
            Y = np.where(pd.Series(Y).astype(str).str.contains("normal"), "normal", "attack")
            Y = pd.factorize(Y)[0]
            XY = pd.DataFrame(np.hstack([X, Y.reshape(len(Y), 1)])).drop_duplicates()
            XY = ClassRebalancer._binary_class_rebalancing(data=XY, y_name=(XY.shape[1] - 1))
            X = XY.drop([(XY.shape[1] - 1)], axis=1).to_numpy()
            X = PreProcessingHelper.one_hot_encode_variables(X)
            Y = XY[XY.shape[1] - 1].to_numpy().astype(int)
        elif dataset_name == "diabetes":
            toy_dataset = sklearn_datasets.load_diabetes()
            X = toy_dataset.data
            Y = toy_dataset.target
        elif dataset_name == "cali":
            toy_dataset = sklearn_datasets.fetch_california_housing()
            X = toy_dataset.data
            Y = toy_dataset.target
        elif dataset_name == "friedman1":
            X, Y = sklearn_datasets.make_friedman1(n_samples=2000, n_features=15, random_state=10)
        elif dataset_name == "census":
            raw_data = pd.read_csv(Filing.input_parent_folder_path + "/datasets/census/census.data", sep=',')
            raw_data_formatted = pd.DataFrame()
            for l in range(0, raw_data.shape[1]):
                try:
                    raw_data_formatted = pd.concat([raw_data_formatted, raw_data.iloc[:, l].astype(float)], axis=1)
                except:
                    raw_data_formatted = pd.concat([raw_data_formatted, pd.get_dummies(raw_data.iloc[:, l])], axis=1)
            raw_data_formatted = raw_data_formatted.drop([" <=50K"], axis=1).reset_index(drop=True)

            raw_data_formatted = ClassRebalancer._binary_class_rebalancing(data=raw_data_formatted, y_name=" >50K")
            X = (raw_data_formatted.drop([" >50K"], axis=1)).to_numpy()
            Y = raw_data_formatted[" >50K"].to_numpy()
            Y = Y.astype('int')
            # undersampling_index = list(np.random.choice(len(X), 5000, replace=False))
            # X = X[undersampling_index, :]
            # Y = Y[undersampling_index]
        elif dataset_name == "cancer" or dataset_name == "tabular_test_data":
            toy_dataset = sklearn_datasets.load_breast_cancer()
            X = toy_dataset.data
            Y = toy_dataset.target
        elif dataset_name == "digits":
            toy_dataset = sklearn_datasets.load_digits()
            X = toy_dataset.data / 16
            Y = toy_dataset.target
        elif dataset_name == "20news":
            toy_dataset = sklearn_datasets.fetch_20newsgroups()
            X = np.array(toy_dataset.data)
            Y = toy_dataset.target

            undersampling_index = list(np.random.choice(len(X), 2000, replace=False))
            X = X[undersampling_index]
            Y = Y[undersampling_index]
        elif dataset_name == "covertype":
            raw_data = pd.read_csv(Filing.parent_folder_path + "/datasets/covertype/covtype.data", sep=',')

            X = raw_data.iloc[:, :-1].to_numpy()
            Y = raw_data.iloc[:, -1].to_numpy()

            undersampling_index = list(np.random.choice(len(X), 20, replace=False))
            X = X[undersampling_index, :]
            Y = Y[undersampling_index]
            Y = Y.astype('int')

        elif dataset_name == "olivetti":
            toy_dataset = sklearn_datasets.fetch_olivetti_faces()
            X = toy_dataset.data
            Y = toy_dataset.target
        elif dataset_name == "mnist" or dataset_name == "convolutional_test_data":
            (x_train, y_train), (x_test, y_test) = keras_datasets.mnist.load_data()
            x_train = x_train / 255
            x_test = x_test / 255

            if dataset_name == "convolutional_test_data":
                undersampling_index = list(np.random.choice(len(x_train), 200, replace=False))
                x_train = x_train[undersampling_index, :]
                y_train = y_train[undersampling_index]
                y_train = y_train.astype('int')

                undersampling_index = list(np.random.choice(len(x_train), 200, replace=False))
                x_test = x_test[undersampling_index, :]
                y_test = y_test[undersampling_index]
                y_test = y_test.astype('int')

            if return_pre_split_tuple:
                return x_train, y_train, x_test, y_test
            else:
                return x_train, y_train
        elif dataset_name == "cifar10":
            (x_train, y_train), (x_test, y_test) = keras_datasets.cifar10.load_data()
            # undersampling_index = list(np.random.choice(len(x_train), 200, replace=False))
            # x_train = x_train[undersampling_index, :]
            x_train = x_train / 255
            x_test = x_test / 255
            # y_train = y_train[undersampling_index]
            if return_pre_split_tuple:
                return x_train, y_train, x_test, y_test
            else:
                return x_train, y_train

        elif dataset_name == "colorectal":
            colorectal_raw_data = tfds.load("colorectal_histology", split='train')
            images = []
            labels = []
            # Iterate over a dataset
            for i, image_dict in enumerate(tfds.as_numpy(colorectal_raw_data)):
                images.append(image_dict["image"])
                labels.append(image_dict["label"])
            # images = tf.image.resize(images, [32,32]).numpy()

            X = np.array(images)
            Y = np.array(labels)
            # undersampling_index = list(np.random.choice(len(X), 10000, replace=False))
            # X = X[undersampling_index, :]
            X = X / 255
            # Y = Y[undersampling_index]

            return X, Y

        elif dataset_name == "svhn":
            svhn_raw_data = tfds.load("svhn_cropped", split='train')
            images = []
            labels = []
            for i, image_dict in enumerate(tfds.as_numpy(svhn_raw_data)):
                images.append(image_dict["image"])
                labels.append(image_dict["label"])
            # images = tf.image.resize(images, [32,32]).numpy()

            X = np.array(images)
            Y = np.array(labels)
            undersampling_index = list(np.random.choice(len(X), 30000, replace=False))
            X = X[undersampling_index, :]
            X = X / 255
            Y = Y[undersampling_index]

            return X, Y


        elif dataset_name == "stl10":
            stl10_raw_data_train = tfds.load("stl10", split='train')
            stl10_raw_data_test = tfds.load("stl10", split='test')
            train_images = []
            train_labels = []
            for i, image_dict in enumerate(tfds.as_numpy(stl10_raw_data_train)):
                train_images.append(image_dict["image"])
                train_labels.append(image_dict["label"])
            test_images = []
            test_labels = []
            for i, image_dict in enumerate(tfds.as_numpy(stl10_raw_data_test)):
                test_images.append(image_dict["image"])
                test_labels.append(image_dict["label"])
            X = np.array(train_images + test_images)
            Y = np.array(train_labels + test_labels)
            # undersampling_index = list(np.random.choice(len(X), 20000, replace=False))
            # X = X[undersampling_index, :]
            X = X / 255
            # Y = Y[undersampling_index]

            return X, Y


        elif dataset_name == "cifar100":
            (x_train, y_train), (x_test, y_test) = keras_datasets.cifar100.load_data()
            undersampling_index = list(np.random.choice(len(x_train), 20000, replace=False))
            x_train = x_train[undersampling_index, :]
            x_train = x_train / 255
            y_train = y_train[undersampling_index]

            return x_train, y_train
        elif dataset_name == "fashion_mnist":
            (x_train, y_train), (x_test, y_test) = keras_datasets.fashion_mnist.load_data()
            x_train = x_train / 255
            x_test = x_test / 255
            # y_train = y_train[undersampling_index]
            if return_pre_split_tuple:
                return x_train, y_train, x_test, y_test
            else:
                return x_train, y_train

        elif dataset_name == "imdb":
            raw_data = pd.read_csv(Filing.input_parent_folder_path + "/datasets/imdb/imdb_dataset.csv")
            X = raw_data["review"].to_numpy()
            Y = raw_data["sentiment"].to_numpy()
            Y[Y == "negative"] = int(0)
            Y[Y == "positive"] = int(1)

            # undersampling_index = list(np.random.choice(len(X), 25000, replace=False))
            # X = X[undersampling_index]
            # Y = Y[undersampling_index]
            Y = Y.astype('int')
        return X, Y

    def get_parameters(self):
        if "mlp" in str(self.model).lower():
            solver_list = ['adam', 'sgd']
            learning_rate_list = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
            alpha_list = [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 10]
            layer_size = list(np.arange(4, 56, 4)) + [60, 80, 100, 120, 140]
            n_layers = [2, 3, 4, 5]
            parameter_dict = {'solver': solver_list,
                              'learning_rate_init': learning_rate_list,
                              'alpha': alpha_list,
                              'n_layers': n_layers,
                              'layer_size': layer_size
                              }

        elif "conv" in str(self.model).lower():
            solver_list = ['adam', 'sgd']
            learning_rate_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
            drop_out_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            n_layers = [2, 3, 4]
            layer_size = list(range(16, 257,
                                    16))  # list(filter(lambda x: x < 250, list(np.round(np.random.lognormal(0, 0.5, 500) * 64).astype(int))))
            # convolution_size = [3]
            # pooling_size = [2]
            # batch_norm_list = [0]  # yes or no
            dense_layer_1_neurons_list = [100, 200, 512]
            dense_layer_2_neurons_list = [0, 0, 0, 0, 50, 100]

            parameter_dict = {'solver': solver_list,
                              'learning_rate': learning_rate_list,
                              'drop_out_rate': drop_out_rate_list,
                              'n_layers': n_layers,
                              'layer_size': layer_size,
                              # 'convolution_size': convolution_size,
                              # 'pooling_size': pooling_size,
                              # 'batch_norm': batch_norm_list,
                              'dl1_neurons': dense_layer_1_neurons_list,
                              'dl2_neurons': dense_layer_2_neurons_list,
                              }

        return parameter_dict

    @staticmethod
    def train_val_test_split(X,
                             y,
                             OOS_split=0.3,
                             normalize=True,
                             random_state=None) -> Union[np.array, np.array, np.array, np.array]:
        if random_state is not None:
            np.random.seed(random_state)
        OOS_index = list(np.random.choice(len(X), round(len(X) * OOS_split), replace=False))
        IS_index = list(np.setdiff1d(np.arange(len(X)), OOS_index))
        X_OOS = X[OOS_index, :]
        y_OOS = y[OOS_index]
        X_IS = X[IS_index, :]
        y_IS = y[IS_index]

        if normalize:
            scaler = StandardScaler()
            scaler.fit(X_IS)
            X_IS = scaler.transform(X_IS)
            X_OOS = scaler.transform(X_OOS)

        return X_OOS, y_OOS, X_IS, y_IS

    @staticmethod
    def tf_idf_train_val_test_split(X, y, OOS_split=0.3, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        OOS_index = list(np.random.choice(len(X), round(len(X) * OOS_split), replace=False))
        IS_index = list(np.setdiff1d(np.arange(len(X)), OOS_index))
        try:
            X_OOS = X[OOS_index]
            y_OOS = y[OOS_index]
            X_IS = X[IS_index]
            y_IS = y[IS_index]
        except:
            print("Value Error: X data must be a single column containing full string sentences per row.")

        tfidf_vectorizer, tfidf_object = NLPEncoder.fit_BOW_vectorizer(X=X_IS)
        X_IS = NLPEncoder.apply_BOW_vectorizer(vectorizer=tfidf_vectorizer, tfidf_object=tfidf_object, X=X_IS)
        X_OOS = NLPEncoder.apply_BOW_vectorizer(vectorizer=tfidf_vectorizer, tfidf_object=tfidf_object, X=X_OOS)

        return X_OOS, y_OOS, X_IS, y_IS

    def get_hyperparameter_combinations(self, parameter_grid):
        print("Creating hyperparameter space...")
        if self.random_state is not None:
            random.seed(self.random_state)

        try:
            os.mkdir("_storage/initializers")
        except:
            pass

        cached_combination_file_path = "_storage/initializers/" + str(self.model) + "_hyperparameter_combination_df.pkl"
        if os.path.exists(cached_combination_file_path) and abs(
                time.time() - os.path.getmtime(cached_combination_file_path)) < 60 * 60 * 24 * 365:
            hyperparameter_tuple_randomized = pd.read_pickle(cached_combination_file_path)
            print("Found combination initializer that is less than 365 days old, using cached initializer...")

        elif "mlp" in str(self.model).lower() or "conv" in str(self.model).lower():

            for i in tqdm(range(0, 100000)):
                parameter_combination = []
                parameter_combination_columns = []
                for key in parameter_grid.keys():  # NOTE: number of layers parameter must be ordered before layer size in the param dict for this to work
                    if key == 'layer_size' or key == 'convolution_size' or key == 'pooling_size':
                        for j in range(1, max(parameter_grid["n_layers"]) + 1):
                            if j <= n_layers_cached:
                                if key == "layer_size" and j == 1:
                                    parameter = random.choice(list(filter(lambda x: x <= 32, parameter_grid[key])))
                                # elif key == "pooling_size" and j == 1:
                                #     parameter = random.choice([0, 2])
                                elif key == "pooling_size" and j >= 4:  # specific to cifar starting size of 32x32 and pooling of 2x2 up until layer 4
                                    parameter = 0
                                else:
                                    parameter = random.choice(parameter_grid[key])
                                parameter_combination.append(parameter)
                                if "mlp" in str(self.model).lower():
                                    parameter_combination_columns.append("layer_" + str(j))
                                elif "conv" in str(self.model).lower():
                                    if key == "layer_size":
                                        parameter_combination_columns.append("l" + str(j) + "_convolutions")
                                    elif key == "convolution_size":
                                        parameter_combination_columns.append("l" + str(j) + "_size")
                                    elif key == "pooling_size":
                                        parameter_combination_columns.append("p" + str(j) + "_size")
                            else:
                                parameter = 0
                                parameter_combination.append(parameter)
                                if "mlp" in str(self.model).lower():
                                    parameter_combination_columns.append("layer_" + str(j))
                                elif "conv" in str(self.model).lower():
                                    if key == "layer_size":
                                        parameter_combination_columns.append("l" + str(j) + "_convolutions")
                                    elif key == "convolution_size":
                                        parameter_combination_columns.append("l" + str(j) + "_size")
                                    elif key == "pooling_size":
                                        parameter_combination_columns.append("p" + str(j) + "_size")
                    else:
                        parameter = random.choice(parameter_grid[key])
                        parameter_combination.append(parameter)
                        parameter_combination_columns.append(key)
                    if key == 'n_layers':
                        n_layers_cached = parameter
                if i == 0:
                    hyperparameter_tuple = pd.DataFrame(parameter_combination).transpose()
                    hyperparameter_tuple.columns = parameter_combination_columns
                else:
                    hyperparameter_tuple.loc[len(hyperparameter_tuple)] = np.transpose(parameter_combination)

            # hyperparameter_tuple = hyperparameter_tuple.reset_index(drop = True)
            hyperparameter_tuple = hyperparameter_tuple.drop_duplicates()

            hyperparameter_tuple[["adam", "sgd"]] = pd.get_dummies(hyperparameter_tuple[
                                                                       "solver"])
            hyperparameter_tuple = hyperparameter_tuple[
                ~(hyperparameter_tuple["adam"] + hyperparameter_tuple["sgd"] == 0)]
            hyperparameter_tuple = hyperparameter_tuple[
                ~(hyperparameter_tuple["adam"] + hyperparameter_tuple["sgd"] == 2)]
            hyperparameter_tuple = hyperparameter_tuple.drop(["solver"], axis=1)

            hyperparameter_tuple_randomized = hyperparameter_tuple.sample(frac=1,
                                                                          random_state=self.random_state).reset_index(
                drop=True)
            hyperparameter_tuple_randomized.to_pickle(cached_combination_file_path)

        print("Hyperparameter space successfully loaded...")

        return hyperparameter_tuple_randomized

    @staticmethod
    def pivot_classes(y, n_classes):
        pivoted_array = np.zeros((len(y), n_classes))
        for i in range(0, len(y)):
            pivoted_array[i, y[i]] = 1
        return pivoted_array

    @staticmethod
    def loss_profile(y_pred, y_pred_proba, y_obs, n_classes, prediction_type):
        if prediction_type == "regression":
            accuracy = np.nan
            entropy = np.nan
            mse = np.sum((y_obs - y_pred) ** 2) / len(y_obs)
            rmse = math.sqrt(mse)
        elif prediction_type == "classification" or prediction_type == "nlp_classification":
            try:
                if y_pred_proba.shape[1] == 2:
                    entropy = metrics.log_loss(y_obs, y_pred_proba[:, 1])
                else:
                    one_hot_encoded_y_obs = HyperRegCV.pivot_classes(y=y_obs, n_classes=n_classes)
                    scorer = tf.keras.losses.CategoricalCrossentropy()
                    entropy = scorer(one_hot_encoded_y_obs, y_pred_proba).numpy()
                accuracy = metrics.accuracy_score(y_obs, y_pred)

                mse = np.nan
                rmse = np.nan

            except:
                accuracy = np.nan
                entropy = np.nan
                mse = np.nan
                rmse = np.nan

        loss_profile = {"accuracy_score": accuracy, "log_loss": entropy, "mean_squared_error": mse,
                        "root_mean_squared_error": rmse}

        return loss_profile

    @staticmethod
    def loss_metric_direction(loss_metric):
        if loss_metric == 'accuracy_score':
            direction = 'direct'
        elif loss_metric == 'log_loss':
            direction = 'inverse'
        elif loss_metric == 'mean_squared_error':
            direction = 'inverse'
        elif loss_metric == 'root_mean_squared_error':
            direction = 'inverse'
        return direction

    def build_hyperparameter_logger(self, hyperparameter_combinations):

        hyperparameter_performance_record = hyperparameter_combinations.copy()
        for i in range(0, hyperparameter_performance_record.shape[1]):
            hyperparameter_performance_record.iloc[:, i] = np.nan
        hyperparameter_performance_record["accuracy"] = np.nan
        hyperparameter_performance_record["accuracy_score"] = np.nan
        hyperparameter_performance_record["log_loss"] = np.nan
        hyperparameter_performance_record["variance"] = np.nan
        hyperparameter_performance_record["runtime"] = np.nan
        hyperparameter_performance_record["CI_breach"] = np.nan
        hyperparameter_performance_record["point_predictor_MSE"] = np.nan
        hyperparameter_performance_record["loss_profile_dict"] = np.nan

        return hyperparameter_performance_record

    @staticmethod
    def tuplify_network_layer_sizes(
            combination):  # NOTE: assumes column names are sorted from layer 1 to layer N, otherwise will not create ordered list
        layer_tuple = ()
        for column_name in list(combination.index):
            if "layer" in column_name.lower():
                if combination[column_name] != 0:
                    layer_tuple = layer_tuple + (int(combination[column_name]),)
        return layer_tuple

    def _get_validation_loss(self, combination, X_IS, y_IS, custom_loss_function, prediction_type,
                             validating_framework='train_test_split', train_val_split=0.6, k_fold_splits=None,
                             n_classes=2, presplit_X_y_data_tuple=None):

        if validating_framework == 'train_test_split' or validating_framework == 'batch_mean':
            if presplit_X_y_data_tuple is not None:
                X_train, Y_train, X_val, Y_val = presplit_X_y_data_tuple
            else:
                X_val, Y_val, X_train, Y_train = HyperRegCV.train_val_test_split(X=X_IS,
                                                                                 y=y_IS,
                                                                                 OOS_split=train_val_split,
                                                                                 normalize=False,
                                                                                 random_state=self.random_state)

            if "mlp" in str(self.model).lower():
                if "reg" in str(self.model).lower():
                    hidden_layers = HyperRegCV.tuplify_network_layer_sizes(combination)
                    fitted_model = MLPRegressor(solver=HyperRegCV.solver_mapper(combination),
                                                learning_rate_init=combination["learning_rate_init"],
                                                alpha=combination["alpha"], hidden_layer_sizes=hidden_layers,
                                                random_state=self.random_state).fit(X_train, Y_train)
                elif "clas" in str(self.model).lower():
                    hidden_layers = HyperRegCV.tuplify_network_layer_sizes(combination)
                    fitted_model = MLPClassifier(solver=HyperRegCV.solver_mapper(combination),
                                                 learning_rate_init=combination["learning_rate_init"],
                                                 alpha=combination["alpha"], hidden_layer_sizes=hidden_layers,
                                                 random_state=self.random_state).fit(X_train, Y_train)

            elif "conv" in str(self.model).lower():
                fitted_model = ConvNet(solver=HyperRegCV.solver_mapper(combination),
                                       learning_rate=combination['learning_rate'],
                                       drop_out_rate=combination['drop_out_rate'],
                                       # batch_norm=int(combination['batch_norm']),
                                       layer_1_tuple=(
                                           int(combination['l1_convolutions']), 3),  # int(combination['l1_size'])),
                                       dense_layer_1=int(combination['dl1_neurons']),
                                       # epochs=int(combination['epochs']),
                                       # layer_1_pooling=int(combination["p1_size"]),
                                       # layer_2_pooling=int(combination["p2_size"]),
                                       # layer_3_pooling=int(combination["p3_size"]),
                                       # layer_4_pooling=int(combination["p4_size"]),
                                       layer_2_tuple=(
                                           int(combination['l2_convolutions']), 3),  # int(combination['l2_size'])),
                                       layer_3_tuple=(
                                           int(combination['l3_convolutions']), 3),  # int(combination['l3_size'])),
                                       layer_4_tuple=(
                                           int(combination['l4_convolutions']), 3),  # int(combination['l4_size'])),
                                       dense_layer_2=int(combination['dl2_neurons']),
                                       random_state=self.random_state
                                       )
                fitted_model.fit(X_train, Y_train, X_val, Y_val)

            if validating_framework == 'train_test_split':
                y_pred = fitted_model.predict(X_val)
                if prediction_type == "classification" or prediction_type == "nlp_classification":
                    y_pred_proba = fitted_model.predict_proba(X_val)
                else:
                    y_pred_proba = None

                loss_profile = HyperRegCV.loss_profile(y_pred=y_pred, y_pred_proba=y_pred_proba, y_obs=Y_val,
                                                       n_classes=n_classes, prediction_type=prediction_type)

                final_loss = loss_profile[custom_loss_function]
                final_variance = np.nan
                loss_direction = HyperRegCV.loss_metric_direction(custom_loss_function)

            elif validating_framework == 'batch_mean':
                step = 24
                batch_log = []
                for i in range(0, len(X_val), step):
                    if step + i >= len(X_val):
                        break
                    batch_X = X_val[0 + i:step + i, :]
                    batch_y = Y_val[0 + i:step + i]

                    y_pred = fitted_model.predict(batch_X)
                    if prediction_type == "classification" or prediction_type == "nlp_classification":
                        y_pred_proba = fitted_model.predict_proba(batch_X)
                    else:
                        y_pred_proba = None

                    loss_profile = HyperRegCV.loss_profile(y_pred=y_pred, y_pred_proba=y_pred_proba, y_obs=batch_y,
                                                           n_classes=n_classes)
                    batch_loss = loss_profile[custom_loss_function]
                    batch_log.append(batch_loss)

                final_loss = np.mean(np.array(batch_log))
                final_variance = math.sqrt(np.sum(((np.array(batch_log) - np.mean(np.array(batch_log))) ** 2)) / (
                        len(np.array(batch_log)) - 1))
                loss_direction = HyperRegCV.loss_metric_direction(custom_loss_function)

        elif validating_framework == 'k_fold':
            kf = KFold(n_splits=k_fold_splits, random_state=self.random_state, shuffle=True)
            K_fold_performance_record = pd.DataFrame(np.zeros((k_fold_splits, 2)))
            K_fold_performance_record.columns = ["fold", "accuracy"]
            fold = 1
            for train_index, test_index in kf.split(X_IS):
                X_train, X_val = X_IS[train_index, :], X_IS[test_index, :]
                Y_train, Y_val = y_IS[train_index], y_IS[test_index]

                if "mlp" in str(self.model).lower():
                    if "reg" in str(self.model).lower():
                        hidden_layers = HyperRegCV.tuplify_network_layer_sizes(combination)
                        fitted_model = MLPRegressor(solver=HyperRegCV.solver_mapper(combination),
                                                    learning_rate_init=combination["learning_rate_init"],
                                                    alpha=combination["alpha"], hidden_layer_sizes=hidden_layers,
                                                    random_state=self.random_state).fit(X_train, Y_train)
                    elif "clas" in str(self.model).lower():
                        hidden_layers = HyperRegCV.tuplify_network_layer_sizes(combination)
                        fitted_model = MLPClassifier(solver=HyperRegCV.solver_mapper(combination),
                                                     learning_rate_init=combination["learning_rate_init"],
                                                     alpha=combination["alpha"], hidden_layer_sizes=hidden_layers,
                                                     random_state=self.random_state).fit(X_train, Y_train)
                elif "conv" in str(self.model).lower():
                    fitted_model = ConvNet(solver=HyperRegCV.solver_mapper(combination),
                                           learning_rate=combination['learning_rate'],
                                           drop_out_rate=combination['drop_out_rate'],
                                           # batch_norm=int(combination['batch_norm']),
                                           layer_1_tuple=(
                                               int(combination['l1_convolutions']), 3),  # int(combination['l1_size'])),
                                           dense_layer_1=int(combination['dl1_neurons']),
                                           # epochs=int(combination['epochs']),
                                           # layer_1_pooling=int(combination["p1_size"]),
                                           # layer_2_pooling=int(combination["p2_size"]),
                                           # layer_3_pooling=int(combination["p3_size"]),
                                           # layer_4_pooling=int(combination["p4_size"]),
                                           layer_2_tuple=(
                                               int(combination['l2_convolutions']), 3),  # int(combination['l2_size'])),
                                           layer_3_tuple=(
                                               int(combination['l3_convolutions']), 3),  # int(combination['l3_size'])),
                                           layer_4_tuple=(
                                               int(combination['l4_convolutions']), 3),  # int(combination['l4_size'])),
                                           dense_layer_2=int(combination['dl2_neurons']),
                                           random_state=self.random_state
                                           )
                    fitted_model.fit(X_train, Y_train)
                y_pred = fitted_model.predict(X_val)
                if prediction_type == "classification" or prediction_type == "nlp_classification":
                    y_pred_proba = y_pred_proba = fitted_model.predict_proba(X_val)
                else:
                    y_pred_proba = None

                loss_profile = HyperRegCV.loss_profile(y_pred=y_pred, y_pred_proba=y_pred_proba, y_obs=Y_val,
                                                       n_classes=n_classes)
                loss = loss_profile[custom_loss_function]
                loss_direction = HyperRegCV.loss_metric_direction(custom_loss_function)

                K_fold_performance_record.iloc[(fold - 1), 0] = fold
                K_fold_performance_record.iloc[(fold - 1), 1] = loss

                fold = fold + 1

            cross_fold_performance = K_fold_performance_record.mean()
            final_loss = float(cross_fold_performance["accuracy"])
            final_variance = np.nan

        if "conv" in str(self.model).lower():
            self.n_epochs_used = fitted_model.n_epochs_used

        # if np.isnan(final_loss):
        #     final_loss = None
        # print("out of the box validation function IS loss: ", final_loss)

        return final_loss, final_variance, loss_direction, loss_profile

    def _get_OOS_loss(self, combination, X_IS, y_IS, X_OOS, y_OOS, custom_loss_function, prediction_type,
                      batch_data=False, n_classes=2):
        if "mlp" in str(self.model).lower():
            if "reg" in str(self.model).lower():
                hidden_layers = HyperRegCV.tuplify_network_layer_sizes(combination)
                fitted_model = MLPRegressor(solver=HyperRegCV.solver_mapper(combination),
                                            learning_rate_init=combination["learning_rate_init"],
                                            alpha=combination["alpha"], hidden_layer_sizes=hidden_layers,
                                            random_state=self.random_state).fit(X_IS, y_IS)
            elif "clas" in str(self.model).lower():
                hidden_layers = HyperRegCV.tuplify_network_layer_sizes(combination)
                fitted_model = MLPClassifier(solver=HyperRegCV.solver_mapper(combination),
                                             learning_rate_init=combination["learning_rate_init"],
                                             alpha=combination["alpha"], hidden_layer_sizes=hidden_layers,
                                             random_state=self.random_state).fit(X_IS, y_IS)
        elif "conv" in str(self.model).lower():
            fitted_model = ConvNet(solver=HyperRegCV.solver_mapper(combination),
                                   learning_rate=combination['learning_rate'],
                                   drop_out_rate=combination['drop_out_rate'],
                                   # batch_norm=int(combination['batch_norm']),
                                   layer_1_tuple=(
                                       int(combination['l1_convolutions']), int(combination['l1_size'])),
                                   dense_layer_1=int(combination['dl1_neurons']),
                                   # epochs=int(combination['epochs']),
                                   # layer_1_pooling=int(combination["p1_size"]),
                                   # layer_2_pooling=int(combination["p2_size"]),
                                   # layer_3_pooling=int(combination["p3_size"]),
                                   # layer_4_pooling=int(combination["p4_size"]),
                                   layer_2_tuple=(
                                       int(combination['l2_convolutions']), 3),  # int(combination['l2_size'])),
                                   layer_3_tuple=(
                                       int(combination['l3_convolutions']), 3),  # int(combination['l3_size'])),
                                   layer_4_tuple=(
                                       int(combination['l4_convolutions']), 3),  # int(combination['l4_size'])),
                                   dense_layer_2=int(combination['dl2_neurons']),
                                   random_state=self.random_state
                                   )
            fitted_model.fit(X_IS, y_IS)

        if not batch_data:
            y_pred = fitted_model.predict(X_OOS)
            if prediction_type == "classification" or prediction_type == "nlp_classification":
                y_pred_proba = fitted_model.predict_proba(X_OOS)
            else:
                y_pred_proba = None
            loss_profile = HyperRegCV.loss_profile(y_pred=y_pred, y_pred_proba=y_pred_proba,
                                                   y_obs=y_OOS, n_classes=n_classes, prediction_type=prediction_type)
            final_loss = loss_profile[custom_loss_function]
            final_variance = np.nan
            loss_direction = HyperRegCV.loss_metric_direction(custom_loss_function)

        else:
            step = 24
            batch_log = []
            for i in range(0, len(X_OOS), step):
                if step + i >= len(X_OOS):
                    break
                batch_X = X_OOS[0 + i:step + i, :]
                batch_y = y_OOS[0 + i:step + i]

                y_pred = fitted_model.predict(batch_X)
                if prediction_type == "classification" or prediction_type == "nlp_classification":
                    y_pred_proba = fitted_model.predict_proba(batch_X)
                else:
                    y_pred_proba = None

                loss_profile = HyperRegCV.loss_profile(y_pred=y_pred, y_pred_proba=y_pred_proba, y_obs=batch_y,
                                                       n_classes=n_classes)
                batch_loss = loss_profile[custom_loss_function]
                batch_log.append(batch_loss)

            final_loss = np.mean(np.array(batch_log))
            final_variance = np.var(np.array(batch_log)) / math.sqrt(len(np.array(batch_log)))
            loss_direction = HyperRegCV.loss_metric_direction(custom_loss_function)

        return final_loss, final_variance, loss_direction, loss_profile

    def fit(self,
            X,
            y,
            hyper_reg_model,
            custom_loss_function='accuracy_score',
            min_training_iterations=20,
            tolerance=20,
            early_stop=60,
            early_timeout=3600,
            OOS_split=0.3,
            train_val_split=0.6,
            validating_framework='train_test_split',
            CP_scorer='lr_mad',
            k_fold_splits=None,
            confidence_level=0.8,
            conformal_retraining_frequency=5,
            prediction_type="classification",
            track_out_of_sample_performance=False,
            presplit_X_y_data_tuple=None,
            verbose=False):
        if custom_loss_function is None:
            if prediction_type == "regression":
                custom_loss_function = "mean_squared_error"
            else:
                custom_loss_function = "accuracy_score"

        parameter_grid = self.get_parameters()
        hyperparameter_tuple_ordered = self.get_hyperparameter_combinations(parameter_grid=parameter_grid)
        hyperparameter_performance_record = self.build_hyperparameter_logger(
            hyperparameter_combinations=hyperparameter_tuple_ordered)
        OOS_optimal_performance_per_iteration = self.build_hyperparameter_logger(
            hyperparameter_combinations=hyperparameter_tuple_ordered)

        n_classes = len(np.unique(np.array(y)))

        if presplit_X_y_data_tuple is not None:
            X_IS, y_IS, X_OOS, y_OOS = presplit_X_y_data_tuple
        elif "nlp" not in prediction_type.lower():
            # TODO: below set normalization to true for normal data,and false for image data, in future handle automatically
            X_OOS, y_OOS, X_IS, y_IS = HyperRegCV.train_val_test_split(X=X, y=y, OOS_split=OOS_split,
                                                                       normalize=False, random_state=self.random_state)
        else:
            X_OOS, y_OOS, X_IS, y_IS = HyperRegCV.tf_idf_train_val_test_split(X=X, y=y, OOS_split=OOS_split,
                                                                              random_state=self.random_state)

        i = 0
        last_retraining_iteration_counter = 0
        start_time = time.time()
        blacklisted_combination_df = np.array([])
        no_sample_idx = []
        for row in range(0, len(hyperparameter_tuple_ordered)):

            combination = hyperparameter_tuple_ordered.iloc[row, :]
            # TODO: IMPORTANT TO ID ALSO FOR THE RANDOM SEARCH FIT METHOD
            if i <= min_training_iterations:
                if verbose:
                    print(combination)

                if i == 0:
                    primary_model_runtime_log = TimeLogger()
                else:
                    primary_model_runtime_log.resume_runtime()

                validation_loss, validation_variance, loss_direction, validation_loss_profile = self._get_validation_loss(
                    combination=combination, X_IS=X_IS, y_IS=y_IS, custom_loss_function=custom_loss_function,
                    prediction_type=prediction_type,
                    validating_framework=validating_framework, train_val_split=train_val_split, n_classes=n_classes,
                    presplit_X_y_data_tuple=presplit_X_y_data_tuple)

                if np.isnan(validation_loss):
                    blacklisted_combination_df = np.append(blacklisted_combination_df, combination)
                    no_sample_idx.append(row)
                    continue
                elif isinstance(hyperparameter_performance_record["accuracy"].median(), int) or isinstance(
                        hyperparameter_performance_record["accuracy"].median(), float):
                    if prediction_type == "regression" and abs(validation_loss) > abs(
                            np.median(y_IS)) * 3:
                        blacklisted_combination_df = np.append(blacklisted_combination_df, combination)
                        no_sample_idx.append(row)
                        continue

                logged_combination = combination.copy()
                logged_combination['accuracy'] = validation_loss
                logged_combination['accuracy_score'] = validation_loss_profile["accuracy_score"]
                logged_combination['log_loss'] = validation_loss_profile["log_loss"]
                logged_combination['variance'] = validation_variance
                # if "conv" in str(self.model).lower() and self.n_epochs_used is not None:
                #     logged_combination['epochs'] = self.n_epochs_used
                log_time = time.time()
                log_elapse = log_time - start_time
                logged_combination['runtime'] = log_elapse
                logged_combination['CI_breach'] = np.nan
                logged_combination['point_predictor_MSE'] = np.nan
                logged_combination['loss_profile_dict'] = validation_loss_profile
                hyperparameter_performance_record.iloc[i, :] = logged_combination
                primary_model_runtime_log.pause_runtime()
                if i == min_training_iterations:
                    total_primary_model_RS_runtime = primary_model_runtime_log.return_runtime()
                    primary_model_RS_runtime_per_iter = total_primary_model_RS_runtime / (min_training_iterations + 1)

                no_sample_idx.append(row)

                if verbose:
                    print(
                        f"Iteration: {i} | Time Elapsed: {log_elapse} | Validation Loss: {validation_loss} | Validation Accuracy: {validation_loss_profile['accuracy_score']}")

            elif i > min_training_iterations:
                hyperparameter_performance_record_cached = hyperparameter_performance_record.iloc[:i, :]
                hyperparameter_performance_record_cached_clean = hyperparameter_performance_record_cached.dropna(
                    subset=[
                        "accuracy"])  # TODO: this is just a workaround for when convnet throws np nan loss, because the config of the network was too complex or odd and it made nan predictions

                hyperparameter_X = hyperparameter_performance_record_cached_clean.drop(
                    ["accuracy", "accuracy_score", "log_loss", "variance", "runtime", "CI_breach",
                     "point_predictor_MSE", "loss_profile_dict"],
                    axis=1)
                hyperparameter_Y = hyperparameter_performance_record_cached_clean["accuracy"]
                if len(hyperparameter_X) < 50:
                    hyperreg_OOS_split = 5 / len(hyperparameter_X)
                else:
                    hyperreg_OOS_split = 20 / len(hyperparameter_X)

                # TODO: TOGGLE ON AND OFF OUTLIER REMOVAL
                if custom_loss_function == "log_loss":  # or prediction_type == "regression" or
                    outlier_remover = OutlierRemover(method="y_IQR")
                    outlier_remover.fit_outlier_remover(X=np.array(hyperparameter_X), y=np.array(hyperparameter_Y))
                    hyperparameter_X, hyperparameter_Y = outlier_remover.apply_outlier_remover(
                        X=np.array(hyperparameter_X), y=np.array(hyperparameter_Y))

                HR_X_OOS, HR_y_OOS, HR_X_IS, HR_y_IS = HyperRegCV.train_val_test_split(X=np.array(hyperparameter_X),
                                                                                       y=np.array(hyperparameter_Y),
                                                                                       OOS_split=hyperreg_OOS_split,
                                                                                       normalize=False,
                                                                                       random_state=self.random_state)

                if (i == min_training_iterations + 1) or (
                        i - last_retraining_iteration_counter >= conformal_retraining_frequency):
                    scaler = StandardScaler()
                    scaler.fit(HR_X_IS)
                    HR_X_space = scaler.transform(hyperparameter_tuple_ordered.to_numpy())
                    HR_X_IS = scaler.transform(HR_X_IS)
                    HR_X_OOS = scaler.transform(HR_X_OOS)
                    hyperparameter_X_norm = scaler.transform(
                        hyperparameter_X)  # TODO: for this one if you're using it and retraining on the whole dataset then must use the whole datset as the scaler.fit earlier in code

                    # HR_CP_fitted_model = hyper_reg_model.fit(
                    #     HR_X_IS, HR_y_IS)
                    # HR_fitted_model = hyper_reg_model.fit(
                    #     hyperparameter_X_norm, hyperparameter_Y)

                    if i == min_training_iterations + 1:
                        stored_best_hyperparameter_config = None

                    conformer = Conformal(model=hyper_reg_model,
                                          scoring=CP_scorer,
                                          X_obs=np.array(HR_X_OOS),
                                          y_obs=np.array(HR_y_OOS),
                                          X_train=np.array(HR_X_IS),
                                          y_train=np.array(HR_y_IS),
                                          X_obs_train=np.array(hyperparameter_X_norm),
                                          y_obs_train=np.array(hyperparameter_Y),
                                          X_full=np.array(HR_X_space),
                                          random_state=self.random_state,
                                          previous_best_hyperparameter_config=stored_best_hyperparameter_config)

                    if i == min_training_iterations + 1:
                        CP_quantile, hyperreg_model_runtime_per_iter = conformer.conformal_quantile(
                            confidence_level=confidence_level)
                    else:
                        runtime_optimized_combinations = ConformalRuntimeOptimizer.get_optimal_number_of_secondary_model_parameter_combinations(
                            primary_model_runtime=primary_model_RS_runtime_per_iter,
                            secondary_model_runtime=hyperreg_model_runtime_per_iter,
                            secondary_model_retraining_freq=conformal_retraining_frequency,
                            secondary_model_runtime_as_frac_of_primary_model_runtime=1.5)

                        CP_quantile, hyperreg_model_runtime_per_iter_new = conformer.conformal_quantile(
                            confidence_level=confidence_level, n_of_param_combinations=runtime_optimized_combinations)

                        if hyperreg_model_runtime_per_iter_new is not None:
                            hyperreg_model_runtime_per_iter = hyperreg_model_runtime_per_iter_new

                    baseline_accuracies, CP_intervals, CP_bounds = conformer.generate_confidence_intervals(
                        conformal_quantile=CP_quantile)

                    stored_best_hyperparameter_config = conformer.best_hyperparameter_config
                    # if "forest" in str(hyper_reg_model).lower():
                    #     PlotHelper.plot_sorted_conformal_variance(baseline_accuracies=baseline_accuracies,
                    #                                               CP_intervals=CP_intervals,
                    #                                               CP_scorer=CP_scorer,
                    #                                               hyper_reg_model=hyper_reg_model,
                    #                                               confidence_level=confidence_level)

                    if loss_direction == 'direct':
                        CP_bound = CP_bounds["max_bound"].to_numpy()
                    elif loss_direction == 'inverse':
                        CP_bound = CP_bounds["min_bound"].to_numpy()

                    last_retraining_iteration_counter = i

                if loss_direction == 'direct':
                    if no_sample_idx != []:
                        CP_bound[no_sample_idx] = -10000000
                    maximal_idx = np.argmax(CP_bound)
                    no_sample_idx.append(maximal_idx)
                elif loss_direction == 'inverse':
                    if no_sample_idx != []:
                        CP_bound[no_sample_idx] = 10000000
                    maximal_idx = np.argmin(CP_bound)
                    no_sample_idx.append(maximal_idx)

                maximal_parameter = hyperparameter_tuple_ordered.reset_index(drop=True).iloc[maximal_idx,
                                    :]  # NOTE: using the non normalized one now because i need to feed rael parmeters later to the base model

                validation_loss, validation_variance, loss_direction, validation_loss_profile = self._get_validation_loss(
                    combination=maximal_parameter, X_IS=X_IS, y_IS=y_IS, custom_loss_function=custom_loss_function,
                    prediction_type=prediction_type,
                    validating_framework=validating_framework, train_val_split=train_val_split, n_classes=n_classes,
                    presplit_X_y_data_tuple=presplit_X_y_data_tuple)

                if np.isnan(validation_loss) or (prediction_type == "regression" and abs(validation_loss) > abs(
                        np.median(y_IS)) * 3):
                    blacklisted_combination_df = np.append(blacklisted_combination_df, maximal_parameter)
                    # NOTE: here we don't append to no_sample_idx because it's already been appended previously in the maximal_idx section above
                    continue

                # print("IS primary model loss: ", validation_loss)

                maximal_parameter_logged = maximal_parameter.copy()
                maximal_parameter_logged['accuracy'] = validation_loss
                maximal_parameter_logged['accuracy_score'] = validation_loss_profile["accuracy_score"]
                maximal_parameter_logged['log_loss'] = validation_loss_profile["log_loss"]
                maximal_parameter_logged['variance'] = validation_variance
                # if "conv" in str(self.model).lower() and self.n_epochs_used is not None:
                #     maximal_parameter_logged['epochs'] = self.n_epochs_used
                log_time = time.time()
                log_elapse = log_time - start_time
                maximal_parameter_logged['runtime'] = log_elapse
                if (maximal_parameter_logged['accuracy'] > CP_bounds["max_bound"].to_numpy()[maximal_idx]) or (
                        maximal_parameter_logged['accuracy'] < CP_bounds["min_bound"].to_numpy()[maximal_idx]):
                    maximal_parameter_logged['CI_breach'] = 1
                else:
                    maximal_parameter_logged['CI_breach'] = 0
                maximal_parameter_logged['point_predictor_MSE'] = conformer.best_point_predictor_MSE
                maximal_parameter_logged['loss_profile_dict'] = validation_loss_profile
                hyperparameter_performance_record.iloc[i, :] = maximal_parameter_logged
                # At each inner loop, predict on OOS using optimal parameters
                # for plot of optimal performance over number of iterations

                if verbose:
                    print(
                        f"Iteration: {i} | Time Elapsed: {log_elapse} | Validation Loss: {validation_loss} | Validation Accuracy: {validation_loss_profile['accuracy_score']}")

            if track_out_of_sample_performance:

                if loss_direction == 'direct':
                    optimal_idx = (hyperparameter_performance_record['accuracy']).argmax()
                elif loss_direction == 'inverse':
                    optimal_idx = (hyperparameter_performance_record['accuracy']).argmin()
                optimal_parameters = hyperparameter_performance_record.iloc[optimal_idx, :]

                optimal_parameters_conv_modified = optimal_parameters.copy()
                if "conv" in str(self.model).lower() and self.n_epochs_used is not None:
                    optimal_parameters_conv_modified['epochs'] = self.n_epochs_used

                if validating_framework == 'batch_mean':
                    optimal_loss, optimal_variance, loss_direction, optimal_loss_profile = self._get_OOS_loss(
                        combination=optimal_parameters_conv_modified,
                        X_IS=X_IS, y_IS=y_IS, X_OOS=X_OOS,
                        y_OOS=y_OOS,
                        custom_loss_function=custom_loss_function,
                        prediction_type=prediction_type,
                        batch_data=True,
                        n_classes=n_classes)
                else:
                    # for m in range(0, 3):
                    optimal_loss, optimal_variance, loss_direction, optimal_loss_profile = self._get_OOS_loss(
                        combination=optimal_parameters,
                        X_IS=X_IS, y_IS=y_IS,
                        X_OOS=X_OOS,
                        y_OOS=y_OOS,
                        custom_loss_function=custom_loss_function,
                        prediction_type=prediction_type,
                        batch_data=False,
                        n_classes=n_classes)
                    # print("repetition test: ", optimal_loss)
                optimal_parameters_logged = optimal_parameters.copy()
                optimal_parameters_logged["accuracy"] = optimal_loss
                optimal_parameters_logged["accuracy_score"] = optimal_loss_profile["accuracy_score"]
                optimal_parameters_logged["log_loss"] = optimal_loss_profile["log_loss"]
                optimal_parameters_logged['variance'] = optimal_variance
                log_time = time.time()
                log_elapse = log_time - start_time
                optimal_parameters_logged['runtime'] = log_elapse
                optimal_parameters_logged['CI_breach'] = np.nan
                optimal_parameters_logged['point_predictor_MSE'] = np.nan
                optimal_parameters_logged['loss_profile_dict'] = optimal_loss_profile
                OOS_optimal_performance_per_iteration.iloc[i, :] = optimal_parameters_logged

                # print("cleaned optimal IS loss: ", hyperparameter_performance_record['accuracy'].iloc[optimal_idx])
                if verbose:
                    print(i, log_elapse, "th iteration OOS Accuracy:", optimal_loss)  # , end='\r')

            # set tolerance:
            if i > (min_training_iterations + tolerance) and (
                    round(OOS_optimal_performance_per_iteration["accuracy"].iloc[i - tolerance:i].mean(), 4) == \
                    round(OOS_optimal_performance_per_iteration["accuracy"].iloc[i], 4)):
                OOS_optimal_performance_per_iteration = OOS_optimal_performance_per_iteration.iloc[:i, :]
                hyperparameter_performance_record = hyperparameter_performance_record.iloc[:i, :]
                OOS_optimal_performance_per_iteration["95_CI"] = (
                        1.645 * OOS_optimal_performance_per_iteration["variance"])
                hyperparameter_performance_record["95_CI"] = (1.645 * hyperparameter_performance_record["variance"])
                break
            if (early_stop is not None and i > early_stop) or (log_elapse is not None and log_elapse > early_timeout):
                OOS_optimal_performance_per_iteration = OOS_optimal_performance_per_iteration.iloc[:i, :]
                hyperparameter_performance_record = hyperparameter_performance_record.iloc[:i, :]
                OOS_optimal_performance_per_iteration["95_CI"] = (
                        1.645 * OOS_optimal_performance_per_iteration["variance"])
                hyperparameter_performance_record["95_CI"] = (1.645 * hyperparameter_performance_record["variance"])
                break

            i = i + 1

        return OOS_optimal_performance_per_iteration, hyperparameter_performance_record

    def get_true_loss_profile(self, X, y, OOS_split=0.3):
        if "mlp" in str(self.model).lower():

            parameter_grid = self.get_parameters()
            hyperparameter_tuple_ordered = self.get_hyperparameter_combinations(parameter_grid=parameter_grid)
            hyperparameter_performance_record = self.build_hyperparameter_logger(
                hyperparameter_combinations=hyperparameter_tuple_ordered)

            i = 0
            for row in tqdm(range(0, len(hyperparameter_tuple_ordered))):
                combination = hyperparameter_tuple_ordered.iloc[row, :]
                X_OOS, y_OOS, X_IS, y_IS = HyperRegCV.train_val_test_split(X=X, y=y, OOS_split=OOS_split,
                                                                           random_state=self.random_state)

                hidden_layers = HyperRegCV.tuplify_network_layer_sizes(combination)
                if "reg" in str(self.model).lower():
                    optimal_fitted_model = MLPRegressor(solver=HyperRegCV.solver_mapper(combination),
                                                        learning_rate_init=combination["learning_rate_init"],
                                                        alpha=combination["alpha"],
                                                        hidden_layer_sizes=hidden_layers,
                                                        random_state=self.random_state).fit(
                        X_IS, y_IS)
                elif "clas" in str(self.model).lower():
                    optimal_fitted_model = MLPClassifier(solver=HyperRegCV.solver_mapper(combination),
                                                         learning_rate_init=combination["learning_rate_init"],
                                                         alpha=combination["alpha"],
                                                         hidden_layer_sizes=hidden_layers,
                                                         random_state=self.random_state).fit(
                        X_IS, y_IS)
                optimal_y_pred = optimal_fitted_model.predict(X_OOS)
                if prediction_type == "classification" or prediction_type == "nlp_classification":
                    optimal_y_pred_proba = optimal_fitted_model.predict_proba(X_OOS)
                else:
                    optimal_y_pred_proba = None
                loss_profile = HyperRegCV.loss_profile(y_pred=optimal_y_pred, y_pred_proba=optimal_y_pred_proba,
                                                       y_obs=y_OOS, n_classes=n_classes)
                optimal_loss = loss_profile[custom_loss_function]
                loss_direction = HyperRegCV.loss_metric_direction(custom_loss_function)

                hyperparameter_performance_record.iloc[i, :] = [HyperRegCV.solver_mapper(combination),
                                                                combination["learning_rate_init"],
                                                                combination["alpha"],
                                                                str(hidden_layers),
                                                                combination["layer_1"],
                                                                combination["layer_2"],
                                                                combination["layer_3"],
                                                                combination["adam"],
                                                                combination["sgd"], optimal_loss]
                i = i + 1
        return hyperparameter_performance_record

    def fit_random_search(self, X, y, custom_loss_function=None, n_searches=60, max_runtime=3600, OOS_split=0.3,
                          train_val_split=0.6,
                          validating_framework='train_test_split', k_fold_splits=None,
                          prediction_type="classification",
                          track_out_of_sample_performance=False,
                          presplit_X_y_data_tuple=None,
                          verbose=False):
        if custom_loss_function is None:
            if prediction_type == "regression":
                custom_loss_function = "mean_squared_error"
            else:
                custom_loss_function = "accuracy_score"

        parameter_grid = self.get_parameters()
        hyperparameter_tuple_ordered = self.get_hyperparameter_combinations(parameter_grid=parameter_grid)
        hyperparameter_performance_record = self.build_hyperparameter_logger(
            hyperparameter_combinations=hyperparameter_tuple_ordered)
        OOS_optimal_performance_per_iteration = self.build_hyperparameter_logger(
            hyperparameter_combinations=hyperparameter_tuple_ordered)

        n_classes = len(np.unique(np.array(y)))

        if presplit_X_y_data_tuple is not None:
            X_IS, y_IS, X_OOS, y_OOS = presplit_X_y_data_tuple
        elif "nlp" not in prediction_type.lower():
            # TODO: below set normalization to true for normal data,and false for image data, in future handle automatically
            X_OOS, y_OOS, X_IS, y_IS = HyperRegCV.train_val_test_split(X=X, y=y, OOS_split=OOS_split,
                                                                       normalize=False, random_state=self.random_state)
        else:
            X_OOS, y_OOS, X_IS, y_IS = HyperRegCV.tf_idf_train_val_test_split(X=X, y=y, OOS_split=OOS_split,
                                                                              random_state=self.random_state)

        i = 0
        start_time = time.time()
        for row in range(0, len(hyperparameter_tuple_ordered)):

            combination = hyperparameter_tuple_ordered.iloc[row, :]

            validation_loss, validation_variance, loss_direction, validation_loss_profile = self._get_validation_loss(
                combination=combination,
                X_IS=X_IS, y_IS=y_IS,
                custom_loss_function=custom_loss_function,
                prediction_type=prediction_type,
                validating_framework=validating_framework,
                train_val_split=train_val_split,
                n_classes=n_classes,
                presplit_X_y_data_tuple=presplit_X_y_data_tuple)

            if np.isnan(validation_loss):
                continue
            elif isinstance(hyperparameter_performance_record["accuracy"].median(), int) or isinstance(
                    hyperparameter_performance_record["accuracy"].median(), float):
                if prediction_type == "regression" and abs(validation_loss) > abs(
                        np.median(y_IS)) * 3:
                    continue

            logged_combination = combination.copy()
            logged_combination['accuracy'] = validation_loss
            logged_combination['accuracy_score'] = validation_loss_profile["accuracy_score"]
            logged_combination['log_loss'] = validation_loss_profile["log_loss"]
            logged_combination['variance'] = validation_variance
            # if "conv" in str(self.model).lower() and self.n_epochs_used is not None:
            #     logged_combination['epochs'] = self.n_epochs_used
            log_time = time.time()
            log_elapse = log_time - start_time
            logged_combination['runtime'] = log_elapse
            logged_combination['CI_breach'] = np.nan
            logged_combination['point_predictor_MSE'] = np.nan
            logged_combination['loss_profile_dict'] = validation_loss_profile
            hyperparameter_performance_record.iloc[i, :] = logged_combination

            if verbose:
                print(
                    f"Iteration: {i} | Time Elapsed: {log_elapse} | Validation Loss: {validation_loss} | Validation Accuracy: {validation_loss_profile['accuracy_score']}")

            if track_out_of_sample_performance:

                if loss_direction == 'direct':
                    optimal_idx = (hyperparameter_performance_record['accuracy']).argmax()
                elif loss_direction == 'inverse':
                    optimal_idx = (hyperparameter_performance_record['accuracy']).argmin()
                optimal_parameters = hyperparameter_performance_record.iloc[optimal_idx, :]

                optimal_parameters_conv_modified = optimal_parameters.copy()
                if "conv" in str(self.model).lower() and self.n_epochs_used is not None:
                    optimal_parameters_conv_modified['epochs'] = self.n_epochs_used

                if validating_framework == 'batch_mean':
                    optimal_loss, optimal_variance, loss_direction, optimal_loss_profile = self._get_OOS_loss(
                        combination=optimal_parameters,
                        X_IS=X_IS, y_IS=y_IS,
                        X_OOS=X_OOS,
                        y_OOS=y_OOS,
                        custom_loss_function=custom_loss_function,
                        prediction_type=prediction_type,
                        batch_data=True,
                        n_classes=n_classes)
                else:
                    optimal_loss, optimal_variance, loss_direction, optimal_loss_profile = self._get_OOS_loss(
                        combination=optimal_parameters,
                        X_IS=X_IS, y_IS=y_IS,
                        X_OOS=X_OOS,
                        y_OOS=y_OOS,
                        custom_loss_function=custom_loss_function,
                        prediction_type=prediction_type,
                        batch_data=False,
                        n_classes=n_classes)

                optimal_parameters_logged = optimal_parameters.copy()
                optimal_parameters_logged["accuracy"] = optimal_loss
                optimal_parameters_logged["accuracy_score"] = optimal_loss_profile["accuracy_score"]
                optimal_parameters_logged["log_loss"] = optimal_loss_profile["log_loss"]
                optimal_parameters_logged['variance'] = optimal_variance
                log_time = time.time()
                log_elapse = log_time - start_time
                optimal_parameters_logged['runtime'] = log_elapse
                optimal_parameters_logged['CI_breach'] = np.nan
                optimal_parameters_logged['point_predictor_MSE'] = np.nan
                optimal_parameters_logged['loss_profile_dict'] = optimal_loss_profile
                OOS_optimal_performance_per_iteration.iloc[i, :] = optimal_parameters_logged

                if verbose:
                    print(i, log_elapse, "th iteration OOS Accuracy:", optimal_loss, end='\r')

            if i > n_searches or (log_elapse is not None and log_elapse > max_runtime):
                OOS_optimal_performance_per_iteration = OOS_optimal_performance_per_iteration.iloc[:i, :]
                hyperparameter_performance_record = hyperparameter_performance_record.iloc[:i, :]
                OOS_optimal_performance_per_iteration["95_CI"] = (
                        1.645 * OOS_optimal_performance_per_iteration["variance"])
                hyperparameter_performance_record["95_CI"] = (1.645 * hyperparameter_performance_record["variance"])
                break
            i = i + 1

        return OOS_optimal_performance_per_iteration, hyperparameter_performance_record

    @staticmethod
    def solver_mapper(hyperparameter_tuple_row):
        if hyperparameter_tuple_row["adam"] == 1:
            return "adam"
        if hyperparameter_tuple_row["sgd"] == 1:
            return "sgd"


class ConvNet:
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
        return "ConvNet()"

    def __repr__(self):
        return "ConvNet()"

    def fit(self, X, y, val_X=None, val_Y=None):
        if self.random_state is not None:
            random.seed(1234)
            np.random.seed(1234)
            tf.random.set_seed(1234)

        model = models.Sequential()
        input_shape = list(X.shape)
        input_shape.remove(max(input_shape))
        try:
            channels = input_shape[2]
        except:
            channels = 1

        model.add(
            layers.Conv2D(self.layer_1_tuple[0], (self.layer_1_tuple[1], self.layer_1_tuple[1]), activation='relu',
                          padding='same'
                          # , kernel_initializer='he_uniform'
                          # bias_initializer=initializers.Zeros()
                          , input_shape=tuple([(set([x for x in (X.shape) if (X.shape).count(x) > 1])).pop(),
                                               (set([x for x in (X.shape) if (X.shape).count(x) > 1])).pop(),
                                               channels])))
        # if self.batch_norm == 1:
        #     model.add(layers.BatchNormalization())
        # model.add(
        #     layers.Conv2D(self.layer_1_tuple[0], (self.layer_1_tuple[1], self.layer_1_tuple[1]), activation='relu',
        #                   padding='same'
        #                   # ,kernel_initializer='he_uniform'
        #                   # bias_initializer=initializers.Zeros()
        #                   ))
        if self.batch_norm == 1:
            model.add(layers.BatchNormalization())
        if self.layer_1_pooling is not None:
            if self.layer_1_pooling != 0:
                model.add(layers.MaxPooling2D((self.layer_1_pooling, self.layer_1_pooling)))
                if self.batch_norm == 1:
                    model.add(layers.BatchNormalization())
        if self.layer_2_tuple is not None:
            if self.layer_2_tuple[0] != 0 and self.layer_2_tuple[1] != 0:
                model.add(layers.Conv2D(self.layer_2_tuple[0], (self.layer_2_tuple[1], self.layer_2_tuple[1])
                                        # , kernel_initializer='he_uniform'                                        #                                              seed=self.random_state),
                                        # bias_initializer=initializers.Zeros()
                                        , activation='relu',
                                        padding='same'))
                # if self.batch_norm == 1:
                #     model.add(layers.BatchNormalization())
                # model.add(layers.Conv2D(self.layer_2_tuple[0], (self.layer_2_tuple[1], self.layer_2_tuple[1])
                #                         # , kernel_initializer='he_uniform'                                        #                                              seed=self.random_state),
                #                         # bias_initializer=initializers.Zeros()
                #                         , activation='relu',
                #                         padding='same'))
                if self.batch_norm == 1:
                    model.add(layers.BatchNormalization())
        if self.layer_2_pooling is not None:
            if self.layer_2_pooling != 0:
                model.add(layers.MaxPooling2D((self.layer_2_pooling, self.layer_2_pooling)))
                if self.batch_norm == 1:
                    model.add(layers.BatchNormalization())
        if self.layer_3_tuple is not None:
            if self.layer_3_tuple[0] != 0 and self.layer_3_tuple[1] != 0:
                model.add(layers.Conv2D(self.layer_3_tuple[0], (self.layer_3_tuple[1], self.layer_3_tuple[1])
                                        # , kernel_initializer='he_uniform'                                        #                                              seed=self.random_state),
                                        # bias_initializer=initializers.Zeros()
                                        , activation='relu',
                                        padding='same'))
                # if self.batch_norm == 1:
                #     model.add(layers.BatchNormalization())
                # model.add(layers.Conv2D(self.layer_3_tuple[0], (self.layer_3_tuple[1], self.layer_3_tuple[1])
                #                         # , kernel_initializer='he_uniform'                                        #                                              seed=self.random_state),
                #                         # bias_initializer=initializers.Zeros()
                #                         , activation='relu',
                #                         padding='same'))
                if self.batch_norm == 1:
                    model.add(layers.BatchNormalization())
        if self.layer_3_pooling is not None:
            if self.layer_3_pooling != 0:
                model.add(layers.MaxPooling2D((self.layer_3_pooling, self.layer_3_pooling)))
                if self.batch_norm == 1:
                    model.add(layers.BatchNormalization())
        if self.layer_4_tuple is not None:
            if self.layer_4_tuple[0] != 0 and self.layer_4_tuple[1] != 0:
                model.add(layers.Conv2D(self.layer_4_tuple[0], (self.layer_4_tuple[1], self.layer_4_tuple[1])
                                        # , kernel_initializer='he_uniform'                                        #                                              seed=self.random_state),
                                        # bias_initializer=initializers.Zeros()
                                        , activation='relu',
                                        padding='same'))
                # if self.batch_norm == 1:
                #     model.add(layers.BatchNormalization())
                # model.add(layers.Conv2D(self.layer_4_tuple[0], (self.layer_4_tuple[1], self.layer_4_tuple[1])
                #                         # , kernel_initializer='he_uniform'                                        #                                              seed=self.random_state),
                #                         # bias_initializer=initializers.Zeros()
                #                         , activation='relu',
                #                         padding='same'))
                if self.batch_norm == 1:
                    model.add(layers.BatchNormalization())
        if self.layer_4_pooling is not None:
            if self.layer_4_pooling != 0:
                model.add(layers.MaxPooling2D((self.layer_4_pooling, self.layer_4_pooling)))
                if self.batch_norm == 1:
                    model.add(layers.BatchNormalization())

        model.add(layers.Flatten())
        model.add(layers.Dense(self.dense_layer_1, activation='relu'
                               # , kernel_initializer='he_uniform'
                               # bias_initializer=initializers.Zeros()
                               ))
        model.add(tf.keras.layers.Dropout(self.drop_out_rate))  # , seed=self.random_state))
        if self.dense_layer_2 is not None:
            if self.dense_layer_2 != 0:
                model.add(layers.Dense(self.dense_layer_2, activation='relu'
                                       # , kernel_initializer='he_uniform'                                       #                                              seed=self.random_state),
                                       # bias_initializer=initializers.Zeros()
                                       ))
                model.add(tf.keras.layers.Dropout(self.drop_out_rate))  # , seed=self.random_state))

        model.add(layers.Dense(len(np.unique(y)), activation='softmax'))

        if self.solver == 'adam':
            optimizer_config = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.solver == 'sgd':
            optimizer_config = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        model.compile(loss='sparse_categorical_crossentropy',
                      # we're using from logits equals true in this loss function because we didn't add a softmax layer at the end of the sequential model api, if you later add this, remember to change this loss to simply crossentropy string, or just set logit to false
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


# import pandas as pd
# raw_data = pd.read_csv(Filing.parent_folder_path + "/datasets/covertype/covtype.data", sep=',')

class PreProcessingHelper:
    @staticmethod
    def one_hot_encode_variables(X):
        X_encoded = X.copy()
        X_final = np.zeros(len(X_encoded)).reshape(len(X_encoded), 1)
        for i in range(0, X_encoded.shape[1]):
            try:
                X_encoded[:, i].astype(float)
                X_final = np.hstack([X_final, X_encoded[:, i].reshape(len(X_encoded[:, i]), 1)])
            except:
                X_append = pd.get_dummies(pd.Series(X_encoded[:, i])).to_numpy()
                X_final = np.hstack([X_final, X_append])
        X_final = X_final[:, 1:]
        return X_final


class ClassRebalancer:
    @staticmethod
    def _binary_class_rebalancing(data, y_name, type="majority_undersampling"):
        if type == "majority_undersampling":
            majority_class = data[y_name].mode()[0]
            minority_class_filtered_dataset = data[data[y_name] != majority_class]
            majority_class_filtered_dataset = data[data[y_name] == majority_class]
            majority_class_filtered_dataset = majority_class_filtered_dataset.sample(
                n=len(minority_class_filtered_dataset),
                replace=True, random_state=1234)
            rebalanced_data = pd.concat([minority_class_filtered_dataset, majority_class_filtered_dataset], axis=0)
        elif type == "minority_oversampling":
            majority_class = data[y_name].mode()[0]
            minority_class_filtered_dataset = data[data[y_name] != majority_class]
            majority_class_filtered_dataset = data[data[y_name] == majority_class]
            minority_class_filtered_dataset = minority_class_filtered_dataset.sample(
                n=len(majority_class_filtered_dataset),
                replace=True, random_state=1234)
            rebalanced_data = pd.concat([minority_class_filtered_dataset, majority_class_filtered_dataset], axis=0)

        return rebalanced_data

    @staticmethod
    def _multi_label_class_rebalancing(data, y_name, type="majority_undersampling"):
        if type == "majority_undersampling":
            lowest_represented_class = data[y_name].value_counts().index[-1]
            lowest_represented_class_filtered_dataset = data[data[y_name] == lowest_represented_class]
            rebalanced_data = lowest_represented_class_filtered_dataset.copy()
            for label in data[y_name].unique():
                if label != lowest_represented_class:
                    filtered_dataset = data[data[y_name] == label]
                    resampled_filtered_dataset = filtered_dataset.sample(
                        n=len(lowest_represented_class_filtered_dataset),
                        replace=True, random_state=1234)
                    rebalanced_data = pd.concat([rebalanced_data, resampled_filtered_dataset], axis=0)

        return rebalanced_data


class OutlierRemover:
    def __init__(self, method):
        self.method = method

        self.q_low_arr = []
        self.q_hi_arr = []
        self.IQR_arr = []

    def fit_outlier_remover(self, X, y):

        if self.method == "X_IQR":
            for i in range(0, X.shape[1]):
                q_low = np.quantile(X[:, i], 0.25)
                q_hi = np.quantile(X[:, i], 0.75)
                IQR = abs(q_hi - q_low)
                self.q_low_arr.append[q_low]
                self.q_hi_arr.append[q_hi]
                self.IQR_arr.append[IQR]

        elif self.method == "y_IQR":
            self.q_low_arr = np.quantile(y, 0.25)
            self.q_hi_arr = np.quantile(y, 0.75)
            self.IQR_arr = abs(self.q_hi_arr - self.q_low_arr)

    def apply_outlier_remover(self, X, y):
        if self.method == "X_IQR":
            idx_list = []
            for i in range(0, X.shape[1]):
                idx_low = list(np.where(X[:, i] < (self.q_low_arr[i] - 1.5 * self.IQR_arr[i]))[0])
                idx_hi = list(np.where(X[:, i] > (self.q_hi_arr[i] + 1.5 * self.IQR_arr[i]))[0])
                idx = idx_low + idx_hi
                idx_list = idx_list + idx

            idx_keep = list(set(list(range(0, len(X)))) - set(idx_list))
            X_outlier_cleaned = X[idx_keep, :]
            y_outlier_cleaned = y[idx_keep]

        elif self.method == "y_IQR":
            idx_low = list(
                np.where(y < (self.q_low_arr - 1.5 * self.IQR_arr))[0])  # TODO: revert to 1.5 or make thisparameter
            idx_hi = list(np.where(y > (self.q_hi_arr + 1.5 * self.IQR_arr))[0])
            idx = idx_hi  # idx_low + idx_hi

            idx_keep = list(set(list(range(0, len(X)))) - set(idx))
            X_outlier_cleaned = X[idx_keep, :]
            y_outlier_cleaned = y[idx_keep]

        return X_outlier_cleaned, y_outlier_cleaned

# (x_train, y_train), (x_test, y_test) = keras_datasets.cifar10.load_data()
# x_train = x_train / 255
# x_test = x_test / 255
# undersampling_index = list(np.random.choice(len(x_train), 10000, replace=False))
# x_train = x_train[undersampling_index, :]
# y_train = y_train[undersampling_index]
#
# ConvNet(solver='adam',
#         learning_rate=0.001,
#         drop_out_rate=0.3,
#         layer_1_tuple=(32, 3),
#         dense_layer_1=256,
#         batch_norm=0,
#         layer_1_pooling=0,
#         layer_2_pooling=2,
#         layer_3_pooling=2,
#         layer_4_pooling=2,
#         layer_2_tuple=(128, 3),
#         layer_3_tuple=(255, 3),
#         layer_4_tuple=(255, 3),
#         dense_layer_2=None,
#         random_state=1234).fit(x_train, y_train, x_test, y_test)
