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
from sklearn import metrics
from sklearn.model_selection import KFold
import tensorflow as tf

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler

from SCHO.conformal_methods import Conformal
from SCHO.utils.runtime_eval import TimeLogger
from SCHO.utils.runtime_eval import ConformalRuntimeOptimizer
from nlp_helper import NLPEncoder
from SCHO.wrappers.keras_wrappers import CNNClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
from multiprocessing import Pool


class SeqTune:
    def __init__(self, model, hyperparameter_space=None, random_state=None):
        self.model = model
        self.random_state = random_state
        self.hyperparameter_space = hyperparameter_space

        self._hyperparameter_performance_record = None
        self.best_params_ = None
        self.best_score_ = None
        self._best_model = None
        self._n_epochs_used = None

    def get_default_hyperparameter_space(self):
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

        elif "cnn" in str(self.model).lower():
            solver_list = ['adam', 'sgd']
            learning_rate_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
            drop_out_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            n_layers = [2, 3, 4]
            layer_size = list(range(16, 257, 16))
            dense_layer_1_neurons_list = [100, 200, 512]
            dense_layer_2_neurons_list = [0, 0, 0, 0, 50, 100]

            parameter_dict = {'solver': solver_list,
                              'learning_rate': learning_rate_list,
                              'drop_out_rate': drop_out_rate_list,
                              'n_layers': n_layers,
                              'layer_size': layer_size,
                              'dl1_neurons': dense_layer_1_neurons_list,
                              'dl2_neurons': dense_layer_2_neurons_list
                              }

        elif "forest" in str(self.model).lower():
            n_estimators_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700]
            min_samples_split_list = [2, 5, 10, 20, 30]
            # max_features_list = ["sqrt", "log2", "auto"]

            parameter_dict = {'n_estimators': n_estimators_list,
                              'min_samples_split': min_samples_split_list}
            # 'max_features': max_features_list}

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

        elif "mlp" in str(self.model).lower() or "cnn" in str(self.model).lower():

            for i in tqdm(range(0, 200000)):
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
                                elif "cnn" in str(self.model).lower():
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
                                elif "cnn" in str(self.model).lower():
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

        else:
            for i in tqdm(range(0, 10000)):
                parameter_combination = []
                parameter_combination_columns = []
                for key in parameter_grid.keys():
                    parameter = random.choice(parameter_grid[key])
                    parameter_combination.append(parameter)
                    parameter_combination_columns.append(key)
                if i == 0:
                    hyperparameter_tuple = pd.DataFrame(parameter_combination).transpose()
                    hyperparameter_tuple.columns = parameter_combination_columns
                else:
                    hyperparameter_tuple.loc[len(hyperparameter_tuple)] = np.transpose(parameter_combination)

            hyperparameter_tuple = hyperparameter_tuple.drop_duplicates()

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
        elif prediction_type == "classification":
            try:
                if y_pred_proba.shape[1] == 2:
                    entropy = metrics.log_loss(y_obs, y_pred_proba[:, 1])
                else:
                    one_hot_encoded_y_obs = SeqTune.pivot_classes(y=y_obs, n_classes=n_classes)
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

    def build_hyperparameter_logger(self, hyperparameter_combinations, stripped=False):

        hyperparameter_performance_record_df = hyperparameter_combinations.copy()
        for i in range(0, hyperparameter_performance_record_df.shape[1]):
            hyperparameter_performance_record_df.iloc[:, i] = np.nan
        hyperparameter_performance_record_df["accuracy"] = np.nan
        if not stripped:
            hyperparameter_performance_record_df["accuracy_score"] = np.nan
            hyperparameter_performance_record_df["log_loss"] = np.nan
            hyperparameter_performance_record_df["variance"] = np.nan
            hyperparameter_performance_record_df["runtime"] = np.nan
            hyperparameter_performance_record_df["CI_breach"] = np.nan
            hyperparameter_performance_record_df["point_predictor_MSE"] = np.nan
            hyperparameter_performance_record_df["loss_profile_dict"] = np.nan

        return hyperparameter_performance_record_df

    @staticmethod
    def tuplify_network_layer_sizes(
            combination):  # TODO: assumes column names are sorted from layer 1 to layer N, otherwise will not create ordered list, change to reduce this vulnerability
        layer_tuple = ()
        for column_name in list(combination.index):
            if "layer_" in column_name.lower():
                if int(combination[column_name]) != 0:
                    layer_tuple = layer_tuple + (int(combination[column_name]),)
        return layer_tuple

    def combination_row_2_input_dict(self, combination_row):
        if "mlp" in str(self.model).lower():
            hidden_layers = SeqTune.tuplify_network_layer_sizes(combination_row)
            solver = SeqTune.solver_mapper(combination_row)

            # TODO: hard coded now, but make it able to accept more inputs from the ocmbination_row, only issue is remember to remove automatically entries that ewre just meant to be used by SCHO, eg. discretization of hidden layers
            input_dict = {"solver": try_numeric(solver),
                          "learning_rate_init": try_numeric(combination_row["learning_rate_init"]),
                          "alpha": try_numeric(combination_row["alpha"]),
                          "hidden_layer_sizes": try_numeric(hidden_layers),
                          "random_state": self.random_state}

        elif "cnn" in str(self.model).lower():
            solver = SeqTune.solver_mapper(combination_row)

            # TODO: hard coded now, but make it able to accept more inputs from the ocmbination_row, only issue is remember to remove automatically entries that ewre just meant to be used by SCHO, eg. discretization of hidden layers
            input_dict = {"solver": try_numeric(solver),
                          "learning_rate": try_numeric(combination_row["learning_rate"]),
                          "drop_out_rate": try_numeric(combination_row["drop_out_rate"]),
                          "layer_1_tuple": (int(combination_row['l1_convolutions']), 3),
                          "dense_layer_1": int(combination_row['dl1_neurons']),
                          "layer_2_tuple": (int(combination_row['l2_convolutions']), 3),
                          "layer_3_tuple": (int(combination_row['l3_convolutions']), 3),
                          "layer_4_tuple": (int(combination_row['l4_convolutions']), 3),
                          "dense_layer_2": int(combination_row['dl2_neurons']),
                          "random_state": self.random_state}
        else:
            input_dict = {}
            combination_df = combination_row.reset_index()
            combination_df.columns = ["parameter", "value"]
            for m in range(0, len(combination_df)):
                try:
                    if combination_df["value"].iloc[m] % 1 == 0:
                        value = int(combination_df["value"].iloc[m])
                except:
                    try:
                        value = float(combination_df["value"].iloc[m])
                    except:
                        value = combination_df["value"].iloc[m]
                input_dict[combination_df["parameter"].iloc[m]] = value
        return input_dict

    def _get_validation_loss(self, combination, X_IS, y_IS, custom_loss_function, prediction_type,
                             validating_framework='train_test_split', train_val_split=0.6, k_fold_splits=None,
                             n_classes=2, presplit_X_y_data_tuple=None):

        combination_input_dict = self.combination_row_2_input_dict(combination)

        if validating_framework == 'train_test_split' or validating_framework == 'batch_mean':
            if presplit_X_y_data_tuple is not None:
                X_train, Y_train, X_val, Y_val = presplit_X_y_data_tuple
            else:
                X_val, Y_val, X_train, Y_train = SeqTune.train_val_test_split(X=X_IS,
                                                                              y=y_IS,
                                                                              OOS_split=train_val_split,
                                                                              normalize=False,
                                                                              random_state=self.random_state)

            if "cnn" in str(self.model).lower():
                fitted_model = CNNClassifier(**combination_input_dict)
                fitted_model.fit(X_train, Y_train, X_val, Y_val)

            else:
                fitted_model = eval(str(self.model).split("(")[0] + "(**" + str(combination_input_dict) + ")").fit(
                    X_train, Y_train)

            if validating_framework == 'train_test_split':
                y_pred = fitted_model.predict(X_val)
                if prediction_type == "classification":
                    y_pred_proba = fitted_model.predict_proba(X_val)
                else:
                    y_pred_proba = None

                loss_profile = SeqTune.loss_profile(y_pred=y_pred, y_pred_proba=y_pred_proba, y_obs=Y_val,
                                                    n_classes=n_classes, prediction_type=prediction_type)

                final_loss = loss_profile[custom_loss_function]
                final_variance = np.nan
                loss_direction = SeqTune.loss_metric_direction(custom_loss_function)

            elif validating_framework == 'batch_mean':
                step = 24
                batch_log = []
                for i in range(0, len(X_val), step):
                    if step + i >= len(X_val):
                        break
                    batch_X = X_val[0 + i:step + i, :]
                    batch_y = Y_val[0 + i:step + i]

                    y_pred = fitted_model.predict(batch_X)
                    if prediction_type == "classification":
                        y_pred_proba = fitted_model.predict_proba(batch_X)
                    else:
                        y_pred_proba = None

                    loss_profile = SeqTune.loss_profile(y_pred=y_pred, y_pred_proba=y_pred_proba, y_obs=batch_y,
                                                        n_classes=n_classes)
                    batch_loss = loss_profile[custom_loss_function]
                    batch_log.append(batch_loss)

                final_loss = np.mean(np.array(batch_log))
                final_variance = math.sqrt(np.sum(((np.array(batch_log) - np.mean(np.array(batch_log))) ** 2)) / (
                        len(np.array(batch_log)) - 1))
                loss_direction = SeqTune.loss_metric_direction(custom_loss_function)

        elif validating_framework == 'k_fold':
            kf = KFold(n_splits=k_fold_splits, random_state=self.random_state, shuffle=True)
            K_fold_performance_record = pd.DataFrame(np.zeros((k_fold_splits, 2)))
            K_fold_performance_record.columns = ["fold", "accuracy"]
            fold = 1
            for train_index, test_index in kf.split(X_IS):
                X_train, X_val = X_IS[train_index, :], X_IS[test_index, :]
                Y_train, Y_val = y_IS[train_index], y_IS[test_index]

                if "cnn" in str(self.model).lower():
                    fitted_model = CNNClassifier(**combination_input_dict)
                    fitted_model.fit(X_train, Y_train, X_val, Y_val)

                else:
                    fitted_model = eval(str(self.model).split("(")[0] + "(**" + str(combination_input_dict) + ")").fit(
                        X_train, Y_train)

                y_pred = fitted_model.predict(X_val)
                if prediction_type == "classification":
                    y_pred_proba = y_pred_proba = fitted_model.predict_proba(X_val)
                else:
                    y_pred_proba = None

                loss_profile = SeqTune.loss_profile(y_pred=y_pred, y_pred_proba=y_pred_proba, y_obs=Y_val,
                                                    n_classes=n_classes)
                loss = loss_profile[custom_loss_function]
                loss_direction = SeqTune.loss_metric_direction(custom_loss_function)

                K_fold_performance_record.iloc[(fold - 1), 0] = fold
                K_fold_performance_record.iloc[(fold - 1), 1] = loss

                fold = fold + 1

            cross_fold_performance = K_fold_performance_record.mean()
            final_loss = float(cross_fold_performance["accuracy"])
            final_variance = np.nan

        if "cnn" in str(self.model).lower():
            self._n_epochs_used = fitted_model.n_epochs_used

        return final_loss, final_variance, loss_direction, loss_profile

    @staticmethod
    def get_batch_number(prediction_type, y):
        if prediction_type == "regression":
            desired_batch_size = 60
        elif prediction_type == "classification":
            desired_batch_size = 30 * len(np.unique(np.array(y)))
        batch_number = int(math.floor(len(y) / desired_batch_size))
        return batch_number

    @staticmethod
    def get_batches(X, y, batch_number):
        X_batch_list = []
        y_batch_list = []
        for i in range(0, len(X), math.floor(len(X) / batch_number)):
            X_batch_list.append(X[0 + i:math.floor(len(X) / batch_number) + i,
                                :])  # TODO: sort out the indexes better than floor here and in other method
            y_batch_list.append(y[0 + i:math.floor(len(y) / batch_number) + i])
        return X_batch_list, y_batch_list

    def fit(self,
            X,
            y,
            hyper_reg_model,
            custom_loss_function='accuracy_score',
            min_training_iterations=20,
            tolerance=20,
            early_stop=60,
            early_timeout=3600,
            train_val_split=0.6,
            validating_framework='train_test_split',
            CP_scorer='lr_mad',
            k_fold_splits=None,
            confidence_level=0.8,
            conformal_retraining_frequency=5,
            prediction_type="classification",
            presplit_X_y_data_tuple=None,
            enforced_batch_number=30,
            verbose=False):
        if custom_loss_function is None:
            if prediction_type == "regression":
                custom_loss_function = "mean_squared_error"
            else:
                custom_loss_function = "accuracy_score"

        if self.hyperparameter_space is not None:
            parameter_grid = self.hyperparameter_space
        else:
            parameter_grid = self.get_default_hyperparameter_space()
        hyperparameter_tuple_ordered = self.get_hyperparameter_combinations(parameter_grid=parameter_grid)
        hyperparameter_performance_record = self.build_hyperparameter_logger(
            hyperparameter_combinations=hyperparameter_tuple_ordered)

        n_classes = len(np.unique(np.array(y)))

        i = 0
        last_retraining_iteration_counter = 0
        start_time = time.time()
        blacklisted_combination_df = np.array([])
        no_sample_idx = []

        for row in range(0, len(hyperparameter_tuple_ordered)):
            if i == 0:
                primary_model_runtime_log = TimeLogger()
            else:
                primary_model_runtime_log.resume_runtime()

            if enforced_batch_number == None:
                batch_n = SeqTune.get_batch_number(prediction_type=prediction_type, y=y)
            else:
                batch_n = enforced_batch_number
            X_batches, y_batches = SeqTune.get_batches(X=X, y=y, batch_number=batch_n)
            if (i == 0) or (
                    i - last_retraining_iteration_counter >= conformal_retraining_frequency):
                if i == 0:
                    batch_combinations = hyperparameter_tuple_ordered.sample(n=batch_n,
                                                                             random_state=self.random_state).reset_index(
                        drop=True)
                else:
                    batch_combinations = next_batch_combinations.copy()
                batch_performance_record = self.build_hyperparameter_logger(
                    hyperparameter_combinations=batch_combinations, stripped=True)
                print("Running batch search...")
                for batch_index in tqdm(range(0, len(batch_combinations))):
                    X_batch = X_batches[batch_index]
                    y_batch = y_batches[batch_index]
                    validation_loss, validation_variance, loss_direction, validation_loss_profile = self._get_validation_loss(
                        combination=batch_combinations.iloc[batch_index, :], X_IS=X_batch, y_IS=y_batch,
                        custom_loss_function=custom_loss_function,
                        prediction_type=prediction_type,
                        validating_framework=validating_framework, train_val_split=train_val_split, n_classes=n_classes)
                    logged_batch_combination = batch_combinations.iloc[batch_index, :]
                    logged_batch_combination['accuracy'] = validation_loss
                    batch_performance_record.iloc[batch_index, :] = logged_batch_combination.copy()
                print(batch_performance_record["accuracy"].max())

                primary_model_RS_runtime_per_iter = 2  # TODO, come back and fix properly, reintigrate counter

                # if i == min_training_iterations:
                #     total_primary_model_RS_runtime = primary_model_runtime_log.return_runtime()
                #     primary_model_RS_runtime_per_iter = total_primary_model_RS_runtime / (min_training_iterations + 1)
                #     print("THIS1:", primary_model_RS_runtime_per_iter)

                # no_sample_idx.append(row)

                # if verbose:
                #     print(
                #         f"Iteration: {i} | Time Elapsed: {log_elapse} | Validation Loss: {validation_loss} | Validation Accuracy: {validation_loss_profile['accuracy_score']}")

            batch_performance_record_cached = batch_performance_record.copy()
            batch_performance_record_cached_clean = batch_performance_record_cached.dropna(
                subset=[
                    "accuracy"])  # TODO: this is just a workaround for when convnet throws np nan loss, because the config of the network was too complex or odd and it made nan predictions

            hyperparameter_X = batch_performance_record_cached_clean.drop(
                ["accuracy"],
                axis=1)
            hyperparameter_Y = batch_performance_record_cached_clean["accuracy"]
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

            HR_X_OOS, HR_y_OOS, HR_X_IS, HR_y_IS = SeqTune.train_val_test_split(X=np.array(hyperparameter_X),
                                                                                y=np.array(hyperparameter_Y),
                                                                                OOS_split=hyperreg_OOS_split,
                                                                                normalize=False,
                                                                                random_state=self.random_state)

            if (i == 0) or (
                    i - last_retraining_iteration_counter >= conformal_retraining_frequency):
                scaler = StandardScaler()
                scaler.fit(HR_X_IS)
                HR_X_space = scaler.transform(hyperparameter_tuple_ordered.to_numpy())
                HR_X_IS = scaler.transform(HR_X_IS)
                HR_X_OOS = scaler.transform(HR_X_OOS)
                hyperparameter_X_norm = scaler.transform(
                    hyperparameter_X)  # TODO: for this one if you're using it and retraining on the whole dataset then must use the whole datset as the scaler.fit earlier in code

                if i == 0:
                    point_estimator_stored_best_hyperparameter_config = None
                    variance_estimator_stored_best_hyperparameter_config = None

                conformer = Conformal(point_estimator=hyper_reg_model,
                                      variance_estimator=CP_scorer,
                                      X_obs=np.array(HR_X_OOS),
                                      y_obs=np.array(HR_y_OOS),
                                      X_train=np.array(HR_X_IS),
                                      y_train=np.array(HR_y_IS),
                                      X_obs_train=np.array(hyperparameter_X_norm),
                                      y_obs_train=np.array(hyperparameter_Y),
                                      X_full=np.array(HR_X_space),
                                      random_state=self.random_state,
                                      point_estimator_previous_best_hyperparameter_config=point_estimator_stored_best_hyperparameter_config,
                                      variance_estimator_previous_best_hyperparameter_config=variance_estimator_stored_best_hyperparameter_config)

                if i == 0:
                    CP_quantile, hyperreg_model_runtime_per_iter = conformer.conformal_quantile(
                        confidence_level=confidence_level)
                else:
                    runtime_optimized_combinations = ConformalRuntimeOptimizer.get_optimal_number_of_secondary_model_parameter_combinations(
                        primary_model_runtime=primary_model_RS_runtime_per_iter,
                        secondary_model_runtime=hyperreg_model_runtime_per_iter,
                        secondary_model_retraining_freq=conformal_retraining_frequency,
                        secondary_model_runtime_as_frac_of_primary_model_runtime=0.5)

                    CP_quantile, hyperreg_model_runtime_per_iter_new = conformer.conformal_quantile(
                        confidence_level=confidence_level, n_of_param_combinations=runtime_optimized_combinations)

                    if hyperreg_model_runtime_per_iter_new is not None:
                        hyperreg_model_runtime_per_iter = hyperreg_model_runtime_per_iter_new

                baseline_accuracies, CP_intervals, CP_bounds = conformer.generate_confidence_intervals(
                    conformal_quantile=CP_quantile)

                point_estimator_stored_best_hyperparameter_config = conformer.point_estimator_best_hyperparameter_config
                variance_estimator_stored_best_hyperparameter_config = conformer.variance_estimator_best_hyperparameter_config
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
                top_n_idx = np.argpartition(CP_bound, -batch_n)[-batch_n:]
                no_sample_idx.append(maximal_idx)
            elif loss_direction == 'inverse':
                if no_sample_idx != []:
                    CP_bound[no_sample_idx] = 10000000
                maximal_idx = np.argmin(CP_bound)
                top_n_idx = np.argpartition(CP_bound, batch_n)[:batch_n]
                no_sample_idx.append(maximal_idx)

            maximal_parameter = hyperparameter_tuple_ordered.reset_index(drop=True).iloc[maximal_idx,
                                :]  # NOTE: using the non normalized one now because i need to feed rael parmeters later to the base model
            next_batch_combinations = hyperparameter_tuple_ordered.reset_index(drop=True).iloc[top_n_idx,
                                      :]  # NOTE: using the non normalized one now because i need to feed rael parmeters later to the base model

            validation_loss, validation_variance, loss_direction, validation_loss_profile = self._get_validation_loss(
                combination=maximal_parameter, X_IS=X, y_IS=y, custom_loss_function=custom_loss_function,
                prediction_type=prediction_type,
                validating_framework=validating_framework, train_val_split=train_val_split, n_classes=n_classes,
                presplit_X_y_data_tuple=presplit_X_y_data_tuple)

            if np.isnan(validation_loss) or (prediction_type == "regression" and abs(validation_loss) > abs(
                    np.median(y)) * 3):
                blacklisted_combination_df = np.append(blacklisted_combination_df, maximal_parameter)
                # NOTE: here we don't append to no_sample_idx because it's already been appended previously in the maximal_idx section above
                continue

            # print("IS primary model loss: ", validation_loss)

            maximal_parameter_logged = maximal_parameter.copy()
            maximal_parameter_logged['accuracy'] = validation_loss
            maximal_parameter_logged['accuracy_score'] = validation_loss_profile["accuracy_score"]
            maximal_parameter_logged['log_loss'] = validation_loss_profile["log_loss"]
            maximal_parameter_logged['variance'] = validation_variance

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

            # set tolerance:
            if i > (min_training_iterations + tolerance) and (
                    round(hyperparameter_performance_record["accuracy"].iloc[i - tolerance:i].mean(), 4) == \
                    round(hyperparameter_performance_record["accuracy"].iloc[i], 4)):
                hyperparameter_performance_record = hyperparameter_performance_record.iloc[:i, :]
                hyperparameter_performance_record["95_CI"] = (1.645 * hyperparameter_performance_record["variance"])
                break
            if (early_stop is not None and i > early_stop) or (log_elapse is not None and log_elapse > early_timeout):
                hyperparameter_performance_record = hyperparameter_performance_record.iloc[:i, :]
                hyperparameter_performance_record["95_CI"] = (1.645 * hyperparameter_performance_record["variance"])
                break

            i = i + 1

        hyperparameter_performance_record_parameters_only = hyperparameter_performance_record.drop(
            ["accuracy", "accuracy_score", "log_loss", "variance", "runtime", "CI_breach",
             "point_predictor_MSE", "loss_profile_dict"],
            axis=1)
        self._hyperparameter_performance_record = hyperparameter_performance_record
        if loss_direction == 'direct':
            self.best_score_ = hyperparameter_performance_record["accuracy"].max()
            self.best_params_ = self.combination_row_2_input_dict(
                hyperparameter_performance_record_parameters_only.iloc[
                hyperparameter_performance_record["accuracy"].idxmax(), :])
        elif loss_direction == 'inverse':
            self.best_score_ = hyperparameter_performance_record["accuracy"].min()
            self.best_params_ = self.combination_row_2_input_dict(
                hyperparameter_performance_record_parameters_only.iloc[
                hyperparameter_performance_record["accuracy"].idxmin(), :])

        # TODO: make this the default instead of explicitly checking for name of model object

        self._best_model = eval(str(self.model).split("(")[0] + "(**" + str(self.best_params_) + ")").fit(X, y)

    def fit_random_search(self, X, y, custom_loss_function=None, n_searches=60, max_runtime=3600,
                          train_val_split=0.6,
                          validating_framework='train_test_split', k_fold_splits=None,
                          prediction_type="classification",
                          presplit_X_y_data_tuple=None,
                          verbose=False):
        if custom_loss_function is None:
            if prediction_type == "regression":
                custom_loss_function = "mean_squared_error"
            else:
                custom_loss_function = "accuracy_score"

        if self.hyperparameter_space is not None:
            parameter_grid = self.hyperparameter_space
        else:
            parameter_grid = self.get_default_hyperparameter_space()
        hyperparameter_tuple_ordered = self.get_hyperparameter_combinations(parameter_grid=parameter_grid)
        hyperparameter_performance_record = self.build_hyperparameter_logger(
            hyperparameter_combinations=hyperparameter_tuple_ordered)

        n_classes = len(np.unique(np.array(y)))

        i = 0
        start_time = time.time()
        for row in range(0, len(hyperparameter_tuple_ordered)):

            combination = hyperparameter_tuple_ordered.iloc[row, :]

            validation_loss, validation_variance, loss_direction, validation_loss_profile = self._get_validation_loss(
                combination=combination,
                X_IS=X, y_IS=y,
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
                        np.median(y)) * 3:
                    continue

            logged_combination = combination.copy()
            logged_combination['accuracy'] = validation_loss
            logged_combination['accuracy_score'] = validation_loss_profile["accuracy_score"]
            logged_combination['log_loss'] = validation_loss_profile["log_loss"]
            logged_combination['variance'] = validation_variance

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

            if i > n_searches or (log_elapse is not None and log_elapse > max_runtime):
                hyperparameter_performance_record = hyperparameter_performance_record.iloc[:i, :]
                hyperparameter_performance_record["95_CI"] = (1.645 * hyperparameter_performance_record["variance"])
                break
            i = i + 1

        hyperparameter_performance_record_parameters_only = hyperparameter_performance_record.drop(
            ["accuracy", "accuracy_score", "log_loss", "variance", "runtime", "CI_breach",
             "point_predictor_MSE", "loss_profile_dict"],
            axis=1)
        self._hyperparameter_performance_record = hyperparameter_performance_record
        if loss_direction == 'direct':
            self.best_score_ = hyperparameter_performance_record["accuracy"].max()
            self.best_params_ = self.combination_row_2_input_dict(
                hyperparameter_performance_record_parameters_only.iloc[
                hyperparameter_performance_record["accuracy"].idxmax(), :])
        elif loss_direction == 'inverse':
            self.best_score_ = hyperparameter_performance_record["accuracy"].min()
            self.best_params_ = self.combination_row_2_input_dict(
                hyperparameter_performance_record_parameters_only.iloc[
                hyperparameter_performance_record["accuracy"].idxmin(), :])

        # TODO: make this the default instead of explicitly checking for name of model object
        self._best_model = eval(str(self.model).split("(")[0] + "(**" + str(self.best_params_) + ")").fit(X, y)

    def predict(self, X):
        return self._best_model.predict(X)

    @staticmethod
    def solver_mapper(hyperparameter_tuple_row):
        if hyperparameter_tuple_row["adam"] == 1:
            return "adam"
        if hyperparameter_tuple_row["sgd"] == 1:
            return "sgd"


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


def try_numeric(value):
    try:
        if float(value) % 1 == 0:
            return int(value)
        else:
            return float(value)
    except:
        return value
