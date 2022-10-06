import math
import random

import numpy as np
import pandas as pd
from SCHO.utils.runtime_eval import TimeLogger

from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


class Conformal:

    def __init__(self,
                 model,
                 scoring,
                 X_obs,
                 y_obs,
                 X_train,
                 y_train,
                 X_obs_train,
                 y_obs_train,
                 X_full,
                 random_state,
                 previous_best_hyperparameter_config=None,
                 verbose=False):
        self.model = model
        self.scoring = scoring
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.X_train = X_train
        self.y_train = y_train
        self.X_obs_train = X_obs_train
        self.y_obs_train = y_obs_train
        self.X_full = X_full
        self.random_state = random_state
        self.previous_best_hyperparameter_config = previous_best_hyperparameter_config
        self.verbose = verbose

        self.predictor_model = None
        self.var_model = None
        self.quant_reg_hi = None
        self.quant_reg_lo = None
        self.random_quantile_model = None
        self.quantile_forest_percentile = None
        self.best_hyperparameter_config = None
        self.best_point_predictor_MSE = None

    @staticmethod
    def nearest_neighbour_finder(input_point, point_space):
        distance_log = []
        for candidate_point in point_space:
            distance = np.linalg.norm(np.array(input_point) - np.array(candidate_point))
            distance_log.append(distance)
        nearest_neighbour_idx = np.argmin(np.array(distance_log))
        nearest_neighbour_distance = min(distance_log)
        nearest_neighbour = point_space[nearest_neighbour_idx, :]

        return nearest_neighbour_idx, nearest_neighbour_distance, nearest_neighbour

    def tune_fit_hyperreg_model(self, X, y, n_of_param_combinations=None):
        if n_of_param_combinations is None or self.previous_best_hyperparameter_config is None or n_of_param_combinations > 1:
            tuning_helper = TuningHelper(model=self.model, random_state=self.random_state,
                                         previous_best_hyperparameter_config=self.previous_best_hyperparameter_config)
            parameter_grid = tuning_helper.get_hyperreg_model_parameters()
            hyperparameter_tuple_ordered = tuning_helper.get_hyperreg_parameter_combinations(
                parameter_grid=parameter_grid,
                n_of_param_combinations=n_of_param_combinations)
            hyperparameter_performance_record = tuning_helper.build_hyperreg_hyperparameter_logger(
                hyperparameter_combinations=hyperparameter_tuple_ordered)

            hyperreg_model_runtime_log = TimeLogger()
            parameter_row = 0
            for row in range(0, len(hyperparameter_tuple_ordered)):
                combination = hyperparameter_tuple_ordered.iloc[row, :]
                validation_loss = tuning_helper._get_validation_loss(
                    combination=combination, X=X, y=y)

                logged_combination = combination.copy()
                logged_combination['accuracy'] = validation_loss
                hyperparameter_performance_record.iloc[parameter_row, :] = logged_combination
                parameter_row = parameter_row + 1
            hyperreg_model_runtime_log.pause_runtime()
            hyperreg_model_total_runtime = hyperreg_model_runtime_log.return_runtime()
            hyperreg_model_runtime_per_iter = hyperreg_model_total_runtime / (len(hyperparameter_tuple_ordered) + 1)

            if self.verbose:
                print(
                    f"Best Achieved SCHO Point Estimator Accuracy: {hyperparameter_performance_record['accuracy'].min()}")
            optimal_idx = (hyperparameter_performance_record['accuracy']).argmin()
            optimal_parameters = hyperparameter_performance_record.iloc[optimal_idx, :]

            self.best_point_predictor_MSE = hyperparameter_performance_record['accuracy'].min()

        if n_of_param_combinations == 1 and self.previous_best_hyperparameter_config is not None:
            optimal_parameters = self.previous_best_hyperparameter_config
            hyperreg_model_runtime_per_iter = None

        self.best_hyperparameter_config = optimal_parameters

        if "mlp" in str(self.model).lower():
            hidden_layers = TuningHelper.tuplify_network_layer_sizes(optimal_parameters)
            optimal_model = MLPRegressor(solver=TuningHelper.solver_mapper(optimal_parameters),
                                         learning_rate_init=optimal_parameters["learning_rate_init"],
                                         alpha=optimal_parameters["alpha"], hidden_layer_sizes=hidden_layers,
                                         random_state=self.random_state).fit(X, y)
        elif "forest" in str(self.model).lower():
            if optimal_parameters["max_features"] is not None:
                try:
                    max_features_value = float(optimal_parameters["max_features"])
                except:
                    max_features_value = str(optimal_parameters["max_features"])
            else:
                max_features_value = None
            optimal_model = RandomForestRegressor(n_estimators=int(optimal_parameters["n_estimators"]),
                                                  max_features=max_features_value,
                                                  min_samples_split=int(optimal_parameters["min_samples_split"]),
                                                  min_samples_leaf=int(optimal_parameters["min_samples_leaf"]),
                                                  random_state=self.random_state).fit(
                X,
                y)
        elif "neigh" in str(self.model).lower():
            optimal_model = KNeighborsRegressor(n_neighbors=int(optimal_parameters["n_neighbors"])).fit(X, y)

        elif "svr" in str(self.model).lower():
            optimal_model = SVR(kernel=str(optimal_parameters["kernel"]),
                                degree=int(optimal_parameters["degree"]),
                                C=optimal_parameters["C"]).fit(X, y)

        elif "boosting" in str(self.model).lower():
            optimal_model = GradientBoostingRegressor(learning_rate=optimal_parameters["learning_rate"],
                                                      n_estimators=int(optimal_parameters["n_estimators"]),
                                                      min_samples_split=int(optimal_parameters["min_samples_split"]),
                                                      min_samples_leaf=int(optimal_parameters["min_samples_leaf"]),
                                                      max_depth=int(optimal_parameters["max_depth"]),
                                                      random_state=self.random_state).fit(X, y)

        elif "gaussian" in str(self.model).lower():
            optimal_model = GaussianProcessRegressor(kernel=eval(optimal_parameters["kernel"]),
                                                     random_state=self.random_state).fit(X, y)

        return optimal_model, hyperreg_model_runtime_per_iter

    def conformal_quantile(self, confidence_level,
                           tune_hypermodel=True,
                           n_of_param_combinations=None):  # scoring: deviation, knn_deviation, varfit_deviation
        if tune_hypermodel or "mlp" in str(self.model).lower() or "svr" in str(self.model).lower() or "boosting" in str(
                self.model).lower():
            predictor_model, hyperreg_model_runtime_per_iter = self.tune_fit_hyperreg_model(X=self.X_train,
                                                                                            y=self.y_train,
                                                                                            n_of_param_combinations=n_of_param_combinations)
        else:
            predictor_model = self.model.fit(X=self.X_train, y=self.y_train)
        self.predictor_model = predictor_model
        y_obs_pred = np.array(predictor_model.predict(self.X_obs))

        if self.scoring == 'deviation':
            nonconformity_score = abs(np.array(self.y_obs) - y_obs_pred)
            nonconformity_percentile = np.percentile(nonconformity_score, confidence_level * 100)

        if self.scoring == 'lr_mad':
            var_train = abs(np.array(self.y_train) - np.mean(np.array(self.y_train)))
            var_model = LinearRegression().fit(self.X_train, var_train)
            self.var_model = var_model

            var_array = var_model.predict(self.X_obs)
            var_array = np.array([max(x, 0) for x in var_array])
            nonconformity_score = abs(np.array(self.y_obs) - y_obs_pred) / var_array
            nonconformity_percentile = np.percentile(nonconformity_score, confidence_level * 100)

        if self.scoring == 'rf_mad':
            var_train = abs(np.array(self.y_train) - np.mean(np.array(self.y_train)))
            # var_model = RandomForestRegressor().fit(self.X_train, var_train)
            parameters = {'n_estimators': [10, 50, 100, 200]}
            var_model = GridSearchCV(RandomForestRegressor(random_state=self.random_state), parameters)
            var_model.fit(self.X_train, var_train)
            self.var_model = var_model

            var_array = var_model.predict(self.X_obs)
            var_array = np.array([max(x, 0) for x in var_array])
            nonconformity_score = abs(np.array(self.y_obs) - y_obs_pred) / var_array
            nonconformity_percentile = np.percentile(nonconformity_score, confidence_level * 100)

        if self.scoring == 'knn_deviation':
            var_list = []
            for i in range(0, len(self.X_obs)):
                search_space = np.delete(self.X_obs, i, axis=0)
                observation = self.X_obs[i, :]
                _, nearest_neighbour_distance, _ = Conformal.nearest_neighbour_finder(input_point=observation,
                                                                                      point_space=search_space)
                # nearest_neighbour_var = self.y_obs.iloc[i] - self.y_obs.iloc[nearest_neighbour_idx]
                var_list.append(nearest_neighbour_distance)

            var_array = np.array(var_list)
            var_array = np.array([max(x, 0) for x in var_array])
            nonconformity_score = abs(np.array(self.y_obs) - y_obs_pred) / var_array
            nonconformity_percentile = np.percentile(nonconformity_score, confidence_level * 100)

        if self.scoring == 'linear_cqr':
            parameters = {"alpha": [0, 0.001, 0.005, 0.01, 0.1]}
            quant_reg_lo = GridSearchCV(QuantileRegressor(quantile=0.05), parameters,
                                        scoring=make_scorer(mean_pinball_loss, alpha=0.05))
            quant_reg_lo.fit(self.X_train, self.y_train)
            # quant_reg_lo = QuantileRegressor(quantile=(1 - confidence_level)).fit(self.X_train, self.y_train)
            self.quant_reg_lo = quant_reg_lo
            quant_reg_hi = GridSearchCV(QuantileRegressor(quantile=0.95), parameters,
                                        scoring=make_scorer(mean_pinball_loss, alpha=0.95))
            quant_reg_hi.fit(self.X_train, self.y_train)
            # quant_reg_hi = QuantileRegressor(quantile=confidence_level).fit(self.X_train, self.y_train)
            self.quant_reg_hi = quant_reg_hi

            var_array = quant_reg_hi.predict(self.X_obs) - quant_reg_lo.predict(self.X_obs)
            var_array = np.array([max(x, 0) for x in var_array])
            nonconformity_score = abs(np.array(self.y_obs) - y_obs_pred) / var_array
            nonconformity_percentile = np.percentile(nonconformity_score, confidence_level * 100)

        if self.scoring == 'gradient_boosted_cqr':
            parameters = {"learning_rate": [0.01, 0.05, 0.1, 0.5]}
            quant_reg_lo = GridSearchCV(GradientBoostingRegressor(loss="quantile", alpha=0.05,
                                                                  random_state=self.random_state),
                                        param_grid=parameters, scoring=make_scorer(mean_pinball_loss, alpha=0.05))
            quant_reg_lo.fit(self.X_train, self.y_train)
            # quant_reg_lo = GradientBoostingRegressor(loss="quantile", alpha=(1 - confidence_level),
            #                                          random_state=self.random_state).fit(self.X_train,
            #                                                                              self.y_train)
            self.quant_reg_lo = quant_reg_lo
            quant_reg_hi = GridSearchCV(GradientBoostingRegressor(loss="quantile", alpha=0.95,
                                                                  random_state=self.random_state),
                                        param_grid=parameters, scoring=make_scorer(mean_pinball_loss, alpha=0.95))
            quant_reg_hi.fit(self.X_train, self.y_train)
            # quant_reg_hi = GradientBoostingRegressor(loss="quantile", alpha=(confidence_level),
            #                                          random_state=self.random_state).fit(
            #     self.X_train,
            #     self.y_train)
            self.quant_reg_hi = quant_reg_hi

            var_array = quant_reg_hi.predict(self.X_obs) - quant_reg_lo.predict(self.X_obs)
            var_array = np.array([max(x, 0) for x in var_array])

            nonconformity_score = abs(np.array(self.y_obs) - y_obs_pred) / var_array
            nonconformity_percentile = np.percentile(nonconformity_score, confidence_level * 100)

        if self.scoring == 'quantile_forest_cqr':
            base_reg = RandomForestRegressor(random_state=self.random_state).fit(self.X_train, self.y_train)
            self.random_quantile_model = base_reg
            self.quantile_forest_percentile = confidence_level

            base_pred = pd.DataFrame()
            for obs_pred in base_reg.estimators_:
                obs_pred_array = pd.Series(obs_pred.predict(self.X_obs))
                base_pred = pd.concat([base_pred, obs_pred_array], axis=1)
            var_array = base_pred.quantile(q=confidence_level, axis=1) - base_pred.quantile(q=(1 - confidence_level),
                                                                                            axis=1)
            var_array = np.array([max(x, 0) for x in var_array])
            nonconformity_score = abs(np.array(self.y_obs) - y_obs_pred) / var_array
            nonconformity_percentile = np.percentile(nonconformity_score, confidence_level * 100)

        return nonconformity_percentile, hyperreg_model_runtime_per_iter

    def generate_confidence_intervals(self, conformal_quantile):
        # predictor_model = self.model.fit(self.X_train, self.y_train) # NOTE: switch to x_obs_train and y_obs_train to retrain on full IS data (but confidence intervals might not hold)
        y_full_pred = np.array(self.predictor_model.predict(self.X_full))

        if self.scoring == 'deviation':
            intervals = np.repeat(conformal_quantile, len(y_full_pred))

            max_bound_y = y_full_pred + intervals
            min_bound_y = y_full_pred - intervals
            y_bounds = pd.DataFrame({'max_bound': max_bound_y, 'min_bound': min_bound_y})

        if self.scoring == 'lr_mad' or self.scoring == 'rf_mad':
            var_array = self.var_model.predict(self.X_full)
            var_array = np.array([max(x, 0) for x in var_array])
            intervals = var_array * conformal_quantile

            max_bound_y = y_full_pred + intervals
            min_bound_y = y_full_pred - intervals
            y_bounds = pd.DataFrame({'max_bound': max_bound_y, 'min_bound': min_bound_y})

        if self.scoring == 'knn_deviation':
            var_list = []
            for i in range(0, len(self.X_full)):
                _, nearest_neighbour_distance, _ = Conformal.nearest_neighbour_finder(input_point=self.X_full[i, :],
                                                                                      point_space=self.X_obs)
                var_list.append(nearest_neighbour_distance)
            var_array = np.array(var_list)
            intervals = var_array * conformal_quantile

            max_bound_y = y_full_pred + intervals
            min_bound_y = y_full_pred - intervals
            y_bounds = pd.DataFrame({'max_bound': max_bound_y, 'min_bound': min_bound_y})

        if self.scoring == 'linear_cqr' or self.scoring == 'gradient_boosted_cqr':
            var_array = self.quant_reg_hi.predict(self.X_full) - self.quant_reg_lo.predict(self.X_full)
            var_array = np.array([max(x, 0) for x in var_array])
            intervals = var_array * conformal_quantile

            max_bound_y = y_full_pred + intervals
            min_bound_y = y_full_pred - intervals
            y_bounds = pd.DataFrame({'max_bound': max_bound_y, 'min_bound': min_bound_y})

        if self.scoring == 'quantile_forest_cqr':
            base_pred = pd.DataFrame()
            for obs_pred in self.random_quantile_model.estimators_:
                obs_pred_array = pd.Series(obs_pred.predict(self.X_full))
                base_pred = pd.concat([base_pred, obs_pred_array], axis=1)
            var_array = base_pred.quantile(q=self.quantile_forest_percentile, axis=1) - base_pred.quantile(
                q=(1 - self.quantile_forest_percentile), axis=1)
            var_array = np.array([max(x, 0) for x in var_array])
            intervals = var_array * conformal_quantile

            max_bound_y = y_full_pred + intervals
            min_bound_y = y_full_pred - intervals
            y_bounds = pd.DataFrame({'max_bound': max_bound_y, 'min_bound': min_bound_y})

        return y_full_pred, intervals, y_bounds


class TuningHelper:
    NEURAL_NETWORK_DEFAULT_PARAMS = 300
    RANDOM_FOREST_DEFAULT_PARAMS = 200
    KNN_DEFAULT_PARAMS = 30
    SVM_DEFAULT_PARAMS = 100
    GBM_DEFAULT_PARAMS = 200
    GP_DEFAULT_PARAMS = 100

    def __init__(self, model, random_state=None, previous_best_hyperparameter_config=None):
        self.model = model
        self.random_state = random_state
        self.previous_best_hyperparameter_config = previous_best_hyperparameter_config  # optional, if you want to keep the best config from previous iteration as the ONE to try as default this time, it's model agnostic

    @staticmethod
    def solver_mapper(hyperparameter_tuple_row):
        if hyperparameter_tuple_row["adam"] == 1:
            return "adam"
        if hyperparameter_tuple_row["sgd"] == 1:
            return "sgd"

    @staticmethod
    def tuplify_network_layer_sizes(
            combination):  # NOTE: assumes column names are sorted from layer 1 to layer N, otherwise will not create ordered list
        layer_tuple = ()
        for column_name in list(combination.index):
            if "layer" in column_name.lower():
                if combination[column_name] != 0:
                    layer_tuple = layer_tuple + (int(combination[column_name]),)
        return layer_tuple

    def _get_validation_loss(self, combination, X, y,
                             k_fold_splits=5):
        kf = KFold(n_splits=k_fold_splits, random_state=self.random_state, shuffle=True)
        K_fold_performance_record = pd.DataFrame(np.zeros((k_fold_splits, 2)))
        K_fold_performance_record.columns = ["fold", "accuracy"]
        fold = 1
        for train_index, test_index in kf.split(X):
            X_train, X_val = X[train_index, :], X[test_index, :]
            Y_train, Y_val = y[train_index], y[test_index]

            if "mlp" in str(self.model).lower():
                hidden_layers = TuningHelper.tuplify_network_layer_sizes(combination)
                fitted_model = MLPRegressor(solver=TuningHelper.solver_mapper(combination),
                                            learning_rate_init=combination["learning_rate_init"],
                                            alpha=combination["alpha"], hidden_layer_sizes=hidden_layers,
                                            random_state=self.random_state).fit(X_train, Y_train)
            elif "forest" in str(self.model).lower():
                try:
                    max_features_value = float(combination["max_features"])
                except:
                    max_features_value = str(combination["max_features"])
                fitted_model = RandomForestRegressor(n_estimators=int(combination["n_estimators"]),
                                                     max_features=max_features_value,
                                                     min_samples_split=int(combination["min_samples_split"]),
                                                     min_samples_leaf=int(combination["min_samples_leaf"]),
                                                     random_state=self.random_state).fit(
                    X_train,
                    Y_train)
            elif "neigh" in str(self.model).lower():
                fitted_model = KNeighborsRegressor(n_neighbors=int(combination["n_neighbors"])).fit(X_train, Y_train)

            elif "svr" in str(self.model).lower():
                fitted_model = SVR(kernel=str(combination["kernel"]),
                                   degree=int(combination["degree"]),
                                   C=combination["C"]).fit(X_train, Y_train)

            elif "boosting" in str(self.model).lower():
                fitted_model = GradientBoostingRegressor(learning_rate=combination["learning_rate"],
                                                         n_estimators=int(combination["n_estimators"]),
                                                         min_samples_split=int(combination["min_samples_split"]),
                                                         min_samples_leaf=int(combination["min_samples_leaf"]),
                                                         max_depth=int(combination["max_depth"]),
                                                         random_state=self.random_state).fit(X_train,
                                                                                             Y_train)

            elif "gaussian" in str(self.model).lower():
                fitted_model = GaussianProcessRegressor(kernel=eval(combination["kernel"]),
                                                        random_state=self.random_state).fit(X_train, Y_train)

            y_pred = fitted_model.predict(X_val)
            try:
                loss = math.sqrt(metrics.mean_squared_error(Y_val, y_pred))
            except:
                loss = np.nan

            K_fold_performance_record.iloc[(fold - 1), 0] = fold
            K_fold_performance_record.iloc[(fold - 1), 1] = loss

            fold = fold + 1

        cross_fold_performance = K_fold_performance_record.mean()
        final_loss = float(cross_fold_performance["accuracy"])

        return final_loss

    def build_hyperreg_hyperparameter_logger(self, hyperparameter_combinations):
        if hyperparameter_combinations is not None:
            hyperparameter_performance_record = hyperparameter_combinations.copy()
            for i in range(0, hyperparameter_performance_record.shape[1]):
                hyperparameter_performance_record.iloc[:, i] = np.nan
            hyperparameter_performance_record["accuracy"] = np.nan

        return hyperparameter_performance_record

    def get_hyperreg_model_parameters(self):
        if "mlp" in str(self.model).lower():
            solver_list = ['adam', 'sgd']
            learning_rate_list = [0.0001, 0.001, 0.01, 0.1]
            alpha_list = [0.0001, 0.001, 0.01, 0.1, 1, 3, 10]
            layer_size = [2, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40]
            n_layers = [2, 3, 4]
            parameter_dict = {'solver': solver_list,
                              'learning_rate_init': learning_rate_list,
                              'alpha': alpha_list,
                              'n_layers': n_layers,
                              'layer_size': layer_size
                              }
        elif "forest" in str(self.model).lower():
            n_estimators = [10, 50, 100, 200, 400]
            max_features = [0.2, 0.4, 0.6, 0.8, 1]  # ['sqrt', 'log2', None]
            min_samples_split = [2, 3, 5]
            min_samples_leaf = [1, 2]
            parameter_dict = {'n_estimators': n_estimators,
                              'max_features': max_features,
                              'min_samples_split': min_samples_split,
                              'min_samples_leaf': min_samples_leaf}

        elif "neigh" in str(self.model).lower():
            n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            parameter_dict = {'n_neighbors': n_neighbors}

        elif "svr" in str(self.model).lower():
            kernel = ['linear', 'poly', 'rbf', 'sigmoid']
            degree = [2, 3, 4]  # ['sqrt', 'log2', None]
            C = [0.001, 0.01, 0.05, 0.1, 0.5, 1]
            parameter_dict = {'kernel': kernel,
                              'degree': degree,
                              'C': C}

        elif "boosting" in str(self.model).lower():
            learning_rate = [0.001, 0.01, 0.1, 1]
            n_estimators = [50, 100, 200]
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 3, 5]
            max_depth = [1, 3, 5]
            parameter_dict = {'learning_rate': learning_rate,
                              'n_estimators': n_estimators,
                              'min_samples_split': min_samples_split,
                              'min_samples_leaf': min_samples_leaf,
                              'max_depth': max_depth
                              }

        elif "gaussian" in str(self.model).lower():
            kernel = ["RBF()", "RationalQuadratic()"]
            parameter_dict = {'kernel': kernel}

        return parameter_dict

    def get_hyperreg_parameter_combinations(self, parameter_grid, n_of_param_combinations=None):
        if self.random_state is not None:
            random.seed(self.random_state)

        if "mlp" in str(self.model).lower():

            if n_of_param_combinations is not None and n_of_param_combinations < TuningHelper.NEURAL_NETWORK_DEFAULT_PARAMS:
                combination_enum = n_of_param_combinations
            else:
                combination_enum = TuningHelper.NEURAL_NETWORK_DEFAULT_PARAMS
            for i in range(0, combination_enum):
                parameter_combination = []
                parameter_combination_columns = []
                for key in parameter_grid.keys():  # NOTE: number of layers parameter must be ordered before layer size in the param dict for this to work
                    if key == 'layer_size':
                        for j in range(1, max(parameter_grid["n_layers"]) + 1):
                            if j <= n_layers_cached:
                                parameter = random.choice(parameter_grid[key])
                                parameter_combination.append(parameter)
                                parameter_combination_columns.append("layer_" + str(j))
                            else:
                                parameter = 0
                                parameter_combination.append(parameter)
                                parameter_combination_columns.append("layer_" + str(j))
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

            if self.previous_best_hyperparameter_config is not None:
                single_default_parameter_tuple = self.previous_best_hyperparameter_config
                hyperparameter_tuple.loc[len(hyperparameter_tuple)] = single_default_parameter_tuple

            hyperparameter_tuple = hyperparameter_tuple.drop_duplicates()

            hyperparameter_tuple[["adam", "sgd"]] = pd.get_dummies(hyperparameter_tuple[
                                                                       "solver"])

            hyperparameter_tuple = hyperparameter_tuple[
                ~(hyperparameter_tuple["adam"] + hyperparameter_tuple["sgd"] == 0)]
            hyperparameter_tuple = hyperparameter_tuple[
                ~(hyperparameter_tuple["adam"] + hyperparameter_tuple["sgd"] == 2)]
            hyperparameter_tuple = hyperparameter_tuple.drop(["solver"], axis=1)

        elif "forest" in str(self.model).lower():
            if n_of_param_combinations is not None and n_of_param_combinations < TuningHelper.RANDOM_FOREST_DEFAULT_PARAMS:
                combination_enum = n_of_param_combinations
            else:
                combination_enum = TuningHelper.RANDOM_FOREST_DEFAULT_PARAMS
            for i in range(0, combination_enum):
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

            if self.previous_best_hyperparameter_config is not None:
                single_default_parameter_tuple = self.previous_best_hyperparameter_config
                hyperparameter_tuple.loc[len(hyperparameter_tuple)] = single_default_parameter_tuple

            hyperparameter_tuple = hyperparameter_tuple.drop_duplicates()

        elif "neigh" in str(self.model).lower():
            if n_of_param_combinations is not None and n_of_param_combinations < TuningHelper.KNN_DEFAULT_PARAMS:
                combination_enum = n_of_param_combinations
            else:
                combination_enum = TuningHelper.KNN_DEFAULT_PARAMS
            for i in range(0, combination_enum):
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

            if self.previous_best_hyperparameter_config is not None:
                single_default_parameter_tuple = self.previous_best_hyperparameter_config
                hyperparameter_tuple.loc[len(hyperparameter_tuple)] = single_default_parameter_tuple

            hyperparameter_tuple = hyperparameter_tuple.drop_duplicates()

        elif "svr" in str(self.model).lower():
            if n_of_param_combinations is not None and n_of_param_combinations < TuningHelper.SVM_DEFAULT_PARAMS:
                combination_enum = n_of_param_combinations
            else:
                combination_enum = TuningHelper.SVM_DEFAULT_PARAMS
            for i in range(0, combination_enum):
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

            if self.previous_best_hyperparameter_config is not None:
                single_default_parameter_tuple = self.previous_best_hyperparameter_config
                hyperparameter_tuple.loc[len(hyperparameter_tuple)] = single_default_parameter_tuple

            hyperparameter_tuple = hyperparameter_tuple.drop_duplicates()

        elif "boosting" in str(self.model).lower():
            if n_of_param_combinations is not None and n_of_param_combinations < TuningHelper.GBM_DEFAULT_PARAMS:
                combination_enum = n_of_param_combinations
            else:
                combination_enum = TuningHelper.GBM_DEFAULT_PARAMS
            for i in range(0, combination_enum):
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

            if self.previous_best_hyperparameter_config is not None:
                single_default_parameter_tuple = self.previous_best_hyperparameter_config
                hyperparameter_tuple.loc[len(hyperparameter_tuple)] = single_default_parameter_tuple

            hyperparameter_tuple = hyperparameter_tuple.drop_duplicates()

        elif "gaussian" in str(self.model).lower():
            if n_of_param_combinations is not None and n_of_param_combinations < TuningHelper.GP_DEFAULT_PARAMS:
                combination_enum = n_of_param_combinations
            else:
                combination_enum = TuningHelper.GP_DEFAULT_PARAMS
            for i in range(0, combination_enum):
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

            if self.previous_best_hyperparameter_config is not None:
                single_default_parameter_tuple = self.previous_best_hyperparameter_config
                hyperparameter_tuple.loc[len(hyperparameter_tuple)] = single_default_parameter_tuple

            hyperparameter_tuple = hyperparameter_tuple.drop_duplicates()

        hyperparameter_tuple_randomized = hyperparameter_tuple.sample(frac=1,
                                                                      random_state=self.random_state).reset_index(
            drop=True)

        return hyperparameter_tuple_randomized
