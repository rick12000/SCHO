import unittest

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier

from SCHO.hyperparameter_tuner import SeqTune, CNNClassifier


class TestReproducibility(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import os
        import warnings
        warnings.filterwarnings("ignore")
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        cls.min_train_iter = 30
        cls.max_iter = 50

        X_tabular_test_data, y_tabular_test_data = SeqTune.get_toy_dataset("tabular_test_data")
        X_convolutional_test_data, y_convolutional_test_data = SeqTune.get_toy_dataset("convolutional_test_data")

        tabular_initialized_model = SeqTune(model=MLPClassifier(), random_state=1234)
        cls.tabular_OOS_HR_log, cls.tabular_IS_HR_log = tabular_initialized_model.fit(X=X_tabular_test_data,
                                                                                      min_training_iterations=cls.min_train_iter,
                                                                                      early_stop=cls.max_iter,
                                                                                      y=y_tabular_test_data,
                                                                                      hyper_reg_model=GradientBoostingRegressor())
        cls.tabular_OOS_RS_log, cls.tabular_IS_RS_log = tabular_initialized_model.fit_random_search(
            X=X_tabular_test_data,
            y=y_tabular_test_data,
            n_searches=cls.max_iter)

        convolutional_initialized_model = SeqTune(model=CNNClassifier(), random_state=1234)
        cls.convolutional_OOS_HR_log, cls.convolutional_IS_HR_log = convolutional_initialized_model.fit(
            min_training_iterations=cls.min_train_iter,
            early_stop=cls.max_iter,
            X=X_convolutional_test_data,
            y=y_convolutional_test_data,
            hyper_reg_model=GradientBoostingRegressor())
        cls.convolutional_OOS_RS_log, cls.convolutional_IS_RS_log = convolutional_initialized_model.fit_random_search(
            X=X_convolutional_test_data,
            y=y_convolutional_test_data,
            n_searches=cls.max_iter)

    def test_tabular_performance_repro(self):
        for i in range(0, TestReproducibility.min_train_iter - 1):
            self.assertEqual(TestReproducibility.tabular_IS_HR_log["accuracy"].iloc[i],
                             TestReproducibility.tabular_IS_RS_log["accuracy"].iloc[i])
        print("Completed tabular performance reproducibility test...")

    def test_convolutional_performance_repro(self):
        for i in range(0, TestReproducibility.min_train_iter - 1):
            self.assertEqual(TestReproducibility.convolutional_IS_HR_log["accuracy"].iloc[i],
                             TestReproducibility.convolutional_IS_RS_log["accuracy"].iloc[i])
        print("Completed convolutional performance reproducibility test...")

    def test_arg_range(self):
        pass

    def test_large_tabular_iter(self):
        X_tabular_test_data, y_tabular_test_data = SeqTune.get_toy_dataset("tabular_test_data")

        tabular_initialized_model = SeqTune(model=MLPClassifier(), random_state=1234)
        large_iter_tabular_OOS_HR_log, large_iter_tabular_IS_HR_log = tabular_initialized_model.fit(
            X=X_tabular_test_data,
            min_training_iterations=30,
            early_stop=300,
            y=y_tabular_test_data,
            hyper_reg_model=GradientBoostingRegressor())


if __name__ == "__main__":
    unittest.main()
