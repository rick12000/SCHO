import time


class ConformalRuntimeOptimizer:
    @staticmethod
    def get_optimal_number_of_secondary_model_parameter_combinations(primary_model_runtime,
                                                                     secondary_model_runtime,
                                                                     secondary_model_retraining_freq,
                                                                     secondary_model_runtime_as_frac_of_primary_model_runtime):
        optimal_n_of_secondary_model_param_combinations = (
                (primary_model_runtime * secondary_model_retraining_freq) / (
                secondary_model_runtime * (1/secondary_model_runtime_as_frac_of_primary_model_runtime)**2))
        optimal_n_of_secondary_model_param_combinations = max(1, int(round(optimal_n_of_secondary_model_param_combinations)))

        # print(f"Optimal Number of Hyperparameters to Test: {optimal_n_of_secondary_model_param_combinations}")
        return optimal_n_of_secondary_model_param_combinations


class TimeLogger:
    def __init__(self):
        self.start_time = time.time()
        self.runtime = 0

    def _elapsed_runtime(self):
        take_time = time.time()
        return abs(take_time - self.start_time)

    def pause_runtime(self):
        self.runtime = self.runtime + self._elapsed_runtime()

    def resume_runtime(self):
        self.start_time = time.time()

    def return_runtime(self):
        return self.runtime
