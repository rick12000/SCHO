import os

import pandas as pd


class Filing:
    input_parent_folder_path = ""
    output_parent_folder_path = "saved_logs/current_log"

    def __init__(self, data_folder_name, plots_folder_name, results_folder_name,
                 dataset_name="unspecified_dataset"):
        self.input_parent_folder_path = Filing.input_parent_folder_path
        self.output_parent_folder_path = Filing.output_parent_folder_path
        self.data_folder_name = data_folder_name
        self.plots_folder_name = plots_folder_name
        self.results_folder_name = results_folder_name
        self.dataset_name = dataset_name
        # TODO: add method to clear all directories before writing to them

    def save_dataframe(self, df, destination_filename):
        if not os.path.exists(self.output_parent_folder_path + "/" + self.dataset_name + "/" + self.data_folder_name):
            os.makedirs(self.output_parent_folder_path + "/" + self.dataset_name + "/" + self.data_folder_name)
        df.to_csv(
            self.output_parent_folder_path + "/" + self.dataset_name + "/" + self.data_folder_name + "/" + destination_filename)

    def load_dataframe(self, df_filename):
        df = pd.read_csv(
            self.output_parent_folder_path + "/" + self.dataset_name + "/" + self.data_folder_name + "/" + df_filename)
        return df


class Analytics:

    @staticmethod
    def summarize_cross_model_performance(cross_model_IS_log):
        cross_model_log = cross_model_IS_log.copy()
        cross_model_log = cross_model_log.drop(["loss_profile_dict"], axis=1)
        cross_model_log = cross_model_log.reset_index(drop=True)

        aggregation = {"runtime": ["max", "count"], "accuracy": [Analytics.get_last, "mean"],
                       "CI_breach": "mean"}
        results_table = cross_model_log.groupby(["secondary_model", "cp_scorer"],
                                                as_index=False).agg(
            aggregation)
        results_table.columns = [''.join(col).strip() for col in results_table.columns.values]
        results_table["runtime_per_iter"] = results_table["runtimemax"] / results_table["runtimecount"]
        adjunct_results_table = \
            cross_model_log.iloc[
            cross_model_log.groupby(["secondary_model", "cp_scorer"]).accuracy.idxmax().fillna(
                0).values, :][
                # fillna with 0 is not good, it only applies to when OOS is not selected as an option in train test split and this returns Nans for the OOS entries, replacing with 0 means arbitrarily selecting the 0th row instead of the idxmax for the OOS data, even though this hould correspond to Nan anyway which is fine
                ["secondary_model", "cp_scorer", "accuracy"]]
        adjunct_results_table = adjunct_results_table.rename(columns={"accuracy": "accuracymax"})
        results_table = pd.merge(results_table, adjunct_results_table, how='left',
                                 on=["secondary_model", "cp_scorer"])
        results_table = results_table.drop(["runtimemax"], axis=1)

        return results_table

    @staticmethod
    def get_last(x):
        return x.tail(1)
