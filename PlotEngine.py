import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DataHandling import Filing


class PlotHelper:
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    color_palette = ['tab:blue', 'tab:orange', 'tab:olive', 'tab:red', 'tab:pink']
    marker_type_list = ["+", "x", "D", "o", "s", "h", "P"]

    @staticmethod
    def compare_search_frameworks(framework_log_1, framework_log_2):
        joint_log = pd.merge(framework_log_1, framework_log_2, right_index=True, left_index=True)
        joint_log = joint_log.rename(columns={"accuracy_x": "Conformal Random Forest", "accuracy_y": "Random Search"})
        joint_log[["Conformal Random Forest", "Random Search"]].plot()
        # plt.grid()
        plt.xlabel("Number of Iterations")
        plt.ylabel("Loss")
        plt.legend(loc="lower right")
        plt.savefig('AccuracyPlot.png')
        plt.show()

    @staticmethod
    def plot_loss_surface_cross_section(x_axis_variable, y_axis_variable,
                                        z_axis_variable, loss_surface_log, hypermodel_log, min_z_level=None):

        ax = plt.axes(projection='3d')
        loss_surface_log_light = loss_surface_log.drop(["hidden_layer_sizes", "solver"], axis=1)
        hypermodel_log_light = hypermodel_log.drop(["hidden_layer_sizes", "solver"], axis=1)

        predicted_loss_optimum = hypermodel_log.iloc[hypermodel_log[z_axis_variable].idxmax(), :]

        non_plottable_metrics = list(loss_surface_log_light.columns)
        non_plottable_metrics.remove(x_axis_variable)
        non_plottable_metrics.remove(y_axis_variable)
        non_plottable_metrics.remove(z_axis_variable)
        for metric_column in non_plottable_metrics:
            loss_surface_log_light = loss_surface_log_light[
                loss_surface_log_light[metric_column] == predicted_loss_optimum[metric_column]]

        if min_z_level is not None:
            loss_surface_log_light = loss_surface_log_light[loss_surface_log_light[z_axis_variable] > min_z_level]

        ax.plot_trisurf(loss_surface_log_light[x_axis_variable], loss_surface_log_light[y_axis_variable],
                        loss_surface_log_light[z_axis_variable],
                        cmap='viridis')
        ax.scatter3D(predicted_loss_optimum[x_axis_variable], predicted_loss_optimum[y_axis_variable],
                     predicted_loss_optimum[z_axis_variable],
                     color="red")

        plt.savefig('3dPlot.png')
        plt.show()

    @staticmethod
    def plot_regret(loss_surface_log, hypermodel_log, secondary_model_log):
        max_accuracy = np.max(np.array(loss_surface_log["accuracy"]))
        joint_log = pd.merge(hypermodel_log, secondary_model_log, right_index=True, left_index=True)
        joint_log["regret_x"] = max_accuracy - joint_log["accuracy_x"]
        joint_log["regret_y"] = max_accuracy - joint_log["accuracy_y"]
        joint_log[["regret_x", "regret_y"]].plot()
        # plt.grid()
        plt.savefig('RegretPlot.png')
        plt.show()

    @staticmethod
    def plot_indexed_acquisition_function(full_parameter_space_predictions, confidence_intervals):
        predictions = np.array(full_parameter_space_predictions)
        index = np.arange(len(predictions))
        plt.plot(index, predictions)
        ci = np.array(confidence_intervals)
        plt.fill_between(index, (predictions - ci), (predictions + ci), color='blue', alpha=0.1)
        plt.pause(0.05)
        plt.show()
        plt.clf()

    @staticmethod
    def plot_sorted_conformal_variance(baseline_accuracies, CP_intervals, CP_scorer, hyper_reg_model, confidence_level,
                                       visually_undersample=True, undersampled_size=500):
        if visually_undersample and len(baseline_accuracies) > undersampled_size:
            plottable_baseline_accuracies = baseline_accuracies[
                np.random.choice(len(baseline_accuracies), undersampled_size, replace=False)]
        else:
            plottable_baseline_accuracies = baseline_accuracies.copy()
        sort_index = np.argsort(plottable_baseline_accuracies)
        sorted_accuracies = plottable_baseline_accuracies[sort_index]
        sorted_intervals = CP_intervals[sort_index]
        plot_index = np.arange(len(sorted_accuracies))
        plt.plot(plot_index, sorted_accuracies, color='darkkhaki', label="predicted loss")
        plt.fill_between(plot_index, (sorted_accuracies - sorted_intervals), (sorted_accuracies + sorted_intervals),
                         color='wheat',
                         label=(str(int(round(confidence_level * 100, 0))) + "% confidence interval"))  # , alpha=0.4)
        plt.ylabel("Log Loss")
        plt.xlabel("Sorted Observations")
        plt.legend(loc="lower right")

        plt.pause(0.05)
        plt.savefig('saved_logs/current_log/plots/ci_plots/' + str(CP_scorer) + "-" + str(
            hyper_reg_model) + "-confidence_intervals_plot.png")
        plt.show()
        plt.clf()

    @staticmethod
    def plot_search_performance_tri_chart(logging_data,
                                          min_aggregation=False,
                                          max_aggregation=False,
                                          plot_confidence=False,
                                          plot_over_time=False,
                                          avg_aggregation=False,
                                          moving_average_window=15,
                                          filer=None,
                                          show=True,
                                          chart_indexing_variable='cp_scorer',
                                          line_indexing_variable='secondary_model',
                                          outcome_variable='accuracy',
                                          plot_point_markers=False,
                                          bounding_array=None,
                                          y_label=None):
        plt.clf()
        plt.figure(figsize=(17, 8))
        letter_enum = ["a", "b", "c", "d", "e", "f", "g"]

        # per plot
        plot_index = 0
        for CP_scorer in logging_data[chart_indexing_variable].unique():
            plot_index = plot_index + 1
            color_palette_index = 0
            for secondary_model in logging_data[line_indexing_variable].unique():
                if outcome_variable == 'point_predictor_MSE' and secondary_model == 'RS':
                    continue

                plt.subplot(1, len(logging_data[chart_indexing_variable].unique()), plot_index)

                model_logging_data = logging_data[(logging_data[line_indexing_variable] == secondary_model) & (
                        logging_data[chart_indexing_variable] == CP_scorer)].reset_index(drop=True)
                predictions = np.array(model_logging_data[outcome_variable])
                ci = np.array(model_logging_data['95_CI'])
                if min_aggregation:
                    for i in range(0, len(predictions)):
                        if i == 0:
                            predictions[i] = predictions[i]
                        else:
                            predictions[i] = min(predictions[0:i + 1])
                            if predictions[i] != predictions[i - 1]:
                                turning_index = i
                            try:
                                ci[i] = ci[turning_index]
                            except:
                                pass
                if max_aggregation:
                    for i in range(0, len(predictions)):
                        if i == 0:
                            predictions[i] = predictions[i]
                        else:
                            predictions[i] = max(predictions[0:i + 1])
                            if predictions[i] != predictions[i - 1]:
                                turning_index = i
                            try:
                                ci[i] = ci[turning_index]
                            except:
                                pass

                if avg_aggregation:
                    for i in range(0, len(predictions)):
                        if i == 0:
                            predictions[i] = predictions[i]
                        elif i < len(predictions) - moving_average_window - 1:
                            predictions[i] = np.nanmean(predictions[i:i + moving_average_window])
                        else:
                            predictions[i] = np.nan
                if plot_over_time:
                    minutes = np.array(model_logging_data["runtime"] / 60)
                    if plot_point_markers:
                        plt.plot(minutes, predictions, color=PlotHelper.color_palette[color_palette_index],
                                 label=str(secondary_model), marker=PlotHelper.marker_type_list[color_palette_index],
                                 linewidth=3, markersize=7.5)
                    else:
                        plt.plot(minutes, predictions, color=PlotHelper.color_palette[color_palette_index],
                                 label=str(secondary_model),
                                 linewidth=3)
                else:
                    index = np.arange(len(predictions))
                    if plot_point_markers:
                        plt.plot(index, predictions, color=PlotHelper.color_palette[color_palette_index],
                                 label=str(secondary_model), marker=PlotHelper.marker_type_list[color_palette_index],
                                 linewidth=3, markersize=7.5)
                    else:
                        plt.plot(index, predictions, color=PlotHelper.color_palette[color_palette_index],
                                 label=str(secondary_model),
                                 linewidth=3)
                if plot_confidence and plot_over_time:
                    plt.fill_between(minutes, (predictions - ci), (predictions + ci), alpha=0.35,
                                     color=PlotHelper.color_palette[color_palette_index])
                elif plot_confidence and not plot_over_time:
                    plt.fill_between(index, (predictions - ci), (predictions + ci), alpha=0.35,
                                     color=PlotHelper.color_palette[color_palette_index])

                if plot_over_time:
                    time_descriptor = "x-axis-minutes"
                    plt.xlabel(f"Minutes\n ({letter_enum[plot_index - 1]}) {CP_scorer}", fontsize=21)
                else:
                    time_descriptor = "x-axis-iterations"
                    plt.xlabel(f"Number of Iterations\n ({letter_enum[plot_index - 1]}) {CP_scorer}", fontsize=21)

                if y_label is not None:
                    plt.ylabel(y_label, fontsize=21)
                else:
                    if min_aggregation:
                        plt.ylabel("Minimum Log Loss", fontsize=21)
                    elif max_aggregation:
                        plt.ylabel("Best Validation Accuracy", fontsize=21)
                    elif avg_aggregation:
                        plt.ylabel("Moving Average Accuracy", fontsize=21)
                    else:
                        plt.ylabel("Log Loss", fontsize=21)

                plt.legend(loc="lower right", prop={'size': 19})
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                if bounding_array is not None:
                    plt.ylim(bounding_array)

                color_palette_index = color_palette_index + 1

            # plt.grid(linestyle='--')

        if min_aggregation:
            aggregation_description = "minimum_aggregation"
        elif max_aggregation:
            aggregation_description = "maximum_aggregation"
        elif avg_aggregation:
            aggregation_description = "average_aggregation"
        else:
            aggregation_description = "no_aggregation"

        my_dpi = 96
        # plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        plt.tight_layout()
        if filer is not None:
            if not os.path.exists(
                    filer.output_parent_folder_path + "/" + filer.dataset_name + "/" + filer.plots_folder_name):
                os.makedirs(filer.output_parent_folder_path + "/" + filer.dataset_name + "/" + filer.plots_folder_name)
            plt.savefig(
                filer.output_parent_folder_path + "/" + filer.dataset_name + "/" + filer.plots_folder_name + "/" + str(
                    outcome_variable) + str(
                    CP_scorer) + "-" + str(
                    aggregation_description) + "-" + time_descriptor + "-" + 'cross_model_plot.png', dpi=my_dpi * 1.5)
        if show:
            plt.show()

    @staticmethod
    def plot_single_scorer_datasets(dataset_names,
                                    CP_scorer,
                                    min_aggregation=False,
                                    max_aggregation=False,
                                    plot_confidence=False,
                                    plot_over_time=False,
                                    avg_aggregation=False,
                                    moving_average_window=15,
                                    show=True):
        plt.clf()
        plt.figure(figsize=(27, 8))

        cross_dataset_cross_model_log = pd.DataFrame()
        for m in range(0, len(dataset_names)):
            dataset_name = dataset_names[m]
            filer = Filing(
                data_folder_name="data",
                plots_folder_name="plots",
                results_folder_name="tables_and_results",
                dataset_name=dataset_name)

            cross_model_OOS_log = filer.load_dataframe(df_filename="OOS_log.csv")
            cross_model_OOS_log["sample_category"] = "OOS"
            cross_model_IS_log = filer.load_dataframe(df_filename="IS_log.csv")
            cross_model_IS_log["sample_category"] = "IS"

            # TODO: TEMP, TOGGLE ON AND OFF
            cross_model_OOS_log["secondary_model"] = cross_model_OOS_log["secondary_model"].str.replace(
                "Dense Neural Network", "DNN Predictor")
            cross_model_OOS_log["secondary_model"] = cross_model_OOS_log["secondary_model"].str.replace("KNN",
                                                                                                        "KNN Predictor")
            cross_model_OOS_log["secondary_model"] = cross_model_OOS_log["secondary_model"].str.replace("Random Search",
                                                                                                        "RS")
            cross_model_OOS_log["secondary_model"] = cross_model_OOS_log["secondary_model"].str.replace("Random Forest",
                                                                                                        "RF Predictor")
            cross_model_IS_log["secondary_model"] = cross_model_IS_log["secondary_model"].str.replace(
                "Dense Neural Network", "DNN Predictor")
            cross_model_IS_log["secondary_model"] = cross_model_IS_log["secondary_model"].str.replace("KNN",
                                                                                                      "KNN Predictor")
            cross_model_IS_log["secondary_model"] = cross_model_IS_log["secondary_model"].str.replace("Random Search",
                                                                                                      "RS")
            cross_model_IS_log["secondary_model"] = cross_model_IS_log["secondary_model"].str.replace("Random Forest",
                                                                                                      "RF Predictor")
            # END OF TEMP TOGGLE

            cross_model_log = pd.concat([cross_model_OOS_log, cross_model_IS_log], axis=0)
            cross_dataset_cross_model_log = cross_dataset_cross_model_log.append(cross_model_log)

        # per plot
        plot_index = 0
        for dataset in cross_dataset_cross_model_log["dataset"].unique():
            plot_index = plot_index + 1
            color_palette_index = 0
            for secondary_model in cross_dataset_cross_model_log["secondary_model"].unique():
                plt.subplot(1, len(cross_dataset_cross_model_log["dataset"].unique()), plot_index)

                model_logging_data = cross_dataset_cross_model_log[
                    (cross_dataset_cross_model_log["secondary_model"] == secondary_model) & (
                            cross_dataset_cross_model_log["dataset"] == dataset) & (
                            cross_dataset_cross_model_log["cp_scorer"] == CP_scorer) & (
                            cross_dataset_cross_model_log["sample_category"] == "IS")].reset_index(drop=True)

                predictions = np.array(model_logging_data['accuracy'])
                minutes = np.array(model_logging_data["runtime"] / 60)
                ci = np.array(model_logging_data['95_CI'])
                if min_aggregation:
                    for i in range(0, len(predictions)):
                        if i == 0:
                            predictions[i] = predictions[i]
                        else:
                            predictions[i] = min(predictions[0:i + 1])
                            if predictions[i] != predictions[i - 1]:
                                turning_index = i
                            try:
                                ci[i] = ci[turning_index]
                            except:
                                pass
                if max_aggregation:
                    for i in range(0, len(predictions)):
                        if i == 0:
                            predictions[i] = predictions[i]
                        else:
                            predictions[i] = max(predictions[0:i + 1])
                            if predictions[i] != predictions[i - 1]:
                                turning_index = i
                            try:
                                ci[i] = ci[turning_index]
                            except:
                                pass

                if avg_aggregation:
                    for i in range(0, len(predictions) - moving_average_window - 1):
                        if i == 0:
                            predictions[i] = predictions[i]
                        else:
                            predictions[i] = np.nanmean(predictions[i:i + moving_average_window])
                    predictions = predictions[:-moving_average_window - 1]
                    minutes = minutes[:-moving_average_window - 1]
                if plot_over_time:
                    plt.plot(minutes, predictions, color=PlotHelper.color_palette[color_palette_index],
                             label=str(secondary_model),
                             linewidth=2)
                else:
                    index = np.arange(len(predictions))
                    plt.plot(index, predictions, color=PlotHelper.color_palette[color_palette_index],
                             label=str(secondary_model),
                             linewidth=2)
                if plot_confidence and plot_over_time:
                    plt.fill_between(minutes, (predictions - ci), (predictions + ci), alpha=0.35,
                                     color=PlotHelper.color_palette[color_palette_index])
                elif plot_confidence and not plot_over_time:
                    plt.fill_between(index, (predictions - ci), (predictions + ci), alpha=0.35,
                                     color=PlotHelper.color_palette[color_palette_index])

                if plot_over_time:
                    time_descriptor = "x-axis-minutes"
                    plt.xlabel("Minutes", fontsize=29)
                else:
                    time_descriptor = "x-axis-iterations"
                    plt.xlabel("Number of Iterations", fontsize=29)

                if min_aggregation:
                    plt.ylabel("Minimum Log Loss", fontsize=29)
                elif max_aggregation:
                    plt.ylabel("Best Validation Accuracy", fontsize=29)
                elif avg_aggregation:
                    plt.ylabel("Moving Average Accuracy", fontsize=29)
                else:
                    plt.ylabel("Minimum Average Log Loss", fontsize=29)

                plt.legend(loc="lower right", prop={'size': 22})
                plt.xticks(fontsize=25)
                plt.yticks(fontsize=25)
                # plt.ylim([0.05, 0.3])

                color_palette_index = color_palette_index + 1

            # plt.grid(linestyle='--')

        if min_aggregation:
            aggregation_description = "minimum_aggregation"
        elif avg_aggregation:
            aggregation_description = "average_aggregation"
        else:
            aggregation_description = "no_aggregation"

        my_dpi = 96
        # plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        plt.tight_layout()
        if filer is not None:
            if not os.path.exists(
                    filer.output_parent_folder_path + "/" + filer.dataset_name + "/" + filer.plots_folder_name):
                os.makedirs(filer.output_parent_folder_path + "/" + filer.dataset_name + "/" + filer.plots_folder_name)
            plt.savefig(
                filer.output_parent_folder_path + "/" + "cross_model_plots" + "/" + filer.plots_folder_name + "/" + ''.join(
                    dataset_names) + "-" + str(
                    aggregation_description) + "-" + time_descriptor + "-" + 'cross_model_plot.png', dpi=my_dpi * 1.5)
        if show:
            plt.show()

    @staticmethod
    def plot_IS_vs_OOS_search_performance_single_chart_per_dataset(cross_model_cross_dataset_logging_data,
                                                                   secondary_model,
                                                                   CP_scorer,
                                                                   min_aggregation=False,
                                                                   max_aggregation=False,
                                                                   plot_confidence=False,
                                                                   plot_over_time=False,
                                                                   avg_aggregation=False,
                                                                   moving_average_window=15,
                                                                   filer=None,
                                                                   show=True):

        plt.clf()

        n_subplots = len(cross_model_cross_dataset_logging_data["dataset"].unique())
        print(n_subplots)
        print(cross_model_cross_dataset_logging_data["dataset"].unique())

        # per plot
        plot_index = 0
        for dataset_name in cross_model_cross_dataset_logging_data["dataset"].unique():
            plot_index = plot_index + 1
            color_palette_index = 0

            dataset_filtered_logging_data = cross_model_cross_dataset_logging_data[
                cross_model_cross_dataset_logging_data["dataset"] == dataset_name]
            secondary_model_list = [secondary_model, "Random Search"]
            for secondary_model in secondary_model_list:
                if secondary_model == "Random Search":
                    linestyle_selection = "-."
                else:
                    linestyle_selection = "-"

                for sample_category in ["IS", "OOS"]:
                    plt.subplot(1, n_subplots, plot_index)

                    model_logging_data = dataset_filtered_logging_data[
                        (dataset_filtered_logging_data["secondary_model"] == secondary_model) & (
                                dataset_filtered_logging_data["cp_scorer"] == CP_scorer) & (
                                dataset_filtered_logging_data["sample_category"] == sample_category)].reset_index(
                        drop=True)
                    predictions = np.array(model_logging_data['accuracy'])
                    ci = np.array(model_logging_data['95_CI'])
                    if sample_category == "IS" and min_aggregation:
                        for i in range(0, len(predictions)):
                            if i == 0:
                                predictions[i] = predictions[i]
                            else:
                                predictions[i] = min(predictions[0:i + 1])
                                if predictions[i] != predictions[i - 1]:
                                    turning_index = i
                                try:
                                    ci[i] = ci[turning_index]
                                except:
                                    pass
                    if sample_category == "IS" and max_aggregation:
                        for i in range(0, len(predictions)):
                            if i == 0:
                                predictions[i] = predictions[i]
                            else:
                                predictions[i] = max(predictions[0:i + 1])
                                if predictions[i] != predictions[i - 1]:
                                    turning_index = i
                                try:
                                    ci[i] = ci[turning_index]
                                except:
                                    pass

                    if sample_category == "IS" and avg_aggregation:
                        for i in range(0, len(predictions) - moving_average_window - 1):
                            if i == 0:
                                predictions[i] = predictions[i]
                            else:
                                predictions[i] = np.nanmean(predictions[i:i + moving_average_window])
                    if plot_over_time:
                        minutes = np.array(model_logging_data["runtime"] / 60)
                        plt.plot(minutes, predictions, color=PlotHelper.color_palette[color_palette_index],
                                 label=str(secondary_model) + " " + str(sample_category), linestyle=linestyle_selection)
                    else:
                        index = np.arange(len(predictions))
                        plt.plot(index, predictions, color=PlotHelper.color_palette[color_palette_index],
                                 label=str(secondary_model) + " " + str(sample_category), linestyle=linestyle_selection)
                    if plot_confidence and plot_over_time:
                        plt.fill_between(minutes, (predictions - ci), (predictions + ci), alpha=0.35,
                                         color=PlotHelper.color_palette[color_palette_index])
                    elif plot_confidence and not plot_over_time:
                        plt.fill_between(index, (predictions - ci), (predictions + ci), alpha=0.35,
                                         color=PlotHelper.color_palette[color_palette_index])

                    if plot_over_time:
                        time_descriptor = "x-axis-minutes"
                        plt.xlabel("Minutes")
                    else:
                        time_descriptor = "x-axis-iterations"
                        plt.xlabel("Number of Iterations")
                    plt.ylabel("Log Loss")
                    plt.legend(loc="lower right", prop={'size': 7})
                    # plt.ylim([0, 0.5])

                    color_palette_index = color_palette_index + 1

                # plt.grid(linestyle='--')

            if min_aggregation:
                aggregation_description = "minimum_aggregation"
            elif avg_aggregation:
                aggregation_description = "average_aggregation"
            else:
                aggregation_description = "no_aggregation"

            my_dpi = 96
            # plt.tight_layout()
            if filer is not None:
                if not os.path.exists(
                        filer.output_parent_folder_path + "/" + filer.dataset_name + "/" + filer.plots_folder_name):
                    os.makedirs(
                        filer.output_parent_folder_path + "/" + filer.dataset_name + "/" + filer.plots_folder_name)
                plt.savefig(
                    filer.output_parent_folder_path + "/" + filer.dataset_name + "/" + filer.plots_folder_name + "/" + str(
                        CP_scorer) + "-" + str(
                        aggregation_description) + "-" + time_descriptor + "-" + 'cross_dataset_plot.png',
                    dpi=my_dpi / 2)
            if show:
                plt.show()
