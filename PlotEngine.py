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
    def plot_loss_surface_cross_section(x_axis_variable,
                                        y_axis_variable,
                                        z_axis_variable,
                                        loss_surface_log,
                                        hypermodel_log,
                                        min_z_level=None):

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
    def plot_sorted_conformal_variance(baseline_accuracies,
                                       CP_intervals,
                                       CP_scorer,
                                       hyper_reg_model,
                                       confidence_level,
                                       visually_undersample=True,
                                       undersampled_size=500):
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
    def plot_multi_chart_figure(logging_data,
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
                # ci = np.array(model_logging_data['95_CI'])
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
                    CP_scorer) + "-" + str(plot_over_time) + "-" + str(
                    aggregation_description) + "-" + time_descriptor + "-" + 'cross_model_plot.png', dpi=my_dpi * 1.5)
        if show:
            plt.show()

        plt.close()

    @staticmethod
    def plot_performance_variance_breach_trifecta(logging_data, plot_over_time):
        plt.clf()
        plt.figure(figsize=(17, 8))
        mycolorpalette = ['tab:pink', 'tab:purple', 'tab:red']

        logging_data["secondary_model_confidence_proxy"] = logging_data["secondary_model"].str.replace(" Predictor",
                                                                                                       "") + " " + (
                                                                   logging_data["confidence_level"].astype(
                                                                       float) * 100).astype(int).astype(
            str) + "% Confidence"
        logging_data.loc[logging_data["secondary_model"] == "RS", ["secondary_model_confidence_proxy"]] = "RS"

        # PLOT 1:
        outcome_variable = "accuracy_score"
        bounding_array = None

        plot_index = 1
        color_palette_index = 0
        for secondary_model in logging_data["secondary_model_confidence_proxy"].unique():

            plt.subplot(1, 3, plot_index)

            model_logging_data = logging_data[
                (logging_data["secondary_model_confidence_proxy"] == secondary_model)].reset_index(
                drop=True)
            predictions = np.array(model_logging_data[accuracy_metric])

            # max:
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

            if plot_over_time:
                minutes = np.array(model_logging_data["runtime"] / 60)
                plt.plot(minutes, predictions, color=mycolorpalette[color_palette_index],
                         label=str(secondary_model),
                         linewidth=3)
            else:
                index = np.arange(len(predictions))
                plt.plot(index, predictions, color=mycolorpalette[color_palette_index],
                         label=str(secondary_model),
                         linewidth=3)
            if plot_over_time:
                plt.xlabel("Minutes", fontsize=21)
            else:
                plt.xlabel("Number of Iterations", fontsize=21)

            plt.ylabel("Best Validation Accuracy", fontsize=21)

            plt.legend(loc="lower right", prop={'size': 19})
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            if bounding_array is not None:
                plt.ylim(bounding_array)

            color_palette_index = color_palette_index + 1

        # PLOT 2:
        avg_aggregation = True
        moving_average_window = 400
        bounding_array = [-0.1, 0.35]  # 0.85]

        plot_index = 2
        color_palette_index = 0
        for secondary_model in logging_data["secondary_model_confidence_proxy"].unique():

            plt.subplot(1, 3, plot_index)

            model_logging_data = logging_data[
                (logging_data["secondary_model_confidence_proxy"] == secondary_model)].reset_index(
                drop=True)
            predictions = np.array(model_logging_data[outcome_variable])

            # avg:
            for i in range(0, len(predictions)):
                if i == 0:
                    predictions[i] = predictions[i]
                elif i < len(predictions) - moving_average_window - 1:
                    predictions[i] = np.std(predictions[i:i + moving_average_window])
                else:
                    predictions[i] = np.nan
            if plot_over_time:
                minutes = np.array(model_logging_data["runtime"] / 60)
                plt.plot(minutes, predictions, color=mycolorpalette[color_palette_index],
                         label=str(secondary_model),
                         linewidth=3)
            else:
                index = np.arange(len(predictions))
                plt.plot(index, predictions, color=mycolorpalette[color_palette_index],
                         label=str(secondary_model),
                         linewidth=3)
            if plot_over_time:
                plt.xlabel("Minutes", fontsize=21)
            else:
                plt.xlabel("Number of Iterations", fontsize=21)

            plt.ylabel("Validation Accuracy: 400 Iteration Rolling Standard Deviation", fontsize=21)

            plt.legend(loc="lower right", prop={'size': 19})
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            if bounding_array is not None:
                plt.ylim(bounding_array)

            color_palette_index = color_palette_index + 1

        # PLOT 3:
        bounding_array = [-0.1, 0.8]  # 0.65]
        outcome_variable = "CI_breach"
        logging_data = logging_data[logging_data["secondary_model"] != "RS"]

        plot_index = 3
        color_palette_index = 0
        for secondary_model in logging_data["secondary_model_confidence_proxy"].unique():

            plt.subplot(1, 3, plot_index)

            model_logging_data = logging_data[
                (logging_data["secondary_model_confidence_proxy"] == secondary_model)].reset_index(
                drop=True)
            predictions = np.array(model_logging_data[outcome_variable])

            # avg:
            for i in range(0, len(predictions)):
                if i == 0:
                    predictions[i] = predictions[i]
                elif i < len(predictions) - moving_average_window - 1:
                    predictions[i] = np.nanmean(predictions[i:i + moving_average_window])
                else:
                    predictions[i] = np.nan

            if plot_over_time:
                minutes = np.array(model_logging_data["runtime"] / 60)
                plt.plot(minutes, predictions, color=mycolorpalette[color_palette_index],
                         label=str(secondary_model),
                         linewidth=3)
            else:
                index = np.arange(len(predictions))
                plt.plot(index, predictions, color=mycolorpalette[color_palette_index],
                         label=str(secondary_model),
                         linewidth=3)
            if plot_over_time:
                plt.xlabel("Minutes", fontsize=21)
            else:
                plt.xlabel("Number of Iterations", fontsize=21)

            plt.ylabel("CI Breach Rate: 400 Iteration Rolling Mean", fontsize=21)

            plt.legend(loc="lower right", prop={'size': 19})
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            if bounding_array is not None:
                plt.ylim(bounding_array)

            color_palette_index = color_palette_index + 1

        my_dpi = 96
        # plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        plt.tight_layout()
        if filer is not None:
            if not os.path.exists(
                    filer.output_parent_folder_path + "/" + filer.dataset_name + "/" + filer.plots_folder_name):
                os.makedirs(filer.output_parent_folder_path + "/" + filer.dataset_name + "/" + filer.plots_folder_name)
            plt.savefig(
                filer.output_parent_folder_path + "/" + filer.dataset_name + "/" + filer.plots_folder_name + "/" + 'tri_confidence_chart.png',
                dpi=my_dpi * 1.5)
