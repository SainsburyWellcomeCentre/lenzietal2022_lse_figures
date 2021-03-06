import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact, ranksums, iqr

from shared import (
    a4figure,
    save_dir,
    format_general_axis,
    default_colors,
    create_panel_if_needed,
    add_figure_labels,
    add_axes,
)

from load import load_track_data as data
import cricket_plotting_fcns


def plot_fig_3b(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    n_flees, total_n_trials, p_val = fig_3b_data()

    colors = [default_colors["ih_ivc_7day"], default_colors["lse"]]

    plot_fig_3_bars(ax, n_flees, total_n_trials, p_val, colors, bars_empty=True)

    plt.ylabel("Auditory Threat-evoked\nEscape Probability (%)", color="k", fontsize=9)

    ax.text(
        0,
        -10,
        "Naive",
        color=default_colors["ih_ivc_7day"],
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9,
        rotation=60,
    )
    ax.text(
        1,
        -10,
        "LSE",
        color=default_colors["lse"],
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9,
        rotation=60,
    )


def plot_fig_3d(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    n_flees, total_n_trials, p_val = fig_3d_data()

    colors = [default_colors["pre_test_immediate"], default_colors["pre_test_24hr"]]

    plot_fig_3_bars(ax, n_flees, total_n_trials, p_val, colors, bars_empty=False)

    plt.ylabel("Pre-LSE\nEscape Probability (%)", color="k", fontsize=9)

    ax.text(
        0.5,
        140,
        "Pre-test",
        color="k",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=10,
    )
    ax.text(
        0,
        -10,
        "< 0.2 hrs",
        color=default_colors["pre_test_immediate"],
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9,
        rotation=60,
    )
    ax.text(
        1,
        -10,
        "> 24 hrs",
        color=default_colors["pre_test_24hr"],
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9,
        rotation=60,
    )


def plot_fig_3e(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    n_flees, total_n_trials, p_val, _ = fig_3e_data()
    colors = [
        default_colors["pre_test_immediate"],
        default_colors["pre_test_24hr"],
        default_colors["lse"],
    ]

    plot_fig_3_bars(ax, n_flees, total_n_trials, p_val, colors, bars_empty=True)

    plt.ylabel("Post-LSE\nEscape Probability (%)", color="k", fontsize=9)

    ax.text(
        1,
        140,
        "Post-test",
        color="k",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=10,
    )
    ax.text(
        0,
        -10,
        "< 0.2 hrs",
        color=default_colors["pre_test_immediate"],
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9,
        rotation=60,
    )
    ax.text(
        1,
        -10,
        "> 24 hrs",
        color=default_colors["pre_test_24hr"],
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9,
        rotation=60,
    )
    ax.text(
        2,
        -10,
        "No pre-\ntest",
        color=default_colors["lse"],
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9,
        rotation=60,
    )


def plot_fig_3_bars(ax, n_flees, total_n_trials, p_val, colors, bars_empty=False):

    prc_flee = [100 * float(x) / y for x, y in zip(n_flees, total_n_trials)]

    for i, bar_height in enumerate(prc_flee):
        x = np.array([i + 0.3, i + 0.3, i - 0.3, i - 0.3])
        y = np.array([0, bar_height, bar_height, 0])
        h = plt.fill(x, y, clip_on=False)
        if bars_empty:
            h[0].set_facecolor("none")
            h[0].set_edgecolor(colors[i])
        else:
            h[0].set_facecolor(colors[i])
            h[0].set_edgecolor(colors[i])

    for i in range(len(prc_flee)):
        if prc_flee[i] < 30:
            ax.text(
                i,
                prc_flee[i] + 5,
                f"{n_flees[i]}/{total_n_trials[i]}",
                horizontalalignment="center",
                verticalalignment="bottom",
                color="k",
                fontsize=8,
                rotation=90,
            )
        else:
            ax.text(
                i,
                prc_flee[i] - 3,
                f"{n_flees[i]}/{total_n_trials[i]}",
                horizontalalignment="center",
                verticalalignment="top",
                color="k",
                fontsize=8,
                rotation=90,
            )

    plt.xlim([-0.5, len(prc_flee) - 0.5])

    plt.xticks([])
    plt.ylim([0, 100])
    plt.yticks(range(0, 101, 50))
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible("k")
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.yaxis.set_ticks_position("left")

    ax.grid(False)

    ax.yaxis.label.set_fontsize(10)
    ax.xaxis.label.set_fontsize(10)
    for item in ax.get_yticklabels() + ax.get_xticklabels():
        item.set_fontsize(8)

    # print stats lines
    for i in range(len(prc_flee)):
        for j in range(i + 1, len(prc_flee)):
            if p_val[i, j] < 0.05:
                plt.plot(
                    [i, j],
                    2 * [max(prc_flee) + (j - i) * 18],
                    color="k",
                    clip_on=False,
                    linewidth=1,
                    solid_capstyle="butt",
                )
                if p_val[i, j] > 0.01:
                    p_str = "*"
                elif p_val[i, j] > 0.001:
                    p_str = "**"
                else:
                    p_str = "***"
                ax.text(
                    (i + j) / 2,
                    max(prc_flee) + (j - i) * 18,
                    p_str,
                    color="k",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=8,
                )


def fig_3b_data():
    group_ids = ["auditory_naive", "auditory_lse"]
    dfs = [data.dataframe[x] for x in group_ids]

    n_flees, total_n_trials, p_val = fig_3_bar_data(dfs)

    return n_flees, total_n_trials, p_val


def fig_3d_data():

    group_ids = ["pre_test_immediate_pre", "pre_test_24hr_pre"]
    dfs = [data.dataframe[x] for x in group_ids]

    n_flees, total_n_trials, p_val = fig_3_bar_data(dfs)

    return n_flees, total_n_trials, p_val


def fig_3e_data():

    group_ids = ["pre_test_immediate_post", "pre_test_24hr_post", "pre_test_none_post"]
    dfs = [data.dataframe[x] for x in group_ids]

    n_flees, total_n_trials, p_val = fig_3_bar_data(dfs)

    return n_flees, total_n_trials, p_val, group_ids


def fig_3_bar_data(dfs):

    n_flees = [sum(np.array(x["is_flee"])) for x in dfs]
    total_n_trials = [x.shape[0] for x in dfs]

    p_val = np.ones(shape=(len(n_flees), len(n_flees)))
    for i in range(len(n_flees)):
        for j in range(i + 1, len(n_flees)):
            _, p_val[i, j] = fisher_exact(
                [
                    [n_flees[i], n_flees[j]],
                    [total_n_trials[i] - n_flees[i], total_n_trials[j] - n_flees[j]],
                ]
            )

    return n_flees, total_n_trials, p_val


def print_stats():

    _, p = fisher_exact([[13, 11], [2, 1]])

    print(f"AUDITORY: naive vs. LSE: p = {p:.2f}")

    n_flees, total_n_trials, p_val, group_ids = fig_3e_data()

    print(
        f"{group_ids[0]}: {n_flees[0]}/{total_n_trials[0]} vs. "
        f"{group_ids[2]}: {n_flees[2]}/{total_n_trials[2]} "
        f"p = {p_val[0, 2]}"
    )

    print(
        f"{group_ids[1]}: {n_flees[1]}/{total_n_trials[1]} vs. "
        f"{group_ids[2]}: {n_flees[2]}/{total_n_trials[2]} "
        f"p = {p_val[1, 2]}"
    )

    # cricket statistics
    initial_duration = 10  # minutes

    def n_events_in_initial_period(group_id):
        is_event_in_initial_period = [
            np.logical_and(x >= 0, x < initial_duration)
            for x in cricket_plotting_fcns.event_times[group_id]
        ]
        return [sum(x) for x in is_event_in_initial_period]

    _, p = ranksums(
        n_events_in_initial_period("naive"), n_events_in_initial_period("lse")
    )

    print(
        f"median number of bouts in first {initial_duration} minutes for naive mice"
        f' = {np.median(n_events_in_initial_period("naive"))} (IQR={iqr(n_events_in_initial_period("naive"))})'
        f" (n = 5) vs. median for LSE mice"
        f' = {np.median(n_events_in_initial_period("lse"))} (IQR={iqr(n_events_in_initial_period("lse"))})'
        f" (n = 6), p = {p:.2f}, Wilcoxon rank-sum test"
    )

    def n_retreats_in_initial_period(group_id):

        event_times = cricket_plotting_fcns.event_times[group_id]
        reaches_shelter = [
            np.array(x) for x in cricket_plotting_fcns.reaches_shelter[group_id]
        ]

        is_event_in_initial_period = [
            np.logical_and(x >= 0, x < initial_duration) for x in event_times
        ]
        n_retreats = []
        for i in range(len(is_event_in_initial_period)):
            n_retreats.append(sum(reaches_shelter[i][is_event_in_initial_period[i]]))
        return n_retreats

    _, p = ranksums(
        n_retreats_in_initial_period("naive"), n_retreats_in_initial_period("lse")
    )

    print(
        f"median number of retreats in first {initial_duration} minutes for naive mice"
        f' = {np.median(n_retreats_in_initial_period("naive"))} (IQR={iqr(n_retreats_in_initial_period("naive"))})'
        f" (n = 5) vs. median for LSE mice"
        f' = {np.median(n_retreats_in_initial_period("lse"))} (IQR={iqr(n_retreats_in_initial_period("lse"))})'
        f" (n = 6), p = {p:.2f}, Wilcoxon rank-sum test"
    )


def main():

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"

    h_fig = a4figure()

    labels = {
        "a": [37, 297 - 73, 0, 0],
        "b": [37, 297 - 140, 0, 0],
        "c": [73, 297 - 140, 0, 0],
        "d": [73, 297 - 164, 0, 0],
        "e": [116, 297 - 164, 0, 0],
    }

    axis_positions = {
        "a_naive_tracks": [59, 297 - 126, 20, 50],
        "a_lse_tracks": [82, 297 - 126, 20, 50],
        "a_naive_events": [124, 297 - 79, 47, 8.3333],
        "a_lse_events": [124, 297 - 90, 47, 10],
        "a_histograms": [124, 297 - 105, 47, 12],
        "a_line_plots": [124, 297 - 126, 47, 16],
        "b": [50, 297 - 189, 20, 48],
        "d": [92, 297 - 189, 20, 22],
        "e": [135, 297 - 189, 30, 22],
    }

    # add the legend labels
    add_figure_labels(h_fig, labels)

    # create the axes
    axes_dict = add_axes(h_fig, axis_positions)

    # format the axes
    for panel_id in [
        "b",
        "d",
        "e",
        "a_naive_events",
        "a_lse_events",
        "a_histograms",
        "a_line_plots",
    ]:
        format_general_axis(axes_dict[panel_id])

    cricket_plotting_fcns.plot_example_tracks(axes_dict["a_naive_tracks"], "naive")
    cricket_plotting_fcns.plot_example_tracks(axes_dict["a_lse_tracks"], "lse")

    cricket_plotting_fcns.plot_events(axes_dict["a_naive_events"], "naive")
    cricket_plotting_fcns.plot_events(axes_dict["a_lse_events"], "lse")

    cricket_plotting_fcns.plot_histogram(axes_dict["a_histograms"], "naive")
    cricket_plotting_fcns.plot_histogram(axes_dict["a_histograms"], "lse")

    cricket_plotting_fcns.plot_lines(axes_dict["a_line_plots"], "naive")
    cricket_plotting_fcns.plot_lines(axes_dict["a_line_plots"], "lse")

    plot_fig_3b(fig=h_fig, axis=axes_dict["b"])
    plot_fig_3d(fig=h_fig, axis=axes_dict["d"])
    plot_fig_3e(fig=h_fig, axis=axes_dict["e"])

    print_stats()

    h_fig.savefig(str(save_dir / "figure_3.pdf"))

    plt.show()


if __name__ == "__main__":
    main()
