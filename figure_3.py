import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact

from shared import (
    a4figure,
    save_dir,
    format_general_axis,
    default_colors,
    create_panel_if_needed,
    add_figure_labels,
    add_axes,
)

import load_track_data as data
import cricket_plotting_fcns


def fig_3b_data():

    group_ids = ["pre_test_immediate_pre", "pre_test_24hr_pre"]
    dfs = [data.dataframe[x] for x in group_ids]

    n_flees, total_n_trials, p_val = fig_3_bar_data(dfs)

    return n_flees, total_n_trials, p_val


def fig_3c_data():

    group_ids = [
        "pre_test_immediate_post",
        "pre_test_24hr_post",
        "pre_test_none_post",
    ]
    dfs = [data.dataframe[x] for x in group_ids]

    n_flees, total_n_trials, p_val = fig_3_bar_data(dfs)

    return n_flees, total_n_trials, p_val


def fig_3d_data():

    # these numbers have been checked but ideally we would fix this to extract from the raw data
    n_flees = [13, 11]
    total_n_trials = [15, 12]

    p_val = np.ones(shape=(2, 2))
    _, p_val[0, 1] = fisher_exact(
        [
            [n_flees[0], n_flees[1]],
            [total_n_trials[0] - n_flees[0], total_n_trials[1] - n_flees[1]],
        ]
    )

    return n_flees, total_n_trials, p_val


def fig_3_bar_data(dfs):

    n_flees = [sum(np.array(x["is_flee"])) for x in dfs]
    total_n_trials = [x.shape[0] for x in dfs]

    p_val = np.ones(shape=(len(n_flees), len(n_flees)))
    for i in range(len(n_flees)):
        for j in range(i + 1, len(n_flees)):
            _, p_val[i, j] = fisher_exact(
                [
                    [n_flees[i], n_flees[j]],
                    [
                        total_n_trials[i] - n_flees[i],
                        total_n_trials[j] - n_flees[j],
                    ],
                ]
            )

    return n_flees, total_n_trials, p_val


def plot_fig_3b(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    n_flees, total_n_trials, p_val = fig_3b_data()

    colors = [
        default_colors["pre_test_immediate"],
        default_colors["pre_test_24hr"],
    ]

    plot_fig_3_bars(
        ax, n_flees, total_n_trials, p_val, colors, bars_empty=False
    )

    plt.ylabel("Pre-LSIE\nEscape Probability (%)", color="k", fontsize=9)

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


def plot_fig_3c(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    n_flees, total_n_trials, p_val = fig_3c_data()
    colors = [
        default_colors["pre_test_immediate"],
        default_colors["pre_test_24hr"],
        default_colors["lsie"],
    ]

    plot_fig_3_bars(
        ax, n_flees, total_n_trials, p_val, colors, bars_empty=True
    )

    plt.ylabel("Post-LSIE\nEscape Probability (%)", color="k", fontsize=9)

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
        color=default_colors["lsie"],
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9,
        rotation=60,
    )


def plot_fig_3d(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    n_flees, total_n_trials, p_val = fig_3d_data()

    colors = [default_colors["ih_ivc_7day"], default_colors["lsie"]]

    plot_fig_3_bars(
        ax, n_flees, total_n_trials, p_val, colors, bars_empty=True
    )

    plt.ylabel(
        "Auditory Threat-evoked\nEscape Probability (%)", color="k", fontsize=9
    )

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
        "LSIE",
        color=default_colors["lsie"],
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=9,
        rotation=60,
    )


def plot_fig_3_bars(
    ax, n_flees, total_n_trials, p_val, colors, bars_empty=False
):

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


def main():

    mpl.rcParams[
        "pdf.fonttype"
    ] = 42  # save text elements as text and not shapes
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"

    h_fig = a4figure()

    labels = {
        "a": [37, 297 - 55, 0, 0],
        "b": [37, 297 - 88, 0, 0],
        "c": [78, 297 - 88, 0, 0],
        "d": [134, 297 - 88, 0, 0],
        "e": [37, 297 - 134, 0, 0],
    }

    axis_positions = {
        "b": [56, 297 - 111, 20, 22],
        "c": [99, 297 - 111, 30, 22],
        "d": [157, 297 - 111, 20, 22],
        "e_naive_tracks": [69, 297 - 190, 20, 50],
        "e_lsie_tracks": [92, 297 - 190, 20, 50],
        "e_naive_events": [130, 297 - 143, 47, 8.3333],
        "e_lsie_events": [130, 297 - 154, 47, 10],
        "e_histograms": [130, 297 - 169, 47, 12],
        "e_line_plots": [130, 297 - 190, 47, 16],
    }

    # add the legend labels
    add_figure_labels(h_fig, labels)

    # create the axes
    axes_dict = add_axes(h_fig, axis_positions)

    # format the axes
    for panel_id in [
        "b",
        "c",
        "d",
        "e_naive_events",
        "e_lsie_events",
        "e_histograms",
        "e_line_plots",
    ]:
        format_general_axis(axes_dict[panel_id])

    plot_fig_3b(fig=h_fig, axis=axes_dict["b"])
    plot_fig_3c(fig=h_fig, axis=axes_dict["c"])
    plot_fig_3d(fig=h_fig, axis=axes_dict["d"])

    cricket_plotting_fcns.plot_example_tracks(
        axes_dict["e_naive_tracks"], "naive"
    )
    cricket_plotting_fcns.plot_example_tracks(
        axes_dict["e_lsie_tracks"], "lsie"
    )

    cricket_plotting_fcns.plot_events(axes_dict["e_naive_events"], "naive")
    cricket_plotting_fcns.plot_events(axes_dict["e_lsie_events"], "lsie")

    cricket_plotting_fcns.plot_histogram(axes_dict["e_histograms"], "naive")
    cricket_plotting_fcns.plot_histogram(axes_dict["e_histograms"], "lsie")

    cricket_plotting_fcns.plot_lines(axes_dict["e_line_plots"], "naive")
    cricket_plotting_fcns.plot_lines(axes_dict["e_line_plots"], "lsie")

    h_fig.savefig(str(save_dir / "figure_3.pdf"))

    plt.show()


if __name__ == "__main__":
    main()
