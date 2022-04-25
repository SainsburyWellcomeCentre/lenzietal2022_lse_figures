import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import fisher_exact, ranksums

from shared import (
    a4figure,
    create_panel_if_needed,
    scatterball_size,
    format_track_axis,
    format_general_axis,
    default_colors,
    save_dir,
    x_axis_spots,
    track_plot_data,
    plot_tracks_general,
    add_figure_labels,
    add_axes,
    n_flees_by_mouse,
    GROUP_IDS_FIG1,
    N_GROUPS_FIG1,
)

from load import load_track_data as data


def plot_supp_1b_upper(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    t, tracks, linestyle = track_plot_data(data.dataframe["gh_enriched"])

    plot_tracks_general(
        t,
        tracks,
        linestyle,
        fig=fig,
        axis=ax,
        color_override=default_colors["gh_enriched"],
    )

    ax.text(
        2.25,
        55,
        "Group housed, enriched pen",
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    # text indicating which line is which
    ax.text(
        1.5,
        -5,
        "Escape",
        color=default_colors["gh_enriched"],
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=8,
    )
    ax.text(
        6.5,
        50,
        "No reaction",
        color=default_colors["non_flee"],
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=8,
    )


def plot_supp_1b_lower(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    t, tracks, linestyle = track_plot_data(data.dataframe["ih_enriched"])

    plot_tracks_general(
        t,
        tracks,
        linestyle,
        fig=fig,
        axis=ax,
        color_override=default_colors["ih_enriched"],
    )

    ax.text(
        2.25,
        55,
        "Individually housed, enriched pen",
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="bottom",
    )


def plot_supp_1c(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    n_flees, total_n_trials, p_val = fig_supp_1c_data()

    colors = [default_colors[x] for x in ["gh_enriched", "ih_enriched", "ih_ivc_1mth"]]
    label_str = ["GH enriched", "IH enriched", "IH IVC (1 mth)"]

    n_flees = n_flees[:-1]
    total_n_trials = total_n_trials[:-1]

    percentages = 100 * n_flees / total_n_trials

    for i in range(3):
        plt.bar(i, percentages[i], color=colors[i])

    for i in range(3):
        ax.text(
            i,
            percentages[i] - 3,
            f"{n_flees[i]:.0f}/{total_n_trials[i]:.0f}",
            horizontalalignment="center",
            verticalalignment="top",
            color="k",
            fontsize=8,
            rotation=90,
        )

    ax.spines["bottom"].set_visible(False)

    plt.ylabel("Escape Probability (%)")
    plt.ylim([0, 100])
    plt.yticks(np.arange(0, 101, 20))
    plt.xlim([-0.5, 2.5])
    plt.xticks(range(3), label_str, rotation=40, ha="right")

    # print stats
    for i in range(len(percentages)):
        for j in range(i + 1, len(percentages)):
            if p_val[i, j] < 0.05:
                plt.plot(
                    [i + 0.05, j - 0.05],
                    2 * [100 + (j - i) * 15],
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
                    100 + (j - i) * 15,
                    p_str,
                    color="k",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=8,
                )


def plot_supp_1d(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    percentage_lists, _, _, _ = fig_supp_1d_data()

    x_limits = [-0.5, 3.5]
    y_limits = [0, 100]
    y_tick_spacing = 20

    for i, group_id in enumerate(percentage_lists):
        x = np.arange(len(percentage_lists[group_id][:-1])) - 0.2 + i * 0.2
        plt.bar(
            x,
            percentage_lists[group_id][:-1],
            width=0.2,
            facecolor=default_colors[group_id],
            edgecolor="none",
        )
        plt.bar(
            len(percentage_lists[group_id]) - 1.2 + i * 0.2,
            percentage_lists[group_id][-1],
            width=0.2,
            facecolor=default_colors[group_id],
            edgecolor="none",
        )

    ax.spines["bottom"].set_visible(False)
    plt.ylabel("% mice")

    plt.ylim(y_limits)
    plt.yticks(np.arange(y_limits[0], y_limits[1] + 1, y_tick_spacing))
    plt.xlim(x_limits)

    plt.xticks([0, 1, 2, 3])
    plt.xlabel("# escapes (/3 test trials)")

    # legend
    legend_box_x = [x - 1 for x in [1, 1.3, 1.3, 1]]
    plt.fill(legend_box_x, [70, 70, 74, 74], color=default_colors["ih_ivc_1mth"])
    plt.fill(legend_box_x, [80, 80, 84, 84], color=default_colors["ih_enriched"])
    plt.fill(legend_box_x, [90, 90, 94, 94], color=default_colors["gh_enriched"])
    ax.text(
        0.5,
        72,
        "IH IVC (1 mth)",
        color=default_colors["ih_ivc_1mth"],
        fontsize=7,
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.text(
        0.5,
        82,
        "IH enriched",
        color=default_colors["ih_enriched"],
        fontsize=7,
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.text(
        0.5,
        92,
        "GH enriched",
        color=default_colors["gh_enriched"],
        fontsize=7,
        horizontalalignment="left",
        verticalalignment="center",
    )


def plot_supp_1e(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    avg_speed, sem_speed, p_val, group_ids = fig_supp_1e_data(
        group_ids=["gh_enriched", "ih_pooled"], index_group="ih_pooled"
    )

    x_limits = [-0.5, 4.5]
    y_limits = [20, 80]
    y_tick_space = 10

    colors = [default_colors[x] for x in group_ids]

    ih_pooled = group_ids.index("ih_pooled")
    for i, s in enumerate(avg_speed):

        plt.plot(range(5), s, color=colors[i], linewidth=0.75, zorder=-1)

        for j, a in enumerate(s):
            if a is not None:
                plt.plot(
                    [j, j],
                    [s[j] - sem_speed[i][j], s[j] + sem_speed[i][j]],
                    color=colors[i],
                    linewidth=0.5,
                    zorder=-1,
                )
        if i == ih_pooled:
            plt.scatter(
                range(5),
                s,
                s=scatterball_size(2),
                facecolors="w",
                edgecolors=colors[i],
                linewidths=0.5,
                zorder=1,
            )
        else:
            plt.scatter(
                range(5), s, s=scatterball_size(2), facecolors=colors[i], linewidths=0.5
            )

    for ii in range(len(avg_speed) - 1):
        for jj in range(5):

            if avg_speed[ii][jj] is None:
                continue

            if p_val[ii, jj] < 0.05:
                if p_val[ii, jj] > 0.01:
                    p_str = "*"
                elif p_val[ii, jj] > 0.001:
                    p_str = "**"
                else:
                    p_str = "***"
            else:
                p_str = ""

            ax.text(
                jj,
                avg_speed[ii][jj] - 5,
                p_str,
                color="k",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=8,
            )

    ax.spines["bottom"].set_visible(False)
    plt.ylabel("Avg. Peak Escape\nSpeed (cm/s)", color="k")

    plt.ylim(y_limits)
    plt.yticks(np.arange(y_limits[0], y_limits[1] + 1, y_tick_space))
    plt.xlim(x_limits)
    legend_box_x = [x - 1 for x in [2.5, 2.8, 2.8, 2.5]]
    plt.fill(legend_box_x, [69, 69, 72, 72], color=default_colors["ih_pooled"])
    plt.fill(legend_box_x, [75, 75, 78, 78], color=default_colors["gh_enriched"])

    ax.text(
        2.0,
        70,
        "IH pooled",
        color=default_colors["ih_pooled"],
        fontsize=7,
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.text(
        2.0,
        76,
        "GH enriched",
        color=default_colors["gh_enriched"],
        fontsize=7,
        horizontalalignment="left",
        verticalalignment="center",
    )
    x_axis_spots(ax, y_limits)


def fig_supp_1c_data():
    n_flees = np.zeros(N_GROUPS_FIG1)
    total_n_trials = np.zeros(N_GROUPS_FIG1)
    for i in range(N_GROUPS_FIG1):
        dataframe = data.dataframe[GROUP_IDS_FIG1[i]]
        n_flees[i] = sum(dataframe["is_flee"])
        total_n_trials[i] = dataframe.shape[0]

    # stats
    p_val = np.zeros(shape=(N_GROUPS_FIG1, N_GROUPS_FIG1))
    for i in range(N_GROUPS_FIG1):
        for j in range(i + 1, N_GROUPS_FIG1):
            o, p = fisher_exact(
                [
                    [n_flees[i], n_flees[j]],
                    [total_n_trials[i] - n_flees[i], total_n_trials[j] - n_flees[j]],
                ]
            )
            p_val[i, j] = p

    return n_flees, total_n_trials, p_val


def fig_supp_1d_data():

    group_ids = ["gh_enriched", "ih_enriched", "ih_ivc_1mth"]

    n_trials = 3

    percentage_lists = {k: np.zeros(n_trials + 1) for k in group_ids}
    count_lists = {k: np.zeros(n_trials + 1) for k in group_ids}
    n_mice = {k: [] for k in group_ids}

    for group_id in group_ids:

        dataframe = data.dataframe[group_id]

        mouse_ids = dataframe["mouse_id"].unique()
        n_mice[group_id] = len(mouse_ids)

        for mouse_id in mouse_ids:
            n_escapes, _ = n_flees_by_mouse(dataframe, mouse_id)
            count_lists[group_id][n_escapes] += 1

        percentage_lists[group_id] = 100 * count_lists[group_id] / n_mice[group_id]

    return percentage_lists, count_lists, n_mice, group_ids


def fig_supp_1e_data(group_ids, index_group):

    n_looms = 5
    n_groups = len(group_ids)

    all_speeds = []
    avg_speed = []
    sem_speed = []

    for group_id in group_ids:

        if group_id == "ih_pooled":
            dataframe = pd.DataFrame()
            for k in ["ih_ivc_7day", "ih_ivc_1mth", "ih_enriched"]:
                df = data.dataframe[k]
                dataframe = dataframe.append(df, ignore_index=True)
        else:
            dataframe = data.dataframe[group_id]

        flees = np.array(dataframe["is_flee"])
        speeds = np.array(dataframe["peak_speed"])
        last_loom = np.array(dataframe["last_loom"])

        speeds = speeds[flees]
        last_loom = last_loom[flees]

        all_speeds.append([])
        avg_speed.append([])
        sem_speed.append([])

        for j in range(n_looms):

            s = speeds[last_loom == (j + 1)]

            if len(s) > 0:

                all_speeds[-1].append(s)
                avg_speed[-1].append(np.mean(s))
                if len(s) > 1:  # this is to prevent warning in np.std
                    sem_speed[-1].append(np.std(s, ddof=1) / np.sqrt(len(s)))
                else:
                    sem_speed[-1].append(0)

            else:

                all_speeds[-1].append(None)
                avg_speed[-1].append(None)
                sem_speed[-1].append(None)

    ih_ivc_7day_idx = group_ids.index(index_group)
    p_val = np.ones(shape=(n_groups, n_looms))
    # compare all to last group ('ih_ivc_7day')
    for i in range(n_groups - 1):
        for j in range(n_looms):
            if all_speeds[i][j] is not None and all_speeds[-1][j] is not None:
                if len(all_speeds[i][j]) > 1:
                    [_, p_val[i, j]] = ranksums(
                        all_speeds[i][j], all_speeds[ih_ivc_7day_idx][j]
                    )

    return avg_speed, sem_speed, p_val, group_ids


def main():

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"

    h_fig = a4figure()

    offset_y = 35
    offset_x = 15
    # position of legend labels, 297 - X to be like Illustrator
    labels = {
        "a": [20 + offset_x, 297 - 30 - offset_y, 0, 0],
        "b": [20 + offset_x, 297 - 90 - offset_y, 0, 0],
        "c": [108 + offset_x, 297 - 30 - offset_y, 0, 0],
        "d": [108 + offset_x, 297 - 90 - offset_y, 0, 0],
        "e": [108 + offset_x, 297 - 145 - offset_y, 0, 0],
    }

    axis_positions = {
        "b_upper": [37 + offset_x, 297 - 127 - offset_y, 62, 32],
        "b_lower": [37 + offset_x, 297 - 182 - offset_y, 62, 32],
        "c": [122 + offset_x, 297 - 67 - offset_y, 32, 32],
        "d": [122 + offset_x, 297 - 127 - offset_y, 32, 32],
        "e": [122 + offset_x, 297 - 182 - offset_y, 32, 32],
    }

    # add the legend labels
    add_figure_labels(h_fig, labels)

    # create the axes
    axes_dict = add_axes(h_fig, axis_positions)

    # format the axes
    track_axes = ["b_upper", "b_lower"]
    normal_axes = ["c", "d", "e"]
    for panel_id in track_axes:
        format_track_axis(axes_dict[panel_id])
    for panel_id in normal_axes:
        format_general_axis(axes_dict[panel_id])

    # plot the data in the figure panels
    plot_supp_1b_upper(fig=h_fig, axis=axes_dict["b_upper"])
    plot_supp_1b_lower(fig=h_fig, axis=axes_dict["b_lower"])
    plot_supp_1c(fig=h_fig, axis=axes_dict["c"])
    plot_supp_1d(fig=h_fig, axis=axes_dict["d"])
    plot_supp_1e(fig=h_fig, axis=axes_dict["e"])

    h_fig.savefig(str(save_dir / "figure_1_supplement.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
