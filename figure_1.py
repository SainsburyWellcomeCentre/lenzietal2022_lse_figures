import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums, iqr, fisher_exact

from figure_1_supplement import fig_supp_1c_data
from load.load_track_data import mice_by_day
from shared import (
    a4figure,
    create_panel_if_needed,
    scatterball_size,
    format_track_axis,
    format_general_axis,
    default_colors,
    save_dir,
    x_axis_spots,
    robustness_of_mouse,
    track_timebase,
    timebase_to_show,
    track_plot_data,
    plot_tracks_general,
    ARENA_SIZE_CM,
    nice_p_string,
    add_figure_labels,
    add_axes,
    DATA_DIRECTORY,
    GROUP_IDS_FIG1,
)

from load import load_track_data as data


def plot_fig_1c(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    t, track = fig_1c_data()

    scale_bar_x = [5.5, 6.5]
    scale_bar_y = 30

    track_labels_position = {
        "b1": [-0.8, 25, "center"],
        "b2": [0.2, 40, "left"],
        "b3": [0.6, 20, "left"],
        "b4": [1.5, 5, "center"],
    }

    plt.plot(t, track, color="k", linewidth=0.8)

    # scale bar
    plt.plot(
        scale_bar_x,
        2 * [scale_bar_y],
        "k",
        linewidth=1,
        clip_on=False,
        solid_capstyle="butt",
    )
    ax.text(
        np.mean(scale_bar_x),
        scale_bar_y - 1.5,
        "1s",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
    )

    # manually place points along curve
    for key, val in track_labels_position.items():
        ax.text(
            val[0],
            val[1],
            key,
            color="k",
            horizontalalignment=val[2],
            verticalalignment="bottom",
            fontsize=10,
        )


def plot_fig_1d_and_1e(axis_label, fig=None, axis=None):
    if axis_label == "d":
        data_key = "ih_ivc_7day"
        color = "k"
        title = "Individually housed, IVC (2-7 days)"
    elif axis_label == "e":
        data_key = "ih_ivc_1mth"
        color = default_colors[data_key]
        title = "Individually housed, IVC (>28 days)"
    ax, _ = create_panel_if_needed(fig, axis)

    t, tracks, linestyle = track_plot_data(data.dataframe[data_key])

    plot_tracks_general(t, tracks, linestyle, fig=fig, axis=ax, color_override=color)

    ax.text(
        2.25,
        55,
        title,
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    plt.plot(
        [5.5, 6.5], [-5, -5], "k", linewidth=1, clip_on=False, solid_capstyle="butt"
    )
    ax.text(
        6,
        -5 - 1.5,
        "1s",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
    )

    # text indicating which line is which
    ax.text(
        4.5,
        5,
        "Escape",
        color=color,
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
    ax.text(
        6.5,
        28,
        "Freeze",
        color=default_colors["flee"],
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=8,
    )


def plot_fig_1f(n_flees, total_n_trials, fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    percentages = [100 * float(x) / y for x, y in zip(n_flees, total_n_trials)]

    x = np.arange(2, 9)
    x[-1] = 9
    plt.bar(x[:-1], percentages[:-1], width=0.8, facecolor="none", edgecolor="k")
    plt.bar(x[-1], percentages[-1], width=0.8, facecolor=default_colors["ih_ivc_1mth"])

    for i, (x_val, percent) in enumerate(zip(x, percentages)):
        ax.text(
            x_val + 0.09,
            percent - 3,
            f"{n_flees[i]}/{total_n_trials[i]}",
            horizontalalignment="center",
            verticalalignment="top",
            color="k",
            fontsize=8,
            rotation=90,
        )

    plt.ylabel("Escape (%)")
    plt.xticks(x)
    xlabels = list(x[:-1])
    xlabels.extend(["1 mth"])
    ax.set_xticklabels(xlabels)
    plt.xlabel("Days Isolated")


def plot_fig_1g(axis_label, fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)
    group_ids = GROUP_IDS_FIG1[-2:][::-1]
    if axis_label == "speed":
        lists, p_val = fig_1g_data(group_ids)
    elif axis_label == "robustness":
        lists, p_val = escape_robustness_data(group_ids)
    else:
        raise ValueError("axis_label must be 'speed' or 'robustness'")

    colors = ["k", default_colors["ih_ivc_1mth"]]
    label_str = ["IH IVC\n(2-7 days)", "IH IVC\n(1 mth)"]

    for i, y in enumerate(lists):
        x_jitter = i - 0.2 + 0.4 * np.random.rand(len(y))
        plt.scatter(
            x_jitter,
            y,
            s=scatterball_size(1),
            facecolor="none",
            edgecolor=colors[i],
            alpha=0.6,
            clip_on=False,
        )
        plt.plot(
            [i - 0.3, i + 0.3],
            [np.median(y), np.median(y)],
            color="k",
            linewidth=1,
            solid_capstyle="butt",
        )

    ax.spines["bottom"].set_visible(False)
    plt.xlim([-0.5, len(lists) - 0.5])
    plt.xticks(range(2), label_str, rotation=0, ha="center")

    if axis_label == "speed":
        plt.ylabel("Escape Vigour\n(peak speed, cm/s)")
        plt.ylim([0, 120])
        plt.yticks(np.arange(0, 121, 30))
        start = 100
        space = 15
    elif axis_label == "robustness":
        plt.ylabel("Escape Robustness (a.u.)")
        yl = [-5, 800]
        plt.ylim(yl)
        plt.yticks(np.arange(0, 801, 200))
        start = yl[1]
        space = 100
    else:
        return

    # stats
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            if p_val[i, j] < 0.05:
                plt.plot(
                    [i + 0.05, j - 0.05],
                    2 * [start + (j - i) * space],
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
                    start + (j - i) * space,
                    p_str,
                    color="k",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=8,
                )


def plot_fig_1h(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    percentage_list = fig_1h_data()

    x_limits = [-1, 7.5]
    y_limits = [0, 60]
    y_tick_space = 20
    bar_width = 0.4

    for i, group_id in enumerate(percentage_list):
        x = np.arange(len(percentage_list[group_id][:-1])) - bar_width + i * bar_width
        if group_id == "gh_enriched":
            face_color = edge_color = default_colors[group_id]
            zorder = 0
        elif group_id == "ih_ivc_7day":
            face_color = "none"
            edge_color = "k"
            zorder = 100
        else:
            return
        plt.bar(
            x,
            percentage_list[group_id][:-1],
            width=bar_width,
            facecolor=face_color,
            edgecolor=edge_color,
            clip_on=False,
            zorder=zorder,
        )
        plt.bar(
            len(percentage_list[group_id]) - bar_width + i * bar_width,
            percentage_list[group_id][-1],
            width=bar_width,
            facecolor=face_color,
            edgecolor=edge_color,
            clip_on=False,
            zorder=zorder,
        )

    ax.spines["bottom"].set_visible(False)
    plt.ylabel("% Escape Trials")
    plt.ylim(y_limits)
    plt.yticks(np.arange(y_limits[0], y_limits[1] + 1, y_tick_space))
    plt.xlim(x_limits)

    y_offset = x_axis_spots(ax, y_limits, offset=0.2)
    ax.text(
        5.75,
        y_offset,
        "No\nescape",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=8,
    )

    # legend
    gh_offset = 50
    ih_offset = 55
    plt.fill(
        [1, 1.7, 1.7, 1],
        [gh_offset, gh_offset, gh_offset + 2.4, gh_offset + 2.4],
        facecolor=default_colors["gh_enriched"],
        edgecolor="none",
        clip_on=False,
    )
    plt.fill(
        [1, 1.7, 1.7, 1],
        [ih_offset, ih_offset, ih_offset + 2.4, ih_offset + 2.4],
        facecolor="none",
        edgecolor="k",
        clip_on=False,
    )

    ax.text(
        2,
        gh_offset + 1.2,
        "GH enriched",
        color=default_colors["gh_enriched"],
        fontsize=7,
        horizontalalignment="left",
        verticalalignment="center",
        clip_on=False,
    )
    ax.text(
        2,
        ih_offset + 1.2,
        "IH IVC (2-7 days)",
        color="k",
        fontsize=7,
        horizontalalignment="left",
        verticalalignment="center",
        clip_on=False,
    )


def fig_1c_data():

    track = np.load(DATA_DIRECTORY / "example_tracks" / "ca281_1_trial_0.npy")
    t = track_timebase[timebase_to_show]
    track = ARENA_SIZE_CM * track[timebase_to_show]

    return t, track


def fig_1f_data():

    n_flees = []
    total_n_trials = []

    for key, val in mice_by_day.items():
        idx = [np.any(a in val) for a in data.dataframe["ih_ivc_7day"]["mouse_id"]]
        flees = data.dataframe["ih_ivc_7day"]["is_flee"][idx]
        n_flees.append(sum(flees))
        total_n_trials.append(len(flees))

    n_flees.append(sum(data.dataframe["ih_ivc_1mth"]["is_flee"]))
    total_n_trials.append(len(data.dataframe["ih_ivc_1mth"]))

    # stats
    n_groups = len(mice_by_day.keys())
    p_val = np.zeros(shape=(n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            o, p = fisher_exact(
                [
                    [n_flees[i], n_flees[j]],
                    [
                        total_n_trials[i] - n_flees[i],
                        total_n_trials[j] - n_flees[j],
                    ],
                ]
            )
            p_val[i, j] = p

    return n_flees, total_n_trials, p_val


def fig_1g_data(group_ids):
    n_groups = len(group_ids)
    speed_lists = []
    for group_id in group_ids:
        dataframe = data.dataframe[group_id]
        flees = np.array(dataframe["is_flee"])
        speeds = np.array(dataframe["peak_speed"])
        speed_lists.append(speeds[flees])

    # stats
    p_val = np.zeros(shape=(n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            _, p_val[i, j] = ranksums(speed_lists[i], speed_lists[j])

    return speed_lists, p_val


def fig_1h_data():

    group_ids = ["ih_ivc_7day", "gh_enriched"]

    percentage_lists = {k: [] for k in group_ids}

    for group_id in group_ids:

        dataframe = data.dataframe[group_id]
        total_n_trials = dataframe.shape[0]

        for i in range(1, 6):
            df_escapes = dataframe[dataframe["is_flee"]]
            fraction_on_this_loom = sum(df_escapes["last_loom"] == i) / total_n_trials
            percentage_lists[group_id].append(100 * fraction_on_this_loom)

        percentage_lists[group_id].append(
            100 * (total_n_trials - sum(dataframe["is_flee"])) / total_n_trials
        )

    return percentage_lists


def latency_data(group_ids):
    n_groups = len(group_ids)
    latency_lists = []
    for group_id in group_ids:
        dataframe = data.dataframe[group_id]
        flees = np.array(dataframe["is_flee"])
        latencies = np.array(dataframe["latency"])
        latency_lists.append(latencies[flees])

    # stats
    p_val = np.zeros(shape=(n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            _, p_val[i, j] = ranksums(latency_lists[i], latency_lists[j])

    return latency_lists, p_val


def escape_robustness_data(group_ids):
    n_groups = len(group_ids)
    robustness_lists = []
    for group_id in group_ids:
        dataframe = data.dataframe[group_id]
        mouse_ids = list(set(dataframe["mouse_id"]))
        robustness_lists.append([robustness_of_mouse(dataframe, x) for x in mouse_ids])

    # stats
    p_val = np.zeros(shape=(n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            _, p_val[i, j] = ranksums(robustness_lists[i], robustness_lists[j])

    return robustness_lists, p_val


def print_stats_fig_1f(n_flees, total_n_trials, stats):

    label_str = [f"Isolated for {k} days" for k in mice_by_day.keys()]

    for i in range(len(label_str)):
        for j in range(i + 1, len(label_str)):
            p_str = nice_p_string(stats[i, j])
            print(
                f"{label_str[i]} ({n_flees[i]:.0f}/{total_n_trials[i]:.0f}) vs {label_str[j]} "
                f"({n_flees[j]:.0f}/{total_n_trials[j]:.0f}), p={p_str}, Fisher's exact test"
            )


def print_stats():

    group_ids = ["gh_enriched", "ih_enriched", "ih_ivc_1mth", "ih_ivc_7day"]
    n_flees, total_n_trials, p_val = fig_supp_1c_data()

    print("\n\n\nESCAPE PROBABILITY\n\n\n")
    for i in range(len(n_flees) - 1):
        for j in range(i + 1, len(n_flees)):
            nice_p = nice_p_string(p_val[i, j])
            print(
                f"{group_ids[i]}: {n_flees[i]:.0f} / {total_n_trials[i]:.0f} vs "
                f"{group_ids[j]}: {n_flees[j]:.0f} / {total_n_trials[j]:.0f} "
                f"p = {nice_p}"
            )

    print("\n\n\nLATENCY\n\n\n")
    latency_lists, p_val = latency_data(GROUP_IDS_FIG1)
    display_metric_stats(group_ids, p_val, latency_lists)

    print("\n\n\nSPEED\n\n\n")
    speed_lists, p_val = fig_1g_data(GROUP_IDS_FIG1)
    display_metric_stats(group_ids, p_val, speed_lists)

    print("\n\n\nROBUSTNESS\n\n\n")
    robustness_lists, p_val = escape_robustness_data(GROUP_IDS_FIG1)
    display_metric_stats(group_ids, p_val, robustness_lists)

    print("\n\n\nISOLATION DAYS\n\n\n")
    n_flees, total_n_trials, p_val = fig_1f_data()
    print_stats_fig_1f(n_flees, total_n_trials, p_val)


def display_metric_stats(group_ids, p_val, metric_lists):
    for i in range(len(metric_lists) - 1):
        for j in range(i + 1, len(metric_lists)):
            nice_p = nice_p_string(p_val[i, j])
            print(
                f"{group_ids[i]}: {np.median(metric_lists[i]):.2f}a.u. IQR={iqr(metric_lists[i]):.2f}a.u. vs "
                f"{group_ids[j]}: {np.median(metric_lists[j]):.2f}a.u. IQR={iqr(metric_lists[j]):.2f}a.u. "
                f"p = {nice_p}"
            )


def main():

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"

    h_fig = a4figure()
    offset_y = 45
    offset_x = 3

    # position of legend labels, 297 - X to be like Illustrator
    labels = {
        "a": [20 + offset_x, 297 - 30 - offset_y, 0, 0],
        "b": [50 + offset_x, 297 - 52 - offset_y, 0, 0],
        "c": [20 + offset_x, 297 - 78 - offset_y, 0, 0],
        "d": [108 + offset_x, 297 - 30 - offset_y, 0, 0],
        "e": [108 + offset_x, 297 - 78 - offset_y, 0, 0],
        "f": [20 + offset_x, 297 - 130 - offset_y, 0, 0],
        "g": [85 + offset_x, 297 - 130 - offset_y, 0, 0],
        "h": [122 + offset_x, 297 - 130 - offset_y, 0, 0],
    }

    axis_positions = {
        "c": [37 + offset_x, 297 - 117 - offset_y, 62, 32],
        "d": [117 + offset_x, 297 - 67 - offset_y, 62, 32],
        "e": [117 + offset_x, 297 - 117 - offset_y, 62, 32],
        "f": [37 + offset_x, 297 - 162 - offset_y, 43, 32],
        "g": [100 + offset_x, 297 - 162 - offset_y, 23, 32],
        "h": [137 + offset_x, 297 - 162 - offset_y, 48, 32],
    }

    # add the legend labels
    add_figure_labels(h_fig, labels)

    # create the axes
    axes_dict = add_axes(h_fig, axis_positions)

    # format the axes
    track_axes = ["c", "d", "e"]
    normal_axes = ["f", "g", "h"]
    for panel_id in track_axes:
        format_track_axis(axes_dict[panel_id])
    for panel_id in normal_axes:
        format_general_axis(axes_dict[panel_id])

    # plot the data in the figure panels
    plot_fig_1c(fig=h_fig, axis=axes_dict["c"])
    plot_fig_1d_and_1e(fig=h_fig, axis=axes_dict["d"], axis_label="d")
    plot_fig_1d_and_1e(fig=h_fig, axis=axes_dict["e"], axis_label="e")

    plot_fig_1g("speed", fig=h_fig, axis=axes_dict["g"])
    plot_fig_1h(fig=h_fig, axis=axes_dict["h"])

    n_flees, total_n_trials, _ = fig_1f_data()
    plot_fig_1f(n_flees, total_n_trials, fig=h_fig, axis=axes_dict["f"])

    print_stats()

    h_fig.savefig(str(save_dir / "figure_1.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
