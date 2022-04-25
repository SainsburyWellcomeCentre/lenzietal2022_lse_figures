import math
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from shared import LOOM_ONSETS_S, GROUP_IDS_FIG2
from scipy.stats import wilcoxon, ranksums, fisher_exact

from load import load_head_direction_data as hd_data, load_track_data as data

from shared import (
    save_dir,
    a4figure,
    add_figure_labels,
    add_axes,
    create_panel_if_needed,
    format_track_axis,
    format_general_axis,
    default_colors,
    scatterball_size,
    track_plot_data,
    plot_tracks_general,
    track_timebase,
    timebase_to_show,
    track_display_limits,
    x_axis_spots,
    default_dash_size,
    arrow_parameters,
)


def convert_track_from_string(speeds):
    speeds = [np.array(t.strip("[]").split(" ")) for t in speeds]
    speeds = [np.array([float(x) for x in t if x != ""]) for t in speeds]
    return speeds


def plot_fig_2b(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    t, tracks, linestyle = track_plot_data(data.dataframe["lse"])

    plot_tracks_general(t, tracks, linestyle, fig=fig, axis=ax)

    ax.text(
        2.25,
        55,
        "Post LSE test trials",
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    # scale bar
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
        6.5,
        5,
        "Escape",
        color=default_colors["flee"],
        horizontalalignment="right",
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


def plot_fig_2c_left(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    plot_fig_2c("naive")


def plot_fig_2c_right(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    plot_fig_2c("lse")


def plot_fig_2c(group_id):

    if group_id == "naive":
        title_str = "Naive"
        col = default_colors["ih_ivc_7day"]
    else:
        title_str = "Post LSE"
        col = default_colors["lse"]

    line_length = 4
    line_width_pt = 0.5  # pt
    arrow_size_frac = 0.5  # fraction
    axis_width_mm = 28
    axis_width_units = 20

    arrow_params = arrow_parameters(
        axis_width_mm, axis_width_units, line_width_pt, arrow_size_frac
    )

    start_positions = np.stack(hd_data.start_positions[group_id])
    theta_before = np.array(hd_data.pre_loom_angle[group_id])
    theta_after = np.array(hd_data.post_loom_angle[group_id])

    x = start_positions[:, 0]
    y = 20 * (start_positions[:, 1] + 2) / (22 - 3.5)

    for i in range(len(theta_before)):

        d_pos_before = line_length * np.array(
            [
                np.cos((math.pi / 180) * theta_before[i]),
                np.sin((math.pi / 180) * theta_before[i]),
            ]
        )
        d_pos_after = line_length * np.array(
            [
                np.cos((math.pi / 180) * theta_after[i]),
                np.sin((math.pi / 180) * theta_after[i]),
            ]
        )

        plt.plot(
            20 - (y[i] - np.array([d_pos_before[1], 0])),
            x[i] + np.array([d_pos_before[0], 0]),
            color=[0.4, 0.4, 0.4],
            linewidth=0.5,
            dashes=[4, 2],
            alpha=0.6,
            clip_on=False,
        )

        if hd_data.response_type[group_id][i] == 2:  # don't do anything for freezes

            pass

        elif (
            hd_data.response_type[group_id][i] == 0
        ):  # for no reaction plot a dashed line - requires annoying work

            plt.plot(
                20 - (y[i] + np.array([0, 0.9 * d_pos_after[1]])),
                x[i] - np.array([0, 0.9 * d_pos_after[0]]),
                color=col,
                linewidth=line_width_pt,
                linestyle=(0, (4, 2)),
                clip_on=False,
            )

            plt.arrow(
                20 - y[i] - 0.9 * d_pos_after[1],
                x[i] - 0.9 * d_pos_after[0],
                -0.1 * d_pos_after[1],
                -0.1 * d_pos_after[0],
                head_width=arrow_params["head_width"],
                head_length=arrow_params["head_length"],
                length_includes_head=True,
                facecolor=col,
                edgecolor="none",
                clip_on=False,
                linestyle=(0, (4, 2)),
            )

        else:  # else it's an escape

            plt.arrow(
                20 - y[i],
                x[i],
                -d_pos_after[1],
                -d_pos_after[0],
                width=arrow_params["line_width"],
                head_width=arrow_params["head_width"],
                head_length=arrow_params["head_length"],
                length_includes_head=True,
                facecolor=col,
                edgecolor="none",
                clip_on=False,
            )

    # plot the positions at loom onset
    for i in range(len(theta_before)):

        ball_col = "k" if hd_data.response_type[group_id][i] == 2 else "w"
        plt.scatter(
            20 - y[i],
            x[i],
            s=scatterball_size(1.93),
            facecolors=ball_col,
            edgecolors="k",
            linewidth=0.5,
            clip_on=False,
            zorder=100,
        )

    # plot the delta angles
    for i in range(len(theta_before)):

        d_arrow = 5 * np.array(
            [
                np.cos((math.pi / 180) * theta_after[i]),
                np.sin((math.pi / 180) * theta_after[i]),
            ]
        )

        if hd_data.response_type[group_id][i] == 2:

            pass

        elif hd_data.response_type[group_id][i] == 0:

            plt.plot(
                10 - np.array([0, 0.9 * d_arrow[1]]),
                20 - np.array([0, 0.9 * d_arrow[0]]),
                color=col,
                linewidth=line_width_pt,
                linestyle=(0, (4, 2)),
                clip_on=False,
            )

            plt.arrow(
                10 - 0.9 * d_arrow[1],
                20 - 0.9 * d_arrow[0],
                -0.1 * d_arrow[1],
                -0.1 * d_arrow[0],
                head_width=arrow_params["head_width"],
                head_length=arrow_params["head_length"],
                length_includes_head=True,
                facecolor=col,
                edgecolor="none",
                clip_on=False,
                linestyle=(0, (4, 2)),
            )
        else:

            plt.arrow(
                10,
                20,
                -d_arrow[1],
                -d_arrow[0],
                width=arrow_params["line_width"],
                head_width=arrow_params["head_width"],
                head_length=arrow_params["head_length"],
                length_includes_head=True,
                facecolor=col,
                edgecolor="none",
                clip_on=False,
            )

    plt.scatter(
        10,
        10,
        s=scatterball_size(8.467),
        facecolors=default_colors["shelter"],
        edgecolors=default_colors["shelter"],
        clip_on=False,
    )
    plt.text(
        10,
        10,
        "S",
        color="k",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
    )

    plt.xlim([0, 20])
    plt.ylim([10, 40])
    plt.title(title_str, color=col, fontsize=8)

    plt.axis("off")


def plot_fig_2d_speed(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    t, speed_avg, speed_std = fig_2c_data()

    ctl_up = np.nansum((speed_avg["ih_ivc_7day"], speed_std["ih_ivc_7day"]), axis=0)
    ctl_down = np.nansum((speed_avg["ih_ivc_7day"], -speed_std["ih_ivc_7day"]), axis=0)
    lse_up = np.nansum((speed_avg["lse"], speed_std["lse"]), axis=0)
    lse_down = np.nansum((speed_avg["lse"], -speed_std["lse"]), axis=0)

    yl = np.array([-60, 60])
    for loom in LOOM_ONSETS_S:
        plt.plot([loom, loom], yl, "k", dashes=default_dash_size, linewidth=0.5)

    plt.fill_between(
        t,
        ctl_down,
        ctl_up,
        facecolor=default_colors["ih_ivc_7day"],
        edgecolor="none",
        alpha=0.3,
        clip_on=False,
    )
    plt.plot(t, speed_avg["ih_ivc_7day"], color=default_colors["ih_ivc_7day"])
    plt.fill_between(
        t,
        lse_down,
        lse_up,
        facecolor=default_colors["lse"],
        edgecolor="none",
        alpha=0.3,
        clip_on=False,
    )
    plt.plot(t, speed_avg["lse"], color=default_colors["lse"])

    plt.ylabel("Speed (cm/s)")

    plt.ylim(yl)
    plt.yticks(np.arange(yl[0], yl[1] + 1, 30))
    plt.xlim(track_display_limits)
    plt.xticks([])
    plt.plot(
        [5.5, 6.5], [-50, -50], "k", linewidth=1, clip_on=False, solid_capstyle="butt"
    )
    ax.text(
        6,
        -50 + 3.6,
        "1s",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
    )
    ax.spines["bottom"].set_color("none")

    ax.text(
        4,
        40,
        "Naive",
        color=default_colors["ih_ivc_7day"],
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=8,
    )  # -60+(120.0/50)*60
    ax.text(
        4,
        52,
        "Post LSE",
        color=default_colors["lse"],
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=8,
    )  # -60+(120.0/50)*55

    plt.scatter(
        -1,
        60,
        s=scatterball_size(8.467),
        facecolors=default_colors["shelter"],
        edgecolors=default_colors["shelter"],
        clip_on=False,
    )
    plt.text(
        -1,
        60,
        "S",
        color="k",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
    )
    plt.arrow(-1, 15, 0, 15, color="k", head_width=0.2, head_length=4)
    ax.text(
        -1.4,
        22.5,
        "Toward",
        color="k",
        horizontalalignment="right",
        verticalalignment="center",
        fontsize=8,
        rotation=90,
    )
    plt.arrow(-1, -30, 0, -15, color="k", head_width=0.2, head_length=4)
    ax.text(
        -1.4,
        -37.5,
        "Away",
        color="k",
        horizontalalignment="right",
        verticalalignment="center",
        fontsize=8,
        rotation=90,
    )

    ax.invert_yaxis()


def plot_fig_2e(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    n_flees_to_each_loom, n_freezes, total_n_trials = fig_2e_data()

    freeze_offset = {"ih_ivc_7day": 7, "lse": 8}
    no_response_offset = {"ih_ivc_7day": 9.5, "lse": 10.5}
    face_color = {"ih_ivc_7day": "none", "lse": default_colors["lse"]}
    edge_color = {
        "ih_ivc_7day": default_colors["ih_ivc_7day"],
        "lse": default_colors["lse"],
    }
    linestyle = {"ih_ivc_7day": (0, (3, 1.6)), "lse": "solid"}

    for group_id in GROUP_IDS_FIG2:

        n_no_responses = int(
            total_n_trials[group_id]
            - sum(n_flees_to_each_loom[group_id])
            - n_freezes[group_id]
        )
        prc_flees_to_each_loom = (
            100 * n_flees_to_each_loom[group_id] / total_n_trials[group_id]
        )
        prc_freezes = 100 * n_freezes[group_id] / total_n_trials[group_id]
        prc_no_response = 100 * float(n_no_responses) / total_n_trials[group_id]

        for ii in range(len(n_flees_to_each_loom[group_id])):
            if n_flees_to_each_loom[group_id][ii] != 0:
                plt.bar(
                    ii,
                    prc_flees_to_each_loom[ii],
                    facecolor=face_color[group_id],
                    edgecolor=edge_color[group_id],
                    linestyle=linestyle[group_id],
                    linewidth=0.5,
                    clip_on=False,
                )

        if n_freezes[group_id] != 0:
            plt.bar(
                freeze_offset[group_id],
                prc_freezes,
                facecolor=face_color[group_id],
                edgecolor=edge_color[group_id],
                linestyle=linestyle[group_id],
                linewidth=0.5,
                clip_on=False,
            )

        plt.bar(
            no_response_offset[group_id],
            prc_no_response,
            facecolor=face_color[group_id],
            edgecolor=edge_color[group_id],
            linestyle=linestyle[group_id],
            linewidth=0.5,
            clip_on=False,
        )

        ax.text(
            freeze_offset[group_id],
            prc_freezes,
            f"{int(n_freezes[group_id])}/{total_n_trials[group_id]}",
            fontsize=8,
            color="k",
            rotation=90,
            horizontalalignment="center",
            verticalalignment="bottom",
        )

        if group_id == "ih_ivc_7day":
            ax.text(
                no_response_offset[group_id],
                prc_no_response + 3,
                f"{n_no_responses}/{total_n_trials[group_id]}",
                fontsize=8,
                color="k",
                rotation=90,
                horizontalalignment="center",
                verticalalignment="bottom",
            )
        else:
            ax.text(
                no_response_offset[group_id],
                prc_no_response - 3,
                f"{n_no_responses}/{total_n_trials[group_id]}",
                fontsize=8,
                color="w",
                rotation=90,
                horizontalalignment="center",
                verticalalignment="top",
            )

    ax.spines["bottom"].set_color("none")
    ax.tick_params(axis="x", length=0)

    yl = [0, 100]
    plt.ylabel("Response Freq. (%)")
    plt.ylim(yl)
    plt.yticks(np.arange(0, 101, 20))
    plt.xlim([-1, 14])

    x_axis_spots(ax, yl)

    plt.xticks([7.5, 10], ["Freezing", "No reaction"], rotation=40, ha="right")

    ax.text(
        2,
        85,
        "Naive",
        color=default_colors["ih_ivc_7day"],
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=8,
    )
    ax.text(
        2,
        75,
        "Post LSE",
        color=default_colors["lse"],
        horizontalalignment="left",
        verticalalignment="center",
        fontsize=8,
    )


def plot_fig_2f(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    time_labels = ["7 min", "24 hrs", "72 hrs", "7 days", "8 days", "2 weeks"]

    (
        t,
        percentage_list,
        n_mice,
        prc_flees_ih_ivc_7day,
        n_ih_ivc_7day_mice,
        _,
        _,
    ) = fig_2f_data()

    plt.scatter(
        t,
        percentage_list,
        clip_on=False,
        facecolors="none",
        edgecolors=default_colors["lse"],
    )

    xl = [-24 * 60, 14 * 24 * 70]
    p = np.polyfit(t, percentage_list, 1)
    y_fit = [p[0] * xl[0] + p[1], p[0] * xl[1] + p[1]]

    plt.plot(
        xl,
        2 * [prc_flees_ih_ivc_7day],
        color=default_colors["ih_ivc_7day"],
        linewidth=0.5,
        dashes=[4, 4],
    )
    plt.plot(
        xl,
        y_fit,
        color=default_colors["lse"],
        linewidth=0.5,
        dashes=[4, 4],
        clip_on=False,
    )

    ax.text(
        (xl[0] + xl[1]) / 2.0,
        prc_flees_ih_ivc_7day,
        f"% of non-escapes in naive mice\n("
        + "%.1f" % prc_flees_ih_ivc_7day
        + f"%; n={int(n_ih_ivc_7day_mice)})",
        fontsize=8,
        color=default_colors["ih_ivc_7day"],
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    plt.ylabel("Escape\nSuppression (%)")
    plt.xlabel("LSE Post-test Interval")
    plt.ylim([0, 100])
    plt.yticks(np.arange(0, 101, 20))
    plt.xlim(xl)
    plt.xticks(t, time_labels, rotation=40, ha="right")
    ax.spines["bottom"].set_color("none")
    ax.xaxis.label.set_color(default_colors["lse"])
    ax.tick_params(axis="x", colors=default_colors["lse"])

    for i, x in enumerate(n_mice):
        ax.text(
            t[i],
            percentage_list[i] - 5,
            f"n = {x}",
            fontsize=8,
            rotation=90,
            horizontalalignment="center",
            verticalalignment="top",
            color=default_colors["lse"],
        )


def fig_2c_data():

    speed_avg = {k: [] for k in GROUP_IDS_FIG2}
    speed_std = {k: [] for k in GROUP_IDS_FIG2}

    for group_id in GROUP_IDS_FIG2:

        speeds = data.dataframe[group_id]["speed"]
        speeds = convert_track_from_string(speeds)

        speed_avg[group_id] = -np.nanmean(np.stack(speeds), axis=0)[timebase_to_show]
        speed_std[group_id] = -np.nanstd(np.stack(speeds), axis=0)[timebase_to_show]

    t = track_timebase[timebase_to_show]

    return t, speed_avg, speed_std


def fig_2e_data():

    n_flees_to_each_loom = {k: np.zeros(5) for k in GROUP_IDS_FIG2}
    n_freezes = {k: 0 for k in GROUP_IDS_FIG2}
    total_n_trials = {k: [] for k in GROUP_IDS_FIG2}

    for group_id in GROUP_IDS_FIG2:

        dataframe = data.dataframe[group_id]

        total_n_trials[group_id] = dataframe.shape[0]

        flees = np.array(dataframe["is_flee"])
        freeze = np.array(dataframe["is_freeze"])
        last_loom = np.array(dataframe["last_loom"])

        last_loom = last_loom[flees]

        for i in range(1, 6):
            n_flees_to_each_loom[group_id][i - 1] = sum(last_loom == i)

        n_freezes[group_id] = sum(freeze)

    return n_flees_to_each_loom, n_freezes, total_n_trials


def fig_2f_data():
    t = [7, 24 * 60, 72 * 60, 7 * 24 * 60, 8 * 24 * 60, 14 * 24 * 60]

    percentage_list = []
    n_non_flees = []
    total_n_trials = []
    n_mice = []

    for key, mouse_ids in data.longitudinal_lse_ids.items():
        df_ids = data.dataframe["lse"]["mouse_id"]
        idx = [np.any(a in mouse_ids) for a in df_ids]
        flees = data.dataframe["lse"]["is_flee"][idx]
        n_non_flees.append(sum(idx) - sum(flees))
        total_n_trials.append(sum(idx))
        percentage_list.append(100 * (1 - float(sum(flees)) / sum(idx)))
        n_mice.append(sum([int(np.any(df_ids == m)) for m in mouse_ids]))

    prc_flees_ih_ivc_7day = 100 - 100 * sum(
        data.dataframe["ih_ivc_7day"]["is_flee"]
    ) / len(data.dataframe["ih_ivc_7day"])
    n_ih_ivc_7day_mice = len(data.mouse_ids["ih_ivc_7day"])

    return (
        t,
        percentage_list,
        n_mice,
        prc_flees_ih_ivc_7day,
        n_ih_ivc_7day_mice,
        n_non_flees,
        total_n_trials,
    )


def print_stats():

    baseline_limits = [-1, 0]
    response_limits = [0, 1]

    # speeds
    baseline_frames = np.logical_and(
        track_timebase >= baseline_limits[0], track_timebase < baseline_limits[1]
    )
    response_frames = np.logical_and(
        track_timebase >= response_limits[0], track_timebase < response_limits[1]
    )

    def max_min(a):
        return max(np.abs(a))

    lse_speeds = data.dataframe["lse"].speed
    lse_speeds = convert_track_from_string(lse_speeds)
    ih_ivc_7day_speeds = data.dataframe["ih_ivc_7day"].speed
    ih_ivc_7day_speeds = convert_track_from_string(ih_ivc_7day_speeds)

    lse_baseline = [max_min(x[baseline_frames]) for x in lse_speeds]
    lse_response = [max_min(x[response_frames]) for x in lse_speeds]

    ih_ivc_7day_baseline = [max_min(x[baseline_frames]) for x in ih_ivc_7day_speeds]
    ih_ivc_7day_response = [max_min(x[response_frames]) for x in ih_ivc_7day_speeds]

    _, lse_baseline_vs_response_p_val = wilcoxon(lse_baseline, lse_response)
    _, ih_ivc_7day_baseline_vs_response_p_val = wilcoxon(
        ih_ivc_7day_baseline, ih_ivc_7day_response
    )

    _, ih_ivc_7day_vs_lse_p_val = ranksums(lse_response, ih_ivc_7day_response)

    print(
        f"LSE baseline vs response: "
        f"{np.nanmedian(lse_baseline):.2f} (n={sum(~np.isnan(np.array(lse_baseline)))}) "
        f"vs {np.nanmedian(lse_response):.2f} (n={sum(~np.isnan(np.array(lse_response)))}) "
        f" p={lse_baseline_vs_response_p_val:.2e}, Wilcoxon paired"
    )

    print(
        f"IH IVC 7day baseline vs response: "
        f"{np.nanmedian(ih_ivc_7day_baseline):.2f} (n={sum(~np.isnan(np.array(ih_ivc_7day_baseline)))}) "
        f"vs {np.nanmedian(ih_ivc_7day_response):.2f} (n={sum(~np.isnan(np.array(ih_ivc_7day_response)))}) "
        f" p={ih_ivc_7day_baseline_vs_response_p_val:.2e}, Wilcoxon paired"
    )

    print(
        f"LSE response vs IH IVC 7day response: "
        f"{np.nanmedian(lse_response):.2f} (n={sum(~np.isnan(np.array(lse_response)))}) "
        f"vs {np.nanmedian(ih_ivc_7day_response):.2f} (n={sum(~np.isnan(np.array(ih_ivc_7day_response)))}) "
        f" p={ih_ivc_7day_vs_lse_p_val:.2e}, Wilcoxon paired"
    )

    n_flees_to_each_loom, n_freezes, total_n_trials = fig_2e_data()

    n_non_escapes = {
        "ih_ivc_7day": total_n_trials["ih_ivc_7day"]
        - sum(n_flees_to_each_loom["ih_ivc_7day"]),
        "lse": total_n_trials["lse"] - sum(n_flees_to_each_loom["lse"]),
    }

    _, p = fisher_exact(
        [
            [n_non_escapes["ih_ivc_7day"], n_non_escapes["lse"]],
            [
                total_n_trials["ih_ivc_7day"] - n_non_escapes["ih_ivc_7day"],
                total_n_trials["lse"] - n_non_escapes["lse"],
            ],
        ]
    )

    print(
        f'LSE n non escapes ({n_non_escapes["lse"]:.0f}/{total_n_trials["lse"]:.0f})'
        f'({100*n_non_escapes["lse"]/total_n_trials["lse"]:.1f}%) vs '
        f'IH IVC 7 day ({n_non_escapes["ih_ivc_7day"]:.0f}/{total_n_trials["ih_ivc_7day"]:.0f}), '
        f'({100*n_non_escapes["ih_ivc_7day"]/total_n_trials["ih_ivc_7day"]:.1f}%)'
        f"p={p:.2e}"
    )

    _, p = fisher_exact(
        [
            [n_freezes["ih_ivc_7day"], n_freezes["lse"]],
            [
                total_n_trials["ih_ivc_7day"] - n_freezes["ih_ivc_7day"],
                total_n_trials["lse"] - n_freezes["lse"],
            ],
        ]
    )

    print(
        f'LSE n freezes ({n_freezes["lse"]:.0f}/{total_n_trials["lse"]:.0f}) vs '
        f'IH IVC 7 day ({n_freezes["ih_ivc_7day"]:.0f}/{total_n_trials["ih_ivc_7day"]:.0f}), '
        f"p={p:.2e}"
    )

    lse_delta = np.abs(
        np.array(hd_data.post_loom_angle["lse"])
        - np.array(hd_data.pre_loom_angle["lse"])
    )
    naive_delta = np.abs(
        np.array(hd_data.post_loom_angle["naive"])
        - np.array(hd_data.pre_loom_angle["naive"])
    )

    lse_med_delta = np.median(lse_delta)
    naive_med_delta = np.median(naive_delta)

    print(
        f"median absolute change in heading direction: "
        f"LSE: {lse_med_delta:.1f} degrees (n={len(lse_delta)}) vs "
        f"naive = {naive_med_delta:.1f} degrees (n={len(naive_delta)}),"
        "p < 0.001, Kuiper's test"
    )

    t, _, _, _, _, n_non_lse_flees, total_n_lse_trials = fig_2f_data()

    for i in range(len(n_non_lse_flees)):

        _, p = fisher_exact(
            [
                [n_non_escapes["ih_ivc_7day"], n_non_lse_flees[i]],
                [
                    total_n_trials["ih_ivc_7day"] - n_non_escapes["ih_ivc_7day"],
                    total_n_lse_trials[i] - n_non_lse_flees[i],
                ],
            ]
        )

        print(
            f"post-LSE test time {t[i]/60}: {n_non_lse_flees[i]}/{total_n_lse_trials[i]} vs"
            f' {n_non_escapes["ih_ivc_7day"]}/{total_n_trials["ih_ivc_7day"]}'
            f" p={p:1e}, Fisher"
        )


def main():

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"

    h_fig = a4figure()

    # position of legend labels
    labels = {
        "a": [23, 297 - 65, 0, 0],
        "b": [103, 297 - 65, 0, 0],
        "c": [23, 297 - 65 - 55, 0, 0],
        "d": [103, 297 - 65 - 55, 0, 0],
        "e": [23, 297 - 65 - 2 * 55, 0, 0],
        "f": [103, 297 - 65 - 2 * 55, 0, 0],
    }

    axis_positions = {
        "b": [120, 297 - 100, 62, 32],
        "c_left": [30, 297 - 162, 28, (30 * 28) / 20],
        "c_right": [64, 297 - 162, 28, (30 * 28) / 20],
        "d": [118, 297 - 155, 62, 32],
        "e": [40, 297 - 210, 62, 32],
        "f": [120, 297 - 210, 62, 32],
    }

    # add the legend labels
    add_figure_labels(h_fig, labels)

    # create the axes
    axes_dict = add_axes(h_fig, axis_positions)

    # format the axes
    format_track_axis(axes_dict["b"])
    for panel_id in ["d", "e", "f"]:
        format_general_axis(axes_dict[panel_id])

    plot_fig_2b(fig=h_fig, axis=axes_dict["b"])
    plot_fig_2d_speed(fig=h_fig, axis=axes_dict["d"])
    plot_fig_2c_left(fig=h_fig, axis=axes_dict["c_left"])
    plot_fig_2c_right(fig=h_fig, axis=axes_dict["c_right"])
    plot_fig_2e(fig=h_fig, axis=axes_dict["e"])
    plot_fig_2f(fig=h_fig, axis=axes_dict["f"])

    print_stats()

    h_fig.savefig(str(save_dir / "figure_2.pdf"))

    plt.show()


if __name__ == "__main__":
    main()
