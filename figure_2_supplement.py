import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from shared import (
    format_track_axis,
    format_general_axis,
    add_figure_labels,
    add_axes,
    a4figure,
    create_panel_if_needed,
    track_plot_data,
    plot_tracks_general,
    default_colors,
    x_axis_spots,
    save_dir,
    EXAMPLE_TRACKS_DIR,
)

from load import load_track_data as data

mids_with_shelter = ["1116285", "1116286", "1116287", "1116289"]
mids_no_shelter = ["1114188", "1114189", "1114307", "1114186"]


def plot_example_exploration_track(h_ax, mouse_id, group):
    color = default_colors[group]
    plt.sca(h_ax)
    savedir = pathlib.Path(EXAMPLE_TRACKS_DIR)
    x = np.load(str(savedir / f"{mouse_id}_x.npy"))
    y = np.load(str(savedir / f"{mouse_id}_y.npy"))
    snips = list(range(0, len(x), 900))[::4]

    for a, b in zip(snips[:-1], snips[1:]):
        plt.plot(y[a:b], x[a:b], alpha=1, color=color, linewidth=0.3)

    plt.ylim([0.45, 1])
    plt.xlim([0.05, 0.4])

    shelter_radius = 0.1
    sheltery = 0.23
    shelterx = 0.66

    if group == "lse_with_shelter":
        shelter = plt.Circle(
            (sheltery, shelterx),
            shelter_radius,
            facecolor=default_colors["shelter"],
            zorder=0,
            linewidth=2,
            edgecolor="none",
        )

        h_ax.add_patch(shelter)
        h_ax.text(
            sheltery,
            shelterx,
            "S",
            color=default_colors["shelter"],
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
    h_ax.set_aspect("equal")

    h_ax.get_xaxis().set_ticks([])
    h_ax.get_yaxis().set_ticks([])
    plt.axis("off")


def plot_fig_supp_2a(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    t, tracks, linestyle = track_plot_data(data.dataframe["lse_with_shelter"])

    plot_tracks_general(t, tracks, linestyle, fig=fig, axis=ax)

    ax.text(
        2.25,
        55,
        "LSE with access to shelter",
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="bottom",
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


def fig_supp_2b_data():

    group_ids = ["lse", "lse_with_shelter"]
    n_flees_to_each_loom = {k: np.zeros(5) for k in group_ids}
    n_freezes = {k: 0 for k in group_ids}
    total_n_trials = {k: [] for k in group_ids}

    for group_id in group_ids:

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


def plot_fig_supp_2b_and_c(
    fig=None, axis=None, escapes=True, group_ids=["lse_with_shelter", "lse"]
):

    ax, _ = create_panel_if_needed(fig, axis)

    n_flees_to_each_loom, n_freezes, total_n_trials = fig_supp_2b_data()

    freeze_offset = {"lse_with_shelter": 7, "lse": 8}
    no_response_offset = {"lse_with_shelter": 9.5, "lse": 10.5}
    face_color = {"lse_with_shelter": "none", "lse": default_colors["lse"]}
    edge_color = {
        "lse_with_shelter": default_colors["lse_with_shelter"],
        "lse": default_colors["lse"],
    }
    linestyle = {"lse_with_shelter": (0, (3, 1.6)), "lse": "solid"}

    if escapes:
        for group_id in group_ids:

            prc_flees_to_each_loom = (
                100 * n_flees_to_each_loom[group_id] / total_n_trials[group_id]
            )

            for ii in range(len(n_flees_to_each_loom[group_id])):
                plt.bar(
                    ii,
                    prc_flees_to_each_loom[ii],
                    facecolor=face_color[group_id],
                    edgecolor=edge_color[group_id],
                    linestyle=linestyle[group_id],
                    linewidth=0.5,
                    clip_on=False,
                )

        ax.spines["bottom"].set_color("none")
        ax.tick_params(axis="x", length=0)

        yl = [0, 10]
        plt.ylim(yl)
        plt.yticks(np.arange(0, 11, 2))
        plt.xlim([-1, 6])
        plt.xticks([])

        if group_id == "lse_with_shelter":
            ax.text(
                0.1,
                6,
                "Shelter present",
                color=default_colors[group_id],
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=8,
            )
            x_axis_spots(ax, yl)
        else:
            ax.text(
                0.1,
                6,
                "Shelter absent",
                color=default_colors[group_id],
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=8,
            )
            plt.ylabel("% Escape Trials", loc="top")

    else:
        for group_id in group_ids:
            n_no_responses = int(
                total_n_trials[group_id]
                - sum(n_flees_to_each_loom[group_id])
                - n_freezes[group_id]
            )
            prc_freezes = 100 * n_freezes[group_id] / total_n_trials[group_id]
            prc_no_response = 100 * float(n_no_responses) / total_n_trials[group_id]

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

            if group_id == "lse":
                ax.text(
                    no_response_offset[group_id],
                    prc_no_response - 30,
                    f"{n_no_responses}/{total_n_trials[group_id]}",
                    fontsize=8,
                    color="w",
                    rotation=90,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )
            else:
                ax.text(
                    no_response_offset[group_id] + 0.1,
                    prc_no_response - 3,
                    f"{n_no_responses}/{total_n_trials[group_id]}",
                    fontsize=8,
                    color="k",
                    rotation=90,
                    horizontalalignment="center",
                    verticalalignment="top",
                )

        ax.spines["bottom"].set_color("none")
        ax.tick_params(axis="x", length=0)

        plt.ylabel("Response Freq. (%)")
        yl = [0, 10]
        plt.ylim(yl)
        plt.yticks(np.arange(0, 101, 20))
        plt.xlim([6, 14])

        plt.xticks([7.5, 10], ["Freezing", "No reaction"], rotation=40, ha="right")


def main():

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.family"] = "sans-serif"

    h_fig = a4figure()
    offset_y = 45
    offset_x = 3
    # position of legend labels, 297 - X to be like Illustrator
    labels = {
        "a": [20 + offset_x, 297 - offset_y, 0, 0],
        "b": [20 + offset_x, 297 - 78 - offset_y, 0, 0],
        "c": [100 + offset_x, 297 - 78 - offset_y, 0, 0],
        "d": [140 + offset_x, 297 - 78 - offset_y, 0, 0],
    }

    axis_positions = {
        "a": [37 + offset_x, 297 - 117 - offset_y, 62, 32],
        "b_upper": [115 + offset_x, 297 - 142, 31, 14],
        "b_lower": [115 + offset_x, 297 - 163, 31, 14],
        "c_right": [155 + offset_x, 297 - 117 - offset_y, 31, 32],
        "1114188": [37 + offset_x, 297 - 81, 20, 50],
        "1114189": [60 + offset_x, 297 - 81, 20, 50],
        "1114307": [83 + offset_x, 297 - 81, 20, 50],
        "1114186": [106 + offset_x, 297 - 81, 20, 50],
        "1116285": [37 + offset_x, 297 - 116, 20, 50],
        "1116286": [60 + offset_x, 297 - 116, 20, 50],
        "1116287": [83 + offset_x, 297 - 116, 20, 50],
        "1116289": [106 + offset_x, 297 - 116, 20, 50],
    }

    # add the legend labels
    add_figure_labels(h_fig, labels)

    # create the axes
    axes_dict = add_axes(h_fig, axis_positions)

    # format the axes
    track_axes = ["a"]
    normal_axes = ["b_upper", "b_lower", "c_right"]
    for panel_id in track_axes:
        format_track_axis(axes_dict[panel_id])
    for panel_id in normal_axes:
        format_general_axis(axes_dict[panel_id])
    plot_fig_supp_2a(fig=h_fig, axis=axes_dict["a"])
    plot_fig_supp_2b_and_c(
        fig=h_fig,
        axis=axes_dict["b_upper"],
        escapes=True,
        group_ids=[
            "lse",
        ],
    )
    plot_fig_supp_2b_and_c(
        fig=h_fig,
        axis=axes_dict["b_lower"],
        escapes=True,
        group_ids=[
            "lse_with_shelter",
        ],
    )
    plot_fig_supp_2b_and_c(
        fig=h_fig,
        axis=axes_dict["c_right"],
        escapes=False,
        group_ids=["lse", "lse_with_shelter"],
    )

    for mid in mids_no_shelter:
        plot_example_exploration_track(
            mouse_id=mid, h_ax=axes_dict[f"{mid}"], group="lse"
        )
    for mid in mids_with_shelter:
        plot_example_exploration_track(
            mouse_id=mid, h_ax=axes_dict[f"{mid}"], group="lse_with_shelter"
        )

    h_fig.savefig(str(save_dir / "figure_2_supplement.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
