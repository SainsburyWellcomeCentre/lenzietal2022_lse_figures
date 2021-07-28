import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact, ranksums, iqr

from looming_spots.db import loom_trial_group

from shared import a4figure, \
    create_panel_if_needed, \
    scatterball_size, \
    format_track_axis, \
    format_general_axis, \
    default_colors, \
    save_dir, \
    x_axis_spots, \
    robustness_of_mouse, \
    track_timebase, \
    timebase_to_show, \
    n_points, \
    track_plot_data, \
    plot_tracks_general, \
    ARENA_SIZE_CM, \
    nice_p_string, \
    add_figure_labels, \
    add_axes

import load_track_data as data


def fig_1c_data(mouse_id='CA281_1', trial_n=0):

    group = loom_trial_group.MouseLoomTrialGroup(mouse_id)
    track = group.pre_test_trials()[trial_n].normalised_x_track[0:n_points]

    t = track_timebase[timebase_to_show]
    track = ARENA_SIZE_CM * track[timebase_to_show]

    return t, track


def fig_1e_data():

    group_ids = ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth']

    percentage_lists = {k: [] for k in group_ids}

    for group_id in group_ids:

        dataframe = data.dataframe[group_id]
        total_n_trials = dataframe.shape[0]

        for i in range(1, 6):
            fraction_on_this_loom = sum(dataframe['last_loom'] == i)/total_n_trials
            percentage_lists[group_id].append(100 * fraction_on_this_loom)

        percentage_lists[group_id].append(100 * (total_n_trials - sum(dataframe['is_flee'])) / total_n_trials)

    return percentage_lists


def fig_1f_data():

    group_ids = ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth', 'ih_ivc_7day']
    n_groups = len(group_ids)

    n_flees = np.zeros(n_groups)
    total_n_trials = np.zeros(n_groups)
    for i in range(n_groups):
        dataframe = data.dataframe[group_ids[i]]
        n_flees[i] = sum(dataframe['is_flee'])
        total_n_trials[i] = dataframe.shape[0]

    # stats
    p_val = np.zeros(shape=(n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            o, p = fisher_exact([[n_flees[i], n_flees[j]],
                                 [total_n_trials[i] - n_flees[i], total_n_trials[j] - n_flees[j]]])
            p_val[i, j] = p

    return n_flees, total_n_trials, p_val


def fig_1g_data():

    group_ids = ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth', 'ih_ivc_7day']
    n_groups = len(group_ids)

    speed_lists = []
    for group_id in group_ids:
        dataframe = data.dataframe[group_id]
        flees = np.array(dataframe['is_flee'])
        speeds = np.array(dataframe['peak_speed'])
        speed_lists.append(speeds[flees])

    # stats
    p_val = np.zeros(shape=(n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            print(speed_lists[i])
            print(speed_lists[j])
            _, p_val[i, j] = ranksums(speed_lists[i], speed_lists[j])

    return speed_lists, p_val


def latency_data():

    group_ids = ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth', 'ih_ivc_7day']
    n_groups = len(group_ids)

    dfs = [data.dataframe[k] for k in group_ids]

    latency_lists = []
    for df in dfs:
        flees = np.array(df['classified as flee'])
        latency_lists.append(np.array(df['latency peak detect'])[flees])

    # stats
    p_val = np.zeros(shape=(n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            _, p_val[i, j] = ranksums(latency_lists[i], latency_lists[j])

    return latency_lists, p_val


def fig_1h_data():

    group_ids = ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth', 'ih_ivc_7day']
    n_groups = len(group_ids)

    robustness_lists = []
    for group_id in group_ids:
        dataframe = data.dataframe[group_id]
        mouse_ids = list(set(dataframe['mouse_id']))
        robustness_lists.append([robustness_of_mouse(dataframe, x) for x in mouse_ids])

    # stats
    p_val = np.zeros(shape=(n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            _, p_val[i, j] = ranksums(robustness_lists[i], robustness_lists[j])

    return robustness_lists, p_val


def fig_1j_data():

    percentage_list = []

    dataframe = data.dataframe['ih_ivc_7day']
    total_n_trials = dataframe.shape[0]

    for i in range(1, 6):
        fraction_on_this_loom = sum(dataframe['last_loom'] == i)/total_n_trials
        percentage_list.append(100 * fraction_on_this_loom)

    percentage_list.append(100 * (total_n_trials - sum(dataframe['is_flee'])) / total_n_trials)

    return percentage_list


def fig_1k_data():

    group_ids = ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth', 'ih_ivc_7day']

    n_looms = 5
    n_groups = len(group_ids)

    all_speeds = []
    avg_speed = []
    sem_speed = []

    for group_id in group_ids:

        dataframe = data.dataframe[group_id]

        flees = np.array(dataframe['is_flee'])
        speeds = np.array(dataframe['peak_speed'])
        last_loom = np.array(dataframe['last_loom'])

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

    p_val = np.ones(shape=(n_groups, n_looms))
    # compare all to last group ('ih_ivc_7day')
    for i in range(n_groups-1):
        for j in range(n_looms):
            if all_speeds[i][j] is not None and all_speeds[-1][j] is not None:
                if len(all_speeds[i][j]) > 1:
                    [_, p_val[i, j]] = ranksums(all_speeds[i][j], all_speeds[3][j])

    return avg_speed, sem_speed, p_val


def plot_fig_1c(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    t, track = fig_1c_data()

    scale_bar_x = [5.5, 6.5]
    scale_bar_y = 30

    track_labels_position = {'b1': [-0.8, 25, 'center'],
                             'b2': [0.2, 40, 'left'],
                             'b3': [0.6, 20, 'left'],
                             'b4': [1.5, 5, 'center']}

    plt.plot(t, track, color='k', linewidth=0.8)

    # scale bar
    plt.plot(scale_bar_x, 2 * [scale_bar_y], 'k', linewidth=1, clip_on=False, solid_capstyle='butt')
    ax.text(np.mean(scale_bar_x), scale_bar_y - 1.5, '1s',
            horizontalalignment='center', verticalalignment='top', fontsize=10)

    # manually place points along curve
    for key, val in track_labels_position.items():
        ax.text(val[0], val[1], key, color='k', horizontalalignment=val[2], verticalalignment='bottom', fontsize=10)


def plot_fig_1d_upper(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    t, tracks, linestyle = track_plot_data(data.dataframe['gh_enriched'])

    plot_tracks_general(t, tracks, linestyle, fig=fig, axis=ax)

    ax.text(2.25, 55, 'Group housed, enriched pen',
            fontsize=10, horizontalalignment='center', verticalalignment='bottom')

    # text indicating which line is which
    ax.text(1.5, -5, 'Escape', color=default_colors['flee'], horizontalalignment='left',
                     verticalalignment='center', fontsize=8)
    ax.text(6.5, 50, 'No reaction', color=default_colors['non_flee'], horizontalalignment='right',
                     verticalalignment='bottom', fontsize=8)


def plot_fig_1d_middle(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    t, tracks, linestyle = track_plot_data(data.dataframe['ih_enriched'])

    plot_tracks_general(t, tracks, linestyle, fig=fig, axis=ax)

    ax.text(2.25, 55, 'Individually housed, enriched pen',
            fontsize=10, horizontalalignment='center', verticalalignment='bottom')


def plot_fig_1d_lower(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    t, tracks, linestyle = track_plot_data(data.dataframe['ih_ivc_1mth'])

    scale_bar_x = [5.5, 6.5]
    scale_bar_y = 25

    plot_tracks_general(t, tracks, linestyle, fig=fig, axis=ax)

    ax.text(2.25, 55, 'Individually housed, IVC (1 month)',
            fontsize=10, horizontalalignment='center', verticalalignment='bottom')

    # scale bar
    plt.plot(scale_bar_x, 2 * [scale_bar_y], 'k', linewidth=1, clip_on=False, solid_capstyle='butt')
    ax.text(np.mean(scale_bar_x), scale_bar_y - 1.5, '1s',
            horizontalalignment='center', verticalalignment='top', fontsize=10)


def plot_fig_1e(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    percentage_list = fig_1e_data()

    x_limits = [-0.5, 7.5]
    y_limits = [0, 100]
    y_tick_spacing = 20

    for i, group_id in enumerate(percentage_list):
        x = np.arange(len(percentage_list[group_id][:-1])) - 0.2 + i * 0.2
        plt.bar(x, percentage_list[group_id][:-1], width=0.2, facecolor=default_colors[group_id], edgecolor='none')
        plt.bar(len(percentage_list[group_id]) - 0.2 + i * 0.2, percentage_list[group_id][-1],
                width=0.2, facecolor=default_colors[group_id], edgecolor='none')

    ax.spines['bottom'].set_visible(False)
    plt.ylabel('% Escape Trials')

    plt.ylim(y_limits)
    plt.yticks(np.arange(y_limits[0], y_limits[1] + 1, y_tick_spacing))
    plt.xlim(x_limits)

    y_offset = x_axis_spots(ax, y_limits)

    ax.text(6, y_offset, 'No\nescape',
            horizontalalignment='center', verticalalignment='top', fontsize=8)

    # legend
    plt.fill([1, 1.7, 1.7, 1], [70, 70, 74, 74], color=default_colors['ih_ivc_1mth'])
    plt.fill([1, 1.7, 1.7, 1], [80, 80, 84, 84], color=default_colors['ih_enriched'])
    plt.fill([1, 1.7, 1.7, 1], [90, 90, 94, 94], color=default_colors['gh_enriched'])
    ax.text(2, 72, 'IH IVC (1 mth)', color=default_colors['ih_ivc_1mth'], fontsize=7,
            horizontalalignment='left', verticalalignment='center')
    ax.text(2, 82, 'IH enriched', color=default_colors['ih_enriched'], fontsize=7,
            horizontalalignment='left', verticalalignment='center')
    ax.text(2, 92, 'GH enriched', color=default_colors['gh_enriched'], fontsize=7,
            horizontalalignment='left', verticalalignment='center')


def plot_fig_1f(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    n_flees, total_n_trials, p_val = fig_1f_data()

    colors = [default_colors[x] for x in ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth']]
    label_str = ['GH enriched', 'IH enriched', 'IH IVC']

    n_flees = n_flees[:-1]
    total_n_trials = total_n_trials[:-1]

    percentages = 100 * n_flees / total_n_trials

    for i in range(3):
        plt.bar(i, percentages[i], color=colors[i])

    for i in range(3):
        ax.text(i, percentages[i] - 3, f'{n_flees[i]:.0f}/{total_n_trials[i]:.0f}',
                horizontalalignment='center', verticalalignment='top', color='k', fontsize=8, rotation=90)

    ax.spines['bottom'].set_visible(False)

    plt.ylabel('Escape Probability (%)')
    plt.ylim([0, 100])
    plt.yticks(np.arange(0, 101, 20))
    plt.xlim([-0.5, 2.5])
    plt.xticks(range(3), label_str, rotation=40, ha='right')

    # print stats
    for i in range(len(percentages)):
        for j in range(i + 1, len(percentages)):
            if p_val[i, j] < 0.05:
                plt.plot([i + 0.05, j - 0.05], 2 * [100 + (j - i) * 15],
                         color='k', clip_on=False, linewidth=1, solid_capstyle='butt')
                if p_val[i, j] > 0.01:
                    p_str = '*'
                elif p_val[i, j] > 0.001:
                    p_str = '**'
                else:
                    p_str = '***'
                ax.text((i + j) / 2, 100 + (j - i) * 15, p_str, color='k', horizontalalignment='center',
                        verticalalignment='bottom', fontsize=8)


def plot_fig_1g_and_h(axis_label, fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    if axis_label == 'speed':
        lists, p_val = fig_1g_data()
    elif axis_label == 'robustness':
        lists, p_val = fig_1h_data()
    else:
        raise ValueError("axis_label must be 'speed' or 'robustness'")

    colors = [default_colors[x] for x in ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth']]
    label_str = ['GH enriched', 'IH enriched', 'IH IVC']

    lists = lists[:-1]

    for i, y in enumerate(lists):
        x_jitter = i - 0.2 + 0.4 * np.random.rand(len(y))
        plt.scatter(x_jitter, y, s=scatterball_size(1), facecolor='none', edgecolor=colors[i], alpha=0.6)
        plt.plot([i - 0.3, i + 0.3], [np.median(y), np.median(y)], color='k', linewidth=1, solid_capstyle='butt')

    ax.spines['bottom'].set_visible(False)
    plt.xlim([-0.5, len(lists)-0.5])
    plt.xticks(range(3), label_str, rotation=40, ha='right')

    if axis_label == 'speed':
        plt.ylabel('Escape Vigour\n(peak speed, cm/s)')
        plt.ylim([0, 120])
        plt.yticks(np.arange(0, 121, 30))
        start = 100
        space = 15
    elif axis_label == 'robustness':
        plt.ylabel('Escape Robustness (a.u.)')
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
                plt.plot([i + 0.05, j - 0.05], 2 * [start + (j - i) * space],
                         color='k', clip_on=False, linewidth=1, solid_capstyle='butt')
                if p_val[i, j] > 0.01:
                    p_str = '*'
                elif p_val[i, j] > 0.001:
                    p_str = '**'
                else:
                    p_str = '***'
                ax.text((i + j) / 2, start + (j - i) * space, p_str, color='k', horizontalalignment='center',
                        verticalalignment='bottom', fontsize=8)


def plot_fig_1i(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    t, tracks, linestyle = track_plot_data(data.dataframe['ih_ivc_7day'])

    plot_tracks_general(t, tracks, linestyle, fig=fig, axis=ax)

    ax.text(2.25, 55, 'Individually housed, IVC (up to 7 days)',
            fontsize=10, horizontalalignment='center', verticalalignment='bottom')

    plt.plot([5.5, 6.5], [-5, -5], 'k', linewidth=1, clip_on=False, solid_capstyle='butt')
    ax.text(6, -5 - 1.5, '1s', horizontalalignment='center', verticalalignment='top', fontsize=10)

    # text indicating which line is which
    ax.text(1.5, -5, 'Escape', color=default_colors['flee'],
            horizontalalignment='left', verticalalignment='center', fontsize=8)
    ax.text(6.5, 50, 'No reaction', color=default_colors['non_flee'],
            horizontalalignment='right', verticalalignment='bottom', fontsize=8)
    ax.text(6.5, 28, 'Freeze', color=default_colors['flee'],
            horizontalalignment='right', verticalalignment='top', fontsize=8)


def plot_fig_1j(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    percentage_list = fig_1j_data()

    x_limits = [-0.5, 7.5]
    y_limits = [0, 60]
    y_tick_space = 20

    for j in range(len(percentage_list[:-1])):
        if percentage_list[j] != 0:
            plt.bar(j, percentage_list[j],
                    width=0.8, facecolor='none', edgecolor=default_colors['ih_ivc_7day'], clip_on=False)
    plt.bar(len(percentage_list), percentage_list[-1],
            width=0.8, facecolor='none', edgecolor=default_colors['ih_ivc_7day'], clip_on=False)

    ax.spines['bottom'].set_visible(False)
    plt.ylabel('% Escape Trials')
    plt.ylim(y_limits)
    plt.yticks(np.arange(y_limits[0], y_limits[1] + 1, y_tick_space))
    plt.xlim(x_limits)

    y_offset = x_axis_spots(ax, y_limits)
    ax.text(len(percentage_list), y_offset, 'No\nescape',
            horizontalalignment='center', verticalalignment='top', fontsize=8)


def plot_fig_1k(fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    avg_speed, sem_speed, p_val = fig_1k_data()

    x_limits = [-0.5, 4.5]
    y_limits = [20, 80]
    y_tick_space = 10

    colors = [default_colors[x] for x in ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth', 'ih_ivc_7day']]

    for i, s in enumerate(avg_speed):
        plt.plot(range(5), s, color=colors[i], linewidth=0.5, zorder=-1)
        for j, a in enumerate(s):
            if a is not None:
                plt.plot([j, j], [s[j] - sem_speed[i][j], s[j] + sem_speed[i][j]],
                         color=colors[i], linewidth=0.5, zorder=-1)
        if i == 3:
            plt.scatter(range(5), s, s=scatterball_size(1.8),
                        facecolors='w', edgecolors=colors[i], linewidths=0.5, zorder=1)
        else:
            plt.scatter(range(5), s, s=scatterball_size(1.2), facecolors=colors[i], linewidths=0.5)

    for ii in range(3):
        for jj in range(5):

            if avg_speed[ii][jj] is None:
                continue

            if p_val[ii, jj] < 0.05:
                if p_val[ii, jj] > 0.01:
                    p_str = '*'
                elif p_val[ii, jj] > 0.001:
                    p_str = '**'
                else:
                    p_str = '***'
            else:
                p_str = ''

            ax.text(jj, avg_speed[ii][jj] - 5, p_str, color='k', horizontalalignment='center',
                    verticalalignment='top', fontsize=8)

    ax.spines['bottom'].set_visible(False)
    plt.ylabel('Avg. Peak Escape\nSpeed (cm/s)', color='k')

    plt.ylim(y_limits)
    plt.yticks(np.arange(y_limits[0], y_limits[1] + 1, y_tick_space))
    plt.xlim(x_limits)

    x_axis_spots(ax, y_limits)


def print_stats_fig_1f(n_flees, total_n_trials, stats):

    label_str = ['GH enriched', 'IH enriched', 'IH IVC 1mth', 'IH IVC 7 day']

    for i in range(len(label_str)):
        for j in range(i+1, len(label_str)):
            p_str = nice_p_string(stats[i, j])
            print(f'{label_str[i]} ({n_flees[i]:.0f}/{total_n_trials[i]:.0f}) vs {label_str[j]} '
                  f'({n_flees[j]:.0f}/{total_n_trials[j]:.0f}), p={p_str}, Fisher\'s exact test')


def print_stats_fig_1g_and_h(lists, stats, axis_label):

    label_str = ['GH enriched', 'IH enriched', 'IH IVC 1mth', 'IH IVC 7 day']

    if axis_label == 'speed':
        print('-----------------SPEED-----------------')
        uom = 'cm/s'
        n_str = 'escapes'
    elif axis_label == 'robustness':
        print('--------------ROBUSTNESS---------------')
        uom = 'a.u.'
        n_str = 'mice'
    elif axis_label == 'latency':
        print('----------------LATENCY-----------------')
        uom = 's'
        n_str = 'escapes'
    else:
        return

    for i in range(len(label_str)):
        for j in range(i+1, len(label_str)):

            n_1 = len(lists[i])
            n_2 = len(lists[j])

            median_1 = np.median(np.array(lists[i]))
            median_2 = np.median(np.array(lists[j]))

            iqr_1 = iqr(np.array(lists[i]))
            iqr_2 = iqr(np.array(lists[j]))

            p_str = nice_p_string(stats[i, j])

            print(f'{label_str[i]} (median={median_1:.2f}{uom}, iqr={iqr_1:.2f}{uom}, n={n_1} {n_str}]) '
                  f'vs {label_str[j]} (median={median_2:.2f}{uom}, iqr={iqr_2:.2f}{uom}, n={n_2} {n_str}), '
                  f'p={p_str}, ranksum test')


def main():

    mpl.rcParams['pdf.fonttype'] = 42  # save text elements as text and not shapes
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"

    h_fig = a4figure()

    # position of legend labels, 297 - X to be like Illustrator
    labels = {'a': [20, 297 - 30, 0, 0],
              'b': [50, 297 - 52, 0, 0],
              'c': [20, 297 - 78, 0, 0],
              'd': [20, 297 - 126, 0, 0],
              'e': [106, 297 - 30, 0, 0],
              'f': [146, 297 - 30, 0, 0],
              'g': [108, 297 - 95, 0, 0],
              'h': [146, 297 - 95, 0, 0],
              'i': [108, 297 - 156, 0, 0],
              'j': [108, 297 - 206, 0, 0],
              'k': [146, 297 - 206, 0, 0]}

    axis_positions = {'c': [37, 297 - 112, 62, 32],
                      'd_upper': [37, 297 - 157, 62, 32],
                      'd_middle': [37, 297 - 207, 62, 32],
                      'd_lower': [37, 297 - 257, 62, 32],
                      'e': [115, 297 - 67, 34, 32],
                      'f': [162, 297 - 67, 23, 32],
                      'g': [117, 297 - 130, 23, 32],
                      'h': [164, 297 - 130, 23, 32],
                      'i': [117, 297 - 188, 62, 32],
                      'j': [117, 297 - 238, 34, 32],
                      'k': [164, 297 - 238, 23, 32]}

    # add the legend labels
    add_figure_labels(h_fig, labels)

    # create the axes
    axes_dict = add_axes(h_fig, axis_positions)

    # format the axes
    track_axes = ['c', 'd_upper', 'd_middle', 'd_lower', 'i']
    normal_axes = ['e', 'f', 'g', 'h', 'j', 'k']
    for panel_id in track_axes:
        format_track_axis(axes_dict[panel_id])
    for panel_id in normal_axes:
        format_general_axis(axes_dict[panel_id])

    # plot the data in the figure panels
    plot_fig_1c(fig=h_fig, axis=axes_dict['c'])
    plot_fig_1d_upper(fig=h_fig, axis=axes_dict['d_upper'])
    plot_fig_1d_middle(fig=h_fig, axis=axes_dict['d_middle'])
    plot_fig_1d_lower(fig=h_fig, axis=axes_dict['d_lower'])
    plot_fig_1e(fig=h_fig, axis=axes_dict['e'])
    plot_fig_1f(fig=h_fig, axis=axes_dict['f'])
    plot_fig_1g_and_h('speed', fig=h_fig, axis=axes_dict['g'])
    plot_fig_1g_and_h('robustness', fig=h_fig, axis=axes_dict['h'])
    plot_fig_1i(fig=h_fig, axis=axes_dict['i'])
    plot_fig_1j(fig=h_fig, axis=axes_dict['j'])
    plot_fig_1k(fig=h_fig, axis=axes_dict['k'])

    h_fig.savefig(f'{save_dir}\\figure_1.pdf')
    plt.show()


if __name__ == '__main__':
    main()
