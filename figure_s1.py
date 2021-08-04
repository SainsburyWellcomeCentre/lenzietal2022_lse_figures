import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import ranksums

from shared import a4figure, \
    mmpos2normpos, \
    scatterball_size, \
    format_general_axis, \
    save_dir, \
    default_colors, \
    create_panel_if_needed, robustness_of_mouse

import load_track_data as data


day2_isolated_ctrl = ['CA159_1', 'CA159_4', 'CA160_1', 'CA301_1', 'CA301_2', 'CA301_3', '1114085', '1114086', '1114087',
                      '1114088', '1114089', '1114290', '1114291', '1114292', '1114293', '1114294']
day3_isolated_ctrl = ['CA160_2', 'CA160_5', 'CA112_5', 'CA113_1', 'CA131_1', 'CA131_2', 'CA131_4', 'CA131_5', 'CA132_1',
                      'CA132_2', 'CA132_3', 'CA132_4', 'CA473_1']
day4_isolated_ctrl = ['CA475_2', 'CA482_1', 'CA482_2', 'CA482_3', 'CA482_4']
day5_isolated_ctrl = ['CA475_4', 'CA476_1']
day6_isolated_ctrl = ['CA188_4', 'CA113_2', 'CA113_3', 'CA113_4', 'CA473_2', 'CA473_3', 'CA473_5']
day7_isolated_ctrl = ['CA475_5']

mice_by_day = {2: day2_isolated_ctrl,
               3: day3_isolated_ctrl,
               4: day4_isolated_ctrl,
               5: day5_isolated_ctrl,
               6: day6_isolated_ctrl,
               7: day7_isolated_ctrl}


def fig_s1a_data():

    n_flees = []
    total_n_trials = []

    for key, val in mice_by_day.items():
        idx = [np.any(a in val) for a in data.dataframe['ih_ivc_7day']['mouse_id']]
        flees = data.dataframe['ih_ivc_7day']['is_flee'][idx]
        n_flees.append(sum(flees))
        total_n_trials.append(len(flees))

    return n_flees, total_n_trials


def fig_s1b_data():

    group_ids = ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth', 'ih_ivc_7day']
    dfs = [data.dataframe[x] for x in group_ids]

    speed_lists = []
    for df in dfs:
        flees = np.array(df['is_flee'])
        speed_lists.append(np.array(df['peak_speed'])[flees])

    # stats
    p_val = np.zeros(shape=(len(dfs), len(dfs)))
    for i in range(len(dfs)):
        for j in range(i + 1, len(dfs)):
            _, p_val[i, j] = ranksums(speed_lists[i], speed_lists[j])

    return speed_lists, p_val


def fig_s1c_data():

    group_ids = ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth', 'ih_ivc_7day']
    dfs = [data.dataframe[x] for x in group_ids]

    robustness_lists = []
    for df in dfs:
        mouse_ids = list(set(df['mouse_id']))
        robustness_lists.append([robustness_of_mouse(df, x) for x in mouse_ids])

    # stats
    p_val = np.zeros(shape=(len(dfs), len(dfs)))
    for i in range(len(dfs)):
        for j in range(i + 1, len(dfs)):
            _, p_val[i, j] = ranksums(robustness_lists[i], robustness_lists[j])

    return robustness_lists, p_val


def plot_fig_s1a(n_flees, total_n_trials, fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    percentages = [100 * float(x) / y for x, y in zip(n_flees, total_n_trials)]

    x = np.arange(2, 8)
    plt.bar(x, percentages, width=0.8, facecolor='k')

    for i in range(len(percentages)):
        ax.text(2 + i, percentages[i]-3, f'{n_flees[i]}/{total_n_trials[i]}', horizontalalignment='center',
                verticalalignment='top', color='w', fontsize=8, rotation=90)

    plt.ylabel('Escape (%)')
    plt.xticks(x)
    plt.xlabel('Days Isolated')


def plot_fig_s1b_and_s1c(lists, p_val, axis_label, fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    label_str = ['GH enriched', 'IH enriched', 'IH IVC (1 mth)', 'IH IVC (<= 7 days)']
    colors = [default_colors[x] for x in ['gh_enriched', 'ih_enriched', 'ih_ivc_1mth', 'ih_ivc_7day']]

    for i, y in enumerate(lists):
        x_jitter = i - 0.2 + 0.4 * np.random.rand(len(y))
        plt.scatter(x_jitter, y, s=scatterball_size(1), facecolor='none', edgecolor=colors[i],
                    alpha=0.6, clip_on=False)
        plt.plot([i - 0.3, i + 0.3], [np.median(y), np.median(y)], color='k', linewidth=1,
                 solid_capstyle='butt', clip_on=False)

    ax.spines['bottom'].set_visible(False)
    plt.xlim([-0.5, len(lists)-0.5])
    plt.xticks(range(len(lists)), label_str, rotation=40, ha='right')

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


def main():

    h_fig = a4figure()

    labels = {'a': [15, 297 - 130, 0, 0],
              'b': [98, 297 - 130, 0, 0],
              'c': [145, 297 - 130, 0, 0]}

    axis_positions = {'a': [29, 297 - 166, 62, 32],
                      'b': [112, 297 - 166, 31, 32],
                      'c': [163, 297 - 166, 31, 32]}

    # add the legend labels
    for letter in labels:
        pos = mmpos2normpos(labels[letter])
        h_fig.text(pos[0], pos[1], letter, color='k', horizontalalignment='left', verticalalignment='bottom',
                   fontsize=18, fontweight='bold')

    # create the axes
    axes_dict = dict.fromkeys(axis_positions)
    for panel_id in axis_positions:
        ax_pos = mmpos2normpos(axis_positions[panel_id])
        axes_dict[panel_id] = h_fig.add_axes(ax_pos)

    # format the axes
    for panel_id in axis_positions:
        format_general_axis(axes_dict[panel_id])

    n_flees, total_n_trials = fig_s1a_data()
    plot_fig_s1a(n_flees, total_n_trials, fig=h_fig, axis=axes_dict['a'])

    speed_lists, stats_s1b = fig_s1b_data()
    robustness_lists, stats_s1c = fig_s1c_data()

    plot_fig_s1b_and_s1c(speed_lists, stats_s1b, 'speed', fig=h_fig, axis=axes_dict['b'])
    plot_fig_s1b_and_s1c(robustness_lists, stats_s1c, 'robustness', fig=h_fig, axis=axes_dict['c'])

    h_fig.savefig(str(save_dir / "supplementary_figure_1.pdf"))
    plt.show()


if __name__ == '__main__':
    main()
