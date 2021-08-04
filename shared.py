import os.path

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from scipy.ndimage import gaussian_filter

from looming_spots.constants import ARENA_SIZE_CM, \
    CLASSIFICATION_LATENCY, \
    FRAME_RATE, LOOM_ONSETS_S

from path_config import figure_path


date_str = datetime.now().strftime('%Y%m%d')
save_dir = os.path.join(figure_path, date_str)

mm_per_point = 0.352778
inches_per_mm = 0.0393701
a4_size = (8.27, 11.69)

default_dash_size = [5.6, 5.6]

shelter_size = 10

default_colors = {'shelter': [234/255.0, 228/255.0, 198/255.0],
                  'flee': [0/255.0, 0/255.0, 0/255.0],
                  'non_flee': [160/255.0, 160/255.0, 160/255.0],
                  'ih_ivc_7day': [37/255.0, 116/255.0, 49/255.0],
                  'lsie': [76/255.0, 36/255.0, 23/255.0],
                  'gh_enriched': [55.0/255.0, 126.0/255, 184.0/255.0],
                  'ih_enriched': [218.0/255.0, 141.0/255, 53.0/255.0],
                  'ih_ivc_1mth': [152.0/255.0, 78.0/255.0, 163.0/255.0],
                  'pre_test_none': [37/255.0, 116/255.0, 49/255.0],
                  'pre_test_24hr': [128/255.0, 176/255.0, 211/255.0],
                  'pre_test_immediate': [248/255.0, 179/255.0, 102/255.0]}

n_points = 600
track_display_limits = [-2, 6.5]
track_timebase = (np.arange(n_points) - 200) / 30
timebase_to_show = np.logical_and(track_timebase > track_display_limits[0], track_timebase < track_display_limits[1])
time_after_return_to_show = 0.5


def create_save_dir():
    if os.path.isdir(save_dir):
        os.mkdir(save_dir)


def add_figure_labels(h_fig, labels):
    for letter in labels:
        pos = mmpos2normpos(labels[letter])
        h_fig.text(pos[0], pos[1], letter, color='k', horizontalalignment='left', verticalalignment='bottom',
                   fontsize=18, fontweight='bold')


def add_axes(h_fig, axis_positions):
    axes_dict = dict.fromkeys(axis_positions)
    for panel_id in axis_positions:
        ax_pos = mmpos2normpos(axis_positions[panel_id])
        axes_dict[panel_id] = h_fig.add_axes(ax_pos)
    return axes_dict


def create_axis(axis_position=None, fig=None):

    # create figure
    if fig is None:
        fig = a4figure()
    else:
        plt.figure(fig.number)

    if axis_position is None:
        axis_position = [65, 118.5, 80, 60]

    ax_pos = mmpos2normpos(axis_position)
    ax = fig.add_axes(ax_pos)

    return fig, ax


def create_panel_if_needed(fig=None, axis=None):

    # create figure
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)

    if axis is None:
        ax = plt.axes()
    else:
        ax = axis
        plt.sca(ax)

    return ax, fig


def a4figure(ori='portrait'):

    if ori == 'portrait':
        h_fig = plt.figure(figsize=a4_size)
    else:
        h_fig = plt.figure(figsize=a4_size[::-1])

    return h_fig


def mm2inches(mm):
    return inches_per_mm * mm


def mmpos2normpos(pos):

    from_left_mm, from_bottom_mm, width_mm, height_mm = pos

    from_left_inches = mm2inches(from_left_mm)
    from_bottom_inches = mm2inches(from_bottom_mm)
    width_inches = mm2inches(width_mm)
    height_inches = mm2inches(height_mm)

    from_left_norm = from_left_inches / a4_size[0]
    from_bottom_norm = from_bottom_inches / a4_size[1]
    width_norm = width_inches / a4_size[0]
    height_norm = height_inches / a4_size[1]

    return [from_left_norm, from_bottom_norm, width_norm, height_norm]


def scatterball_size(dot_diameter_mm):
    return (dot_diameter_mm / mm_per_point) ** 2


def format_track_axis(ax):

    plt.sca(ax)

    format_general_axis(ax)

    # shelter location
    plt.fill_between(track_display_limits, [0, 0], 2 * [shelter_size],
                     facecolor=default_colors['shelter'], edgecolor=default_colors['shelter'])

    # loom onsets
    for loom in LOOM_ONSETS_S:
        plt.plot([loom, loom], [0, ARENA_SIZE_CM], 'k', dashes=default_dash_size, linewidth=0.5)

    plt.ylabel('Position along\narena (cm)')

    plt.xlim(track_display_limits)
    plt.ylim([0, ARENA_SIZE_CM])

    plt.xticks([])
    plt.yticks(range(0, ARENA_SIZE_CM+1, 10))

    ax.spines['bottom'].set_color('none')


def format_general_axis(ax):

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.grid(False)

    ax.yaxis.label.set_fontsize(10)
    ax.xaxis.label.set_fontsize(10)
    for item in ax.get_yticklabels() + ax.get_xticklabels():
        item.set_fontsize(8)


def x_axis_spots(ax, y_limits):

    plt.sca(ax)

    n_spots = 5

    plt.xticks([])

    y_offset = y_limits[0] - (y_limits[1] - y_limits[0]) / 6
    y_offset2 = y_limits[0] - 2 * (y_limits[1] - y_limits[0]) / 6

    [plt.scatter(x, y_offset, s=scatterball_size(3), facecolor='k', clip_on=False) for x in range(n_spots)]

    [ax.text(x, y_offset, str(x + 1), color='w', horizontalalignment='center', verticalalignment='center',
             fontsize=8, fontweight='bold') for x in range(n_spots)]

    ax.text((n_spots-1)/2, y_offset2, 'Last looming spot\npreceding escape',
            horizontalalignment='center', verticalalignment='top', fontsize=8)

    return y_offset2


def n_flees_by_mouse(df, mouse_id):
    n_flees = sum(df[df['mouse_id'] == mouse_id]['is_flee'])
    n_total = np.sum(df['mouse_id'] == mouse_id)

    return n_flees, n_total


def average_speed_of_mouse(df, mouse_id):

    is_flee = df[df['mouse_id'] == mouse_id]['is_flee']
    speed = df[df['mouse_id'] == mouse_id]['peak_speed']
    if sum(is_flee) == 0:
        avg_speed = 0
    else:
        avg_speed = np.mean(speed[is_flee])
    return avg_speed


def average_latency_of_mouse(df, mouse_id):

    is_flee = df[df['mouse_id'] == mouse_id]['is_flee']
    latency = df[df['mouse_id'] == mouse_id]['latency']
    if sum(is_flee) == 0:
        avg_latency = CLASSIFICATION_LATENCY
    else:
        avg_latency = np.mean(latency[is_flee])
    return avg_latency


def robustness_of_mouse(df, mouse_id):
    n_trials_per_mouse = 3
    fraction_flees = float(n_flees_by_mouse(df, mouse_id)[0]) / n_trials_per_mouse
    avg_speed = average_speed_of_mouse(df, mouse_id)
    avg_latency = average_latency_of_mouse(df, mouse_id)
    robustness = avg_speed * fraction_flees / avg_latency
    return robustness


def track_plot_data(dataframe):

    flees = np.array(dataframe['is_flee'])
    freeze = np.array(dataframe['is_freeze'])
    time_to_shelter = np.array(dataframe['time_to_shelter'])

    t = []
    tracks = []
    linestyle = []

    for i, track in enumerate(dataframe['track']):

        if not flees[i]:

            if freeze[i]:
                linestyle.append('freeze')
            else:
                linestyle.append('no reaction')

            t.append(track_timebase[timebase_to_show])
            tracks.append(track[timebase_to_show])

        else:

            linestyle.append('escape')

            idx = np.logical_and(track_timebase > track_display_limits[0],
                                 track_timebase < (time_to_shelter[i] + time_after_return_to_show))
            t.append(track_timebase[idx])
            tracks.append(track[idx])

    return t, tracks, linestyle


def plot_tracks_general(t, tracks, linestyle, fig=None, axis=None):

    ax, _ = create_panel_if_needed(fig, axis)

    for i, track in enumerate(tracks):
        if linestyle[i] == 'no reaction':
            plt.plot(t[i], track, color=default_colors['non_flee'], linewidth=0.8, alpha=0.5)
        elif linestyle[i] == 'freeze':
            plt.plot(t[i], track, color=default_colors['flee'], linewidth=0.8, alpha=0.5, dashes=default_dash_size)

    for i, track in enumerate(tracks):
        if linestyle[i] == 'escape':
            plt.plot(t[i], track, color=default_colors['flee'], linewidth=0.8, alpha=0.5)


def nice_p_string(p_val):

    if p_val > 0.0095:
        p_str = f'{p_val:.2f}'
    elif p_val > 0.00095:
        p_str = f'{p_val:.3f}'
    elif p_val > 0.000095:
        p_str = f'{p_val:.4f}'
    else:
        p_str = f'{p_val:.2e}'
    return p_str


def arrow_parameters(axis_width_mm, axis_width_units, line_width_pt, arrow_size_frac):

    # width and length of arrow in points (as in Illustrator)
    head_width_pt = 10 * line_width_pt * arrow_size_frac
    head_length_pt = 8.6 * line_width_pt * arrow_size_frac

    ax_width_pt = axis_width_mm * (1 / mm_per_point)
    hw_frac = head_width_pt / ax_width_pt
    hl_frac = head_length_pt / ax_width_pt
    lw_frac = line_width_pt / ax_width_pt

    hw_units = axis_width_units * hw_frac
    hl_units = axis_width_units * hl_frac
    lw_units = axis_width_units * lw_frac

    arrow_params = {'head_width': hw_units, 'head_length': hl_units, 'line_width': lw_units}

    return arrow_params
