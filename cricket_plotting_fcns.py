import numpy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from load_cricket_data import body_tracks, \
    cricket_tracks, \
    reaches_shelter, \
    event_times, \
    event_times_flattened, \
    reaches_shelter_flattened, \
    track_type

from shared import default_colors, scatterball_size

sigma = 7
n_examples = {'naive': 6, 'lsie': 5}
example_body_tracks = {'naive': [], 'lsie': []}
example_cricket_tracks = {'naive': [], 'lsie': []}

# get example data tracks
for group_id in body_tracks:
    for i, body_track_group in enumerate(body_tracks[group_id]):

        cricket_track_group = cricket_tracks[group_id][i]

        idx = numpy.where(reaches_shelter[group_id][i])[0][0:n_examples[group_id]]

        these_body_examples = [body_track_group[x] for x in idx]
        example_body_tracks[group_id].extend(these_body_examples)

        these_cricket_examples = [cricket_track_group[x] for x in idx]
        example_cricket_tracks[group_id].extend(these_cricket_examples)


def plot_example_tracks(h_ax, group_name):

    col = default_colors['ih_ivc_7day'] if group_name == 'naive' else default_colors['lsie']
    string = 'Naive' if group_name == 'naive' else 'LSIE'

    plt.sca(h_ax)

    b_tracks = numpy.copy(example_body_tracks[group_name])
    c_tracks = numpy.copy(example_cricket_tracks[group_name])

    if track_type == 'transformed':
        for i, t in enumerate(b_tracks):
            # b_tracks[i][1] = b_tracks[i][1] * (1 - (30 / (240 - b_tracks[i][1])))
            # c_tracks[i][1] = c_tracks[i][1] * (1 - (30 / (240 - c_tracks[i][1])))
            b_tracks[i][1] = 0.4 * (1 - ((240*(1-b_tracks[i][1]/0.4))+30)/240)
            c_tracks[i][1] = 0.4 * (1 - ((240*(1-c_tracks[i][1]/0.4))+30)/240)

    shelter = plt.Circle((0.2, 0.1), 0.1, facecolor=default_colors['shelter'], zorder=0, linewidth=2, edgecolor=None)
    h_ax.add_patch(shelter)
    h_ax.text(0.2, 0.1, 'S', color='k', horizontalalignment='center', verticalalignment='center', fontsize=12)

    [h_ax.scatter(0.4 - x[1][0], x[0][0], s=scatterball_size(1.4),
                  facecolor='none', edgecolor=[0.5, 0.5, 0.5], linewidth=0.2)
     for x in c_tracks]
    [h_ax.scatter(0.4 - x[1][0], x[0][0], s=scatterball_size(0.06),
                  facecolor='none', edgecolor=[0.3, 0.3, 0.3], linewidth=0.2)
     for x in c_tracks]
    [h_ax.plot(gaussian_filter1d(0.4 - x[1], sigma), gaussian_filter1d(x[0], sigma), color=col, linewidth=0.75)
     for x in b_tracks]

    plt.xlim([0, 0.4])
    plt.ylim([0, 1])

    # h_ax.spines['left'].set_visible(False)
    # h_ax.spines['bottom'].set_visible(False)
    # h_ax.spines['right'].set_visible(False)
    # h_ax.spines['top'].set_visible(False)

    h_ax.get_xaxis().set_ticks([])
    h_ax.get_yaxis().set_ticks([])

    h_ax.text(0.2, 1.05, string, fontsize=10, color=col, horizontalalignment='center', verticalalignment='bottom')


def plot_events(h_ax, group_name):

    col = default_colors['ih_ivc_7day'] if group_name == 'naive' else default_colors['lsie']

    plt.sca(h_ax)

    for ii, events in enumerate(event_times[group_name]):
        for j, e in enumerate(events):
            if reaches_shelter[group_name][ii][j]:
                plt.plot([e, e], [ii-0.5, ii+0.5], color=col, linewidth=0.75, solid_capstyle='butt')

    h_ax.spines['left'].set_visible(False)
    h_ax.spines['bottom'].set_visible(False)
    h_ax.get_xaxis().set_ticks([])
    h_ax.get_yaxis().set_ticks([])

    plt.xlim([0, 60])

    if group_name == 'naive':
        h_ax.text(-8, 0, 'Mouse #', fontsize=10, color='k', horizontalalignment='right', verticalalignment='center',
                  rotation=90)
        plt.title('Retreats to shelter', fontsize=10, color='k')
    if group_name == 'lsie':
        plt.plot([45, 55], [-1, -1], color='k',  linewidth=1, clip_on=False, solid_capstyle='butt')
        h_ax.text(50, -2, '10 min', horizontalalignment='center', verticalalignment='top', fontsize=8)


def plot_histogram(h_ax, group_name, underlying_histogram='bouts_to_shelter'):

    plt.sca(h_ax)

    col = default_colors['ih_ivc_7day'] if group_name == 'naive' else default_colors['lsie']

    all_events = event_times_flattened[group_name]
    all_reaches_shelter = reaches_shelter_flattened[group_name]
    events_reaching_shelter = all_events[all_reaches_shelter]

    count_reaching_shelter, bin_edges = numpy.histogram(events_reaching_shelter, bins=int(60 / 4), range=(0, 60))
    count_all = numpy.histogram(all_events, bins=bin_edges)[0]

    if underlying_histogram == 'bouts_to_shelter':
        plt.hist(bin_edges[:-1], bin_edges,
                 weights=count_reaching_shelter/count_reaching_shelter[0],
                 histtype='step',
                 color=col,
                 linewidth=1,
                 clip_on=False)
    else:
        plt.hist(bin_edges[:-1], bin_edges,
                 weights=count_all / max(count_all),
                 histtype='step',
                 color=col,
                 linewidth=1,
                 clip_on=False)

    plt.xlim([0, 60])
    plt.ylim([0, 1])

    if group_name == 'lsie':
        h_ax.text(-8, 0.6, '# retreats\n(scaled)', fontsize=10, color='k',
                  horizontalalignment='right', verticalalignment='center', rotation=90)

    h_ax.spines['left'].set_visible(False)
    h_ax.spines['bottom'].set_visible(False)
    h_ax.get_xaxis().set_ticks([])
    h_ax.get_yaxis().set_ticks([])
    h_ax.set_facecolor('none')


def plot_lines(h_ax, group_name):

    plt.sca(h_ax)

    col = default_colors['ih_ivc_7day'] if group_name == 'naive' else default_colors['lsie']

    all_events = event_times_flattened[group_name]
    all_reaches_shelter = reaches_shelter_flattened[group_name]
    events_reaching_shelter = all_events[all_reaches_shelter]

    count_reaching_shelter, bin_edges = numpy.histogram(events_reaching_shelter, bins=int(60 / 4), range=(0, 60))
    count_all = numpy.histogram(all_events, bins=bin_edges)[0]

    prc_to_shelter = 100*count_reaching_shelter/count_all
    mid_bins = (bin_edges[:-1] + bin_edges[1:])/2

    plt.plot(mid_bins, prc_to_shelter, color=col, linewidth=1)

    plt.yticks([0, 30, 60])
    plt.xlim([0, 60])
    plt.ylim([0, 60])

    h_ax.spines['bottom'].set_visible(False)
    h_ax.get_xaxis().set_ticks([])

    if group_name == 'naive':
        h_ax.text(-8, 25, '# retreats/\n# bouts (%)', fontsize=10, color='k',
                  horizontalalignment='right', verticalalignment='center', rotation=90)

    if group_name == 'naive':
        h_ax.text(50, 60, 'Naive',
                  color=default_colors['ih_ivc_7day'],
                  fontsize=7,
                  horizontalalignment='right',
                  verticalalignment='top')
    else:
        h_ax.text(50, 40, 'LSIE',
                  color=default_colors['lsie'],
                  fontsize=7,
                  horizontalalignment='right',
                  verticalalignment='bottom')
