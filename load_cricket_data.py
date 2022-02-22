import os
import pandas as pd
import numpy as np
import more_itertools as mit
import skimage

from looming_spots.constants import DF_PATH, PROCESSED_DATA_DIRECTORY, TRANSFORMED_DF_PATH

HOUSE_START = 0.3  # 0.3
DISTANCE_THRESHOLD = 0.2
MINIMUM_BOUT_LENGTH = 10
INTER_BOUT_INTERVAL = 50
N_FRAMES_TO_SHOW = 100

track_type = 'transformed'  # raw or transformed

ids = {'naive': ['1114171_20210410_12_52_20',
                 '1114174_20210408_11_25_40',
                 '1114176_20210416_13_18_24',
                 '1114175_20210421_15_34_16',
                 '1114303_20210429_16_04_42'],
       'lsie': ['1114170_20210408_15_49_13',
                '1114177_20210421_11_48_27',
                '1114178_20210417_14_41_09',
                '1114179_20210415_13_08_19',
                '1114309_20210423_14_28_47',
                '1114302_e20210429_10_41_59']}

keys = ['naive', 'lsie']

paths = {'naive': [os.path.join(DF_PATH, x + '.h5') for x in ids['naive']],
         'lsie': [os.path.join(DF_PATH, x + '.h5') for x in ids['lsie']]}

body_tracks = dict.fromkeys(keys)
cricket_tracks = dict.fromkeys(keys)
reaches_shelter = dict.fromkeys(keys)
event_times = dict.fromkeys(keys)

reaches_shelter_flattened = dict.fromkeys(keys)
event_times_flattened = dict.fromkeys(keys)


def transform_raw_tracks(mid, x_in, y_in):

    f_name = os.path.join(PROCESSED_DATA_DIRECTORY, mid[0:7], mid[8:], 'box_corner_coordinates.npy')
    napari_fmt_coords = np.load(f_name)
    napari_fmt_coords = np.roll(napari_fmt_coords, 1, axis=1)
    new_box_coords = np.empty_like(napari_fmt_coords)
    new_box_coords[0] = napari_fmt_coords[1]
    new_box_coords[1] = napari_fmt_coords[0]
    new_box_coords[2] = napari_fmt_coords[2]
    new_box_coords[3] = napari_fmt_coords[3]
    p = skimage.transform.ProjectiveTransform()
    p.estimate(np.array([[0, 240], [0, 0], [600, 240], [600, 0]]), new_box_coords)
    new_track_x = []
    new_track_y = []
    for x, y in zip(x_in, y_in):
        inverse_mapped = p.inverse([x, y])[0]
        new_track_x.append(inverse_mapped[0])
        new_track_y.append(inverse_mapped[1])
    return np.array(new_track_x), np.array(new_track_y)


for group_id in paths:

    body_tracks_group = []
    cricket_tracks_group = []
    event_times_group = []
    reaches_shelter_group = []

    for i, path in enumerate(paths[group_id]):

        print(f'Loading {ids[group_id][i]}')
        df = pd.read_hdf(path)

        if track_type == 'raw':

            HOUSE_START = 0.3
            x_body = df['body']['x']
            y_body = df['body']['y']
            x_cricket = df['cricket']['x']
            y_cricket = df['cricket']['y']

        else:

            HOUSE_START = 0.2
            dlc_padding = 40
            if not os.path.isdir(TRANSFORMED_DF_PATH):
                x_body, y_body = transform_raw_tracks(ids[group_id][i], df['body']['x'] + dlc_padding, df['body']['y']+ dlc_padding)
                x_cricket, y_cricket = transform_raw_tracks(ids[group_id][i], df['cricket']['x']+ dlc_padding, df['cricket']['y']+ dlc_padding)
            else:
                x_body = np.load(os.path.join(TRANSFORMED_DF_PATH, ids[group_id][i], 'x_body.npy'))
                y_body = np.load(os.path.join(TRANSFORMED_DF_PATH, ids[group_id][i], 'y_body.npy'))
                x_cricket = np.load(os.path.join(TRANSFORMED_DF_PATH, ids[group_id][i], 'x_cricket.npy'))
                y_cricket = np.load(os.path.join(TRANSFORMED_DF_PATH, ids[group_id][i], 'y_cricket.npy'))

        body = np.array([1 - (x_body / 600.0), (1 - (y_body / 240.0)) * 0.4])
        cricket = np.array([1 - (x_cricket / 600.0), (1 - (y_cricket / 240.0)) * 0.4])

        distance = np.sqrt(np.sum((body - cricket)**2, axis=0))

        mouse_within_range_of_cricket = distance < DISTANCE_THRESHOLD
        encounters = np.where(mouse_within_range_of_cricket)[0]

        bouts = [list(group) for group in mit.consecutive_groups(list(list(encounters)))]
        bouts = [x for x in bouts if len(x) > MINIMUM_BOUT_LENGTH]

        bout_distances = [distance[x[0]:x[-1]] for x in bouts]
        minimum_pos = [np.argmin(x) for x in bout_distances]
        closest_pos_idx = [x[y] for x, y in zip(bouts, minimum_pos)]

        y = 0
        bout_idx = []
        for x in closest_pos_idx:
            if x - y > INTER_BOUT_INTERVAL:
                bout_idx.append(x)
            y = x

        these_body_tracks = [body[:, start:start + N_FRAMES_TO_SHOW] for start in bout_idx]
        these_cricket_tracks = [cricket[:, start:start + N_FRAMES_TO_SHOW] for start in bout_idx]

        reaches_shelter_group.append([])
        for track in these_body_tracks:
            if np.any(track[0] < HOUSE_START):
                reaches_shelter_group[-1].append(True)
            else:
                reaches_shelter_group[-1].append(False)

        if track_type == 'raw':
            # re-transform tracks
            for ii, t in enumerate(these_body_tracks):
                back_to_raw = [(1 - t[0]) * 600, (1 - t[1]/0.4) * 240]
                re_transformed = np.array([(570 - back_to_raw[0]) / 557, 0.4 * (194 - back_to_raw[1]) / (194 - 35)])
                these_body_tracks[ii] = re_transformed

        body_tracks_group.append(these_body_tracks)
        cricket_tracks_group.append(these_cricket_tracks)

        event_times_group.append(np.array(bout_idx) / 50 / 60)

    body_tracks[group_id] = body_tracks_group
    cricket_tracks[group_id] = cricket_tracks_group
    reaches_shelter[group_id] = reaches_shelter_group
    event_times[group_id] = event_times_group

    reaches_shelter_flattened[group_id] = np.array([y for x in reaches_shelter[group_id] for y in x])
    event_times_flattened[group_id] = np.array([y for x in event_times[group_id] for y in x])
