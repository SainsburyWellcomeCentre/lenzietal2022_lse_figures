import math
import os
import pathlib
import pandas
import numpy

from looming_spots.db.constants import N_LOOMS_PER_STIMULUS

from scipy.signal import medfilt
from path_config import proc_path

import load_track_data as data

window_t = [-1.6, 6]
frame_rate = 100
dlc_padding = 40
position_threshold = 20

group_ids = ['naive', 'lsie']

mouse_ids = {'naive':  ['1114085',
                        '1114086',
                        '1114087',
                        '1114088',
                        '1114089'],
             'lsie': ['1114186',
                      '1114188',
                      '1114189',
                      '1114307',
                      '1114308']}

datetime_ids = {'naive': ['20210402_12_36_18',
                          '20210402_12_54_58',
                          '20210402_13_21_06',
                          '20210402_13_37_46',
                          '20210402_13_59_38'],
                'lsie': ['20210404_17_40_10',
                         '20210404_18_21_04',
                         '20210404_18_55_29',
                         '20210420_19_22_01',
                         '20210420_18_43_26']}

window_frames = [round(x * frame_rate) for x in window_t]
timebase = numpy.arange(window_frames[0], window_frames[1]+1) / frame_rate
zero_frame_idx = numpy.where(timebase == 0)[0][0]


def track_h5_f_name(_mouse_id, datetime_id):
    d_name = os.path.join(proc_path, _mouse_id, datetime_id, '5_label')
    path = pathlib.Path(d_name)
    f_name = str(list(path.glob('*filtered.h5'))[0])
    return f_name


def loom_starts_npy(_mouse_id, datetime_id):
    f_name = os.path.join(proc_path, _mouse_id, datetime_id, 'loom_starts.npy')
    return f_name


def get_loom_starts(_mouse_id, datetime_id):
    f_name = loom_starts_npy(_mouse_id, datetime_id)
    _loom_starts = numpy.load(f_name)
    return _loom_starts


def get_tracks(_mouse_id, datetime_id):
    f_name = track_h5_f_name(_mouse_id, datetime_id)
    df = pandas.read_hdf(f_name)
    df = df[df.keys()[0][0]]
    body_part_labels = ['L_ear', 'R_ear', 'body']
    _track = {}
    for part in body_part_labels:
        t2d = numpy.transpose(numpy.array([df[part]['x'], df[part]['y']])) + dlc_padding
        _track[part] = transform(t2d)

    return _track


def transform(this_track):

    p1 = numpy.array([71, 103])
    p2 = numpy.array([38, 256])
    p3 = numpy.array([560, 103])
    p4 = numpy.array([598, 256])

    x0 = this_track[:, 0]
    y0 = this_track[:, 1]

    xe = p2[0] + (p1[0] - p2[0]) * (y0 - p2[1]) / (p1[1] - p2[1])
    xs = p4[0] + (p3[0] - p4[0]) * (y0 - p2[1]) / (p1[1] - p2[1])

    val = numpy.transpose(numpy.array([50 * (xs - x0) / (xs - xe), 20 * (y0 - p1[1]) / (p2[1] - p1[1])]))

    return val


def unbound_angles(theta_deg):

    idx_sign_change = \
        numpy.concatenate((numpy.array([False]), (numpy.sign(theta_deg[1:]) != numpy.sign(theta_deg[0:-1]))))
    idx_positive_val = theta_deg > 130
    idx_negative_val = theta_deg < -130

    large_positive_sign_changes = numpy.where(idx_sign_change & idx_positive_val)[0]
    large_negative_sign_changes = numpy.where(idx_sign_change & idx_negative_val)[0]

    delta_angle = numpy.zeros(len(theta_deg))

    delta_angle[1:] = theta_deg[1:] - theta_deg[0:-1]
    delta_angle[large_positive_sign_changes] = -(180 - theta_deg[large_positive_sign_changes]) + \
                                               -(180 + theta_deg[large_positive_sign_changes-1])
    delta_angle[large_negative_sign_changes] = (180 + theta_deg[large_negative_sign_changes]) + \
                                               (180 - theta_deg[large_negative_sign_changes - 1])

    new_theta_deg = theta_deg[0] + numpy.cumsum(delta_angle)

    return new_theta_deg


def crop_non_escapes(tracks_naive):

    mask = (response_type['naive'] == 1)
    last_non_nan_idx = numpy.argmax(tracks_naive[:, mask], axis=0)
    last_non_nan_idx = numpy.max(last_non_nan_idx)
    tracks_naive[last_non_nan_idx:] = numpy.nan
    return tracks_naive


def match_lsie_trials(tracks_naive, tracks_lsie):

    assert tracks_naive.shape[1] == tracks_lsie.shape[1]

    # this is a random list from matlab (>> rng(1); >> randperm(15)) which implemented this code originally
    random_pairing = numpy.array([3, 15, 6, 5, 7, 13, 4, 8, 9, 1, 11, 10, 12, 2, 14]) - 1

    n_tracks_ = tracks_naive.shape[1]
    print('Matching each lsie track with one naive track')
    last_idx = numpy.argmax(tracks_naive, axis=0)

    for ii in range(n_tracks_):

        idx_ = random_pairing[ii]
        tracks_lsie[last_idx[idx_]:, ii] = numpy.nan

    return tracks_lsie


def get_head_angle(head_track, body_track):

    head_vector = head_track - body_track
    shelter_vector = numpy.array([-1, 0])

    norm_head_vector = numpy.sqrt(numpy.sum(head_vector ** 2, axis=1))
    norm_shelter_vector = numpy.linalg.norm(shelter_vector)

    shelter_dot_prod = numpy.dot(head_vector, shelter_vector)

    ha_no_sign = (180 / math.pi) * numpy.arccos(shelter_dot_prod / (norm_head_vector * norm_shelter_vector))

    ha_with_sign = numpy.copy(ha_no_sign)
    ha_with_sign[head_vector[:, 1] < 0] = -ha_no_sign[head_vector[:, 1] < 0]
    ha_with_sign = unbound_angles(ha_with_sign)

    return ha_no_sign, ha_with_sign


# first get 'normal' tracking data
response_type = {k: [] for k in group_ids}

for group_id in group_ids:
    for mouse_id in mouse_ids[group_id]:

        if group_id == 'naive':
            dataframe = data.dataframe['ih_ivc_7day']
        else:
            dataframe = data.dataframe['lsie']

        idx = dataframe['mouse_id'] == mouse_id

        for ii in range(dataframe[idx].shape[0]):

            flees = numpy.array(dataframe[idx]['is_flee'])
            freezes = numpy.array(dataframe[idx]['is_freeze'])

            if flees[ii]:
                response_type[group_id].append(1)
            elif freezes[ii]:  #
                response_type[group_id].append(2)
            else:
                response_type[group_id].append(0)

    response_type[group_id] = numpy.array(response_type[group_id])


# get relevant tracks
head_tracks = {k: [] for k in group_ids}
body_tracks = {k: [] for k in group_ids}

for group_id in group_ids:
    for i, mouse_id in enumerate(mouse_ids[group_id]):

        mouse_tracks = get_tracks(mouse_id, datetime_ids[group_id][i])
        loom_starts = get_loom_starts(mouse_id, datetime_ids[group_id][i])

        loom_starts = loom_starts[0::N_LOOMS_PER_STIMULUS]

        for s in loom_starts:

            frame_idx = s + numpy.arange(window_frames[0], window_frames[1]+1)
            valid_idx = (mouse_tracks['body'][frame_idx, 0] > position_threshold) | (timebase < 0)

            body = numpy.transpose(numpy.array([medfilt(mouse_tracks['body'][:, 0], 3),
                                                medfilt(mouse_tracks['body'][:, 1], 3)]))
            L_ear = numpy.transpose(numpy.array([medfilt(mouse_tracks['L_ear'][:, 0], 3),
                                                 medfilt(mouse_tracks['L_ear'][:, 1], 3)]))
            R_ear = numpy.transpose(numpy.array([medfilt(mouse_tracks['R_ear'][:, 0], 3),
                                                 medfilt(mouse_tracks['R_ear'][:, 1], 3)]))

            body = body[frame_idx, :]
            head = (L_ear[frame_idx, :] + R_ear[frame_idx, :]) / 2

            head[~valid_idx, :] = numpy.nan
            body[~valid_idx, :] = numpy.nan

            head_tracks[group_id].append(head)
            body_tracks[group_id].append(body)


head_angle_no_sign = {k: [] for k in group_ids}
head_angle_with_sign = {k: [] for k in group_ids}

start_positions = {k: [] for k in group_ids}
pre_loom_angle = {k: [] for k in group_ids}
post_loom_angle = {k: [] for k in group_ids}


for group_id in group_ids:

    n_frames = timebase.shape[0]
    n_tracks = len(body_tracks[group_id])

    head_angle_no_sign[group_id] = numpy.full((n_frames, n_tracks), numpy.nan)
    head_angle_with_sign[group_id] = numpy.full((n_frames, n_tracks), numpy.nan)

    for i, track in enumerate(body_tracks[group_id]):

        head = head_tracks[group_id][i]
        body = body_tracks[group_id][i]

        start_positions[group_id].append(head[zero_frame_idx, :])

        head_angle_no_sign[group_id][:, i], head_angle_with_sign[group_id][:, i] = get_head_angle(head, body)

    if group_id == 'naive':
        head_angle_no_sign['naive'] = crop_non_escapes(head_angle_no_sign['naive'])
        head_angle_with_sign['naive'] = crop_non_escapes(head_angle_with_sign['naive'])
    else:
        head_angle_no_sign['lsie'] = match_lsie_trials(head_angle_no_sign['naive'], head_angle_no_sign['lsie'])
        head_angle_with_sign['lsie'] = \
            match_lsie_trials(head_angle_with_sign['naive'], head_angle_with_sign['lsie'])

    for i, track in enumerate(body_tracks[group_id]):

        # average angle over 20 frames before loom onset
        pre_loom_angle[group_id].append(
            numpy.mean(head_angle_with_sign[group_id][zero_frame_idx + numpy.arange(-20, 0), i]))

        nan_idx = numpy.where(numpy.isnan(head_angle_no_sign[group_id][:, i]))[0]
        first_nan_idx = nan_idx[0] if len(nan_idx) > 0 else len(head_angle_no_sign[group_id][:, i])
        imax = numpy.argmin(head_angle_no_sign[group_id][zero_frame_idx:first_nan_idx, i])

        # take angle most closely directed to the shelter
        post_loom_angle[group_id].append(head_angle_with_sign[group_id][zero_frame_idx + imax, i])
