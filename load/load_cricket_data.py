import os
import numpy as np
import more_itertools as mit

from shared import TRANSFORMED_DF_PATH, HOUSE_START, ARENA_LENGTH_PX, ARENA_WIDTH_PX

DISTANCE_THRESHOLD = 0.2
MINIMUM_BOUT_LENGTH = 10
INTER_BOUT_INTERVAL = 50
N_FRAMES_TO_SHOW = 100

ids = {
    "naive": [
        "1114171_20210410_12_52_20",
        "1114174_20210408_11_25_40",
        "1114176_20210416_13_18_24",
        "1114175_20210421_15_34_16",
        "1114303_20210429_16_04_42",
    ],
    "lse": [
        "1114170_20210408_15_49_13",
        "1114177_20210421_11_48_27",
        "1114178_20210417_14_41_09",
        "1114179_20210415_13_08_19",
        "1114309_20210423_14_28_47",
        "1114302_20210429_10_41_59",
    ],
}

keys = ["naive", "lse"]

body_tracks = dict.fromkeys(keys)
cricket_tracks = dict.fromkeys(keys)
reaches_shelter = dict.fromkeys(keys)
event_times = dict.fromkeys(keys)

reaches_shelter_flattened = dict.fromkeys(keys)
event_times_flattened = dict.fromkeys(keys)


for group_id in ids.keys():

    body_tracks_group = []
    cricket_tracks_group = []
    event_times_group = []
    reaches_shelter_group = []

    for exp_id in ids[group_id]:

        print(f"Loading {exp_id}")

        x_body = np.load(os.path.join(TRANSFORMED_DF_PATH, exp_id, "x_body.npy"))
        y_body = np.load(os.path.join(TRANSFORMED_DF_PATH, exp_id, "y_body.npy"))
        x_cricket = np.load(os.path.join(TRANSFORMED_DF_PATH, exp_id, "x_cricket.npy"))
        y_cricket = np.load(os.path.join(TRANSFORMED_DF_PATH, exp_id, "y_cricket.npy"))

        body = np.array(
            [1 - (x_body / ARENA_LENGTH_PX), (1 - (y_body / ARENA_WIDTH_PX)) * 0.4]
        )
        cricket = np.array(
            [
                1 - (x_cricket / ARENA_LENGTH_PX),
                (1 - (y_cricket / ARENA_WIDTH_PX)) * 0.4,
            ]
        )

        distance = np.sqrt(np.sum((body - cricket) ** 2, axis=0))

        mouse_within_range_of_cricket = distance < DISTANCE_THRESHOLD
        encounters = np.where(mouse_within_range_of_cricket)[0]

        bouts = [
            list(group) for group in mit.consecutive_groups(list(list(encounters)))
        ]
        bouts = [x for x in bouts if len(x) > MINIMUM_BOUT_LENGTH]

        bout_distances = [distance[x[0] : x[-1]] for x in bouts]
        minimum_pos = [np.argmin(x) for x in bout_distances]
        closest_pos_idx = [x[y] for x, y in zip(bouts, minimum_pos)]

        y = 0
        bout_idx = []
        for x in closest_pos_idx:
            if x - y > INTER_BOUT_INTERVAL:
                bout_idx.append(x)
            y = x

        these_body_tracks = [
            body[:, start : start + N_FRAMES_TO_SHOW] for start in bout_idx
        ]
        these_cricket_tracks = [
            cricket[:, start : start + N_FRAMES_TO_SHOW] for start in bout_idx
        ]

        reaches_shelter_group.append([])
        for track in these_body_tracks:
            if np.any(track[0] < HOUSE_START):
                reaches_shelter_group[-1].append(True)
            else:
                reaches_shelter_group[-1].append(False)

        body_tracks_group.append(these_body_tracks)
        cricket_tracks_group.append(these_cricket_tracks)

        event_times_group.append(np.array(bout_idx) / 50 / 60)

    body_tracks[group_id] = body_tracks_group
    cricket_tracks[group_id] = cricket_tracks_group
    reaches_shelter[group_id] = reaches_shelter_group
    event_times[group_id] = event_times_group

    reaches_shelter_flattened[group_id] = np.array(
        [y for x in reaches_shelter[group_id] for y in x]
    )
    event_times_flattened[group_id] = np.array(
        [y for x in event_times[group_id] for y in x]
    )
