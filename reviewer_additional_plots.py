from looming_spots.db import loom_trial_group, tracks
import matplotlib.pyplot as plt


def get_lsie_exploration_tracks(mids):
    all_tracks = []
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        start_trial = mtg.lsie_trials()[0]
        lsie_start = start_trial.sample_number
        lsie_end = mtg.lsie_trials()[-1].sample_number
        track = tracks.Track(start_trial.folder, start_trial.path, lsie_start, lsie_end, start_trial.session.x_track())
        all_tracks.append(track)
    return all_tracks

def get_heatmap(tracks):
    pass


def plot_lsie_with_shelter_escape_result(mids):
    for mid in mids:
        mtg = loom_trial_group.MouseLoomTrialGroup(mid)
        for t in mtg.post_test_trials():
            t.plot_track()
        plt.show()
