This repo contains code to replicate the figures appearing in:

**Lenzi et al. (2022) *Threat history controls escape behaviour in mice***

The code relies on preprocessed data (doi: 10.17632/mcht8r94yv.1) that were generated using the [`looming_spots`](https://github.com/SainsburyWellcomeCentre/looming_spots.git) repo (doi: 10.5281/zenodo.6483482)

The path to the data `DATA_DIRECTORY` can be set in `shared.py`. 

Once these paths have been correctly set and the `looming_spots` package is available, you can run `plot_all_figures.py`.

This will load the data, plot the figures and print them to pdfs in `figure_path/<todays date>`.
