This repo contains code to replicate the figures appearing in:

**Lenzi et al. (202x) *Ethological factors influencing escape behaviour and its adaptive control***

The code relies on the [`looming_spots`](https://github.com/stephenlenzi/looming_spots) repo (commit: `xxxx`; annotated tag: `lsie_paper`)

The paths to the data are set in `path_config.py`.  The paths are:


- `proc_path` -  folder containing the processed data from all experiments including those appearing in the paper. At the time of writing, it is:
    
     ` winstor\swc\margrie\glusterfs\imaging\l\loomer\processed_data`

- `df_path` - folder containing `.h5` files from the cricket experiments.

- `transformed_df_path` - folder containing `.npy` files of transformed tracks from `df_path`

- `figure_path` - a folder for putting pdf files of the figures.



Once these paths have been correctly set and the `looming_spots` package is available, you can run `figures_all.py`.

This will load the data, plot the figures and print them to pdfs in `figure_path/<todays date>`.
