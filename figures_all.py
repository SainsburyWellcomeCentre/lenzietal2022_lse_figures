import os

import figure_1
import figure_2
import figure_3
import figure_s1

from shared import save_dir

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

figure_1.main()
figure_2.main()
figure_3.main()
figure_s1.main()
