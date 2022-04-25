import os

import figure_1
import figure_2
import figure_3
import figure_1_supplement
import figure_2_supplement

from shared import save_dir

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

figure_1.main()
figure_1_supplement.main()
figure_2.main()
figure_2_supplement.main()
figure_3.main()
