import os

import figure_1_revised
#import figure_2
#'import figure_3
import figure_s1_revised

from shared import save_dir

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

figure_1_revised.main()
#figure_2.main()
#figure_3.main()
figure_s1_revised.main()
