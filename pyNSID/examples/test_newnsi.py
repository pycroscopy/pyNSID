import numpy as np
import matplotlib.pyplot as plt


import os, sys
sys.path.append('../pyNSID/')
import pyNSID as nsid


import file_tools_nsid as ft


data = ft.h5_open_file()

view = nsid.viz.plot_stack(data)
plt.show()

view = nsid.viz.plot_image(data)
plt.show()


print('done')