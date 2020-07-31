import numpy as np
import matplotlib.pyplot as plt


import os, sys
sys.path.append('../pyNSID/')
import pyNSID as nsid


import file_tools_nsid as ft

data = ft.h5_open_file()
data2 = data.copy()
data.x.units='mm'
data.units = 'ff'
view = nsid.viz.plot_stack(data)
plt.show()

view = nsid.viz.plot_image(data2)
plt.show()


print('done')