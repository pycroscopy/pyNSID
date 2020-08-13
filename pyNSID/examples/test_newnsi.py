import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.append('../../../sidpy/')
import sidpy as sid


sys.path.append('../pyNSID/')
import pyNSID as nsid


import file_tools_nsid as ft
import h5py
h5_file = h5py.File('test.hf5',mode='a')
if 'Measurement_000' not in h5_file:
    h5_group = h5_file.create_group('Measurement_000/Channel_000')
else:
    h5_group = h5_file['Measurement_000/Channel_000']
    #for key in h5_group:
    #    del h5_group[key]
#h5_group['nDim_Data'] = h5_group['05_EELS Spectrum Image_3layers (dark ref corrected)']
for key in h5_group:
    print(key)
print(h5_group['nDim_Data'].shape)
data = nsid.NSIDask.from_hdf5(h5_group)

#data.to_hdf5(h5_group)
#data = nsid.NSIDask.from_hdf5(h5_group['05_EELS Spectrum Image_3layers (dark ref corrected)'])
h5_file.flush()
h5_file.close()


if data.data_type == 'image_stack':
    view = nsid.viz.plot_stack(data)
elif data.data_type == 'image':
    view = nsid.viz.plot_image(data)
elif data.data_type == 'spectrum_image':
    view = nsid.viz.plot_spectrum_image(data)
    view.set_bin([8,2])
plt.show()


print('done')