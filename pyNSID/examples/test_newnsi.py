import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.append('../../../sidpy/')
import sidpy as sid


sys.path.append('../pyNSID/')
import pyNSID as nsid


import file_tools_nsid as ft
sys.path.append('../../../Image_Distortion-NSID/')
import EELS_tools2a  as eels

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
data = nsid.NSIDask.from_hdf5(h5_group['nDim_Data'])

#data.to_hdf5(h5_group)
#data = nsid.NSIDask.from_hdf5(h5_group['05_EELS Spectrum Image_3layers (dark ref corrected)'])
h5_file.flush()
h5_file.close()

Z = 22
Xsection = eels.get_Xsections(int(Z))
print(Xsection['L3'].keys())
index = 0
edges={}
Z_s = {22: 'L3', 8: 'K1', 42:'L3', 57: 'M5'}
edges[index]={}
edge = Xsection['L3']

for index, key in enumerate(Z_s):
    print()
    Xsection = eels.get_Xsections(int(key))
    edges[index] = {}
    edge = Xsection[Z_s[key]]
    edges[index] = {'Z': int(key), 'symmetry': Z_s[key], 'element': eels.elements[key], 'onset': edge['onset']}
    edges[index]['chemcial_shift'] = 0
    edges[index]['end_exclude']   = edge['onset'] + edge['excl after']
    edges[index]['start_exclude'] = edge['onset'] - edge['excl before']
    edges[index]['areal_density'] = 0.0
    edges[index]['original_onset']  = edge['onset']

print(edges)

alpha = data.attrs['convergence_angle']
beta = data.attrs['collection_angle']
beamkV = data.attrs['acceleration_voltage']

eff_beta = eels.effective_collection_angle(data.energy_scale.values, alpha, beta, beamkV)

if data.data_type == 'image_stack':
    view = nsid.viz.plot_stack(data)
elif data.data_type == 'image':
    view = nsid.viz.plot_image(data)
elif data.data_type == 'spectrum_image':
    view = nsid.viz.plot_spectrum_image(data)
    view.set_bin([8,2])
    print(view.bin_x)
plt.show()

Z = 22
Xsection = eels.get_Xsections(int(Z))
print(Xsection.keys())

self.edges[index] ['Z'] = 22
self.edges[index]['onset'] = onset
self.edges[index]['chemcial_shift'] = onset - self.edges[index]['original_onset']
self.edges[index]['end_exclude'] = excl_end
self.edges[index]['start_exclude'] = excl_start

print('done')