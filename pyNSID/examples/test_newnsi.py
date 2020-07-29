import numpy as np

#from nsidask import nsidask as nsi
import string
import os, sys
sys.path.append('../pyNSID/')
import pyNSID as nsid


import file_tools_nsid as ft
import matplotlib.pyplot as plt


def open_file():
    dset = ft.h5_open_file()
    print(dset)
    data = nsid.io.NSIDask.from_hdf5(dset)

    dset.file.close()
    return(data)

def open_file2():
    dset = ft.h5_open_file()

    data = nsid.io.NSIDask.from_array(dset)
    data.title = dset.attrs['title']
    data.units = dset.attrs['units']
    data.quantity = dset.attrs['quantity']
    data.data_type = dset.attrs['data_type']

    for dim in range(dset.ndim):
        data.set_dimension(dim,ft.Dimension(dset.dims[dim].label, np.array(dset.dims[dim][0]), dset.axes_quantities[dim], dset.axes_units[dim], dset.dimension_types[dim]) )

    data.attrs = dict(dset.attrs)
    print(dset.parent.keys())
    print(dset.parent.attrs.keys())
    if 'original_metadata' in dset.parent:
        data.original_metadata = dict(dset.parent['original_metadata'].attrs)

    dset.file.close()


    return data


data = open_file()
#print(data.z, data.z.quantity , data.z.units)
for key,value in data.attrs.items():
    pass#print(key, value)

def plot_image(data):
    """
    plotting of data according to two axis marked as 'spatial' in the dimensions
    """
    selection = []
    image_dims = []
    for  dim, axis in data.axes.items():
        if axis.dimension_type == 'spatial':
            selection.append(slice(None))
            image_dims.append(dim)
        else:
            selection.append(slice(0,1))
    if len(image_dims) != 2:
        raise ValueError('We need two dimensions with dimension_type spatial to plot an image')

    plt.figure()
    img = plt.imshow(data[tuple(selection)].T, extent = data.get_extent(image_dims))
    plt.xlabel(f"{data.x.quantity} [{data.x.units}]")
    plt.ylabel(f"{data.y.quantity} [{data.y.units}]")
    cbar = plt.colorbar(img)
    cbar.set_label(f"{data.quantity} [{data.units}]")

    plt.show()

plot_image(data)

print('done')