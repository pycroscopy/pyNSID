import numpy as np
import matplotlib.pyplot as plt

import h5py

import sys
sys.path.append('../../../sidpy/')
sys.path.append('../pyNSID/')
import pyNSID as nsid

sys.path.append('../../../Image_Distortion-NSID/')
import file_tools_nsid as ft
import EELS_tools2a  as eels


if True:
    h5_file = h5py.File('test.hf5', mode='a')
else:
    hf_file = ft.h5_open()

if 'Measurement_000' not in h5_file:
    h5_group = h5_file.create_group('Measurement_000/Channel_000')
else:
    h5_group = h5_file['Measurement_000/Channel_000']
for key in h5_group:
    print(key)
print(h5_group['nDim_Data'].shape)
data = nsid.read_nsid_dataset(h5_group['nDim_Data'])
if 'Measurement_000/Channel_001' in h5_file:
    del h5_file['Measurement_000/Channel_001']
h5_group_1 = h5_file.create_group('Measurement_000/Channel_001')
dset = nsid.empty_dataset([3, 3, 4], h5_group_1)

print(dset)

h5_file.flush()
h5_file.close()

Z = 22
Xsection = eels.get_Xsections(int(Z))
print(Xsection['L3'].keys())
index = 0
edges = {}
Z_s = {22: 'L3', 8: 'K1', 42: 'L3', 57: 'M5'}
edges[index] = {}
edge = Xsection['L3']

for index, key in enumerate(Z_s):
    print()
    Xsection = eels.get_Xsections(int(key))
    edges[index] = {}
    edge = Xsection[Z_s[key]]
    edges[index] = {'Z': int(key), 'symmetry': Z_s[key], 'element': eels.elements[key], 'onset': edge['onset']}
    edges[index]['chemical_shift'] = 0
    edges[index]['end_exclude'] = edge['onset'] + edge['excl after']
    edges[index]['start_exclude'] = edge['onset'] - edge['excl before']
    edges[index]['areal_density'] = 0.0
    edges[index]['original_onset'] = edge['onset']+0.


edges[1]['onset'] = 528
edges[1]['chemical_shift'] = edges[1]['onset'] - edges[1]['original_onset']
edges[1]['start_exclude'] = edges[1]['start_exclude'] + edges[1]['chemical_shift']
edges[1]['end_exclude'] = edges[1]['end_exclude'] + edges[1]['chemical_shift']
edges[3]['end_exclude'] = data.energy_scale.values[-10]

print(edges[1])
alpha = data.attrs['convergence_angle']
beta = data.attrs['collection_angle']
beamkV = data.attrs['acceleration_voltage']
eff_beta = eels.effective_collection_angle(data.energy_scale.values, alpha, beta, beamkV)

edges = eels.make_cross_sections(edges, data.energy_scale.values, beamkV, eff_beta)

print(edges)
edges['fit_area'] = {'fit_start': data.energy_scale.values[1], 'fit_end': data.energy_scale.values[-2]}

if data.data_type == 'image_stack':
    view = nsid.viz.plot_stack(data)
elif data.data_type == 'image':
    view = nsid.viz.plot_image(data)
elif data.data_type == 'spectrum_image':
    view = nsid.viz.plot_spectrum_image(data)
    view.set_bin([8, 2])
    print(view.bin_x)
plt.show()
view.x = 0
for y in range(1):  # data.shape[1]-1):
    # view.y = y
    spectrum = np.array(view.get_spectrum())
    edges = eels.fit_edges2(spectrum, data.energy_scale.values, edges)

print(edges['model'].keys())
print(edges['model']['spectrum'].shape)
model = np.array(edges['model']['spectrum'])


from lmfit import Parameters, minimize, report_fit


def gauss(x, amp, cen, sigma):
    """Gaussian lineshape."""
    return amp * np.exp(-(x-cen)**2 / (2.*(sigma/2.3548)**2))


def gauss_dataset(params,  x):
    """Calculate Gaussian lineshape from parameters for data set."""
    sum_gauss = x*0
    for i in range(int(params['number_of_peaks'])):
        amp = params[f'amp_{i}']
        cen = params[f'cen_{i}']
        sig = params[f'sig_{i}']
        sum_gauss = sum_gauss+gauss(x, amp, cen, sig)
    return sum_gauss


def objective(params, x, data2):
    """Calculate total residual for fits of Gaussians to several data sets."""
    resid = data2 - gauss_dataset(params, x)

    # now flatten this to a 1D array, as minimize() needs
    return resid


from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter


def noise_free(spectrum, reference, energy_scale, noise_limit=.1, peak_prominence=0.01, noise_level=.45):
    difference = (spectrum-reference)/spectrum.sum()*30000.
    fit_params = Parameters()

    number_of_peaks = 0
    total = 0  # np.abs(difference/np.sqrt(np.abs(spectrum))).sum()
    gauss_out = spectrum*0.
    diff = spectrum
    j = 0
    out = 0
    while np.abs(np.abs(total) - np.abs(diff/np.sqrt(np.abs(spectrum))).sum()) > noise_limit:
        diff = difference-gauss_out
        j += 1
        total = np.abs(diff/np.sqrt(np.abs(spectrum))).sum()
        blurred2 = gaussian_filter(np.abs(diff)/np.sqrt(np.abs(spectrum)), sigma=3)
        peaks, _ = find_peaks(blurred2, prominence=peak_prominence)
        if len(peaks) == 0:
            break
        for i, peak in enumerate(peaks):
            fit_params.add(f'amp_{i+number_of_peaks}', value=diff[peak]*1.1, min=-1000, max=1000)
            fit_params.add(f'cen_{i+number_of_peaks}', value=energy_scale[peak], min=energy_scale[peak]*0.8-.5,
                           max=energy_scale[peak]*1.2+.5)
            fit_params.add(f'sig_{i+number_of_peaks}', value=.5, min=.1, max= 30)
        number_of_peaks += len(peaks)
        fit_params.add('number_of_peaks', number_of_peaks)

        out = minimize(objective, fit_params, args=(energy_scale,difference), **{'ftol': 1e-2})  # , method ='brent')
        fit_params = out.params
        gauss_out = gauss_dataset(out.params, energy_scale)
        diff = difference-gauss_out

        print(f'{j:2d}: {number_of_peaks:3d} peaks, noise: '
              f' {np.abs(diff / np.sqrt(np.abs(spectrum))).sum() / len(spectrum) * 100:.3f} %/channel,'
              f' total Δ : {total - np.abs(diff / np.sqrt(np.abs(spectrum))).sum():.3f}')
        if number_of_peaks > 120:
            break
        if np.abs(diff / np.sqrt(np.abs(spectrum))).sum()/len(spectrum)*100 < noise_level:
            print('noise level reached')
            break

    return gauss_out, out


import time
start_time = time.time()
gauss_out, out = noise_free(spectrum, model, data.energy_scale.values)
print((time.time()-start_time)/60., 'min')

new_model = gauss_out*spectrum.sum()/30000+model
plt.figure()
plt.plot(data.energy_scale.values, spectrum)
plt.plot(data.energy_scale.values, model)
plt.plot(data.energy_scale.values, spectrum-model)
plt.plot(data.energy_scale.values, new_model)
plt.plot(data.energy_scale.values, new_model-spectrum)

plt.show()
