# -*- coding: utf-8 -*-
"""
Utilities for generating static image and line plots of near-publishable quality

Created on Thu May 05 13:29:12 2016

@author: Suhas Somnath, Chris R. Smith
"""
# TODO: All general plotting functions should support data with 1, 2, or 3 spatial dimensions.

from __future__ import division, print_function, absolute_import, unicode_literals

import inspect
import os
import sys
from numbers import Number
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
import dask.array as da

from sidpy.hdf.dtype_utils import is_complex_dtype
from sidpy.base.num_utils import contains_integers, get_exponent
from sidpy.viz.plot_utils import plot_map

if sys.version_info.major == 3:
    unicode = str

default_cmap = plt.cm.viridis


class plot_curve(object):
    def __init__(self, dset, ref_dims, figure =None,**kwargs):

        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp
        print(figure)
        if figure == None:
            self.fig = plt.figure(**fig_args)
        else:
            self.fig = figure

        self.dset = dset
        self.kwargs = kwargs

        # Handle the simple cases first:
        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        if len(ref_dims) != 1:
            print( 'data type not handled yet')
        self.axis = self.fig.add_subplot(1, 1, 1, **fig_args)

        self._update()

    def _update(self):

        if False:#is_complex_dtype(np.array(dset)):
            # Plot real and image
            fig, axes = plt.subplots(nrows=2, **fig_args)

            for axis, ufunc, comp_name in zip(axes.flat, [np.abs, np.angle], ['Magnitude', 'Phase']):
                axis.plot(self.dset.dims[ref_dims][0], ufunc(np.squeeze(curve)), **kwargs)
                if comp_name == 'Magnitude':
                    axis.set_title(self.dset.file.filename.split('/')[-1] + '\n(' + comp_name + ')', pad=15)
                    axis.set_xlabel(self.dset.get_dimension_labels()[ref_dims[0]])# + x_suffix)
                    axis.set_ylabel(self.dset.data_descriptor)
                    axis.ticklabel_format(style='sci', scilimits=(-2, 3))
                else:
                    axis.set_title(comp_name, pad=15)
                    axis.set_ylabel('Phase (rad)')
                    axis.set_xlabel(self.get_dimension_labels()[ref_dims[0]])# + x_suffix)
                    axis.ticklabel_format(style='sci', scilimits=(-2, 3))

            fig.tight_layout()
            return fig, axes

        else:

            self.axis.clear()
            self.axis.plot(self.dset.dims[0][0], self.dset, **self.kwargs)
            self.axis.set_title(self.dset.file.filename.split('/')[-1], pad=15)
            self.axis.set_xlabel(self.dset.get_dimension_labels()[0])# + x_suffix)
            self.axis.set_ylabel(self.dset.data_descriptor)
            self.axis.ticklabel_format(style='sci', scilimits=(-2, 3))
            self.fig.canvas.draw_idle()

class plot_image(object):
    """
    Interactive display of image plot

    The stack can be scrolled through with a mouse wheel or the slider
    The ususal zoom effects of matplotlib apply.
    Works on every backend because it only depends on matplotlib.

    Important: keep a reference to this class to maintain interactive properties so usage is:

    >>view = plot_stack(dataset, {'spatial':[0,1], 'stack':[2]})

    Input:
    ------
    - dset: NSI_dataset
    - dim_dict: dictionary
        with key: "spatial" list of int: dimension of image
    """
    def __init__(self, dset, dim_dict, figure =None,**kwargs):

        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        if figure == None:
            self.fig = plt.figure(**fig_args)
        else:
            self.fig = figure

        self.dset = dset
        extent = self.dset.make_extent(dim_dict['spatial'])

        if is_complex_dtype(self.dset):
            # Plot real and image
            fig, axes = plt.subplots(nrows=2, **fig_args)
            for axis, ufunc, comp_name in zip(axes.flat, [np.abs, np.angle], ['Magnitude', 'Phase']):
                cbar_label = self.data_descriptor
                if comp_name == 'Phase':
                    cbar_label = 'Phase (rad)'
                plot_map(axis, ufunc(np.squeeze(img)), show_xy_ticks=True, show_cbar=True,
                         cbar_label=cbar_label, x_vec=ref_dims[1].values, y_vec=ref_dims[0].values,
                         **kwargs)
                axis.set_title(self.name + '\n(' + comp_name + ')', pad=15)
                axis.set_xlabel(ref_dims[1].name + ' (' + ref_dims[1].units + ')' + suffix[1])
                axis.set_ylabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + suffix[0])
            fig.tight_layout()
            return fig, axes

        else:

            self.axis = self.fig.add_subplot(1,1,1)
            self.img = self.axis.imshow(np.squeeze(self.dset).T, extent=extent, **kwargs)
            self.axis.set_title(self.dset.file.filename.split('/')[-1], pad=15)
            self.axis.set_xlabel(self.dset.get_dimension_labels()[dim_dict['spatial'][0]])# + x_suffix)
            self.axis.set_ylabel(self.dset.get_dimension_labels()[dim_dict['spatial'][1]])
            self.axis.ticklabel_format(style='sci', scilimits=(-2, 3))
            cbar = self.fig.colorbar(self.img)
            cbar.set_label(self.dset.data_descriptor)
            self.fig.tight_layout()
            self.img.axes.figure.canvas.draw_idle()


class  plot_stack(object):
    """
    Interactive display of image stack plot

    The stack can be scrolled through with a mouse wheel or the slider
    The ususal zoom effects of matplotlib apply.
    Works on every backend because it only depends on matplotlib.

    Important: keep a reference to this class to maintain interactive properties so usage is:

    >>view = plot_stack(dataset, {'spatial':[0,1], 'stack':[2]})

    Input:
    ------
    - dset: NSI_dataset
    - dim_dict: dictionary
        with key: "spatial" list of int: dimension of image
        with key: "time" or "stack": list of int: dimension of image stack

    """
    def __init__(self, dset, dim_dict, figure =None,**kwargs):

        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp


        if figure == None:
            self.fig = plt.figure(**fig_args)
        else:
            self.fig = figure


        if len(dset.shape) <3:
            raise KeyError('dataset must have at least three dimensions')
            return

        ### We need one stack dimension and two image dimensions as lists in dictionary
        if 'spatial' not in dim_dict:
            raise KeyError('dimension_dictionary must contain a spatial key')
            return
        image_dims = dim_dict['spatial']
        if len(image_dims)<2:
            raise KeyError('spatial key in dimension_dictionary must be list of length 2')
            return

        if 'stack' not in dim_dict:
            if 'time' in dim_dict:
                stack_dim = dim_dict['time']
            else:
                raise KeyError('dimension_dictionary must contain key stack or time')
                return
        else:
            stack_dim = dim_dict['stack']
        if len(stack_dim) < 1:
            raise KeyError('stack key in dimension_dictionary must be list of length 1')
            return

        if stack_dim[0] != 0 or image_dims != [1,2]:
            ## axes not in expected order, displaying a copy of data with right dimensional oreder:
            self.cube =  np.transpose(dset, (stack_dim[0], image_dims[0],image_dims[1]))
        else:
            self.cube  = dset

        extent = dset.make_extent([image_dims[0],image_dims[1]])

        self.axis = self.fig.add_axes([0.0, 0.2, .9, .7])
        self.ind = 0
        self.img = self.axis.imshow(self.cube[self.ind].T, extent = extent, **kwargs )
        interval = 100 # ms, time between animation frames

        self.number_of_slices= self.cube.shape[0]

        self.axis.set_title('image stack: '+dset.file.filename.split('/')[-1]+'\n use scroll wheel to navigate images')
        self.img.axes.figure.canvas.mpl_connect('scroll_event', self._onscroll)
        self.axis.set_xlabel(dset.get_dimension_labels()[image_dims[0]]);
        self.axis.set_ylabel(dset.get_dimension_labels()[image_dims[1]]);
        cbar = self.fig.colorbar(self.img)
        cbar.set_label(dset.data_descriptor)


        axidx = self.fig.add_axes([0.1, 0.05, 0.55, 0.03])
        self.slider = Slider(axidx, 'image', 0, self.cube.shape[0]-1, valinit=self.ind, valfmt='%d')
        self.slider.on_changed(self._onSlider)
        playax = self.fig.add_axes([0.7, 0.05, 0.09, 0.03])
        self.play_button = Button(playax, 'Play')#, hovercolor='0.975')

        self.play = False


        self.play_button.on_clicked(self._play_slice)

        sumax = self.fig.add_axes([0.8, 0.05, 0.09, 0.03])
        self.sum_button = Button(sumax, 'Average')#, hovercolor='0.975')
        self.sum_button.on_clicked(self._sum_slice)
        self.sum = False

        self.anim = animation.FuncAnimation(self.fig, self._updatefig, interval=200, blit=False, repeat = True)
        self._update()

    def _sum_slice(self,event):
        self.img.set_data(np.average(self.cube, axis = 0).T)
        self.img.axes.figure.canvas.draw_idle()

    def _play_slice(self,event):
        self.play = not self.play
        if self.play:
            self.anim.event_source.start()
        else:
            self.anim.event_source.stop()

    def _onSlider(self, val):
        self.ind = int(self.slider.val+0.5)
        self.slider.valtext.set_text('{}'.format(self.ind))
        self._update()

    def _onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.number_of_slices
        else:
            self.ind = (self.ind - 1) % self.number_of_slices
        self.ind = int(self.ind)
        self.play = False
        self.anim.event_source.stop()
        self.slider.set_val(self.ind)

    def _update(self):
        self.img.set_data(self.cube[int(self.ind)].T)
        self.img.axes.figure.canvas.draw_idle()
        if not self.play:
            self.anim.event_source.stop()

    def _updatefig(self,*args):
        self.ind = (self.ind+1) % self.number_of_slices
        self.slider.set_val(self.ind)

        return self.img

class plot_spectrum_image(object):

    """
    ### Interactive spectrum imaging plot

    """

    def __init__(self, dset,  dim_dict,  figure =None, horizontal = True, **kwargs):

        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        if figure == None:
            self.fig = plt.figure(**fig_args)
        else:
            self.fig = figure

        if len(dset.shape) <3:
            raise KeyError('dataset must have at least three dimensions')
            return

        ### We need one stack dimension and two image dimensions as lists in dictionary
        if 'spatial' not in dim_dict:
            raise KeyError('dimension_dictionary must contain a spatial key')
            return
        image_dims = dim_dict['spatial']
        if len(image_dims)<2:
            raise KeyError('spatial key in dimension_dictionary must be list of length 2')
            return

        if 'spectral' not in dim_dict:
            raise KeyError('dimension_dictionary must contain key stack or time')
            return
        spec_dim = dim_dict['spectral']
        if len(spec_dim) < 1:
            raise KeyError('spectral key in dimension_dictionary must be list of length 1')
            return

        if spec_dim[0] != 2 or image_dims != [0,1]:
            ## axes not in expected order, displaying a copy of data with right dimensional oreder:
            self.cube =  np.transpose(dset, (image_dims[0],image_dims[1], spec_dim[0]))
        else:
            self.cube  = dset

        extent = dset.make_extent([image_dims[0],image_dims[1]])

        self.horizontal = horizontal
        self.x = 0
        self.y = 0
        self.bin_x = 1
        self.bin_y = 1

        sizeX = self.cube.shape[0]
        sizeY = self.cube.shape[1]

        self.energy_scale = dset.dims[spec_dim[0]][0]

        self.extent = [0,sizeX,sizeY,0]
        self.rectangle = [0,sizeX,0,sizeY]
        self.scaleX = 1.0
        self.scaleY = 1.0
        self.analysis = []
        self.plot_legend = False


        if horizontal:
            self.axes = self.fig.subplots(ncols=2)
        else:
            self.axes = self.fig.subplots(nrows=2, **fig_args)

        self.fig.canvas.set_window_title(dset.file.filename.split('/')[-1])
        self.image = np.sum(self.cube, axis=2)

        self.axes[0].imshow(self.image.T, extent = self.extent, **kwargs)
        if horizontal:
            self.axes[0].set_xlabel('distance [pixels]')
        else:
            self.axes[0].set_ylabel('distance [pixels]')
        self.axes[0].set_aspect('equal')

        #self.rect = patches.Rectangle((0,0),1,1,linewidth=1,edgecolor='r',facecolor='red', alpha = 0.2)
        self.rect = patches.Rectangle((0,0),self.bin_x,self.bin_y,linewidth=1,edgecolor='r',facecolor='red', alpha = 0.2)

        self.axes[0].add_patch(self.rect)
        self.intensity_scale = 1.
        self.spectrum = self.get_spectrum()

        self.axes[1].plot(self.energy_scale,self.spectrum)
        self.axes[1].set_title(' spectrum {},{} '.format(self.x, self.y))
        self.xlabel = dset.get_dimension_labels()[spec_dim[0]]
        self.axes[1].set_xlabel(self.xlabel)# + x_suffix)
        self.ylabel = dset.data_descriptor
        self.axes[1].set_ylabel(self.ylabel)
        self.axes[1].ticklabel_format(style='sci', scilimits=(-2, 3))
        self.fig.tight_layout()
        self.cid = self.axes[1].figure.canvas.mpl_connect('button_press_event', self._onclick)

        self.fig.canvas.draw_idle()

    def set_bin(self,bin):

        old_bin_x = self.bin_x
        old_bin_y = self.bin_y
        if isinstance(bin, list):

            self.bin_x = int(bin[0])
            self.bin_y = int(bin[1])

        else:
            self.bin_x = int(bin)
            self.bin_y = int(bin)

        self.rect.set_width(self.rect.get_width()*self.bin_x/old_bin_x)
        self.rect.set_height((self.rect.get_height()*self.bin_y/old_bin_y))
        if self.x+self.bin_x >  self.cube.shape[0]:
            self.x = self.cube.shape[0]-self.bin_x
        if self.y+self.bin_y >  self.cube.shape[1]:
            self.y = self.cube.shape[1]-self.bin_y

        self.rect.set_xy([self.x*self.rect.get_width()/self.bin_x +  self.rectangle[0],
                            self.y*self.rect.get_height()/self.bin_y +  self.rectangle[2]])
        self._update()

    def get_spectrum(self):
        if self.x > self.cube.shape[0]-self.bin_x:
            self.x = self.cube.shape[0]-self.bin_x
        if self.y > self.cube.shape[1]-self.bin_y:
            self.y = self.cube.shape[1]-self.bin_y

        self.spectrum = np.average(self.cube[self.x:self.x+self.bin_x,self.y:self.y+self.bin_y,:], axis=(0,1))
        #* self.intensity_scale[self.x,self.y]
        return   self.spectrum

    def _onclick(self,event):
        self.event = event
        if event.inaxes in [self.axes[0]]:
            x = int(event.xdata)
            y = int(event.ydata)

            x= int(x - self.rectangle[0])
            y= int(y - self.rectangle[2])

            if x>=0 and y>=0:
                if x<=self.rectangle[1] and y<= self.rectangle[3]:
                    self.x = int(x/(self.rect.get_width()/self.bin_x))
                    self.y = int(y/(self.rect.get_height()/self.bin_y))

                    if self.x+self.bin_x >  self.cube.shape[0]:
                        self.x = self.cube.shape[0]-self.bin_x
                    if self.y+self.bin_y >  self.cube.shape[1]:
                        self.y = self.cube.shape[1]-self.bin_y

                    self.rect.set_xy([self.x*self.rect.get_width()/self.bin_x +  self.rectangle[0],
                                      self.y*self.rect.get_height()/self.bin_y +  self.rectangle[2]])
        #self.ax1.set_title(f'{self.x}')
        self._update()

    def _update(self, ev=None):

        xlim = self.axes[1].get_xlim()
        ylim = self.axes[1].get_ylim()
        self.axes[1].clear()
        self.get_spectrum()


        self.axes[1].plot(self.energy_scale,self.spectrum, label = 'experiment')

        self.axes[1].set_title(' spectrum {},{} '.format(self.x, self.y))


        self.axes[1].set_xlim(xlim)
        self.axes[1].set_ylim(ylim)
        self.axes[1].set_xlabel(self.xlabel)
        self.axes[1].set_ylabel(self.ylabel)

        self.fig.canvas.draw_idle()


    def set_legend(self, setLegend):
        self.plot_legend = setLegend

    def get_xy(self):
        return [self.x,self.y]
