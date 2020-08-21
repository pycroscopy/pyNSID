# -*- coding: utf-8 -*-
"""
Simple yet handy HDF5 utilities, independent of the  USID model

Created on Fri May 22, 2020

@author: Gerd Duscher, Suhas Somnath
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import socket
import sys
from platform import platform
from warnings import warn

import dask as da
import h5py
import numpy as np
import collections


sys.path.append('../../../sidpy/')

from sidpy.base.string_utils import validate_single_string_arg, validate_list_of_strings, \
    clean_string_att, get_time_stamp

from sidpy.hdf.hdf_utils import get_auxiliary_datasets, link_h5_obj_as_alias, get_attr, write_book_keeping_attrs, \
    write_simple_attrs, validate_h5_objs_in_same_h5_file, lazy_load_array

if sys.version_info.major == 3:
    unicode = str

__all__ = ['find_results_groups', 'check_and_link_ancillary', 'copy_attributes', 'check_for_old',
             'assign_group_index', 'create_indexed_group', 'create_results_group', 'copy_dataset',
             'copy_linked_objects', 'check_for_matching_attrs']

def find_results_groups(h5_main, tool_name, h5_parent_group=None):
    """
    Finds a list of all groups containing results of the process of name
    `tool_name` being applied to the dataset
    Parameters
    ----------
    h5_main : h5 dataset reference
        Reference to the target dataset to which the tool was applied
    tool_name : String / unicode
        Name of the tool applied to the target dataset
    h5_parent_group : h5py.Group, optional. Default = None
        Parent group under which the results group will be searched for. Use
        this option when the results groups are contained in different HDF5
        file compared to `h5_main`. BY default, this function will search
        within the same group that contains `h5_main`
    Returns
    -------
    groups : list of references to :class:`h5py.Group` objects
        groups whose name contains the tool name and the dataset name
    """
    if not isinstance(h5_main, h5py.Dataset):
        raise TypeError('h5_main should be a h5py.Dataset object')
    tool_name = validate_single_string_arg(tool_name, 'tool_name')

    if h5_parent_group is not None:
        if not isinstance(h5_parent_group, (h5py.File, h5py.Group)):
            raise TypeError("'h5_parent_group' should either be a h5py.File "
                            "or h5py.Group object")
    else:
        h5_parent_group = h5_main.parent

    dset_name = h5_main.name.split('/')[-1]
    groups = []
    for key in h5_parent_group.keys():
        if dset_name in key and tool_name in key and isinstance(h5_parent_group[key], h5py.Group):
            groups.append(h5_parent_group[key])
    return groups


def check_and_link_ancillary(h5_dset, anc_names, h5_main=None, anc_refs=None):
    """
    This function will add references to auxilliary datasets as attributes
    of an input dataset.
    If the entries in anc_refs are valid references, they will be added
    as attributes with the name taken from the corresponding entry in
    anc_names.
    If an entry in anc_refs is not a valid reference, the function will
    attempt to get the attribute with the same name from the h5_main
    dataset
    Parameters
    ----------
    h5_dset : HDF5 Dataset
        dataset to which the attributes will be written
    anc_names : list of str
        the attribute names to be used
    h5_main : HDF5 Dataset, optional
        dataset from which attributes will be copied if `anc_refs` is None
    anc_refs : list of HDF5 Object References, optional
        references that correspond to the strings in `anc_names`
    Returns
    -------
    None
    Notes
    -----
    Either `h5_main` or `anc_refs` MUST be provided and `anc_refs` has the
    higher priority if both are present.
    """
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be a h5py.Dataset object')

    if isinstance(anc_names, (str, unicode)):
        anc_names = [anc_names]
    if isinstance(anc_refs, (h5py.Dataset, h5py.Group, h5py.File,
                             h5py.Reference)):
        anc_refs = [anc_refs]

    if not isinstance(anc_names, (list, tuple)):
        raise TypeError('anc_names should be a list / tuple')
    if h5_main is not None:
        if not isinstance(h5_main, h5py.Dataset):
            raise TypeError('h5_main should be a h5py.Dataset object')
        validate_h5_objs_in_same_h5_file(h5_dset, h5_main)
    if anc_refs is not None:
        if not isinstance(anc_refs, (list, tuple)):
            raise TypeError('anc_refs should be a list / tuple')

    if anc_refs is None and h5_main is None:
        raise ValueError('No objected provided to link as ancillary')

    def __check_and_link_single(h5_obj_ref, target_ref_name):
        if isinstance(h5_obj_ref, h5py.Reference):
            # TODO: Same HDF5 file?
            h5_dset.attrs[target_ref_name] = h5_obj_ref
        elif isinstance(h5_obj_ref, (h5py.Dataset, h5py.Group, h5py.File)):
            validate_h5_objs_in_same_h5_file(h5_obj_ref, h5_dset)
            h5_dset.attrs[target_ref_name] = h5_obj_ref.ref
        elif h5_main is not None:
            h5_anc = get_auxiliary_datasets(h5_main, aux_dset_name=[target_ref_name])
            if len(h5_anc) == 1:
                link_h5_obj_as_alias(h5_dset, h5_anc[0], target_ref_name)
        else:
            warnstring = '{} is not a valid h5py Reference and will be skipped.'.format(repr(h5_obj_ref))
            warn(warnstring)

    if bool(np.iterable(anc_refs) and not isinstance(anc_refs, h5py.Dataset)):
        """
        anc_refs can be iterated over
        """
        for ref_name, h5_ref in zip(anc_names, anc_refs):
            __check_and_link_single(h5_ref, ref_name)
    elif anc_refs is not None:
        """
        anc_refs is just a single value
        """
        __check_and_link_single(anc_refs, anc_names)
    elif isinstance(anc_names, str) or isinstance(anc_names, unicode):
        """
        Single name provided
        """
        __check_and_link_single(None, anc_names)
    else:
        """
        Iterable of names provided
        """
        for name in anc_names:
            __check_and_link_single(None, name)

    h5_dset.file.flush()


def copy_attributes(source, dest, skip_refs=True, verbose=False):
    """
    Copy attributes from one h5object to another
    Parameters
    ----------
    source : h5py.Dataset, :class:`h5py.Group`, or :class:`h5py.File`
        Object containing the desired attributes
    dest : h5py.Dataset, :class:`h5py.Group`, or :class:`h5py.File`
        Object to which the attributes need to be copied to
    skip_refs : bool, optional. default = True
        Whether or not the references (dataset and region) should be skipped
    verbose : bool, optional. Defualt = False
        Whether or not to print logs for debugging
    """
    mesg = 'should be a h5py.Dataset, h5py.Group,or h5py.File object'
    if not isinstance(source, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('source ' + mesg)
    if not isinstance(dest, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('dest ' + mesg)

    skip_dset_refs = skip_refs
    try:
        validate_h5_objs_in_same_h5_file(source, dest)
    except ValueError:
        if not skip_refs:
            warn('Dataset references will not be copied since {} and {} are '
                 'in different files'.format(source, dest))
        skip_dset_refs = True

    for att_name in source.attrs.keys():
        # print(att_name)
        if att_name not in ['DIMENSION_LIST']:
            att_val = get_attr(source, att_name)
            """
            Don't copy references unless asked
            """
            if isinstance(att_val, h5py.Reference) and not isinstance(att_val, h5py.RegionReference):
                if not skip_dset_refs:
                    if verbose:
                        print('dset ref copying ' + att_name)
                    dest.attrs[att_name] = att_val
            elif isinstance(att_val, h5py.RegionReference):
                # handled in dedicated if condition below
                continue
            else:
                # everything else
                if verbose:
                    print('simple copying ' + att_name)
                dest.attrs[att_name] = clean_string_att(att_val)

    if not skip_refs:
        # This can be copied across files without problems
        mesg = 'Could not copy region references to {}.'.format(dest.name)
        if isinstance(dest, h5py.Dataset):
            try:
                if verbose:
                    print('requested reg ref copy')
                # copy_region_refs(source, dest)
                pass  # TODO: activate again

            except TypeError:
                warn(mesg)
        else:
            warn('Cannot copy region references to {}'.format(type(dest)))

    return dest


def check_for_old(h5_base, tool_name, new_parms=None, target_dset=None,
                  h5_parent_goup=None, verbose=False):
    """
    Check to see if the results of a tool already exist and if they
    were performed with the same parameters.
    Parameters
    ----------
    h5_base : h5py.Dataset object
           Dataset on which the tool is being applied to
    tool_name : str
           process or analysis name
    new_parms : dict, optional
           Parameters with which this tool will be performed.
    target_dset : str, optional, default = None
            Name of the dataset whose attributes will be compared against new_parms.
            Default - checking against the group
    h5_parent_goup : h5py.Group, optional. Default = None
            The group to search under. Use this option when `h5_base` and
            the potential results groups (within `h5_parent_goup` are located
            in different HDF5 files. Default - search within h5_base.parent
    verbose : bool, optional, default = False
           Whether or not to print debugging statements
    Returns
    -------
    group : list
           List of all :class:`h5py.Group` objects with parameters matching those in `new_parms`
    """
    if not isinstance(h5_base, h5py.Dataset):
        raise TypeError('h5_base should be a h5py.Dataset object')
    tool_name = validate_single_string_arg(tool_name, 'tool_name')

    if h5_parent_goup is not None:
        if not isinstance(h5_parent_goup, (h5py.File, h5py.Group)):
            raise TypeError("'h5_parent_group' should either be a h5py.File "
                            "or h5py.Group object")
    else:
        h5_parent_goup = h5_base.parent

    if new_parms is None:
        new_parms = dict()
    else:
        if not isinstance(new_parms, dict):
            raise TypeError('new_parms should be a dict')
    if target_dset is not None:
        target_dset = validate_single_string_arg(target_dset, 'target_dset')

    matching_groups = []
    groups = find_results_groups(h5_base, tool_name,
                                 h5_parent_group=h5_parent_goup)

    for group in groups:
        if verbose:
            print('Looking at group - {}'.format(group.name.split('/')[-1]))

        h5_obj = group
        if target_dset is not None:
            if target_dset in group.keys():
                h5_obj = group[target_dset]
            else:
                if verbose:
                    print('{} did not contain the target dataset: {}'.format(group.name.split('/')[-1],
                                                                             target_dset))
                continue

        if check_for_matching_attrs(h5_obj, new_parms=new_parms, verbose=verbose):
            # return group
            matching_groups.append(group)

    return matching_groups


def assign_group_index(h5_parent_group, base_name, verbose=False):
    """
    Searches the parent h5 group to find the next available index for the group
    Parameters
    ----------
    h5_parent_group : :class:`h5py.Group` object
        Parent group under which the new group object will be created
    base_name : str or unicode
        Base name of the new group without index
    verbose : bool, optional. Default=False
        Whether or not to print debugging statements
    Returns
    -------
    base_name : str or unicode
        Base name of the new group with the next available index as a suffix
    """
    if not isinstance(h5_parent_group, h5py.Group):
        raise TypeError('h5_parent_group should be a h5py.Group object')
    base_name = validate_single_string_arg(base_name, 'base_name')

    if len(base_name) == 0:
        raise ValueError('base_name should not be an empty string')

    if not base_name.endswith('_'):
        base_name += '_'

    temp = [key for key in h5_parent_group.keys()]
    if verbose:
        print('Looking for group names starting with {} in parent containing items: '
              '{}'.format(base_name, temp))
    previous_indices = []
    for item_name in temp:
        if isinstance(h5_parent_group[item_name], h5py.Group) and item_name.startswith(base_name):
            previous_indices.append(int(item_name.replace(base_name, '')))
    previous_indices = np.sort(previous_indices)
    if verbose:
        print('indices of existing groups with the same prefix: {}'.format(previous_indices))
    if len(previous_indices) == 0:
        index = 0
    else:
        index = previous_indices[-1] + 1
    return base_name + '{:03d}'.format(index)


def create_indexed_group(h5_parent_group, base_name):
    """
    Creates a group with an indexed name (eg - 'Measurement_012') under h5_parent_group using the provided base_name
    as a prefix for the group's name
    Parameters
    ----------
    h5_parent_group : :class:`h5py.Group` or :class:`h5py.File`
        File or group within which the new group will be created
    base_name : str or unicode
        Prefix for the group name. This need not end with a '_'. It will be added automatically
    Returns
    -------
    """
    if not isinstance(h5_parent_group, (h5py.Group, h5py.File)):
        raise TypeError('h5_parent_group should be a h5py.File or Group object')
    base_name = validate_single_string_arg(base_name, 'base_name')

    group_name = assign_group_index(h5_parent_group, base_name)
    h5_new_group = h5_parent_group.create_group(group_name)
    write_book_keeping_attrs(h5_new_group)
    return h5_new_group


def create_results_group(h5_main, tool_name, h5_parent_group=None):
    """
    Creates a h5py.Group object autoindexed and named as 'DatasetName-ToolName_00x'
    Parameters
    ----------
    h5_main : h5py.Dataset object
        Reference to the dataset based on which the process / analysis is being performed
    tool_name : string / unicode
        Name of the Process / Analysis applied to h5_main
    h5_parent_group : h5py.Group, optional. Default = None
        Parent group under which the results group will be created. Use this
        option to write results into a new HDF5 file. By default, results will
        be written into the same group containing `h5_main`
    Returns
    -------
    h5_group : :class:`h5py.Group`
        Results group which can now house the results datasets
    """
    if not isinstance(h5_main, h5py.Dataset):
        raise TypeError('h5_main should be a h5py.Dataset object')
    if h5_parent_group is not None:
        if not isinstance(h5_parent_group, (h5py.File, h5py.Group)):
            raise TypeError("'h5_parent_group' should either be a h5py.File "
                            "or h5py.Group object")
    else:
        h5_parent_group = h5_main.parent

    tool_name = validate_single_string_arg(tool_name, 'tool_name')

    if '-' in tool_name:
        warn('tool_name should not contain the "-" character. Reformatted name from:{} to '
             '{}'.format(tool_name, tool_name.replace('-', '_')))
    tool_name = tool_name.replace('-', '_')

    group_name = h5_main.name.split('/')[-1] + '-' + tool_name + '_'
    group_name = assign_group_index(h5_parent_group, group_name)

    h5_group = h5_parent_group.create_group(group_name)

    write_book_keeping_attrs(h5_group)

    # Also add some basic attributes like source and tool name. This will allow relaxation of nomenclature restrictions:
    # this are NOT being used right now but will be in the subsequent versions of pyNSID
    write_simple_attrs(h5_group, {'tool': tool_name, 'num_source_dsets': 1})
    # in this case, there is only one source
    if h5_parent_group.file == h5_main.file:
        for dset_ind, dset in enumerate([h5_main]):
            h5_group.attrs['source_' + '{:03d}'.format(dset_ind)] = dset.ref

    return h5_group


def copy_dataset(h5_orig_dset, h5_dest_grp, alias=None, verbose=False):
    """
    Copies the provided HDF5 dataset to the provided destination. This function
    is handy when needing to make copies of datasets to a different HDF5 file.
    Notes
    -----
    This function does NOT copy all linked objects such as ancillary
    datasets. Call `copy_linked_objects` to accomplish that goal.
    Parameters
    ----------
    h5_orig_dset : h5py.Dataset
    h5_dest_grp : h5py.Group or h5py.File object :
        Destination where the duplicate dataset will be created
    alias : str, optional. Default = name from `h5_orig_dset`:
        Name to be assigned to the copied dataset
    verbose : bool, optional. Default = False
        Whether or not to print logs to assist in debugging
    Returns
    -------
    """
    if not isinstance(h5_orig_dset, h5py.Dataset):
        raise TypeError("'h5_orig_dset' should be a h5py.Dataset object")
    if not isinstance(h5_dest_grp, (h5py.File, h5py.Group)):
        raise TypeError("'h5_dest_grp' should either be a h5py.File or "
                        "h5py.Group object")
    if alias is not None:
        validate_single_string_arg(alias, 'alias')
    else:
        alias = h5_orig_dset.name.split('/')[-1]

    if alias in h5_dest_grp.keys():
        if verbose:
            warn('{} already contains an object with the same name: {}'
                 ''.format(h5_dest_grp, alias))
        h5_new_dset = h5_dest_grp[alias]
        if not isinstance(h5_new_dset, h5py.Dataset):
            raise TypeError('{} already contains an object: {} with the desired'
                            ' name which is not a dataset'.format(h5_dest_grp,
                                                                  h5_new_dset))

        da_source = lazy_load_array(h5_orig_dset)
        da_dest = lazy_load_array(h5_new_dset)

        if da_source.shape != da_dest.shape:
            raise ValueError('Existing dataset: {} has a different shape '
                             'compared to the original dataset: {}'
                             ''.format(h5_new_dset, h5_orig_dset))
        if not da.allclose(da_source, da_dest):
            raise ValueError('Existing dataset: {} has different contents'
                             'compared to the original dataset: {}'
                             ''.format(h5_new_dset, h5_orig_dset))
    else:

        kwargs = {'shape': h5_orig_dset.shape,
                  'dtype': h5_orig_dset.dtype,
                  'compression': h5_orig_dset.compression,
                  'chunks': h5_orig_dset.chunks}
        if h5_orig_dset.file.driver == 'mpio':
            if kwargs.pop('compression', None) is not None:
                warn('This HDF5 file has been opened wth the '
                     '"mpio" communicator. mpi4py does not allow '
                     'creation of compressed datasets. Compression'
                     ' kwarg has been removed')
        if verbose:
            print('Creating new HDF5 dataset named: {} at: {} with'
                  ' kwargs: {}'.format(alias, h5_dest_grp,
                                       kwargs))
        h5_new_dset = h5_dest_grp.create_dataset(alias,
                                                 **kwargs)
        if verbose:
            print('dask.array will copy data from source dataset '
                  'to new dataset')
        da.to_hdf5(h5_new_dset.file.filename,
                   {h5_new_dset.name: lazy_load_array(h5_orig_dset)})
    if verbose:
        print('Copying simple attributes of original dataset: {} to '
              'destination dataset: {}'.format(h5_orig_dset, h5_new_dset))

    copy_attributes(h5_orig_dset, h5_new_dset, skip_refs=True)
    # TODO: reinstate copy all region_refs()
    # copy_all_region_refs(h5_orig_dset, h5_new_dset)

    return h5_new_dset


def copy_linked_objects(h5_source, h5_dest, verbose=False):
    """
    Recursively copies datasets linked to the source h5 object to the
    destination h5 object that are be in different HDF5 files.

    This is for copying ancillary datasets to a target dataset that is
    missing ancillary datasets. It is not meant for copying to a Group,
    but that is supported.
    Notes
    -----
    We anticipate this function being used to copy over ancillary datasets
    Parameters
    ----------
    h5_source : h5py.Dataset or h5py.Group object
        Source object
    h5_dest : h5py.Dataset or h5py.Group object
        Destination object
    verbose : bool, optional. Default: False
        Whether or not to print logs for debugging purposes
    """
    try:
        # The following line takes care of object validation
        validate_h5_objs_in_same_h5_file(h5_source, h5_dest)
        same_file = True
    except ValueError:
        same_file = False

    if same_file:
        warn('{} and {} are in the same HDF5 file. Consider copying references'
             ' instead of copying linked objects'.format(h5_source, h5_dest))
        return

    if isinstance(h5_dest, h5py.Group):
        h5_dest_grp = h5_dest
    else:
        h5_dest_grp = h5_dest.parent

    # Now we are working on other files
    for link_obj_name in h5_source.attrs.keys():
        h5_orig_obj = get_attr(h5_source, link_obj_name)
        if isinstance(h5_orig_obj, h5py.Reference) and not \
                isinstance(h5_orig_obj, h5py.RegionReference):
            h5_orig_obj = h5_source.file[h5_orig_obj]
            if verbose:
                print('Attempting to copy object linked to source: {} as {}'
                      ''.format(h5_orig_obj, link_obj_name))
            # Check to see if such a dataset already exist
            if link_obj_name in h5_dest_grp.keys():
                h5_new_obj = h5_dest_grp[link_obj_name]
                warn('An object with the same name: {} already exists in the '
                     'destination group: {}'.format(h5_new_obj, h5_dest_grp.name))
                if type(h5_dest_grp[link_obj_name]) != type(h5_orig_obj):
                    mesg = 'Destination parent: {} already has a child named' \
                           ' {} that is of type: {} which does not match ' \
                           'with that of the object linked with the source ' \
                           'dataset: {}'.format(h5_dest_grp, link_obj_name,
                                                type(h5_orig_obj),
                                                type(h5_new_obj))
                    raise TypeError(mesg)

                elif isinstance(h5_new_obj, h5py.Dataset):
                    _ = copy_dataset(h5_orig_obj, h5_dest_grp,
                                     alias=link_obj_name, verbose=verbose)
                    h5_dest.attrs[link_obj_name] = h5_new_obj.ref
                    continue
                elif isinstance(h5_new_obj, h5py.Group):
                    raise ValueError('Destination already contains another '
                                     'HDF5 group: {} with the same name as '
                                     'the source: {}'.format(h5_new_obj,
                                                             h5_orig_obj))
                else:
                    raise NotImplementedError('Unable to copy {} objects yet'
                                              '. Contact developer if you need'
                                              ' this'
                                              ''.format(type(h5_orig_obj)))
            else:
                if isinstance(h5_orig_obj, h5py.Dataset):
                    h5_new_obj = copy_dataset(h5_orig_obj, h5_dest_grp,
                                              alias=link_obj_name,
                                              verbose=verbose)
                    h5_dest.attrs[link_obj_name] = h5_new_obj.ref
                else:
                    raise NotImplementedError('Unable to copy {} objects yet'
                                              '. Contact developer if you need'
                                              ' this'.format(type(h5_orig_obj)))


def check_for_matching_attrs(h5_obj, new_parms=None, verbose=False):
    """
    Compares attributes in the given H5 object against those in the provided dictionary and returns True if
    the parameters match, and False otherwise
    Parameters
    ----------
    h5_obj : h5py object (Dataset or :class:`h5py.Group`)
        Object whose attributes will be compared against new_parms
    new_parms : dict, optional. default = empty dictionary
        Parameters to compare against the attributes present in h5_obj
    verbose : bool, optional, default = False
       Whether or not to print debugging statements
    Returns
    -------
    tests: bool
        Whether or not all paramters in new_parms matched with those in h5_obj's attributes
    """
    if not isinstance(h5_obj, (h5py.Dataset, h5py.Group, h5py.File)):
        raise TypeError('h5_obj should be a h5py.Dataset, h5py.Group, or h5py.File object')
    if new_parms is None:
        new_parms = dict()
    else:
        if not isinstance(new_parms, dict):
            raise TypeError('new_parms should be a dictionary')

    tests = []
    for key in new_parms.keys():

        if verbose:
            print('Looking for new attribute named: {}'.format(key))

        # HDF5 cannot store None as an attribute anyway. ignore
        if new_parms[key] is None:
            continue

        try:
            old_value = get_attr(h5_obj, key)
        except KeyError:
            # if parameter was not found assume that something has changed
            if verbose:
                print('New parm: {} \t- new parm not in group *****'.format(key))
            tests.append(False)
            break

        if isinstance(old_value, np.ndarray):
            if not isinstance(new_parms[key], collections.Iterable):
                if verbose:
                    print('New parm: {} \t- new parm not iterable unlike old parm *****'.format(key))
                tests.append(False)
                break
            new_array = np.array(new_parms[key])
            if old_value.size != new_array.size:
                if verbose:
                    print('New parm: {} \t- are of different sizes ****'.format(key))
                tests.append(False)
            else:
                try:
                    answer = np.allclose(old_value, new_array)
                except TypeError:
                    # comes here when comparing string arrays
                    # Not sure of a better way
                    answer = []
                    for old_val, new_val in zip(old_value, new_array):
                        answer.append(old_val == new_val)
                    answer = np.all(answer)
                if verbose:
                    print('New parm: {} \t- match: {}'.format(key, answer))
                tests.append(answer)
        else:
            """if isinstance(new_parms[key], collections.Iterable):
                if verbose:
                    print('New parm: {} \t- new parm is iterable unlike old parm *****'.format(key))
                tests.append(False)
                break"""
            answer = np.all(new_parms[key] == old_value)
            if verbose:
                print('New parm: {} \t- match: {}'.format(key, answer))
            tests.append(answer)
    if verbose:
        print('')

    return all(tests)
