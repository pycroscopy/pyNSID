from __future__ import division, print_function, unicode_literals, absolute_import
from typing import Tuple, Dict
import unittest
import os
import sys
import h5py
import numpy as np
import dask.array as da
import tempfile
import sidpy
from sidpy import Dataset, Dimension
from sidpy.hdf.hdf_utils import write_simple_attrs
flatten_dict = sidpy.dict_utils.flatten_dict

sys.path.append("../../")
from pyNSID.io.hdf_utils import find_dataset, read_h5py_dataset, \
    get_all_main, link_as_main, check_if_main


def make_simple_h5_dataset():
    """
    simple h5 dataset with dimesnsion arrays but not attached
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = tmp_dir + 'test.h5'
    h5_file = h5py.File(file_path, 'a')
    h5_group = h5_file.create_group('MyGroup')
    data = np.random.normal(size=(2, 3))
    h5_dataset = h5_group.create_dataset('data', data=data)

    dims = {0: h5_group.create_dataset('a', np.arange(data.shape[0])),
            1: h5_group.create_dataset('b', np.arange(data.shape[1]))}
    return h5_file


def make_simple_nsid_dataset(*args, **kwargs):
    """
    h5 dataset which is fully pyNSID compatible
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = tmp_dir + 'nsid_simple.h5'
    h5_file = h5py.File(file_path, 'a')
    h5_group = h5_file.create_group('MyGroup')

    dsetnames = kwargs.get("dsetnames", ['data'])
    dsetshapes = kwargs.get("dsetshapes")
    if dsetshapes is None:
        dsetshapes = [(2, 3) for i in range(len(dsetnames))]
    for i, d in enumerate(dsetnames):
        data = np.random.normal(size=dsetshapes[i])
        h5_dataset = h5_group.create_dataset(d, data=data)
        
        attrs_to_write = {'quantity': 'quantity',
                          'units': 'units',
                          'pyNSID_version': 'version',
                          'main_data_name': 'title',
                          'data_type': 'UNKNOWN',
                          'modality': 'modality',
                          'source': 'test'}
        if len(args) > 0:
            for k, v in args[0].items():
                if k in attrs_to_write:
                    attrs_to_write[k] = v

        write_simple_attrs(h5_dataset, attrs_to_write)

        dims = {0: h5_group.create_dataset("a{}".format(i), data=np.arange(data.shape[0])),
                1: h5_group.create_dataset("b{}".format(i), data=np.arange(data.shape[1]))}
        for dim, this_dim_dset in dims.items():
            name = this_dim_dset.name.split('/')[-1]
            attrs_to_write = {'name': name, 'units': 'units', 'quantity': 'quantity',
                            'dimension_type': 'dimension_type.name', 'nsid_version': 'test'}

            write_simple_attrs(this_dim_dset, attrs_to_write)

            this_dim_dset.make_scale(name)
            h5_dataset.dims[dim].label = name
            h5_dataset.dims[dim].attach_scale(this_dim_dset)
    return h5_file


def make_nsid_dataset_no_dim_attached():
    """
    except for the dimensions attached the h5 dataset which is fully pyNSID compatible
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = tmp_dir + 'nsid_simple.h5'
    h5_file = h5py.File(file_path, 'a')
    h5_group = h5_file.create_group('MyGroup')
    data = np.random.normal(size=(2, 3))
    h5_dataset = h5_group.create_dataset('data', data=data)
    attrs_to_write = {'quantity': 'quantity', 'units': 'units', 'nsid_version': 'version',
                      'main_data_name': 'title', 'data_type': 'UNKNOWN',
                      'modality': 'modality', 'source': 'test'}

    write_simple_attrs(h5_dataset, attrs_to_write)

    dims = {0: h5_group.create_dataset('a', data=np.arange(data.shape[0])),
            1: h5_group.create_dataset('b', data=np.arange(data.shape[1]))}
    for dim, this_dim_dset in dims.items():
        name = this_dim_dset.name.split('/')[-1]
        attrs_to_write = {'name': name, 'units': 'units', 'quantity': 'quantity',
                          'dimension_type': 'dimension_type.name', 'nsid_version': 'test'}

        write_simple_attrs(this_dim_dset, attrs_to_write)

        this_dim_dset.make_scale(name)
        h5_dataset.dims[dim].label = name
        # h5_dataset.dims[dim].attach_scale(this_dim_dset)
    return h5_file


def make_nsid_length_dim_wrong():
    """
    h5 dataset which is fully pyNSID compatible
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = tmp_dir + 'test.h5'
    h5_file = h5py.File(file_path, 'a')
    h5_group = h5_file.create_group('MyGroup')
    data = np.random.normal(size=(2, 3))
    h5_dataset = h5_group.create_dataset('data', data=data)
    attrs_to_write = {'quantity': 'quantity', 'units': 'units', 'nsid_version': 'version',
                      'main_data_name': 'title', 'data_type': 'data_type.name',
                      'modality': 'modality', 'source': 'test'}

    write_simple_attrs(h5_dataset, attrs_to_write)

    dims = {0: h5_group.create_dataset('a', data=np.arange(data.shape[0]*3)),
            1: h5_group.create_dataset('b', data=np.arange(data.shape[1]))}
    for dim, this_dim_dset in dims.items():
        name = this_dim_dset.name.split('/')[-1]
        attrs_to_write = {'name': name, 'units': 'units', 'quantity': 'quantity',
                          'dimension_type': 'dimension_type.name', 'nsid_version': 'test'}

        write_simple_attrs(this_dim_dset, attrs_to_write)

        this_dim_dset.make_scale(name)
        h5_dataset.dims[dim].label = name
        h5_dataset.dims[dim].attach_scale(this_dim_dset)
    return h5_file


def get_dim_dict(dims: Tuple[int]
                 ) -> Dict[int, h5py.Dataset]:
    h5_f = h5py.File('test2.h5', 'a')
    h5_group = h5_f.create_group('MyGroup2')
    dim_dict = {}
    names = ['X', 'Y', 'Z', 'F']
    for i, d in enumerate(dims):
        dim_dict[i] = h5_group.create_dataset(names[i], data=np.arange(d))
    for dim, this_dim_dset in dim_dict.items():
        name = this_dim_dset.name.split('/')[-1]
        attrs_to_write = {'name': name, 'units': 'units', 'quantity': 'quantity',
                          'dimension_type': 'dimension_type.name', 'nsid_version': 'test'}
        write_simple_attrs(this_dim_dset, attrs_to_write)
    return dim_dict


class TestReadH5pyDataset(unittest.TestCase):

    def test_wrong_input_type(self) -> None:
        dataset = make_simple_h5_dataset()
        err_msg = 'can only read single Dataset'
        with self.assertRaises(TypeError) as context:
            _ = read_h5py_dataset(dataset)
        self.assertTrue(err_msg in str(context.exception))

    def test_valid_nsid_h5_dset(self):
        base = {'quantity': 'Current',
                'units': 'nA',
                # 'pyNSID_version': 'version',
                'title': 'Current-Voltage spectroscopy measurement',
                'data_type': 'SPECTRAL_IMAGE',
                'modality': 'cAFM',
                'source': 'Asylum Research Cypher'}
        h5file = make_simple_nsid_dataset(base)
        dset = read_h5py_dataset(h5file['MyGroup']['data'])
        self.assertIsInstance(dset, sidpy.Dataset)
        # Validate the base attributes first
        for key, expected in base.items():
            actual = getattr(dset, key)

        # Validate the dimensions

        # Validate the main dataset itself

    def test_attrs_title(self) -> None:
        """
        _meta = {'title': 'new_name'}
        h5file = make_simple_nsid_dataset(_meta)
        dset = read_h5py_dataset(h5file['MyGroup']['data'])
        self.assertIsInstance(dset, sidpy.Dataset)
        self.assertTrue(dset.title == 'new_name')
        """
        pass

    def test_attrs_units(self) -> None:
        _meta = {"units": "nA"}
        h5file = make_simple_nsid_dataset(_meta)
        dset = read_h5py_dataset(h5file['MyGroup']['data'])
        self.assertIsInstance(dset, sidpy.Dataset)
        self.assertTrue(dset.units == 'nA')

    def test_attrs_quantity(self):
        _meta = {"quantity": "Current"}
        h5file = make_simple_nsid_dataset(_meta)
        dset = read_h5py_dataset(h5file['MyGroup']['data'])
        self.assertIsInstance(dset, sidpy.Dataset)
        self.assertTrue(dset.quantity == 'Current')

    def test_attrs_datatype(self):
        data_types = ['UNKNOWN', 'SPECTRUM', 'LINE_PLOT', 'LINE_PLOT_FAMILY',
                      'IMAGE', 'IMAGE_MAP', 'IMAGE_STACK', 'SPECTRAL_IMAGE',
                      'IMAGE_4D']
        for dt in data_types:
            _meta = {"data_type": dt}
        h5file = make_simple_nsid_dataset(_meta)
        dset = read_h5py_dataset(h5file['MyGroup']['data'])
        self.assertIsInstance(dset, sidpy.Dataset)
        self.assertTrue(dset.data_type.name == dt)

    def test_attrs_modality(self) -> None:
        _meta = {"modality": "mod"}
        h5file = make_simple_nsid_dataset(_meta)
        dset = read_h5py_dataset(h5file['MyGroup']['data'])
        self.assertIsInstance(dset, sidpy.Dataset)
        self.assertTrue(dset.modality == 'mod')

    def test_attrs_source(self) -> None:
        _meta = {"source": "src"}
        h5file = make_simple_nsid_dataset(_meta)
        dset = read_h5py_dataset(h5file['MyGroup']['data'])
        self.assertIsInstance(dset, sidpy.Dataset)
        self.assertTrue(dset.source == 'src')

    def test_dims(self) -> None:
        h5file = make_simple_nsid_dataset()
        dset = read_h5py_dataset(h5file['MyGroup']['data'])
        self.assertIsInstance(dset, sidpy.Dataset)
        self.assertTrue(dset._axes[0].name == 'a0')
        self.assertTrue(dset._axes[1].name == 'b0')

    def tearDown(self, fname: str = 'test.h5') -> None:
        if os.path.exists(fname):
            os.remove(fname)


class TestGetAllMain(unittest.TestCase):

    def test_invalid_input(self):
        dset = np.random.randn(5, 10, 10)
        err_msg = "parent should be a h5py.File or h5py.Group object"
        with self.assertRaises(TypeError) as context:
            _ = get_all_main(dset)
        self.assertTrue(err_msg in str(context.exception))

    def test_h5_file_instead_of_group(self):
        h5file = make_simple_nsid_dataset()
        dset_list = get_all_main(h5file)
        self.assertTrue(isinstance(dset_list, list))
        self.assertTrue(isinstance(dset_list[0], h5py.Dataset))
        self.assertTrue(isinstance(dset_list[0][()], np.ndarray))

    def test_one_main_dataset(self):
        h5file = make_simple_nsid_dataset()
        h5group = h5file['MyGroup']
        dset_list = get_all_main(h5group)
        self.assertTrue(isinstance(dset_list, list))
        self.assertTrue(isinstance(dset_list[0], h5py.Dataset))
        self.assertTrue(isinstance(dset_list[0][()], np.ndarray))

    def test_multiple_main_dsets_in_same_group(self):
        h5file = make_simple_nsid_dataset(dsetnames=['data1', 'data2'])
        h5group = h5file['MyGroup']
        dset_list = get_all_main(h5group)
        self.assertTrue(isinstance(dset_list, list))
        self.assertEqual(len(dset_list), 2)
        for i, dset in enumerate(dset_list):
            self.assertTrue(isinstance(dset, h5py.Dataset))
            self.assertTrue(isinstance(dset[()], np.ndarray))

    def test_multiple_main_dsets_in_diff_nested_groups(self):
        pass

    def tearDown(self, fname: str = 'test.h5') -> None:
        if os.path.exists(fname):
            os.remove(fname)


class TestFindDataset(unittest.TestCase):
    # This function inherits a good portion of the code from sidpy.
    # We don't yet have the functionality to upconvert to sidpy.Dataset yet

    def test_one_not_main_dataset(self):
        # should return as h5py.Dataset
        pass

    def test_one_is_main_dataset(self):
        # should return the dataset as a sidpy.Dataset object
        pass

    def test_one_is_main_other_is_not(self):
        # Should return one as sidpy.Dataset and the other as h5py.Dataset
        pass


class TestCheckIfMain(unittest.TestCase):

    def setUp(self) -> None:
        self.h5_simple_file = make_simple_h5_dataset()
        self.h5_nsid_simple = make_simple_nsid_dataset()

    def test_not_h5_dataset(self):

        for arg in [np.arange(3),
                    da.from_array(np.arange(3)),
                    self.h5_simple_file['MyGroup'],
                    self.h5_simple_file,
                    ]:
            self.assertFalse(check_if_main(arg))

    def test_dims_missing(self):
        self.assertFalse(check_if_main(self.h5_simple_file['MyGroup']['data']))

    def test_dim_exist_but_scales_not_attached_to_main(self):
        h5_nsid_no_dim_attached = make_nsid_dataset_no_dim_attached()
        self.assertFalse(check_if_main(h5_nsid_no_dim_attached['MyGroup']['data']))

    def test_dim_sizes_not_matching_main(self):
        h5_nsid_wrong_dim_length = make_nsid_length_dim_wrong()
        self.assertFalse(check_if_main(h5_nsid_wrong_dim_length['MyGroup']['data']))

    def test_mandatory_attrs_not_present(self):
        for key in  ['quantity', 'units', 'main_data_name', 'data_type', 'modality', 'source']:
            attribute = self.h5_nsid_simple['MyGroup']['data'].attrs[key]

            del self.h5_nsid_simple['MyGroup']['data'].attrs[key]
            self.assertFalse(check_if_main(self.h5_nsid_simple['MyGroup']['data']))
            self.h5_nsid_simple['MyGroup']['data'].attrs[key] = attribute

    def test_invalid_types_for_str_attrs(self):
        for key in ['quantity', 'units', 'main_data_name', 'data_type', 'modality', 'source']:
            attribute = self.h5_nsid_simple['MyGroup']['data'].attrs[key]
            self.h5_nsid_simple['MyGroup']['data'].attrs[key] = 1
            self.assertFalse(check_if_main(self.h5_nsid_simple['MyGroup']['data']))
            self.h5_nsid_simple['MyGroup']['data'].attrs[key] = attribute

    def test_dset_is_main(self):
        self.assertTrue(check_if_main(self.h5_nsid_simple['MyGroup']['data']))


class TestLinkAsMain(unittest.TestCase):
    # Perhaps this function could call the validate function
    # So some of these tests could actually be in the validate function

    def test_all_dims_in_mem(self):
        # All dimensions are sidpy.Dimension objects
        pass

    def test_all_dims_already_h5_datasets(self):
        pass

    # Need clarification on whether this is supposed to throw an error or not
    #def test_dims_and_h5_main_in_diff_files(self):
    #    self.tearDown()
    #    dims = (2, 3)
    #    h5file = make_simple_nsid_dataset(dsetshapes=[dims])
    #    dset = h5file['MyGroup']['data']
    #    dim_dict = get_dim_dict(dims)
    #    linked = link_as_main(dset, dim_dict)
    #    assert_array_equal(linked.dims[0].values()[1][()], np.arange(2))
    #    assert_array_equal(linked.dims[1].values()[1][()], np.arange(3))

    def test_some_dims_in_mem_others_h5_dsets(self):
        pass

    def test_dim_in_mem_same_name_as_dim_in_h5(self):
        pass

    def test_dim_size_mismatch_main_shape(self):
        dims1 = (2, 3)
        dims2 = (3, 3)
        h5file = make_simple_nsid_dataset(dsetshapes=[dims1])
        dset = h5file['MyGroup']['data']
        dim_dict = get_dim_dict(dims2)
        err_msg = "Dimension 0 has the following error_message"
        with self.assertRaises(TypeError) as context:
            _ = link_as_main(dset, dim_dict)
        self.assertTrue(err_msg in str(context.exception))

    def test_too_many_dims(self):
        dims1 = (2, 3)
        dims2 = (1, 2, 3)
        h5file = make_simple_nsid_dataset(dsetshapes=[dims1])
        dset = h5file['MyGroup']['data']
        dim_dict = get_dim_dict(dims2)
        err_msg = "Incorrect number of dimensions"
        with self.assertRaises(ValueError) as context:
            _ = link_as_main(dset, dim_dict)
        self.assertTrue(err_msg in str(context.exception))

    def test_too_few_dims(self):
        dims1 = (2, 3)
        dims2 = (3,)
        h5file = make_simple_nsid_dataset(dsetshapes=[dims1])
        dset = h5file['MyGroup']['data']
        dim_dict = get_dim_dict(dims2)
        err_msg = "Incorrect number of dimensions"
        with self.assertRaises(ValueError) as context:
            _ = link_as_main(dset, dim_dict)
        self.assertTrue(err_msg in str(context.exception))

    def test_h5_main_invalid_object_type(self):
        dims = (2, 3)
        dataset = Dataset.from_array(np.random.random([*dims]), name="new")
        dim_dict = get_dim_dict(dims)
        err_msg = "h5_main should be a h5py.Dataset object"
        with self.assertRaises(TypeError) as context:
            _ = link_as_main(dataset, dim_dict)
        self.assertTrue(err_msg in str(context.exception))

    def test_dim_dict_invalid_obj_type(self):
        dims = (2, 3)
        h5file = make_simple_nsid_dataset(dsetshapes=[dims])
        dset = h5file['MyGroup']['data']
        dim_dict = [Dimension(np.arange(2), 'X'),
                    Dimension(np.arange(3), 'Y')]
        err_msg = 'dim_dict must be a dictionary'
        with self.assertRaises(TypeError) as context:
            _ = link_as_main(dset, dim_dict)
        self.assertTrue(err_msg in str(context.exception))

    def test_items_in_dim_dict_invalid_obj_type(self):
        dims = (2, 3)
        h5file = make_simple_nsid_dataset(dsetshapes=[dims])
        dset = h5file['MyGroup']['data']
        dim_dict = {0: Dimension(np.arange(2), 'X'),
                    1: Dimension(np.arange(3), 'Y')}
        err_msg = 'Items in dictionary must all  be h5py.Datasets !'
        with self.assertRaises(TypeError) as context:
            _ = link_as_main(dset, dim_dict)
        self.assertTrue(err_msg in str(context.exception))

    def tearDown(self) -> None:
        for fname in ['test.h5', 'test2.h5']:
            if os.path.exists(fname):
                os.remove(fname)


class TestValidateMainDset(unittest.TestCase):

    def base_test(self):
        pass

    def test_imporper_h5_main_type(self):
        pass

    def test_must_be_h5_but_is_not(self):
        pass


if __name__ == '__main__':
    unittest.main()