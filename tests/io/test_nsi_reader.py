from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sys
import unittest
from typing import Type, Tuple

import h5py
import numpy as np

sys.path.append("../pyNSID/")

from pyNSID.io.hdf_io import write_nsid_dataset
from pyNSID.io.hdf_utils import find_dataset
from pyNSID.io.nsi_reader import NSIDReader
from sidpy import Dataset, Dimension



def create_h5group(h5f_name: str, h5g_name: str) -> Type[h5py.Group]:
    mode = 'r+' if os.path.exists(h5f_name) else 'w'
    h5_f = h5py.File(h5f_name, mode)
    h5_group = h5_f.create_group(h5g_name)
    return h5_group


def write_dummy_dset(hf_group: Type[h5py.Group], dims: Tuple[int],
                     main_name: str, **kwargs) -> None:
    dnames = kwargs.get("dnames", np.arange(len(dims)))
    dset = Dataset.from_array(
            np.random.random([*dims]), name="new")
    dnames = kwargs.get("dnames", np.arange(len(dims)))
    for i, d in enumerate(dims):
        dset.set_dimension(i, Dimension(np.arange(d), str(dnames[i])))
    write_nsid_dataset(
        dset, hf_group, main_data_name=main_name)


def get_dset(hf_path: str, dset_name: str) -> Type[h5py.Dataset]:
    hf = h5py.File(hf_path, 'r')
    dset = find_dataset(hf, dset_name)[0]
    return dset


class test_nsid_reader(unittest.TestCase):

    def test_can_read(self) -> None:
        hf_name = "test.hdf5"
        self.tearDown()
        h5group_1 = create_h5group(hf_name, "group1")
        h5group_2 = create_h5group(hf_name, "group2")
        write_dummy_dset(h5group_2, (10, 10, 5), "dset")
        self.assertFalse(NSIDReader(h5group_1).can_read())
        self.assertTrue(NSIDReader(h5group_2).can_read())

    def test_read_single(self) -> None:
        hf_name = "test.hdf5"
        self.tearDown()
        h5group = create_h5group(hf_name, "group")
        write_dummy_dset(h5group, (10, 10, 5), "dset")
        dset = get_dset(hf_name, "dset")
        reader = NSIDReader(h5group)
        d1 = reader.read(dset)
        d2 = reader.read()
        self.assertTrue(isinstance(d1, Dataset))
        self.assertTrue(isinstance(d2, list))
        self.assertTrue(isinstance(d2[0], Dataset))

    def test_read_multi(self) -> None:
        hf_name = "test.hdf5"
        self.tearDown()
        h5group = create_h5group(hf_name, "group")
        for i in range(3):
            write_dummy_dset(
                h5group, (10, 10, 5+i),
                "dset{}".format(i),
                dnames=np.arange(3*i, 3*(i+1)))
        reader = NSIDReader(h5group)
        d_all = reader.read()
        self.assertTrue(isinstance(d_all, list))
        self.assertEqual(len(d_all), 3)
        self.assertEqual(sum([1 for d in d_all if isinstance(d, Dataset)]), 3)
        for i in range(3):
            dset_i = get_dset(hf_name, "dset{}".format(i))
            d_i = reader.read(dset_i)
            self.assertEqual(d_i.shape[-1], 5+i)

    def test_read_all(self) -> None:
        hf_name = "test.hdf5"
        self.tearDown()
        h5group_1 = create_h5group(hf_name, "group1")
        h5group_2 = create_h5group(hf_name, "group2")
        # Write multiple datasets to the first group
        for i in range(5):
            write_dummy_dset(
                h5group_1, (10, 10, 5),
                "dset{}".format(i),
                dnames=np.arange(3*i, 3*(i+1)))
        # write a single dataset to the second group
        write_dummy_dset(h5group_2, (7, 7, 10), "dset")
        # initialize and test reader
        reader = NSIDReader(h5group_1)
        d_all = reader.read_all(recursive=True)
        self.assertEqual(len(d_all), 5)
        self.assertEqual(sum([1 for d in d_all if isinstance(d, Dataset)]), 5)
        d = reader.read_all(recursive=True, parent=h5group_2)
        self.assertEqual(len(d), 1)
        self.assertTrue(isinstance(d[0], Dataset))

    def tearDown(self, fname: str = 'test.hdf5') -> None:
        if os.path.exists(fname):
            os.remove(fname)