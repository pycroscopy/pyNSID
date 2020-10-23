from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
sys.path.append("../pyNSID/")


class TestGetAllMain(unittest.TestCase):

    def test_not_invalid_input(self):
        # Consider passing numpy arrays, HDF5 datasets
        # Expect TypeErrors
        pass

    def test_h5_file_instead_of_group(self):
        pass

    def test_one_main_dataset(self):
        pass

    def test_multiple_main_dsets_in_same_group(self):
        pass

    def test_multiple_main_dsets_in_diff_nested_groups(self):
        pass


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

    def test_not_h5_dataset(self):
        pass

    def test_dims_missing(self):
        pass

    def test_dim_exist_but_scales_not_attached_to_main(self):
        pass

    def test_dim_sizes_not_matching_main(self):
        pass

    def test_mandatory_attrs_not_present(self):
        pass

    def test_invalid_types_for_str_attrs(self):
        pass

    def test_dset_is_main(self):
        pass


class TestLinkAsMain(unittest.TestCase):
    # Perhaps this function could call the validate function
    # So some of these tests could actually be in the validate function

    def test_all_dims_in_mem(self):
        # All dimensions are sidpy.Dimension objects
        pass

    def test_all_dims_already_h5_datasets(self):
        pass

    def test_dims_and_h5_main_in_diff_files(self):
        pass

    def test_some_dims_in_mem_others_h5_dsets(self):
        pass

    def test_dim_in_mem_same_name_as_dim_in_h5(self):
        pass

    def test_dim_size_mismatch_main_shape(self):
        pass

    def test_too_few_dims(self):
        pass

    def test_too_many_dims(self):
        pass

    def test_h5_main_invalid_object_type(self):
        pass

    def test_dim_dict_invalid_obj_type(self):
        pass

    def test_items_in_dim_dict_invalid_obj_type(self):
        pass


class TestValidateMainDset(unittest.TestCase):

    def base_test(self):
        pass

    def test_imporper_h5_main_type(self):
        pass

    def test_must_be_h5_but_is_not(self):
        pass


if __name__ == '__main__':
    unittest.main()
