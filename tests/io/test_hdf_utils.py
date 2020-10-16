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

    def test_ordinary_dataset(self):
        pass

    def test_multiple_datasets_found(self):
        pass

    def test_one_result_is_main_other_is_not(self):
        pass


class TestValidateMainDset(unittest.TestCase):

    def base_test(self):
        pass

    def test_imporper_h5_main_type(self):
        pass

    def test_must_be_h5_but_is_not(self):
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

    def blah(self):
        pass


if __name__ == '__main__':
    unittest.main()
