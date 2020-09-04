
def usid_to_sid(h5_raw, verbose=False):
    """
    converts a usid dataset into a sipy dataset

    Input:
        h5_raw: USID main dataset
        verbose: boolean
    Output:
        sidpy dataset
    """

    if not usid.hdf_utils.check_if_main(h5_raw):
        raise TypeError('This function needs to be provided with a main USID dataset ')

    pd_raw = usid.USIDataset(h5_raw)

    sid_dataset = sidpy.Dataset.from_array(pd_raw.get_n_dim_form())

    descriptor = pd_raw.data_descriptor.split('(')
    sid_dataset.quantity = descriptor[0]
    sid_dataset.units = descriptor[1][:-1]

    for dim, dim_label in enumerate(pd_raw.n_dim_labels):
        if verbose:
            print(dim_label)
        pos_dim = 0
        spec_dim = 0
        if dim_label in pd_raw.pos_dim_labels:
            dim_type = 'spatial'
            dim_values = pd_raw.get_pos_values(dim_label)
            descriptor = pd_raw.pos_dim_descriptors[pos_dim].split('(')
            dim_quantity = descriptor[0]
            dim_units = descriptor[1][:-1]
            pos_dim += 1
        else:
            dim_type = 'spectral'
            dim_values = pd_raw.get_spec_values(dim_label)
            descriptor = pd_raw.spec_dim_descriptors[spec_dim].split('(')
            dim_quantity = descriptor[0]
            dim_units = descriptor[1][:-1]
            spec_dim += 1
        if verbose:
            print('Read dimension {} of  type {} as {} ({})'.format(dim_label, dim_type,
                                                                    dim_quantity, dim_units
                                                                    ))

        sid_dataset.set_dimension(dim, sidpy.Dimension(dim_label, dim_values,
                                                       units=dim_units,
                                                       quantity=dim_quantity,
                                                       dimension_type=dim_type))

    return sid_dataset
