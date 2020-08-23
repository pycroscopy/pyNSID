
import sys
import numpy as np

# Multidimensional Image library
import scipy.ndimage as ndimage
import scipy.constants as const
from scipy.interpolate import interp1d

import skimage
import skimage.registration as registration
from skimage.feature import register_translation
from scipy import fftpack


def get_wave_length(acceleration_voltage):
    """
    Calculates the relativistic corrected de Broglie wave length of an electron

    Input: float
    ------
        acceleration voltage in volt
    Output: float
    -------
        wave length in 1/nm
    """

    eV = const.e * acceleration_voltage
    return const.h/np.sqrt(2*const.m_e*eV*(1+eV/(2*const.m_e*const.c**2)))*10**9


def rebin(im, binning=2):
    """
    rebin an image by the number of pixels in x and y direction given by binning

    Input: numpy array
    ======
            image: numpy array in 2 dimensions

    Output: numpy array
    =======
            binned image
    """
    if len(im.shape) == 2:
        return im.reshape((im.shape[0]//binning, binning, im.shape[1]//binning, binning)).mean(axis=3).mean(1)
    else:
        raise ValueError('not a 2D image')


def cart2pol(points):
    rho = np.linalg.norm(points[:, 0:2], axis=1)
    phi = np.arctan2(points[:, 1], points[:, 0])
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def xy2polar(points, rounding=1e-3):
    """
    Conversion from carthesian to polar coordinates

    the angles and distances are sorted by r and then phi
    The indices of this sort is also returned

    Input:
        points: numpy array with number of points in axis 0 first two elements in axis 1 are x and y
        rounding float, optional:   in significant digits

    returns r,phi, sorted_indices
    """

    r, phi = cart2pol(points)

    phi = phi - phi.min()  # only positive angles
    r = (np.floor(r / rounding)) * rounding  # Remove rounding error differences

    sorted_indices = np.lexsort((phi, r))  # sort first by r and then by phi
    r = r[sorted_indices]
    phi = phi[sorted_indices]

    return r, phi, sorted_indices


def cartesian2polar(x, y, grid, r, t, order=3):
    rr, tt = np.meshgrid(r, t)

    new_x = rr * np.cos(tt)
    new_y = rr * np.sin(tt)

    ix = interp1d(x, np.arange(len(x)))
    iy = interp1d(y, np.arange(len(y)))

    new_ix = ix(new_x.ravel())
    new_iy = iy(new_y.ravel())

    return ndimage.map_coordinates(grid, np.array([new_ix, new_iy]),
                                   order=order).reshape(new_x.shape)


def warp(diff, center):
    # Define original polar grid
    nx = diff.shape[0]
    ny = diff.shape[1]

    x = np.linspace(1, nx, nx, endpoint=True) - center[1]
    y = np.linspace(1, ny, ny, endpoint=True) - center[0]
    z = np.abs(diff)

    # Define new polar grid
    nr = min([center[0], center[1], diff.shape[0] - center[0], diff.shape[1] - center[1]]) - 1
    nt = 360 * 3

    r = np.linspace(1, nr, nr)
    t = np.linspace(0., np.pi, nt, endpoint=False)
    return cartesian2polar(x, y, z, r, t, order=3).T


def calculate_ctf(wave_length, Cs, defocus, k):
    """ Calculate Contrast Transfer Function
    everything in nm
    """
    ctf = np.sin(np.pi * defocus * wave_length * k**2 + 0.5 * np.pi * Cs * wave_length**3 * k**4)
    return ctf


def calculate_Scherzer(wave_length, Cs):
    """
    Calculate the Scherzer defocus. Cs is in mm, lambda is in nm
    # EInput and output in nm
    """
    scherzer = -1.155 * (Cs * wave_length) ** 0.5  # in m
    return scherzer


def rigid_registration(stack):
    """
    Rigid registration of image stack with sub-pixel accuracy
    used phase_cross_correlation from skimage.registration
    (we determine drift from one image to next)

    Input:
        stack: hdf5 group
            image_stack dataset

    Output:
        Registered Stack: numpy array
        drift: numpy array (shape number of images , 2)
            with respect to center image
    """

    nopix = stack.shape[1]
    nopiy = stack.shape[2]
    nimages = stack.shape[0]

    print('Stack contains ', nimages, ' images, each with', nopix, ' pixels in the x-direction and ', nopiy,
          ' pixels in the y-direction')
    fixed = stack[0]
    fft_fixed = np.fft.fft2(fixed)

    relative_drift = [[0., 0.]]
    done = 0

    if QT_available:
        progress = QtWidgets.QProgressDialog("Rigid Registration.", "Abort", 0, nimages)
        progress.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        # progress.setWindowModality(Qt.WindowModal);
        progress.show()

    for i in range(nimages):

        if QT_available:
            progress.setValue(i)
            Qt.QApplication.processEvents()
        else:
            if done < int((i + 1) / nimages * 50):
                done = int((i + 1) / nimages * 50)
                sys.stdout.write('\r')
                # progress output :
                sys.stdout.write("[%-50s] %d%%" % ('*' * done, 2 * done))
                sys.stdout.flush()

        moving = stack[i]
        fft_moving = np.fft.fft2(moving)
        if skimage.__version__[:4] == '0.16':
            shift = register_translation(fft_fixed, fft_moving, upsample_factor=1000, space='fourier')
        else:
            shift = registration.phase_cross_correlation(fft_fixed, fft_moving, upsample_factor=1000, space='fourier')

        fft_fixed = fft_moving
        # print(f'Image number {i:2}  xshift =  {shift[0][0]:6.3f}  y-shift =  {shift[0][1]:6.3f}')

        relative_drift.append(shift[0])
    if QT_available:
        progress.setValue(nimages)
    rig_reg, drift = rig_reg_drift(stack, relative_drift)

    return rig_reg, drift


def rig_reg_drift(dset, rel_drift):
    """
    Uses relative drift to shift images on top of each other
    Shifting is done with shift routine of ndimage from scipy

    is used by Rigid_Registration routine

    Input image_channel with image_stack numpy array
    relative_drift from image to image as list of [shiftx, shifty]

    output stack and drift
    """

    rig_reg = np.zeros(dset.shape)
    # absolute drift
    drift = np.array(rel_drift).copy()

    drift[0] = [0, 0]
    for i in range(drift.shape[0]):
        drift[i] = drift[i - 1] + rel_drift[i]
    center_drift = drift[int(drift.shape[0] / 2)]
    drift = drift - center_drift
    # Shift images
    for i in range(rig_reg.shape[0]):
        # Now we shift
        rig_reg[i, :, :] = ndimage.shift(dset[i], [drift[i, 0], drift[i, 1]], order=3)
    return rig_reg, drift


def crop_image_stack(rig_reg, drift):
    """
    ## Crop images
    """
    xpmin = int(-np.floor(np.min(np.array(drift)[:, 0])))
    xpmax = int(rig_reg.shape[1] - np.ceil(np.max(np.array(drift)[:, 0])))
    ypmin = int(-np.floor(np.min(np.array(drift)[:, 1])))
    ypmax = int(rig_reg.shape[2] - np.ceil(np.max(np.array(drift)[:, 1])))

    return rig_reg[:, xpmin:xpmax, ypmin:ypmax], [xpmin, xpmax, ypmin, ypmax]


def decon_LR(image, probe, verbose=False):
    """
    # This task generates a restored image from an input image and point spread function (PSF) using the algorithm
    # developed independently by Lucy (1974, Astron. J. 79, 745) and Richardson (1972, J. Opt. Soc. Am. 62, 55) and
    # adapted for HST imagery by Snyder (1990, in Restoration of HST Images and Spectra, ST ScI Workshop Proceedings;
    # see also Snyder, Hammoud, & White, JOSA, v. 10, no. 5, May 1993, in press).
    # Additional options developed by Rick White (STScI) are also included.
    #
    # The Lucy-Richardson method can be derived from the maximum likelihood expression for data with a Poisson noise
    # distribution. Thus, it naturally applies to optical imaging data such as HST. The method forces the restored
    # image to be positive, in accord with photon-counting statistics.
    #
    # The Lucy-Richardson algorithm generates a restored image through an iterative method. The essence of the
    # iteration is as follows: the (n+1)th estimate of the restored image is given by the nth estimate of the restored
    # image multiplied by a correction image. That is,
    #
    #                            original data
    #       image    = image    ---------------  * reflect(PSF)
    #            n+1        n     image * PSF
    #                                  n

    # where the *'s represent convolution operators and reflect(PSF) is the reflection of the PSF, i.e.
    # reflect((PSF)(x,y)) = PSF(-x,-y). When the convolutions are carried out using fast Fourier transforms (FFTs),
    # one can use the fact that FFT(reflect(PSF)) = conj(FFT(PSF)), where conj is the complex conjugate operator.
    """

    if len(image) < 1:
        return image

    if image.shape != probe.shape:
        print('Weirdness ', image.shape, ' != ', probe.shape)

    probe_c = np.ones(probe.shape, dtype=np.complex64)
    probe_c.real = probe

    error = np.ones(image.shape, dtype=np.complex64)
    est = np.ones(image.shape, dtype=np.complex64)
    source = np.ones((image.shape), dtype=np.complex64)
    source.real = image

    response_ft = fftpack.fft2(probe_c)

    dE = 100
    dest = 100
    i = 0
    while abs(dest) > 0.0001:  # or abs(dE)  > .025:
        i += 1

        error_old = np.sum(error.real)
        est_old = est.copy()
        error = source / np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(est) * response_ft)))
        est = est * np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(error) * np.conjugate(response_ft))))

        error_new = np.real(np.sum(np.power(error, 2))) - error_old
        dest = np.sum(np.power((est - est_old).real, 2)) / np.sum(est) * 100

        if error_old != 0:
            dE = error_new / error_old * 1.0
        else:
            dE = error_new

        if verbose:
            if verbose:
                print(' LR Deconvolution - Iteration: {0:d} Error: {1:.2f} = change: {2:.5f}%, {3:.5f}%'
                      .format(i, error_new, dE, abs(dest)))
        if i > 1000:
            dE = 0.0
            dest = 0.0
            print('terminate')

    print('\n Lucy-Richardson deconvolution converged in ' + str(i) + '  Iterations')

    return est

##########################################
# Functions Used
##########################################


def MakeProbeG(sizeX, sizeY, widthi, xi, yi):
    sizeX = (sizeX / 2)
    sizeY = (sizeY / 2)
    width = 2 * widthi ** 2
    x, y = np.mgrid[-sizeX:sizeX, -sizeY:sizeY]
    g = np.exp(-((x - xi) ** 2 / float(width) + (y - yi) ** 2 / float(width)))
    probe = g / g.sum()

    return probe


def MakeLorentz(sizeX, sizeY, width, xi, yi):
    sizeX = np.floor(sizeX / 2)
    sizeY = np.floor(sizeY / 2)
    gamma = width
    x, y = np.mgrid[-sizeX:sizeX, -sizeY:sizeY]
    g = gamma / (2 * np.pi) / np.power(((x - xi) ** 2 + (y - yi) ** 2 + gamma ** 2), 1.5)
    probe = g / g.sum()

    return probe
