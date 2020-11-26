from scipy.signal import detrend, fftconvolve
import numpy as np


def apply_normxcorr2(sub_tile):
    """" Mahmoud Al Najar
    :param sub_tile: (cross_shore, long_shore, bands)
    :return img: subtile once normxcorr applied
    """

    def normxcorr2(template, image, mode="full"):
        """ Mahmoud Al Najar
        Input arrays should be floating point numbers.
        :param template: N-D array, of template or filter you are using for cross-correlation.
        Must be less or equal dimensions to image.
        Length of each dimension must be less than length of image.
        :param image: N-D array
        :param mode: Options, "full", "valid", "same"
        full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
        Output size will be image size + 1/2 template size in each dimension.
        valid: The output consists only of those elements that do not rely on the zero-padding.
        same: The output is the same size as image, centered with respect to the ‘full’ output.
        :return: N-D array of same dimensions as image. Size depends on mode parameter.
        """

        # If this happens, it is probably a mistake
        if np.ndim(template) > np.ndim(image) or \
                len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
            print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

        template -= np.mean(template)
        image -= np.mean(image)

        a1 = np.ones(template.shape)
        # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
        ar = np.flipud(np.fliplr(template))
        out = fftconvolve(image, ar.conj(), mode=mode)

        image = fftconvolve(np.square(image), a1, mode=mode) - \
                np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

        # Remove small machine precision errors after subtraction
        image[np.where(image < 0)] = 0

        template = np.sum(np.square(template))
        out = out / np.sqrt(image * template)

        # Remove any divisions by 0 or very close to 0
        out[np.where(np.logical_not(np.isfinite(out)))] = 0

        return out

    cross_shore = sub_tile.shape[0]
    long_shore = sub_tile.shape[1]
    bands = sub_tile.shape[2]
    img = np.zeros(sub_tile.shape)
    for i in range(bands):
        img_t = np.squeeze(sub_tile[:, :, i])
        img_t -= np.ma.median(img_t)
        c0 = normxcorr2(img_t, img_t)
        c2 = normxcorr2(img_t, c0)
        c2 = c2[int(c2.shape[0] / 2 - cross_shore / 2 + 1): int(c2.shape[0] / 2 + long_shore / 2 + 1),
             int(c2.shape[0] / 2 - cross_shore / 2 + 1): int(c2.shape[0] / 2 + long_shore / 2 + 1)]
        img[:, :, i] = c2
    return img


def apply_fft(sub_tile, t_max=25, t_min=5, energy_min_thresh=None, energy_max_thresh=None):
    """
    Compute the fft filtering of a subtile
    :param sub_tile:(np.array) the given subtile to filter
    :param t_max:(int) Max wave periode
    :param t_min:(int) Min wave periode
    :param energy_min_thresh:(float) min energy threshold
    :param energy_max_thresh:(float) max energy threshold
    :return: filtered_sub_tile: Subtile filtered
    """

    flag = 0
    n, m, c = sub_tile.shape
    kx = np.fft.fftshift(np.fft.fftfreq(n, 10))
    ky = np.fft.fftshift(np.fft.fftfreq(m, 10))
    kx = np.repeat(np.reshape(kx, (n, 1)), m, axis=1)
    ky = np.repeat(np.reshape(ky, (1, m)), n, axis=0)
    threshold_min = 1 / (1.56 * t_max ** 2)
    threshold_max = 1 / (1.56 * t_min ** 2)
    filtered_sub_tile = np.zeros(sub_tile.shape)
    kr = np.sqrt(kx ** 2 + ky ** 2)
    kr[kr < threshold_min] = 0
    kr[kr > threshold_max] = 0
    bool_kr = (kr > 0)

    return_max_energy = 0
    for channel in range(c):
        r = sub_tile[:, :, channel]
        r = detrend(detrend(r, axis=1), axis=0)
        fftr = np.fft.fft2(r)
        energy_r = np.fft.fftshift(fftr)
        energy_r *= bool_kr
        max_energy = np.max(np.abs(energy_r))
        if max_energy > return_max_energy:
            return_max_energy = max_energy
        if energy_max_thresh and energy_min_thresh and \
                (max_energy > energy_max_thresh or max_energy < energy_min_thresh):
            flag = 1
        filtered_sub_tile[:, :, channel] = np.real(np.fft.ifft2(np.fft.ifftshift(energy_r)))
    return filtered_sub_tile, flag, return_max_energy


def apply_per_band_min_max_normalization(sub_tile):
    """"sub_tile: (cross_shore, long_shore, bands)"""
    for i in range(sub_tile.shape[2]):
        sub_tile[:, :, i] = (sub_tile[:, :, i] - np.min(sub_tile[:, :, i])) / (
                    np.max(sub_tile[:, :, i]) - np.min(sub_tile[:, :, i]))
    return sub_tile



