# Copyright (c) 2021 Lei Huang, Tianyi Wang
#
# Lei Huang
# huanglei0114@gmail.com
#
# All rights reserved
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .omda import add_colorbar
from .omda import remove_2d_tilt, remove_2d_sphere
from .omda import legendre, zernike
from .omda import qgpu2sc, calculate_2d_height_from_slope
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

def process_hartmanngram(para: dict,
                         hartmanngram: np.ndarray,
                         starting_pixel=np.ndarray([]),
                         thresholding_mode: str = 'adaptive',
                         img_int_thr: int = 0,
                         area_thr: int = 1,
                         ratio_thr: float = 0.05,
                         min_order_u: int = None,
                         max_order_u: int = None,
                         min_order_v: int = None,
                         max_order_v: int = None,
                         edge_exclusion: int = 1,
                         ):
    """
    Process the hartmanngram

    Parameters
    ----------
        para: `dict`
            The user defined parameters in dictionary
        hartmanngram: `numpy.ndarray`
            The hartmanngram
        starting_pixel: `numpy.ndarray`
            The starting pixel for unwrapping
        thresholding_mode: `str`
            The thresholding mode
        img_int_thr: `int`
            Threshold of image intensity
        area_thr: `int`
            Threshold of area
        ratio_thr: `float`
            Threshold of intensity volumn ratio to that of the reference order
        min_order_u: `int`
            The minimum order u
        max_order_u: `int`
            The maximum order u
        min_order_v: `int`
            The minimum order v
        max_order_v: `int`
            The maximum order v
        edge_exclusion: `int`
            The exclusion pixel number around the edge
    Returns
    -------
        wfr: `numpy.ndarray`
            The wavefront
        x2d: `numpy.ndarray`
            The 2D x coordinates
        y2d: `numpy.ndarray`
            The 2D y coordinates
        abr: `numpy.ndarray`
            The aberrration
        coefs_wfr_wl: `numpy.ndarray`
            The coefficients of wavefront in wavelength
        coefs_abr_wl: `numpy.ndarray`
            The coefficients of aberrration in wavelength
        denominator: `numpy.ndarray`
            The denominator
        strehl_ratio: `float`
            The strehl ratio
    """

    # 1. Get the user-defined parameters
    grid_period = para['grid_period']
    dist_mask_to_detector = para['dist_mask_to_detector']
    min_fringe_number = para['lowest_fringe_order']
    centroid_power = para['centroid_power']
    detector_pixel_size = para['detector_pixel_size']
    wavelength = para['wavelength']

    # 2. Analyze the Hartmanngram
    x2d, y2d, sx2d, sy2d, u1d_centroid, v1d_centroid, u2d_centroid, v2d_centroid = \
        analyze_hartmanngram(hartmanngram,
                             dist_mask_to_detector,
                             detector_pixel_size,
                             grid_period,
                             thresholding_mode=thresholding_mode,
                             img_int_thr=img_int_thr,
                             area_thr=area_thr,
                             min_fringe_number=min_fringe_number,
                             starting_pixel=starting_pixel,
                             ratio_thr=ratio_thr,
                             centroid_power=centroid_power,
                             min_order_u=min_order_u,
                             max_order_u=max_order_u,
                             min_order_v=min_order_v,
                             max_order_v=max_order_v,
                             edge_exclusion=edge_exclusion,
                             is_show=True,
                             )

    # 3. Analyze the slopes
    wfr, abr, coefs_wfr_wl, coefs_abr_wl = \
        analyze_hartmann_slopes(x2d,
                                y2d,
                                sx2d,
                                sy2d,
                                wavelength=wavelength,
                                num_of_terms=100,
                                is_show=True,
                                )

    # 4. Calculate the aberration RMS in wavelength
    abr_rms_wl, denominator, strehl_ratio = calculate_abr_rms_wl(abr,
                                                                 wavelength)
    return (wfr, x2d, y2d, abr,
            coefs_wfr_wl, coefs_abr_wl,
            denominator, strehl_ratio)

def analyze_hartmanngram(hartmanngram: np.ndarray,
                         dist_mask_to_detector: float,
                         pixel_size: float = 1.48e-6,
                         grid_period: float = 20e-6,
                         thresholding_mode: str = 'adaptive',
                         img_int_thr: np.uint8 = 0,
                         block_size: int = 31,
                         area_thr: int = 1,
                         min_fringe_number: int = 8,
                         starting_pixel: np.ndarray = None,
                         ratio_thr: float = 0.05,
                         centroid_power: float = 1.7,
                         min_order_u: int = None,
                         max_order_u: int = None,
                         min_order_v: int = None,
                         max_order_v: int = None,
                         edge_exclusion: int = 1,
                         is_show: bool = False,
                         ):
    """
    Calculate the slopes from the Hartmanngram

    Parameters
    ----------
        hartmanngram: `numpy.ndarray`
            The Hartmanngrame image
        dist_mask_to_detector: `float`
            The distance from the Hartmann mask to the Hartmann detector
        pixel_size: `float`
            The pixel size of the detector
        grid_period: `float`
            The period of the Hartmann grid
        thresholding_mode: `str`
            The thresholding mode
        img_int_thr: `np.uint8`
            The threshold of the image intensity
        block_size: `int`
            The size of a pixel neighborhood to calculate a threshold: 3, 5, ...
        area_thr: `int`
            The threshold of the spot area
        min_fringe_number: `int`
            The minimum fringe number that can be analyzed
        starting_pixel: `float`
            The starting pixel in phase unwrapping
        ratio_thr: `float`
            The threshold of intensity volumn ratio to that of reference order
        centroid_power: `float`
            The power parameter used for centroid calculation
        min_order_u: `int`
            The min order in u direction
        max_order_u: `int`
            The max order in u direction
        min_order_v: `int`
            The min order in v direction
        max_order_v: `int`
            The max order in v direction
        edge_exclusion: `int`
            The exclusion pixel number around the edge
        is_show: `bool`
            The flag to show the plots
    Returns
    -------
        x2d_wfr: `numpy.ndarray`
            The x coordinates for wavefront reconstruction
        y2d_wfr: `numpy.ndarray`
            The y coordinates for wavefront reconstruction
        sx2d_wfr: `numpy.ndarray`
            The x-slope for wavefront reconstruction
        sy2d_wfr: `numpy.ndarray`
            The y-slope for wavefront reconstruction
        u1d_centroid: `numpy.ndarray`
            The u coordinates of detectable centroid vector
        v1d_centroid: `numpy.ndarray`
            The v coordinates of detectable centroid vector
        u2d_centroid: `numpy.ndarray`
            The u coordinates of selected centroid array
        v2d_centroid: `numpy.ndarray`
            The v coordinates of selected centroid array
    """

    print('Calculating slopes from the hartmanngram...', end='')
    t0 = time.time()

    nv_img, nu_img = hartmanngram.shape
    u2d_img, v2d_img = np.meshgrid(np.arange(nu_img), np.arange(nv_img))
    x2d_img, y2d_img = u2d_img*pixel_size, v2d_img*pixel_size
    x2d_img_um, y2d_img_um = x2d_img*1e6, y2d_img*1e6

    # 1. Mark the possible spots in the binary image
    hartmanngram = hartmanngram.astype(np.float64)

    # Generate a 8-bit image for labelling purpose only
    img_8bit = (hartmanngram/hartmanngram.max()*(2**8-1)).astype(np.uint8)
    img_8bit_blur = cv2.GaussianBlur(img_8bit, (5, 5), 0)

    if thresholding_mode=='adaptive': # Using adaptive thresholding with gaussian window
        img_bw = cv2.adaptiveThreshold(
            img_8bit_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, -1)

    elif thresholding_mode=='otsu': # Using OTSU's binarization
        img_bw = cv2.threshold(
            img_8bit_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    elif thresholding_mode=='img_int_thr': # Using img_int_thr
        img_bw = cv2.threshold(
            img_8bit_blur, img_int_thr, 255, cv2.THRESH_BINARY)[1]
    else:
        print("Unknown thresholding mode. Use 'adaptive' by default.")
        img_bw = cv2.adaptiveThreshold(
            img_8bit_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, -1)


    # 2. Label the possible spots
    num_of_rois, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img_bw)
    area = stats[:, 4]
    # The largest area is the background which should be excluded
    num_of_valid_rois = sum(area >= area_thr) - 1
    roi_idx_of_valid_rois = np.zeros(num_of_valid_rois)
    num_of_valid_rois = 0
    for roi_idx in range(1, num_of_rois):  # Exclude the background
        if area[roi_idx] >= area_thr:
            roi_idx_of_valid_rois[num_of_valid_rois] = roi_idx
            num_of_valid_rois = num_of_valid_rois + 1
        else:
            labels[labels == roi_idx] = 0

    # 3. Determine the fringe orders by using the FFT method
    # 3.1. Calculate the wrapped phases in u and v from spectrum
    print("calculating the wrapped phase...", end="")
    wpu, wpv, ampu, ampv = calculate_wrapped_phases_uv(hartmanngram,
                                                       u2d_img,
                                                       v2d_img,
                                                       min_fringe_number)

    # 3.2. Calculate the quality image
    quality_map = (ampu + ampv)/2
    quality_img = np.uint8(
        255*(quality_map-np.min(quality_map))/(np.max(quality_map)-np.min(quality_map)))
    # Otsu’s binarization
    quality_img_bw = cv2.threshold(quality_img, 0, 1, cv2.THRESH_OTSU)[1]
    # Predifine a default searching mask for a later figure
    searching_mask = quality_map >= 0

    # 3.3. Determine the starting pixel of phase unwrapping
    if starting_pixel is None:  # If the starting pixel is not given by the user ...

        # Get the centroid of the quality_img_bw
        centroid_u, centroid_v = calculate_centroid(
            quality_img_bw, u2d_img, v2d_img)

        # Calculate the starting pixel of phase unwrapping
        area = np.sum(quality_img_bw*1)  # area of the valid regoin
        r_thr = ((area/np.pi)**0.5)/2  # 1/2 of the equavilent circle radius.
        r2d = ((u2d_img-centroid_u)**2+(v2d_img-centroid_v)**2)**0.5
        searching_mask = r2d < r_thr  # Only search inside the central area
        # The non-hole area has lower quality
        min_quality = np.min(quality_map[searching_mask])
        # Try to put the starting pixel in the non-hole order
        v_min_quality, u_min_quality = np.where(quality_map == min_quality)
        starting_pixel = np.array([v_min_quality, u_min_quality])

    # 3.4. Determine the unwrapped phase and fringe orders
    print("phase unwrapping...", end="")
    uwpu = qgpu2sc(wpu, quality_map, starting_pixel)
    uwpv = qgpu2sc(wpv, quality_map, starting_pixel)
    fringe_orders_u, fringe_orders_v = np.round(
        (uwpu-wpu)/(np.pi*2)), np.round((uwpv-wpv)/(np.pi*2))

    # 4. Check the fringe orders in which the labelled regions belong to
    print("checking fringe orders...", end="")
    orders_of_labelled_rois = np.zeros((num_of_valid_rois, 2))
    for idx_of_valid_roi in range(num_of_valid_rois):
        idx = labels == roi_idx_of_valid_rois[idx_of_valid_roi]
        orders_of_labelled_rois[idx_of_valid_roi, 0] = int(
            np.median(fringe_orders_u[idx]))
        orders_of_labelled_rois[idx_of_valid_roi, 1] = int(
            np.median(fringe_orders_v[idx]))
    complex_orders = orders_of_labelled_rois[:,0] + \
        orders_of_labelled_rois[:, 1]*1j
    unique_orders = np.unique(complex_orders)

    # 5. Calculate the centroids and slopes in vectors
    print("calculating centroids and slopes...", end="")
    # 5.1. Use the example oerder to estimate the sum of intensity
    example_order_u, example_order_v = 2, 3
    sub_img_example = crop_image_with_fringe_orders(
        hartmanngram, fringe_orders_u, fringe_orders_v, order_u=example_order_u, order_v=example_order_v)[0]
    sum_of_sub_img_example = np.sum(sub_img_example)

    # 5.2. Calculate the centroids and slopes in vectors
    sx1d = nans(unique_orders.shape)
    sy1d = nans(unique_orders.shape)
    u1d_centroid = nans(unique_orders.shape)
    v1d_centroid = nans(unique_orders.shape)
    u1d_aperture_center = nans(unique_orders.shape)
    v1d_aperture_center = nans(unique_orders.shape)
    ai1d_sub_img = nans(unique_orders.shape)
    for nIdx in range(unique_orders.size):
        order_u, order_v = np.real(
            unique_orders[nIdx]), np.imag(unique_orders[nIdx])
        if not(order_u == 0 and order_v == 0) and not(np.isnan(order_u) or np.isnan(order_v)):
            try:
                # 5.2.1. Crop image
                sub_img, min_u_in_order_mask, min_v_in_order_mask, \
                    u2d_sub_img, v2d_sub_img, ai_sub_img = \
                    crop_image_with_fringe_orders(hartmanngram,
                                                  fringe_orders_u,
                                                  fringe_orders_v,
                                                  order_u,
                                                  order_v)

                # 5.2.2. Calcualte centroids and slopes in vectors
                if np.sum(sub_img)/sum_of_sub_img_example >= ratio_thr:
                    # If the spot intensity is too weak, exclude it
                    # print('Spot (%d, %d) with Raito=%.2f>=%.2f is included.' % (order_u, order_v, np.sum(sub_img)/sum_of_sub_img_example, ratio_thr))

                    # Calculate the centroids
                    powered_sub_img = sub_img**centroid_power  # Calculate powered image
                    u0d_cen_in_sub_img, v0d_cen_in_sub_img = calculate_centroid(
                        powered_sub_img, u2d_sub_img, v2d_sub_img)
                    u0d_centroid, v0d_centroid = u0d_cen_in_sub_img + \
                        min_u_in_order_mask, v0d_cen_in_sub_img + min_v_in_order_mask

                    # Calculate the reference aperture centers
                    u0d_aperture_center = hartmanngram.shape[1]/2-0.5+(
                        grid_period/pixel_size)*order_u
                    v0d_aperture_center = hartmanngram.shape[0]/2-0.5+(
                        grid_period/pixel_size)*order_v

                    # Calculate the spot location changes
                    du0d, dv0d = u0d_centroid - u0d_aperture_center, v0d_centroid - \
                        v0d_aperture_center  # [pixel] Location changes

                    # Calculate the slopes
                    sx1d[nIdx] = du0d*pixel_size/dist_mask_to_detector  # [rad]
                    sy1d[nIdx] = dv0d*pixel_size/dist_mask_to_detector  # [rad]
                    u1d_centroid[nIdx] = u0d_centroid
                    v1d_centroid[nIdx] = v0d_centroid
                    u1d_aperture_center[nIdx] = u0d_aperture_center
                    v1d_aperture_center[nIdx] = v0d_aperture_center
                    ai1d_sub_img[nIdx] = ai_sub_img
                else:
                    if is_show:
                        print('Spot (%d, %d) with Raito=%.3f<%.3f is excluded.' % (
                            order_u, order_v, np.sum(sub_img)/sum_of_sub_img_example, ratio_thr))

                # 5.2.3. Show one of the sub-images
                if order_u == example_order_u and order_v == example_order_v:
                    if is_show:
                        title = '(order_u, order_v) = (%d, %d)' % (
                            int(order_u), int(order_v))
                        fig, ax = plt.subplots()
                        fig.set_size_inches(8, 8)
                        im = ax.imshow(
                            sub_img, interpolation='nearest', cmap='hot')
                        ax.plot(u0d_cen_in_sub_img, v0d_cen_in_sub_img,
                                'mo', markersize=5, markeredgewidth=2)
                        ax.set_title(title)
                        ax.set_xlabel('Horiztonal pixel')
                        ax.set_ylabel('Vertical pixel')
                        add_colorbar(im, ax=ax)

            except Exception:
                if is_show:
                    print('The order (u, v) = (%d, %d) does not contain a valid spot.' % (
                        order_u, order_v))
                continue
        else:
            if is_show:
                print('The order (u, v) = (%d, %d) does not contain a valid spot.' % (
                    order_u, order_v))

    # 6. Assemble centroids and slopes arrays from vectors
    print("assembling the centroids and slopes...", end="")
    # 6.1. If the range of the orders are not provided by users, it will be
    # calculated automatically. In this case, we will assume the wavefront at
    # the edge is affected by diffrraction, so those edge orders are excluded
    unique_orders_u, unique_orders_v = unique_orders.real, unique_orders.imag
    if min_order_u is None:
        min_order_u = int(np.nanmin(unique_orders_u[:])) + edge_exclusion
    if max_order_u is None:
        max_order_u = int(np.nanmax(unique_orders_u[:])) - edge_exclusion
    if min_order_v is None:
        min_order_v = int(np.nanmin(unique_orders_v[:])) + edge_exclusion
    if max_order_v is None:
        max_order_v = int(np.nanmax(unique_orders_v[:])) - edge_exclusion

    # 6.2. Assemble matrices
    nu_wfr, nv_wfr = max_order_u - min_order_u + 1, max_order_v - min_order_v + 1
    sx2d_wfr = nans((nv_wfr, nu_wfr))
    sy2d_wfr = nans((nv_wfr, nu_wfr))
    u2d_centroid = nans((nv_wfr, nu_wfr))
    v2d_centroid = nans((nv_wfr, nu_wfr))
    u2d_aperture_center = nans((nv_wfr, nu_wfr))
    v2d_aperture_center = nans((nv_wfr, nu_wfr))
    ai2d_sub_img = nans((nv_wfr, nu_wfr))
    for order_v in range(min_order_v, max_order_v + 1):
        for order_u in range(min_order_u, max_order_u + 1):
            m, n = order_v - min_order_v, order_u - min_order_u
            idx = (unique_orders_u == order_u) & (unique_orders_v == order_v)
            if np.any(idx) == True:
                sx2d_wfr[m, n] = sx1d[idx]
                sy2d_wfr[m, n] = sy1d[idx]
                u2d_centroid[m, n] = u1d_centroid[idx]
                v2d_centroid[m, n] = v1d_centroid[idx]
                u2d_aperture_center[m, n] = u1d_aperture_center[idx]
                v2d_aperture_center[m, n] = v1d_aperture_center[idx]
                ai2d_sub_img[m, n] = ai1d_sub_img[idx]

    # 7. Define x and y coordinates and the slope in x-y-z coordinates
    u2d_wfr, v2d_wfr = np.meshgrid(np.arange(
        min_order_u, min_order_u+nu_wfr), np.arange(min_order_v, min_order_v+nv_wfr))
    x2d_wfr, y2d_wfr = u2d_wfr*grid_period, v2d_wfr*grid_period

    # 8. Prepare some variables for figures
    x2d_wfr_um, y2d_wfr_um = x2d_wfr*1e6, y2d_wfr*1e6
    sx2d_wfr_urad, sy2d_wfr_urad = sx2d_wfr*1e6, sy2d_wfr*1e6

    print('completed (lasted', round(time.time() - t0, 6), 's)')

    # Test plots
    if is_show:

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        im = ax.pcolormesh(x2d_img_um, y2d_img_um,
                           hartmanngram, shading='auto', cmap='hot')
        ax.set_aspect("equal")
        ax.set_title("Hartmanngram")
        ax.set_xlabel("x [$\mu$m]")
        ax.set_ylabel("y [$\mu$m]")
        ax.invert_yaxis()
        add_colorbar(im, ax=ax)

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(16, 9)
        ax = axs[0, 0]
        im = ax.pcolormesh(x2d_img_um, y2d_img_um, labels, shading='auto')
        ax.set_aspect("equal")
        ax.set_title("labels")
        ax.set_xlabel("x [$\mu$m]")
        ax.set_ylabel("y [$\mu$m]")
        ax.invert_yaxis()
        add_colorbar(im, ax=ax)

        ax = axs[0, 1]
        im = ax.imshow(hartmanngram, interpolation='nearest', cmap='hot')
        ax.plot(u1d_centroid, v1d_centroid, 'mo',
                markersize=5, markeredgewidth=2)
        ax.plot(u2d_centroid, v2d_centroid, 'gx',
                markersize=5, markeredgewidth=2)
        ax.set_xlabel("u [pixel]")
        ax.set_ylabel("v [pixel]")
        add_colorbar(im, ax=ax)

        ax = axs[1, 0]
        im = ax.pcolormesh(x2d_wfr_um, y2d_wfr_um,
                           sx2d_wfr_urad, shading='auto')
        ax.set_aspect("equal")
        ax.set_title("x-slope")
        ax.set_xlabel("x [$\mu$m]")
        ax.set_ylabel("y [$\mu$m]")
        ax.invert_yaxis()
        cbar = add_colorbar(im, ax=ax)
        cbar.set_label('[$\mu$rad]')

        ax = axs[1, 1]
        im = ax.pcolormesh(x2d_wfr_um, y2d_wfr_um,
                           sy2d_wfr_urad, shading='auto')
        ax.set_aspect("equal")
        ax.set_title("y-slope")
        ax.set_xlabel("x [$\mu$m]")
        ax.set_ylabel("y [$\mu$m]")
        ax.set_title("y-slope")
        ax.invert_yaxis()
        cbar = add_colorbar(im, ax=ax)
        cbar.set_label('[$\mu$rad]')

        fig.tight_layout()

    return (x2d_wfr, y2d_wfr, sx2d_wfr, sy2d_wfr,
            u1d_centroid, v1d_centroid, u2d_centroid, v2d_centroid)


def analyze_hartmann_slopes(x2d: np.ndarray,
                            y2d: np.ndarray,
                            sx2d: np.ndarray,
                            sy2d: np.ndarray,
                            wavelength: float,
                            str_method: str = 'zonal',
                            str_model: str = 'Legendre',
                            num_of_terms: int = 100,
                            is_show: bool = False,
                            ):
    """
    Analyze the Hartmann slopes

    Parameters
    ----------
        x2d: `numpy.ndarray`
            The x coordinates for wavefront reconstruction
        y2d: `numpy.ndarray`
            The y coordinates for wavefront reconstruction
        sx2d: `numpy.ndarray`
            The x-slope for wavefront reconstruction
        sy2d: `numpy.ndarray`
            The y-slope for wavefront reconstruction
        wavelength: `float`
            The wavelength
        str_method: `str`
            The reconstruction method
        str_model: `str`
            The model used in modal method and coefficient calculation
        num_of_terms: `int`
            The number of terms in model
        is_show: `bool`
            The flag to show the plots
    Returns
    -------
        wfr: `numpy.ndarray`
            The reconstructed wavefront
        abr: `numpy.ndarray`
            The reconstructed aberration
        wfr_coefs_wl: `numpy.ndarray`
            The wavefront coefficients in wavelength
        abr_coefs_wl: `numpy.ndarray`
            The aberration coefficients in wavelength
    """

    # 1. Calculate the wavefront from slopes
    if 'legendre' == str_model.lower():
        str_model = "Legendre"
        z2d_wfr, zx, zy, wfr_coefs, zxm3d, zym3d, xn2d, yn2d, xy_norm \
            = legendre.integrate(sx2d, sy2d, x2d, y2d,
                                 j1d=np.arange(1, 1 + num_of_terms))
    elif 'zernike' == str_model.lower():
        str_model = "Zernike"
        z2d_wfr, zx, zy, wfr_coefs, zxm3d, zym3d, xn2d, yn2d, rho_norm \
            = zernike.integrate(sx2d, sy2d, x2d, y2d,
                                j1d=np.arange(1, 1 + num_of_terms))

    if 'zonal' == str_method.lower():
        z2d_wfr = calculate_2d_height_from_slope(sx2d, sy2d, x2d, y2d)

    wfr_coefs_wl = wfr_coefs/wavelength
    wfr = z2d_wfr
    wfr[np.logical_or(np.isnan(sx2d), np.isnan(sy2d))] = np.nan
    wfr_wl = wfr/wavelength

    # 2. Calculate the aberration

    # Remove the best-fit sphere (may need several iterations with de-tilt)
    abr = remove_2d_sphere(x2d, y2d, wfr)[0]
    abr = remove_2d_tilt(x2d, y2d, abr)[0]
    for _ in range(10):
        abr = remove_2d_sphere(x2d, y2d, abr)[0]
        abr = remove_2d_tilt(x2d, y2d, abr)[0]

    j1d = np.arange(1, 1 + wfr_coefs.size)
    if 'legendre' == str_model.lower():  # Calculate aberration with Legendres
        abr_models, abr_coefs = legendre.decompose(abr, xn2d, yn2d, j1d)[0:2]
        abr_coefs[0:3] = 0

    elif 'zernike' == str_model.lower():  # Calculate aberration with Zernikes
        abr_models, abr_coefs = zernike.decompose(abr, xn2d, yn2d, j1d)[0:2]
        abr_coefs[0:3] = 0

    abr_coefs_wl = abr_coefs/wavelength
    abr[np.logical_or(np.isnan(sx2d), np.isnan(sy2d))] = np.nan
    abr = remove_2d_tilt(x2d, y2d, abr)[0]
    abr_wl = abr/wavelength

    abr_models[np.logical_or(np.isnan(sx2d), np.isnan(sy2d))] = np.nan
    abr_models_wl = abr_models/wavelength

    # Test plots
    if is_show:

        x2d_um, y2d_um = x2d*1e6, y2d*1e6

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        im = ax.pcolormesh(x2d_um, y2d_um, wfr_wl, shading='auto')
        ax.set_aspect("equal")
        title = 'Wavefront with {} method'.format(str_method)
        ax.set_title(title)
        ax.set_xlabel('x [$\mu$m]')
        ax.set_ylabel('y [$\mu$m]')
        ax.invert_yaxis()
        cbar = add_colorbar(im, ax=ax)
        cbar.set_label('[$\lambda$]')

        fig, ax = plt.subplots()
        fig.set_size_inches(16, 4)
        width = 0.4
        ax.bar(j1d, wfr_coefs_wl, width, label='Wavefront')
        title = 'Wavefront coefficients \nwith {} models'.format(str_model)
        ax.set_title(title)
        ax.set_xlabel('Terms')
        ax.set_ylabel('Coefficients [$\lambda$]')
        if j1d.size<9: ax.set_xticks(j1d)
        ax.legend()

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        im = ax.pcolormesh(x2d_um, y2d_um, abr_wl, shading='auto')
        ax.set_aspect("equal")
        title = 'Aberration: RMS = {0:.3f}λ = λ/{1:.1f}'.format(np.nanstd(abr_wl), 1/np.nanstd(abr_wl))
        ax.set_title(title)
        ax.set_xlabel('x [$\mu$m]')
        ax.set_ylabel('y [$\mu$m]')
        ax.invert_yaxis()
        cbar = add_colorbar(im, ax=ax)
        cbar.set_label('[$\lambda$]')

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        im = ax.pcolormesh(x2d_um, y2d_um, abr_models_wl, shading='auto')
        ax.set_aspect("equal")
        title = 'Aberration represented by models \nRMS = {0:.3f}λ = λ/{1:.1f}'.format(np.nanstd(abr_models_wl), 1/np.nanstd(abr_models_wl))
        ax.set_title(title)
        ax.set_xlabel('x [$\mu$m]')
        ax.set_ylabel('y [$\mu$m]')
        ax.invert_yaxis()
        cbar = add_colorbar(im, ax=ax)
        cbar.set_label('[$\lambda$]')

        fig, ax = plt.subplots()
        fig.set_size_inches(16, 4)
        width = 0.4
        ax.bar(j1d, abr_coefs_wl, width, label='Aberration')
        title = 'Aberration coefficients \nwith {} models'.format(str_model)
        ax.set_title(title)
        ax.set_xlabel('Terms')
        ax.set_ylabel('Coefficients [$\lambda$]')
        if j1d.size<9: ax.set_xticks(j1d)
        ax.legend()

    return (wfr, abr, wfr_coefs_wl, abr_coefs_wl)

def calculate_abr_rms_wl(abr: np.ndarray, wavelength: float):
    """
    Caclculate the aberration RMS in wavelength

    Parameters
    ----------
        abr: `numpy.ndarray`
            The aberration
        wavelength: `float`
            The wavelength
    Returns
    -------
        abr_rms_wl: `float`
            The aberration RMS in wavelength
        denominator: `float`
            The denominator value in wavelength
        strehl_ratio: `float`
            The strehl ratio
    """
    abr_wl = abr/wavelength
    abr_rms_wl = np.nanstd(abr_wl)

    denominator = 1.0 / abr_rms_wl
    print('Aberration = λ/%.1f RMS' % denominator)

    strehl_ratio = np.exp(-(2*np.pi*abr_rms_wl)**2)
    print('Strehl ratio (Mahajan\'s approximation) = %.4f' % strehl_ratio)

    return abr_rms_wl, denominator, strehl_ratio

def calculate_wrapped_phases_uv(hartmanngram: np.ndarray,
                                u2d_img: np.ndarray,
                                v2d_img: np.ndarray,
                                min_fringe_number: int,
                                ):
    """
    Calculate the wrapped phases in u and v directions

    Parameters
    ----------
        hartmanngram: `numpy.ndarray`
            The hartmanngram to analyze
        u2d_img: `numpy.ndarray`
            The u2d_img coordinates in pixel
        v2d_img: `numpy.ndarray`
            The v2d_img coordinates in pixel
        min_fringe_number: `int`
            The minimum fringe number to analyze
    Returns
    -------
        wpu: `numpy.ndarray`
            The wrapped phase in u direction
        wpv: `numpy.ndarray`
            The wrapped phase in v direction
        ampu: `numpy.ndarray`
            The amplitude in u direction
        ampv: `numpy.ndarray`
            The amplitude in v direction
    """

    # 1. Take the 2D FFT
    spectrum = np.fft.fftshift(np.fft.fft2(hartmanngram))

    # 2. Calculate the wrapped phase in u2d_img direction
    wpu, ampu = calculate_wrapped_phase(spectrum,
                                        u2d_img,
                                        v2d_img,
                                        min_fringe_number,
                                        is_u=True)

    # 3. Calculate wrapped phase in v2d_img direction
    wpv, ampv = calculate_wrapped_phase(spectrum,
                                        u2d_img,
                                        v2d_img,
                                        min_fringe_number,
                                        is_u=False)
    return (wpu, wpv, ampu, ampv)


def calculate_wrapped_phase(spectrum: np.ndarray,
                            u2d_img: np.ndarray,
                            v2d_img: np.ndarray,
                            min_fringe_number: int,
                            is_u: bool = True):
    """
    Calculate the wrapped phase

    Parameters
    ----------
        spectrum: `numpy.ndarray`
            The spectrum of Fourier transform
        u2d_img: `numpy.ndarray`
            The u2d_img coordinates in pixel
        v2d_img: `numpy.ndarray`
            The v2d_img coordinates in pixel
        min_fringe_number: `int`
            The minimum fringe number to analyze
        is_u: `bool` = True
            The flag to show is in the u2d_img direction
    Returns
    -------
        wrapped_phase: `numpy.ndarray`
            The wrapped phase of the filtered complex image
        amplitude: `numpy.ndarray`
            The amplitude of the filtered complex image
    """

    # 1. Set x-y coordinate system
    v0, u0 = spectrum.shape[0]/2, spectrum.shape[1]/2
    un2d, vn2d = u2d_img-u0, v2d_img-v0

    # 2. Calculate the amplitude of the spectrum
    amp = np.abs(spectrum)

    # 3. Calculate the wrapped phase and amplitude of the fileter spectrum
    rn2d = (un2d**2 + vn2d**2)**0.5
    r_mask = rn2d > min_fringe_number
    uv_mask = (vn2d < un2d) & (vn2d > -un2d) \
            if is_u == True else (vn2d > un2d) & (vn2d > -un2d)
    ruv_mask = r_mask & uv_mask
    filtered_amp = amp * ruv_mask
    idx = np.argmax(filtered_amp)
    vc, uc = np.unravel_index(idx, spectrum.shape)

    # In case the second harmonic appears strongest, we can check if half of the
    # strongest frequency is still strong enough
    # If yes, we should get the half frequency which can be the real first order
    # If no, it means what we get is likely the first order already
    # In fact, this is a very tricky to set the fator here!
    uc_half, vc_half = int(np.round((uc-u0)/2+u0)), int(np.round((vc-v0)/2+v0))
    if filtered_amp[vc, uc] < 3 * filtered_amp[vc_half, uc_half]:
        uc, vc = uc_half, vc_half
    unc, vnc = uc - u0, vc - v0
    r_thr = rn2d[vc, uc] / 4
    rc2d = ((un2d-unc)**2 + (vn2d-vnc)**2)**0.5
    mask = rc2d < r_thr
    filtered_complex_img = np.fft.ifft2(np.fft.ifftshift(spectrum * mask))
    wrapped_phase = np.angle(filtered_complex_img)
    amplitude = np.abs(filtered_complex_img)

    return (wrapped_phase, amplitude)


def calculate_centroid(img: np.ndarray,
                       u2d_img: np.ndarray = None,
                       v2d_img: np.ndarray = None):
    """
    Calculate the centroid

    Parameters
    ----------
        img: `numpy.ndarray`
            The image to calcualte the centroid
        u2d_img: `numpy.ndarray`
            The u coordinates
        v2d_img: `numpy.ndarray`
            The v coordinates
    Returns
    -------
        centroid_u: `numpy.ndarray`
            The u coordinate of the centroid
        centroid_v: `numpy.ndarray`
            The v coordinate of the centroid
    """
    if (u2d_img is None) or (v2d_img is None):
        nv_img, nu_img = img.shape
        u2d_img, v2d_img = np.meshgrid(np.arange(nu_img), np.arange(nv_img))
    img_nansum = np.nansum(img)
    u2d_img_nansum, v2d_img_nansum = np.nansum(img*u2d_img), np.nansum(img*v2d_img)

    centroid_u, centroid_v = u2d_img_nansum/img_nansum, v2d_img_nansum/img_nansum
    return (centroid_u, centroid_v)


def crop_image_with_fringe_orders(img: np.ndarray,
                                  fringe_orders_u: np.ndarray,
                                  fringe_orders_v: np.ndarray,
                                  order_u: int = 0,
                                  order_v: int = 1):
    """
    Crop image with fringe orders

    Parameters
    ----------
        img: `numpy.ndarray`
            The image to calcualte the centroid
        fringe_orders_u: `numpy.ndarray`
            The fringe orders in the u direction
        fringe_orders_v: `numpy.ndarray`
            The fringe orders in the v direction
        order_u: `int`
            The order to process in the u direction
        order_v: `int`
            The order to process in the v direction
    Returns
    -------
        sub_img: `numpy.ndarray`
            The sub image
        min_u_in_order_mask: `int`
            The minimum u value in the order mask
        min_v_in_order_mask: `int`
            The minimum v value in the order mask
        u2d_sub_img: `numpy.ndarray`
            The u coordinates of the sub image
        v2d_sub_img: `numpy.ndarray`
            The v coordinates of the sub image
        avg_int_sub_img: `float`
            The average intensity of the sub image
    """
    nv_img, nu_img = img.shape
    u2d_img, v2d_img = np.meshgrid(np.arange(nu_img), np.arange(nv_img))
    order_mask = np.logical_and(
        fringe_orders_u == order_u, fringe_orders_v == order_v)
    u2d_in_order_mask = u2d_img[order_mask == 1]
    v2d_in_order_mask = v2d_img[order_mask == 1]
    min_u_in_order_mask, max_u_in_order_mask = np.min(
        u2d_in_order_mask), np.max(u2d_in_order_mask)
    min_v_in_order_mask, max_v_in_order_mask = np.min(
        v2d_in_order_mask), np.max(v2d_in_order_mask)
    sub_img = img[min_v_in_order_mask:max_v_in_order_mask + 1,
                  min_u_in_order_mask:max_u_in_order_mask + 1].copy()
    sub_mask = order_mask[min_v_in_order_mask:max_v_in_order_mask + 1,
                          min_u_in_order_mask:max_u_in_order_mask + 1].copy()
    u2d_sub_img, v2d_sub_img = np.meshgrid(
        np.arange(sub_img.shape[1]), np.arange(sub_img.shape[0]))
    sub_img_max_v, sub_img_max_u = np.where(sub_img == np.nanmax(sub_img*sub_mask))
    mask_aera = np.sum(order_mask)
    r2d_sub_img = (
        (u2d_sub_img-int(sub_img_max_u[0]))**2 + (v2d_sub_img-int(sub_img_max_v[0]))**2)**0.5
    r_thr = (mask_aera/np.pi)**0.5
    r_mask = r2d_sub_img < r_thr
    sub_img[~r_mask] = 0
    # Calculate the average intensity of sub_img
    avg_int_sub_img = np.sum(sub_img)/np.sum(r_mask)
    return (sub_img, min_u_in_order_mask, min_v_in_order_mask, u2d_sub_img, v2d_sub_img, avg_int_sub_img)


def nans(shape: tuple, dtype: type = np.float64):
    """
    Initalize array with numpy.nan

    Parameters
    ----------
        shape: `tuple`
            The shape of the array
        dtype: `type`
            The data type of the array
    Returns
    -------
        array: `numpy.ndarray`
            The array with numpy.nan

    """
    array = np.empty(shape, dtype)
    array.fill(np.nan)
    return array
