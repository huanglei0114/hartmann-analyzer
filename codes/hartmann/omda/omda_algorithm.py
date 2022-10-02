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

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr

def remove_2d_tilt(x2d: np.ndarray,
                   y2d: np.ndarray,
                   z2d: np.ndarray):
    """
    Remove the 2D tilt

    Parameters
    ----------
        x2d : `numpy.ndarray`
            The 2D x coordinates
        y2d : `numpy.ndarray`
            The 2D y coordinates
        z2d : `numpy.ndarray`
            The 2D z coordinates

    Returns
    -------
        z2d_res: `numpy.ndarray`
            The z2d residuals
        z2d_fit: `numpy.ndarray`
            The fitting results
        coefs: `numpy.ndarray`
            The fitting coefficients
    """
    # Valid data only
    idx = np.isfinite(z2d)
    z1d = z2d[idx]
    x1d = x2d[idx]
    y1d = y2d[idx]

    matrix_h = np.vstack([np.ones(len(x1d)), x1d, y1d]).T
    coefs = np.linalg.lstsq(matrix_h, z1d, rcond=None)[0]

    # Handle all nan case
    if z1d.size==0:
        coefs[:] = np.nan

    z2d_fit = coefs[0] + coefs[1]*x2d + coefs[2]*y2d
    z2d_res = z2d - z2d_fit
    return z2d_res, z2d_fit, coefs


def remove_2d_sphere(x2d: np.ndarray,
                     y2d: np.ndarray,
                     z2d: np.ndarray):
    """
    Remove the 2D sphere

    Parameters
    ----------
        x2d : `numpy.ndarray`
            The 2D x coordinates
        y2d : `numpy.ndarray`
            The 2D y coordinates
        z2d : `numpy.ndarray`
            The 2D z coordinates

    Returns
    -------
        z2d_res: `numpy.ndarray`
            The z2d residuals
        z2d_fit: `numpy.ndarray`
            The fitting results
        coefs: `numpy.ndarray`
            The fitting coefficients
    """
    # Valid data only
    idx = np.isfinite(z2d)
    z1d = z2d[idx]
    x1d = x2d[idx]
    y1d = y2d[idx]

    matrix_h = np.vstack([np.ones(len(x1d)), x1d, y1d, x1d**2+y1d**2]).T
    coefs = np.linalg.lstsq(matrix_h, z1d, rcond=None)[0]

    # Handle all nan case
    if z1d.size==0:
        coefs[:] = np.nan

    z2d_fit = coefs[0] + coefs[1]*x2d + coefs[2]*y2d + \
        coefs[3]*(x2d**2 + y2d**2)
    z2d_res = z2d - z2d_fit
    return z2d_res, z2d_fit, coefs

def calculate_2d_height_from_slope(sx2d: np.ndarray,
                                   sy2d: np.ndarray,
                                   x2d: np.ndarray,
                                   y2d: np.ndarray):
    """
    Calcualte height from slope in 2D

    Parameters
    ----------
        sx2d: `numpy.ndarray`
            The x-slope
        sy2d: `numpy.ndarray`
            The y-slope
        x2d: `numpy.ndarray`
            The x coordinate
        y2d: `numpy.ndarray`
            The y coordinate
    Returns
    -------
        z2d_hfli2: `numpy.ndarray`
            The height map integrated by using the higher-order
            finite-difference-based least-squares integration
    """
    # Generate Matrix matrix_d and matrix_g
    # Calculate size and mask_valid
    nrows, ncols = sx2d.shape
    mask_valid = np.isfinite(sx2d) & np.isfinite(sy2d)

    # Expand in x-direction
    temp_nan_1 = np.full([nrows, 1], np.nan)
    temp_nan_2 = np.full([nrows, 2], np.nan)
    sx2d_ex = np.hstack((temp_nan_1, sx2d, temp_nan_2))
    x2d_ex = np.hstack((temp_nan_1, x2d, temp_nan_2))

    expand_mask_x = np.isnan(sx2d_ex)
    vse = np.array([[1, 1, 0, 1, 0]])
    dilated_expand_mask_x = ndimage.binary_dilation(
        expand_mask_x, structure=vse)
    mask_x = dilated_expand_mask_x[:, 1:-2] & ~expand_mask_x[:, 1:-2]

    # Expand in y-direction
    temp_nan_1 = np.full([1, ncols], np.nan)
    temp_nan_2 = np.full([2, ncols], np.nan)
    sy2d_ex = np.vstack((temp_nan_1, sy2d, temp_nan_2))
    y2d_ex = np.vstack((temp_nan_1, y2d, temp_nan_2))

    expand_mask_y = np.isnan(sy2d_ex)
    vse = np.array([[1, 1, 0, 1, 0]]).T
    dilated_expand_mask_y = ndimage.binary_dilation(
        expand_mask_y, structure=vse)
    maks_y = dilated_expand_mask_y[1:-2, :] & ~expand_mask_y[1:2, :]

    # Compose matrices matrix_dx and matrix_dy
    num = nrows*ncols
    vee = np.ones((1, num))
    data = np.vstack((-vee, vee))
    matrix_dx = sparse.spdiags(data, [0, 1], num, num, format='csr')
    matrix_dy = sparse.spdiags(data, [0, ncols], num, num, format='csr')

    # Compose matrices matrix_gx5 and matrix_gy5
    # Expression with O(h^5)
    matrix_gx5 = (-1/13*sx2d_ex[:, 0:-3]+sx2d_ex[:, 1:-2]+sx2d_ex[:, 2:-1] -
                  1/13*sx2d_ex[:, 3:])*(x2d_ex[:, 2:-1]-x2d_ex[:, 1:-2])*13/24
    matrix_gy5 = (-1/13*sy2d_ex[0:-3, :]+sy2d_ex[1:-2, :]+sy2d_ex[2:-1, :] -
                  1/13*sy2d_ex[3:, :])*(y2d_ex[2:-1, :]-y2d_ex[1:-2, :])*13/24

    # Expression with O(h^3)
    matrix_gx3 = (sx2d_ex[:, 1:-2]+sx2d_ex[:, 2:-1]) * \
        (x2d_ex[:, 2:-1]-x2d_ex[:, 1:-2])/2
    matrix_gy3 = (sy2d_ex[1:-2, :]+sy2d_ex[2:-1, :]) * \
        (y2d_ex[2:-1, :]-y2d_ex[1:-2, :])/2

    # Use O(h^3) values, if O(h^5) is not available
    matrix_gx5[mask_x] = matrix_gx3[mask_x]
    matrix_gy5[maks_y] = matrix_gy3[maks_y]

    del(sx2d_ex, sy2d_ex, x2d_ex, y2d, matrix_gx3, matrix_gy3)

    # Remove nan
    # Compose matrix_d
    matrix_d = sparse.vstack((matrix_dx[np.isfinite(matrix_gx5).flatten(), :],
                              matrix_dy[np.isfinite(matrix_gy5).flatten(), :]))
    del(matrix_dx, matrix_dy)

    # Compose matrix_g
    matrix_g = np.hstack(
        (matrix_gx5[np.isfinite(matrix_gx5)],
         matrix_gy5[np.isfinite(matrix_gy5)])).T
    del(matrix_gx5, matrix_gy5)

    z2d = lsqr(matrix_d, matrix_g)[0]
    del(matrix_d, matrix_g)

    # Reconstructed result
    z2d_hfli2 = np.reshape(z2d, (nrows, ncols))
    z2d_hfli2[np.logical_not(mask_valid)] = np.nan

    return z2d_hfli2