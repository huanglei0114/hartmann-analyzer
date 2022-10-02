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

def legendre_p1(x2d: np.ndarray, orders: np.ndarray):
    """
    Get the P1 type Legendre polynomial modes in 1D and its first derivative

    Parameters
    ----------
        x2d: `numpy.ndarray`
            the 2D x-coordinate, but only in a row or in a column
        orders: `numpy.ndarray`
            the orders of the polynomial modes

    Returns
    -------
        l3d: `numpy.ndarray`
            the legendre polynomial
        lx3d: `numpy.ndarray`
            the first derivative of the legendre polynomial
    """

    order_num = orders.size
    l3d = np.zeros((order_num, x2d.shape[0], x2d.shape[1]))
    lx3d = np.zeros((order_num, x2d.shape[0], x2d.shape[1]))

    for idx in range(order_num):
        # Legendre polynomial
        l3d[idx] = legendre_polynomial(x2d, orders[idx])

        # Derivative
        lx3d[idx] = legendre_derivative(x2d, orders[idx])

    return (l3d, lx3d)


def legendre_polynomial(x2d: np.ndarray, order: int):
    """
    Get the Legendre polynomial in 1D with a specific order

    Parameters
    ----------
        x2d: `numpy.ndarray`
            the 2D x-coordinate, but only in a row or in a column
        order: `int`
            the order of the polynomial

    Returns
    -------
        l2d: `numpy.ndarray`
            the legendre polynomial in 2D shape, but in a row or in a column
    """
    if order == 0:
        l2d = np.ones(x2d.shape)
    elif order == 1:
        l2d = x2d
    else:
        l2d = (2*order-1)/order*x2d*legendre_polynomial(x2d, order-1) \
            - (order-1)/order*legendre_polynomial(x2d, order-2)
    return l2d


def legendre_derivative(x2d: np.ndarray, order: int):
    """
    Get the Legendre polynomial derivative in 1D

    Parameters
    ----------
        x2d: `numpy.ndarray`
            the 2D x-coordinate, but only in a row or in a column
        order: `int`
            the order of the polynomial

    Returns
    -------
        lx2d: `numpy.ndarray`
            the first derivative of the legendre polynomial
    """
    if order == 0:
        lx2d = np.zeros(x2d.shape)
    elif order == 1:
        lx2d = np.ones(x2d.shape)
    else:
        lx2d = (2*order-1)/order*legendre_polynomial(x2d, order-1) \
            + (2*order-1)/order*x2d*legendre_derivative(x2d, order-1) \
            - (order-1)/order*legendre_derivative(x2d, order-2)
    return lx2d


def legendre_xynm(x2d: np.ndarray,
                  y2d: np.ndarray,
                  n1d: np.ndarray,
                  m1d: np.ndarray):
    """
    Calculate Legendres based on (x,y,n,m)

    Parameters
    ----------
        x2d: `numpy.ndarray`
            the normalized 2D x-coordinate
        y2d: `numpy.ndarray`
            the normalized 2D y-coordinate
        n1d: `numpy.ndarray`
            the orders in x direction
        m1d: `numpy.ndarray`
            the orders in y direction
    Returns
    -------
        z2d: `numpy.ndarray`
            the legendre polynomials
        zx2d: `numpy.ndarray`
            the first derivative of the legendre polynomial in x direction
        zy2d: `numpy.ndarray`
            the first derivative of the legendre polynomial in y direction
    """

    # Get the 1D Legendre polynomial and its derivative
    [lm3d, lmy3d] = legendre_p1(y2d, m1d)
    [ln3d, lnx3d] = legendre_p1(x2d, n1d)

    # Convert 1D to 2D polynomials with considering the derivatives
    z3d = ln3d * lm3d
    zx3d = lnx3d * lm3d
    zy3d = ln3d * lmy3d

    return (z3d, zx3d, zy3d)


def legendre_xyjc(x2d: np.ndarray,
                  y2d: np.ndarray,
                  j1d: np.ndarray,
                  c1d: np.ndarray = None):
    """
    Calculate Legendres based on (x,y,j,c)

    Parameters
    ----------
        x2d: `numpy.ndarray`
            the normalized 2D x-coordinate
        y2d: `numpy.ndarray`
            the normalized 2D y-coordinate
        j1d: `numpy.ndarray`
            the orders
        c1d: `numpy.ndarray`
            the coefficients
    Returns
    -------
        z2d: `numpy.ndarray`
            the shape represented with legendre polynomials
        zx2d: `numpy.ndarray`
            the first derivative of the legendre polynomial in x direction
        zy2d: `numpy.ndarray`
            the first derivative of the legendre polynomial in y direction
        zm3d: `numpy.ndarray`
            the legendre polynomial modes
        zxm3d: `numpy.ndarray`
            the first derivative modes of the legendre polynomial in x direction
        zym3d: `numpy.ndarray`
            the first derivative modes of the legendre polynomial in y direction
        n1d: `numpy.ndarray`
            the orders in x direction
        m1d: `numpy.ndarray`
            the orders in y direction
    """

    if c1d is None:
        c1d = np.ones(j1d.shape)

    # Calculate the order vectors (n, m)
    n1d, m1d = convert_j_to_nm(j1d)

    # Calculate the Legendre polynomials and 1st derivatives with coefficients
    zm3d, zxm3d, zym3d = legendre_xynm(x2d, y2d, n1d, m1d)

    # Sum up the arrays
    z2d = np.sum(zm3d*c1d.reshape((-1, 1, 1)), axis=0)
    zx2d = np.sum(zxm3d*c1d.reshape((-1, 1, 1)), axis=0)
    zy2d = np.sum(zym3d*c1d.reshape((-1, 1, 1)), axis=0)

    return (z2d, zx2d, zy2d, zm3d, zxm3d, zym3d, m1d, n1d)


def convert_j_to_nm(j1d: np.ndarray):
    """
    Convert j1d to (n1d, m1d)

    Parameters
    ----------
        j1d: `numpy.ndarray`
            the orders in x direction
        c1d: `numpy.ndarray`
            the order in y direction
    Returns
    -------
        n1d: `numpy.ndarray`
            the orders in x direction
        m1d: `numpy.ndarray`
            the orders in y direction
    """

    b1d = np.ceil(np.sqrt(j1d))
    a1d = b1d**2-j1d+1
    nsm = np.int32(-a1d/2*(1-a1d % 2)+(a1d-1)/2*(a1d % 2))
    nam = np.int32(2*b1d-abs(nsm)-2)
    n1d = np.int32((nam+nsm)/2)
    m1d = np.int32((nam-nsm)/2)
    return (n1d, m1d)


def decompose(z2d: np.ndarray,
              x2d: np.ndarray,
              y2d: np.ndarray,
              j1d: np.ndarray,
              xy_norm: np.ndarray = None):
    """
    Decomposition with Legendre polynomials

    Parameters
    ----------
        z2d: `numpy.ndarray`
            the height distribution
        x2d: `numpy.ndarray`
            the x coordinate
        y2d: `numpy.ndarray`
            the y coordinate
        j1d: `numpy.ndarray`
            the orders
        xy_norm: `numpy.ndarray`
            the normalization paramters in x and y directions
    Returns
    -------
        z2d_rec: `numpy.ndarray`
            the reconstructed height distribution
        coef_est: `numpy.ndarray`
            the estimated coefficients
        zm3d: `numpy.ndarray`
            the legendre modes used in decomposition
        xn2d: `numpy.ndarray`
            the normalized x coordinates
        yn2d: `numpy.ndarray`
            the normalized y coordinates
        xy_norm: `numpy.ndarray`
            the normalization paramters in x and y directions
    """

    if xy_norm is None:
        x_norm = (np.nanmax(x2d)-np.nanmin(x2d))/2
        y_nrom = (np.nanmax(y2d)-np.nanmin(y2d))/2
        xy_norm = np.array([x_norm, y_nrom])
    else:
        x_norm = xy_norm[0]
        y_nrom = xy_norm[1]

    # Normalization
    xn2d = (x2d-np.nanmean(x2d))/x_norm
    yn2d = (y2d-np.nanmean(y2d))/y_nrom

    # Get the modes in height
    zm3d = legendre_xyjc(xn2d, yn2d, j1d)[3]
    zm2d = zm3d.reshape((zm3d.shape[0], -1)).T

    # Get the measured height
    z1d = z2d.ravel()

    # Remove invalid values
    zm2d = zm2d[np.isfinite(z1d), :]
    z1d = z1d[np.isfinite(z1d)]

    # Linear least squares estimation
    coef_est = np.dot(np.linalg.pinv(zm2d), z1d)

    # Use the estimated coefficients to reconstruct the height and slopes
    z2d_rec = legendre_xyjc(xn2d, yn2d, j1d, coef_est)[0]

    return (z2d_rec, coef_est, zm3d, xn2d, yn2d, xy_norm)


def integrate(sx2d: np.ndarray,
              sy2d: np.ndarray,
              x2d: np.ndarray,
              y2d: np.ndarray,
              j1d: np.ndarray = range(1, 1+9),
              xy_norm: np.ndarray = None):
    """
    Integration with Legendre polynomials

    Parameters
    ----------
        sx2d: `numpy.ndarray`
            the x-slope distribution
        sy2d: `numpy.ndarray`
            the y-slope distribution
        x2d: `numpy.ndarray`
            the x coordinate
        y2d: `numpy.ndarray`
            the y coordinate
        j1d: `numpy.ndarray`
            the orders
        xy_norm: `numpy.ndarray`
            the normalization paramters in x and y directions
    Returns
    -------
        z2d_rec: `numpy.ndarray`
            the reconstructed height distribution
        zx2d_rec: `numpy.ndarray`
            the reconstructed x-slope distribution
        zy2d_rec: `numpy.ndarray`
            the reconstructed y-slope distribution
        coef_est: `numpy.ndarray`
            the estimated coefficients
    """

    if xy_norm is None:
        x_norm = (np.nanmax(x2d)-np.nanmin(x2d))/2
        y_nrom = (np.nanmax(y2d)-np.nanmin(y2d))/2
        xy_norm = np.array([x_norm, y_nrom])
    else:
        x_norm = xy_norm[0]
        y_nrom = xy_norm[1]

    # Normalization
    xn2d = (x2d-np.nanmean(x2d))/x_norm
    yn2d = (y2d-np.nanmean(y2d))/y_nrom
    sxn2d = sx2d*x_norm
    syn2d = sy2d*y_nrom

    # Get the modes in slopes
    zxm3d, zym3d = legendre_xyjc(xn2d, yn2d, j1d)[4:6]
    zxm2d = np.reshape(zxm3d, (zxm3d.shape[0], -1)).T
    zym2d = np.reshape(zym3d, (zym3d.shape[0], -1)).T
    sm2d = np.vstack((zxm2d, zym2d))

    # Get the measured slopes
    ssn = np.vstack((sxn2d, syn2d))
    ssn = ssn.ravel()

    # Remove invalid values
    sm2d = sm2d[np.isfinite(ssn), :]
    ssn = ssn[np.isfinite(ssn)]

    # Linear least squares estimation
    coef_est = np.dot(np.linalg.pinv(sm2d), ssn)

    # Use the estimated coefficients to reconstruct the height and slopes
    z2d_rec, zx2d_rec, zy2d_rec = legendre_xyjc(xn2d, yn2d, j1d, coef_est)[0:3]

    return (z2d_rec, zx2d_rec, zy2d_rec, coef_est,
            zxm3d, zym3d, xn2d, yn2d, xy_norm)
