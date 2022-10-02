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
from scipy.special import factorial as fac

def zernike_radial(rho: np.ndarray,
                   n0d: np.ndarray,
                   m0d: np.ndarray):
    """
    Create radial Zernike polynomial on coordinate grid rho

    Parameters
    ----------
        rho: `numpy.ndarray`
            the radial direction
        n0d: `numpy.ndarray`
            the Zernike index in radial direction
        m0d: `numpy.ndarray`
            the Zernike index in azimuthal direction

    Returns
    -------
        height: `numpy.ndarray`
            Radial polynomial with identical shape as roh

    """

    if (np.mod(n0d-m0d, 2) == 1):
        return np.zeros(rho.shape)

    height = np.zeros(rho.shape)
    for k in range(np.int((n0d-m0d)/2)+1):
        height += rho**(n0d-2.0*k) * (-1.0)**k * fac(n0d-k) / \
            (fac(k) * fac((n0d+m0d)/2.0 - k) * fac((n0d-m0d)/2.0 - k))

    return height


def zernike_rtnm(rho: np.ndarray,
                 theta: np.ndarray,
                 n0d: np.ndarray,
                 m0d: np.ndarray):
    """
    Calculate Zernike based on (rho, theta, n0d, m0d)

    Parameters
    ----------
        rho: `numpy.ndarray`
            the radial direction
        theta: `numpy.ndarray`
            the azimuthal direction
        n0d: `numpy.ndarray`
            the Zernike index in radial direction
        m0d: `numpy.ndarray`
            the Zernike index in azimuthal direction

    Returns
    -------
        z2d: `numpy.ndarray`
            the height distribution

    """
    if (m0d > 0):
        z2d = zernike_radial(rho, n0d, m0d) * np.cos(m0d * theta)
    if (m0d < 0):
        z2d = zernike_radial(rho, n0d, -m0d) * np.sin(-m0d * theta)
    if (m0d == 0):
        z2d = zernike_radial(rho, n0d, 0)
    return z2d


def zernike_rtj(r2d: np.ndarray,
                t2d: np.ndarray,
                j0d: int):
    """
    Calculate Zernike based on (rho, theta, j)

    Parameters
    ----------
        r2d: `numpy.ndarray`
            the radial direction
        t2d: `numpy.ndarray`
            the azimuthal direction
        j0d: `int`
            the Zernike order

    Returns
    -------
        z2d: `numpy.ndarray`
            the height distribution
    """

    n0d, m0d = convert_j_to_nm(j0d)
    z2d = zernike_rtnm(r2d, t2d, n0d, m0d)
    return z2d


def zernike_xyjc(x2d: np.ndarray,
                 y2d: np.ndarray,
                 j1d: np.ndarray,
                 c1d: np.ndarray = None):
    """
    Calculate Zernike based on (x, y, j, c)

    Parameters
    ----------
        x2d: `numpy.ndarray`
            the x coordinate
        y2d: `numpy.ndarray`
            the y coordinate
        j1d: `numpy.ndarray`
            the Zernike orders
        c1d: `numpy.ndarray`
            the coefficients
    Returns
    -------
        z2d: `numpy.ndarray`
            the height distribution
    """

    if c1d is None:
        c1d = np.ones(j1d.shape)

    # (x2d,y2d)-->(r2d,t2d)
    r2d = np.sqrt(x2d**2+y2d**2)
    t2d = np.arctan2(y2d, x2d)

    # Validate the coordinates
    t2d[r2d > 1] = np.nan
    r2d[r2d > 1] = np.nan

    # Build list of Zernike modes, these are *not* masked/cropped
    zernikes = [zernike_rtj(r2d, t2d, j0d) for j0d in j1d]
    zm3d = np.array(zernikes)

    """
    Compute the Zernike Cartesian derivatives

    [1] J. Ruoff, and M. Totzeck, "Orientation Zernike polynomials: a useful
    way to describe the polarization effects of optical imaging systems,"
    JOSA (2009)
    """

    zxm3d = np.zeros(zm3d.shape)
    zym3d = np.zeros(zm3d.shape)
    Zx_modes = np.zeros(zm3d.shape)
    Zy_modes = np.zeros(zm3d.shape)

    n1d, m1d = convert_j_to_nm(j1d)
    Cj = n1d+1j*m1d

    # Calculate the parameters
    am = (m1d >= 0)*1 - (m1d < 0)*1

    Nnm = np.sqrt((2-(m1d == 0)*1)*(n1d+1))

    b1 = Nnm*0
    b2 = Nnm*0
    b0 = Nnm*0

    Nnm1 = np.sqrt((2-((m1d-1) == 0)*1)*((n1d-1)+1))
    Nnm2 = np.sqrt((2-((m1d+1) == 0)*1)*((n1d-1)+1))

    temp0 = (2-(m1d == 0)*1)*((n1d-2)+1)
    Nnm0 = Nnm*0
    Nnm0[temp0 > 0] = np.sqrt(temp0[temp0 > 0])

    b1[Nnm1 != 0] = Nnm[Nnm1 != 0]/Nnm1[Nnm1 != 0]
    b2[Nnm2 != 0] = Nnm[Nnm2 != 0]/Nnm2[Nnm2 != 0]
    b0[Nnm0 != 0] = Nnm[Nnm0 != 0]/Nnm0[Nnm0 != 0]
    b0[np.isnan(b0)] = 0

    m1 = am*np.abs(m1d-1)
    m2 = am*np.abs(m1d+1)
    m3 = -am*np.abs(m1d-1)
    m4 = -am*np.abs(m1d+1)

    for k in range(len(j1d)):

        # Deal with the initial conditions
        j1 = Cj == (n1d[k]-1)+m1[k]*1j
        j2 = Cj == (n1d[k]-1)+m2[k]*1j
        j3 = Cj == (n1d[k]-1)+m3[k]*1j
        j4 = Cj == (n1d[k]-1)+m4[k]*1j
        j0 = Cj == (n1d[k]-2)+m1d[k]*1j

        # Check j1
        if np.any(j1):
            Z1 = zm3d[j1]*Nnm[j1]
        else:
            Z1 = np.zeros(x2d.shape)

        # Check j2
        if np.any(j2):
            Z2 = zm3d[j2]*Nnm[j2]
        else:
            Z2 = np.zeros(x2d.shape)

        # Check j3
        if np.any(j3):
            Z3 = zm3d[j3]*Nnm[j3]
        else:
            Z3 = np.zeros(x2d.shape)

        # Check j4
        if np.any(j4):
            Z4 = zm3d[j4]*Nnm[j4]
        else:
            Z4 = np.zeros(x2d.shape)

        # Check j0
        if np.any(j0):
            zx2d = Zx_modes[j0]
            zy2d = Zy_modes[j0]
        else:
            zx2d = np.zeros(x2d.shape)
            zy2d = np.zeros(x2d.shape)

        # Calculate the derivatives
        Zx_temp = n1d[k]*(+b1[k]*Z1 + am[k]*np.sign(m1d[k]+1)
                          * b2[k]*Z2) + b0[k]*zx2d
        Zx_modes[k] = np.real(Zx_temp)
        Zy_temp = n1d[k]*(-am[k]*np.sign(m1d[k]-1)*b1[k]
                          * Z3 + b2[k]*Z4) + b0[k]*zy2d
        Zy_modes[k] = np.real(Zy_temp)

        zxm3d[k] = Zx_modes[k]/Nnm[k]
        zym3d[k] = Zy_modes[k]/Nnm[k]

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
            the orders in xn2d direction
        c1d: `numpy.ndarray`
            the order in yn2d direction
    Returns
    -------
        n1d: `numpy.ndarray`
            the orders in xn2d direction
        m1d: `numpy.ndarray`
            the orders in yn2d direction
    """

    b1d = np.ceil(np.sqrt(j1d))
    a1d = b1d**2-j1d+1
    m1d = np.int32(-a1d/2*(1-a1d % 2)+(a1d-1)/2*(a1d % 2))
    n1d = np.int32(2*(b1d-1)-abs(m1d))
    return (n1d, m1d)


def decompose(z2d: np.ndarray,
              x2d: np.ndarray,
              y2d: np.ndarray,
              j1d: np.ndarray,
              rho_norm: float = None):
    """
    Decomposition with Zernikes

    Parameters
    ----------
        z2d: `numpy.ndarray`
            the height distribution to decompose
        x2d: `numpy.ndarray`
            the x coordinate
        y2d: `numpy.ndarray`
            the y coordinate
        j1d: `numpy.ndarray`
            the orders in xn2d direction
        rho_norm: `float`
            the order in yn2d direction
    Returns
    -------
        z2d_rec: `numpy.ndarray`
            the reconstructed height distribution
        coef_est: `numpy.ndarray`
            the estimated coefficients

    """
    if rho_norm is None:
        rho_norm = set_default_rho(z2d, x2d, y2d)

    # Normalization
    xn2d = x2d/rho_norm
    yn2d = y2d/rho_norm
    r2d = np.sqrt(xn2d**2+yn2d**2)

    # Remove values outside the unit cricle
    xn2d[r2d > 1] = np.nan
    yn2d[r2d > 1] = np.nan
    z2d[r2d > 1] = np.nan

    # Get the modes in height
    zm3d = zernike_xyjc(xn2d, yn2d, j1d)[3]
    zm2d = zm3d.reshape((zm3d.shape[0], -1)).T

    # Get the measured height.
    z1d = z2d.ravel()

    # Remove invalid values.
    zm2d = zm2d[np.isfinite(z1d), :]
    z1d = z1d[np.isfinite(z1d)]

    # Linear least squares estimation.
    coef_est = np.dot(np.linalg.pinv(zm2d), z1d)

    # Use the estimated coefficients to reconstruct the height and slopes.
    z2d_rec = zernike_xyjc(xn2d, yn2d, j1d, coef_est)[0]

    return (z2d_rec, coef_est, zm3d, xn2d, yn2d, rho_norm)


def integrate(sx2d: np.ndarray,
              sy2d: np.ndarray,
              x2d: np.ndarray,
              y2d: np.ndarray,
              j1d: np.ndarray = np.arange(1, 1+9),
              rho_norm: float = None):
    """
    Integration with Zernikes

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

    if rho_norm is None:
        rho_norm = set_default_rho(sx2d+sy2d, x2d, y2d)

    # Normalization
    xn2d = x2d/rho_norm
    yn2d = y2d/rho_norm
    sxn2d = sx2d*rho_norm
    syn2d = sy2d*rho_norm
    r2d = np.sqrt(xn2d**2+yn2d**2)

    # Remove values outside the unit cricle.
    xn2d[r2d > 1] = np.nan
    yn2d[r2d > 1] = np.nan
    sxn2d[r2d > 1] = np.nan
    syn2d[r2d > 1] = np.nan

    # Get the modes in slopes.
    zxm3d, zym3d = zernike_xyjc(xn2d, yn2d, j1d)[4:6]
    zxm2d = np.reshape(zxm3d, (zxm3d.shape[0], -1)).T
    zym2d = np.reshape(zym3d, (zym3d.shape[0], -1)).T
    sm2d = np.vstack((zxm2d, zym2d))

    # Get the measured slopes.
    ssn = np.vstack((sxn2d, syn2d))
    ssn = ssn.ravel()

    # Remove invalid values.
    sm2d = sm2d[np.isfinite(ssn), :]
    ssn = ssn[np.isfinite(ssn)]

    # Linear least squares estimation.
    coef_est = np.dot(np.linalg.pinv(sm2d), ssn)

    # Use the estimated coefficients to reconstruct the height and slopes.
    z2d_rec, zx2d_rec, zy2d_rec = zernike_xyjc(xn2d, yn2d, j1d, coef_est)[0:3]

    return (z2d_rec, zx2d_rec, zy2d_rec, coef_est,
            zxm3d, zym3d, xn2d, yn2d, rho_norm)


def set_default_rho(value2d: np.ndarray, x2d: np.ndarray, y2d: np.ndarray):
    """
    Set the rho_norm value by default

    Parameters
    ----------
        value2d: `numpy.ndarray`
            the height distribution to decompose
        x2d: `numpy.ndarray`
            the x coordinate
        y2d: `numpy.ndarray`
            the y coordinate

    Returns
    -------
        rho_norm: `float`
            the order in yn2d direction
    """
    valid = np.isfinite(value2d)
    rho_norm = np.nanmax(np.sqrt(x2d[valid]**2+y2d[valid]**2))
    return rho_norm
