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

def read_wavefront_sensor_image_dat(filename: str,
                          nu_detector: float = 1024,
                          nv_detector: float = 1024,
                          upsampling: int = 3,
                          ):
    """
    Read the wavefront sensor image in SRW dat format

    Parameters
    ----------
        filename: `str`
            The txt filename which contains the wavefront sensor image
    Returns
    -------
        wavefront_sensor_image: `numpy.ndarray`
            The wavefront sensor image
    """
    intensity = np.loadtxt(filename)
    map_hr = np.flipud(intensity.reshape(nv_detector*upsampling, nu_detector*upsampling))
    temp_map_hr = map_hr.reshape(nv_detector, upsampling, nu_detector, upsampling)
    wavefront_sensor_image = temp_map_hr.mean(axis=3).mean(axis=1) # Binning
    return wavefront_sensor_image


def add_no_noise(intensity_map: np.ndarray):
    """
    Add no noise to the detector image.

    Parameters
    ----------
        intensity_map: `numpy.ndarray`
            The intensity map
    Returns
    -------
        detector_image: `numpy.ndarray`
            The detector image with the shot noise added
    """
    pixel_depth = 14  # Pixel depth of PCO.2000s
    img = np.floor((intensity_map - intensity_map.min()) / intensity_map.ptp()
                   * (2**pixel_depth-1))  # Calculate the  image with integer values
    # Cut the value bigger than 2**pixel_depth-1
    img[img > 2**pixel_depth-1] = 2**pixel_depth-1
    detector_image = img.astype(np.uint16)  # Final image as the Hartamnngram
    return detector_image