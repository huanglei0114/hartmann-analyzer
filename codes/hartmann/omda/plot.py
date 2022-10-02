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


import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1

def add_colorbar(im: plt.cm.ScalarMappable,
                 title: str = None,
                 aspect: int = 10,
                 pad_fraction: float = 1.5,
                 **kwargs):
    """
    Add a vertical color bar to an image plot

    Parameters
    ----------
        im:  `matplotlib.cm.ScalarMappable`
            The image to be described by the colorbar
        title: `str`
            The title of the colorbar
        aspect: `int`
            The aspect ratio between width and height
        pad_fraction: `float`
            The padding fraction
    Returns
    -------
        cbar: `matplotlib.pyplot.colorbar`
            The added colorbar
    """

    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    cbar = im.axes.figure.colorbar(im, cax=cax, **kwargs)

    if title is not None:
        cbar.ax.set_title(title)

    return cbar
