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


def qgpu2sc(wp: np.ndarray,
            quality_map: np.ndarray = np.array([]),
            start: np.ndarray = np.array([]),
            num_of_quality_steps: int = 128,
            ):
    """
    Quality guided phase unwrapping with stack chain

    Parameters
    ----------
        wp: `numpy.ndarray`
            The wrapped phase map
        quality_map: `numpy.ndarray`
            The quality map
        start: `numpy.ndarray`
            The starting pixel of the unwrapping process
        num_of_quality_steps: `int`
            The number of quality steps
    Returns
    -------
        uwp: `numpy.ndarray`
            The unwrapped phase map
    """

    sz = wp.shape
    if quality_map.size==0:
        quality_map = np.mat(np.hanning(sz[0])).T * np.hanning(sz[1])
    quality_thr = np.min(quality_map)
    mask = np.ones(sz, dtype=bool)
    if start.size == 0:
        start = np.where(quality_map == np.max(quality_map))
    start_row, start_col = start
    if start_row.size>1:
        start_row = int(start_row[0])
    if start_col.size>1:
        start_col = int(start_col[0])
    min_q = np.min(quality_map)
    max_q = np.max(quality_map)
    if min_q != max_q:
        quality_map = np.int32(np.round(((quality_map-min_q)/(max_q-min_q))*(num_of_quality_steps-1))+1)
        if quality_thr >= min_q:
            quality_thr = np.round(((quality_thr-min_q)/(max_q-min_q))*(num_of_quality_steps-1))+1
        elif quality_thr < min_q:
            quality_thr = 1
    else:
        quality_map = np.int32(quality_map/min_q)
        quality_thr = 1

    stack_chain = np.int32(np.zeros(num_of_quality_steps+1,))
    uwg_row = np.zeros((wp.size,), dtype=int)
    uwg_col = np.zeros((wp.size,), dtype=int)
    uwd_row = np.zeros((wp.size,), dtype=int)
    uwd_col = np.zeros((wp.size,), dtype=int)
    stack_n = np.zeros((wp.size,))
    uwp = np.zeros_like(wp)
    path_map = np.zeros_like(wp)
    queued_flag = np.zeros(sz, dtype=bool)
    quality_max = int(quality_map[start_row, start_col])
    stack_chain[quality_max] = 1
    pointer = 1
    unwr_order = 0
    uwd_row[stack_chain[quality_max]] = start_row
    uwd_col[stack_chain[quality_max]] = start_col
    uwg_row[stack_chain[quality_max]] = start_row
    uwg_col[stack_chain[quality_max]] = start_col

    path_map[start_row, start_col] = 1
    queued_flag[start_row, start_col] = True
    # Set unwrapping phase as wrapped phase for the starting point.
    uwp[start_row, start_col] = wp[start_row, start_col]
    # When quality_max is higher than quality_thr, flood fill.
    while quality_max >= quality_thr:
        # If stack_chain in quality_max level is currently empty, go to
        # quality_max-1 level.
        if stack_chain[quality_max] == 0:
            quality_max = quality_max-1
        else:
            # Unwrap current point.
            uwdrow = int(uwd_row[stack_chain[quality_max]])
            uwdcol = int(uwd_col[stack_chain[quality_max]])
            a = uwp[uwdrow, uwdcol]
            uwgrow = int(uwg_row[stack_chain[quality_max]])
            uwgcol = int(uwg_col[stack_chain[quality_max]])
            b = wp[uwgrow, uwgcol]
            uwp[uwgrow, uwgcol] = b - 2*np.pi*round((b-a)/(2*np.pi))

            # Temporal row and column of the unwrapping point.
            temp_row = int(uwg_row[stack_chain[quality_max]])
            temp_col = int(uwg_col[stack_chain[quality_max]])

            # Update path_map.
            path_map[temp_row, temp_col] = unwr_order
            unwr_order = unwr_order+1
            stack_chain[quality_max] = stack_n[stack_chain[quality_max]]
            if (temp_row > 0):
                # Check unwrapping state and mask validity.
                if (~queued_flag[temp_row-1, temp_col])\
                        and (mask[temp_row-1, temp_col]):

                    # upper:(row-1,col)
                    uwg_row[pointer] = int(temp_row-1)
                    uwg_col[pointer] = int(temp_col)
                    uwd_row[pointer] = int(temp_row)
                    uwd_col[pointer] = int(temp_col)

                    # Push stack_chain to the stack_n at pointer.
                    stack_n[pointer] = stack_chain[quality_map[uwg_row[pointer],
                                                 uwg_col[pointer]]]
                    # Push pointer to stack_chain.
                    stack_chain[quality_map[uwg_row[pointer],
                                    uwg_col[pointer]]] = pointer

                    # If the quality value of pushed point is bigger than the
                    # current quality_max value, set the quality_max as the
                    # quality value of pushed point.
                    if quality_map[uwg_row[pointer], uwg_col[pointer]] > quality_max:
                        quality_max = quality_map[uwg_row[pointer], uwg_col[pointer]]
                    # Queue the point.
                    queued_flag[uwg_row[pointer], uwg_col[pointer]] = True
                    # pointer++.
                    pointer = pointer+1

            # the nether neighboring point: (row+1,col)
            # Check dimensional validity.
            if (temp_row < sz[0]-1):
                # Check unwrapping state and mask validity.
                if (~queued_flag[temp_row+1, temp_col])\
                        and (mask[temp_row+1, temp_col]):
                    # nether:(row+1,col)
                    uwg_row[pointer] = int(temp_row+1)
                    uwg_col[pointer] = int(temp_col)
                    uwd_row[pointer] = int(temp_row)
                    uwd_col[pointer] = int(temp_col)

                    # Push stack_chain to the stack_n at pointer.
                    stack_n[pointer] = stack_chain[quality_map[uwg_row[pointer],
                                                 uwg_col[pointer]]]
                    # Push pointer to stack_chain.
                    stack_chain[quality_map[uwg_row[pointer],
                                    uwg_col[pointer]]] = pointer

                    # If the quality value of pushed point is bigger than the
                    # current quality_max value, set the quality_max as the
                    # quality value of pushed point.
                    if quality_map[uwg_row[pointer], uwg_col[pointer]] > quality_max:
                        quality_max = quality_map[uwg_row[pointer], uwg_col[pointer]]

                    # Queue the point.
                    queued_flag[uwg_row[pointer], uwg_col[pointer]] = True
                    # pointer++.
                    pointer = pointer+1

            # the left neighboring point: (row,col-1)
            # Check dimensional validity.
            if (temp_col > 0):
                # Check unwrapping state and mask validity.
                if (~queued_flag[temp_row, temp_col-1])\
                        and (mask[temp_row, temp_col-1]):

                    # left:(row,col-1)
                    uwg_row[pointer] = int(temp_row)
                    uwg_col[pointer] = int(temp_col-1)
                    uwd_row[pointer] = int(temp_row)
                    uwd_col[pointer] = int(temp_col)

                    # Push stack_chain to the stack_n at pointer.
                    stack_n[pointer] = stack_chain[quality_map[uwg_row[pointer],
                                                 uwg_col[pointer]]]
                    # Push pointer to stack_chain.
                    stack_chain[quality_map[uwg_row[pointer],
                                    uwg_col[pointer]]] = pointer

                    # If the quality value of pushed point is bigger than the
                    # current quality_max value, set the quality_max as the
                    # quality value of pushed point.
                    if quality_map[uwg_row[pointer], uwg_col[pointer]] > quality_max:
                        quality_max = quality_map[uwg_row[pointer], uwg_col[pointer]]

                    # Queue the point.
                    queued_flag[uwg_row[pointer], uwg_col[pointer]] = True
                    # pointer++.
                    pointer = pointer+1

            # the right neighboring point: (row,col+1)
            # Check dimensional validity.
            if (temp_col < sz[1]-1):
                # Check unwrapping state and mask validity.
                if (~queued_flag[temp_row, temp_col+1])\
                        and (mask[temp_row, temp_col+1]):

                    # right:(row,col+1)
                    uwg_row[pointer] = int(temp_row)
                    uwg_col[pointer] = int(temp_col+1)
                    uwd_row[pointer] = int(temp_row)
                    uwd_col[pointer] = int(temp_col)

                    # Push stack_chain to the stack_n at pointer.
                    stack_n[pointer] = stack_chain[quality_map[uwg_row[pointer],
                                                 uwg_col[pointer]]]
                    # Push pointer to stack_chain.
                    stack_chain[quality_map[uwg_row[pointer],
                                    uwg_col[pointer]]] = pointer

                    # If the quality value of pushed point is bigger than the
                    # current quality_max value, set the quality_max as the
                    # quality value of pushed point.
                    if quality_map[uwg_row[pointer], uwg_col[pointer]] > quality_max:
                        quality_max = quality_map[uwg_row[pointer], uwg_col[pointer]]

                    # Queue the point.
                    queued_flag[uwg_row[pointer], uwg_col[pointer]] = True
                    # pointer++.
                    pointer = pointer+1

    # path_map=(unwr_order-path_map)*mask;
    return uwp
