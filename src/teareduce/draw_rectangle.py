#
# Copyright 2022-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


def draw_rectangle(ax, image_data, x1, x2, y1, y2,
                   color='white', text=False, ndigits=1, fontsize=15):
    """Auxiliary function to display (mean, std) in a rectangle.

    The mean and standard deviation are computed in the rectangle
    defined by x1, x2, y1, and y2.

    Parameters
    ----------
    ax : Axes
        Instance of matplotlib.axes.Axes.
    image_data : 2d numpy array
        Image where the statistical anaylisis will be performed.
    x1 : int
        Lower image index (column number).
    x2 : int
        Upper image index (column number).
    y1 : int
        Lower image index (row number).
    y2 : int
        Upper image index (row number).
    color : str
        Color for text labels.
    text : bool
        If True, display labels with information.
    ndigits : int
        Number of decimal digits.
    fontsize : int
       Size fo text font.

    Returns
    -------
    mean : float
        Mean value computed in the requested image rectangle.
    std : float
        Standard deviation computed in the requested image rectangle.

    """

    mean = image_data[y1:y2, x1:x2].mean()
    std = image_data[y1:y2, x1:x2].std()

    ax.plot((x1, x1), (y1, y2), color, lw=1)
    ax.plot((x2, x2), (y1, y2), color, lw=1)
    ax.plot((x1, x2), (y1, y1), color, lw=1)
    ax.plot((x1, x2), (y2, y2), color, lw=1)

    if text:
        ax.text((x1+x2)/2, y1+(y2-y1)/8, 
                '{:.{prec}f}'.format(mean, prec=ndigits), 
                ha='center', va='center', color=color, fontsize=fontsize)        
        ax.text((x1+x2)/2, y2-(y2-y1)/8,
                '{:.{prec}f}'.format(std, prec=ndigits), 
                ha='center', va='top', color=color, fontsize=fontsize)

    return mean, std
