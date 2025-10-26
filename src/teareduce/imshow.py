#
# Copyright 2022-2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from astropy.units import Unit
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshowme(data, return_objs=False, **kwargs):
    """Simple execution of teareduce.imshow.

    Parameters
    ----------
    data : numpy.ndarray
        2D array to be displayed
    return_objs : bool
        If True, return the Figure, Axes, AxesImage,
        color bar Axes and Colorbar instances.
    **kwargs : dict
        Additional parameters passed to imshow().

    Returns
    -------
    fig : matplotlib.figure.Figure
        Instance of Figure.
    ax : matplotlib.axes.Axes
        Instance of Axes.
    img : matplotlib AxesImage
        Instance returned by ax.imshow()
    cax : matplotlib.axes.Axes or None
        Instance of Axes where the color bar is drawn, or None if
        colorbar is False.
    cbar : matplotlib.colorbar.Colorbar or None
        Instance of Colorbar, or None if colorbar is False.
    """
    fig, ax = plt.subplots()
    img, cax, cbar = imshow(fig=fig, ax=ax, data=data, return_objs=True, **kwargs)
    if return_objs:
        return fig, ax, img, cax, cbar


def imshow(fig=None, ax=None, data=None, ds9mode=False,
           crpix1=1, crval1=None, cdelt1=None, cunit1=None, cunitx=Unit('Angstrom'),
           xlabel=None, ylabel=None, title=None,
           colorbar=True, cblabel='Number of counts',
           return_objs=True,
           **kwargs):
    """Call imshow() with color bar and default labels.

    If crpix1, crval1, cdelt1 and cunit1 are not None, a wavelengh
    scale is also displayed. In this case, the colorbar is not shown
    because there is a conflict (to be solved).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Instance of Figure.
    ax : matplotlib.axes.Axes
        Instance of Axes.
    data : numpy array
        2D array to be displayed.
    ds9mode : bool
        If True, the extent parameter is set to
        [0.5, NAXIS1+0.5, 0.5, NAXIS2+0.5]
        to mimic the DS9 display.
    crpix1 : astropy.units.Quantity
        Float number providing the CRPIX1 value: the reference pixel
        for which CRVAL1 is given.
    crval1 : astropy.units.Quantity
        Float number providing the CRVAL1 value: wavelength at the
        center of the first pixel.
    cdelt1 : astropy.units.Quantity
        Float number providing CDELT1 value: wavelength increment
        per pixel.
    cunit1 : astropy.units.Quantity
        Float number providing CUNIT1: the units employed in the
        wavelength calibration.
    cunitx : astropy.units.Quantity
        Units employed to display the wavelength scale. It can be
        different from cunit1.
    xlabel : str or None
        X label.
    ylabel : str or None
        Y label.
    title : str or None
        Plot title.
    colorbar : bool
        If True, display color bar.
    cblabel : str
        Color bar label.
    return_objs : bool
        If True, return AxesImage, color bar Axes and Colorbar
        instances.
    **kwargs : dict
        Additional parameters passed to imshow().

    Return
    ------
    img : matplotlib AxesImage
        Instance returned by ax.imshow()
    cax : matplotlib.axes.Axes or None
        Instance of Axes where the color bar is drawn, or None if
        colorbar is False.
    cbar : matplotlib.colorbar.Colorbar or None
        Instance of Colorbar, or None if colorbar is False.
    """

    # protections
    if not isinstance(fig, Figure):
        raise ValueError("Unexpected 'fig' argument")
    if not isinstance(ax, Axes):
        raise ValueError("Unexpected 'ax' argument")

    wavecalib = False
    if crpix1 is not None and crval1 is not None and cdelt1 is not None and cunit1 is not None:
        if 'extent' in kwargs:
            raise ValueError('extent parameter can not be used with a wavelength calibration scale')
        if 'aspect' in kwargs:
            raise ValueError('aspect parameter can not be used with a wavelength calibration scale')
        naxis2, naxis1 = data.shape
        xmin, xmax = -0.5, naxis1 - 0.5
        ymin, ymax = -0.5, naxis2 - 0.5
        u_pixel = Unit('pixel')
        xminwv = crval1 + (xmin * u_pixel - crpix1 + 1 * u_pixel) * cdelt1
        xmaxwv = crval1 + (xmax * u_pixel - crpix1 + 1 * u_pixel) * cdelt1
        xminwv = xminwv.to(cunitx).value
        xmaxwv = xmaxwv.to(cunitx).value
        extent = [xminwv, xmaxwv, ymin, ymax]
        xlabel = f'Wavelength ({cunitx})'
        aspect = 'auto'
        wavecalib = True
    else:
        if ds9mode:
            if 'extent' in kwargs:
                raise ValueError('extent parameter can not be used with ds9mode=True')
            naxis2, naxis1 = data.shape
            extent = [0.5, naxis1 + 0.5, 0.5, naxis2 + 0.5]
            if xlabel is None:
                xlabel = 'X pixel (from 1 to NAXIS1)'
            if ylabel is None:
                ylabel = 'Y pixel (from 1 to NAXIS2)'
        else:
            if xlabel is None:
                xlabel = 'X axis (array index)'
            if ylabel is None:
                ylabel = 'Y axis (array index)'
            if 'extent' in kwargs:
                extent = kwargs['extent']
                del kwargs['extent']
            else:
                extent = None
        if 'aspect' in kwargs:
            aspect = kwargs['aspect']
            del kwargs['aspect']
        else:
            aspect = None

    if 'origin' not in kwargs and 'interpolation' not in kwargs:
        img = ax.imshow(data, origin='lower', interpolation='None', extent=extent, aspect=aspect, **kwargs)
    elif 'origin' not in kwargs:
        img = ax.imshow(data, origin='lower', extent=extent, aspect=aspect, **kwargs)
    elif 'interpolation' not in kwargs:
        img = ax.imshow(data, interpolation='None', extent=extent, aspect=aspect, **kwargs)
    else:
        img = ax.imshow(data, extent=extent, aspect=aspect, **kwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # if a wavelength calibration is provided, display the index scale
    # on the top horizontal axis
    if wavecalib:

        def index2coord(i):
            return xminwv + (xmaxwv - xminwv) * i / (naxis1 - 1)

        def coord2index(x):
            return (naxis1 - 1) * (x - xminwv) / (xmaxwv - xminwv)

        ax_top = ax.secondary_xaxis('top', functions=(coord2index, index2coord))
        ax_top.set_xlabel('X axis (array index)')

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=Axes)
        cbar = fig.colorbar(img, cax=cax, label=cblabel)
    else:
        cax = None
        cbar = None

    if return_objs:
        return img, cax, cbar
