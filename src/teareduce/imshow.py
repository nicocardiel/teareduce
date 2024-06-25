#
# Copyright 2022-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import astropy.units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow(fig=None, ax=None, data=None,
           crpix1=1, crval1=None, cdelt1=None, cunit1=None, cunitx=u.Angstrom,
           xlabel=None, ylabel=None, title=None,
           colorbar=True, cblabel='Number of counts',
           **kwargs):
    """Call imshow() with color bar and default labels.

    If crpix1, crval1, cdelt1 and cunit1 are not None, a wavelengh
    scale is also displayed. In this case, the colorbar is not shown
    because there is a conflict (to be solved).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    data : numpy array
        2D array to be displayed.
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

    Return
    ------
    img : matplotlib AxesImage
        Instance returned by ax.imshow()

    """

    if xlabel is None:
        xlabel = 'X axis (array index)'

    if ylabel is None:
        ylabel = 'Y axis (array index)'

    if crpix1 is not None and crval1 is not None and cdelt1 is not None and cunit1 is not None:
        if 'extent' in kwargs:
            raise ValueError('extent parameter can not be used with a wavelength calibration scale')
        if 'aspect' in kwargs:
            raise ValueError('aspect parameter can not be used with a wavelength calibration scale')
        naxis2, naxis1 = data.shape
        xmin, xmax = -0.5, naxis1 - 0.5
        ymin, ymax = -0.5, naxis2 - 0.5
        xminwv = crval1 + (xmin * u.pixel - crpix1 + 1 * u.pixel) * cdelt1
        xmaxwv = crval1 + (xmax * u.pixel - crpix1 + 1 * u.pixel) * cdelt1
        extent = [xminwv.to(cunitx).value, xmaxwv.to(cunitx).value, ymin, ymax]
        xlabel = f'Wavelength ({cunitx})'
        aspect = 'auto'
    else:
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

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax, label=cblabel)
