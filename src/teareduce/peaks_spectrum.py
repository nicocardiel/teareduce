# -*- coding: utf-8 -*-
#
# Copyright 2015-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from pathlib import Path


def find_peaks_spectrum(sx, nwinwidth, deltaflux=0, threshold=0, debugplot=False):
    """Find peaks in array.

    The algorithm imposes that the signal at both sides of the peak
    decreases monotonically.

    Parameters
    ----------
    sx : numpy array
        Input array containing spectrum.
    nwinwidth : int
        Width of the window where each peak must be found.
    deltaflux : float
        Minimum flux difference between the peak and the rest of
        pixels within the searching window.
    threshold : float
        Minimum signal in the peaks.
    debugplot : bool
        If True, display intermediate plots and results.

    Returns
    -------
    ixpeaks : 1d numpy array, int
        Peak locations, in array coordinates (integers).

    """

    if not isinstance(sx, np.ndarray):
        raise ValueError("sx=" + str(sx) + " must be a numpy.ndarray")
    elif sx.ndim != 1:
        raise ValueError("sx.ndim=" + str(sx.ndim) + " must be 1")

    sx_shape = sx.shape
    nmed = nwinwidth//2

    if debugplot:
        print('find_peaks_spectrum> sx shape......:', sx_shape)
        print('find_peaks_spectrum> nwinwidth.....:', nwinwidth)
        print('find_peaks_spectrum> nmed..........:', nmed)
        print('find_peaks_spectrum> data_threshold:', threshold)
        print('find_peaks_spectrum> the first and last', nmed,
              'pixels will be ignored')

    xpeaks = []  # list to store the peaks
    
    if sx_shape[0] < nwinwidth:
        print('find_peaks_spectrum> sx shape......:', sx_shape)
        print('find_peaks_spectrum> nwinwidth.....:', nwinwidth)
        raise ValueError('sx.shape < nwinwidth')
    
    i = nmed
    while i < sx_shape[0] - nmed:
        if sx[i] > threshold:
            peak_ok = True
            j = 0
            loop = True
            while loop:
                if sx[i - nmed + j] > sx[i - nmed + j + 1]:
                    peak_ok = False
                j += 1
                loop = (j < nmed) and peak_ok
            if peak_ok:
                j = nmed + 1
                loop = True
                while loop:
                    if sx[i - nmed + j - 1] < sx[i - nmed + j]:
                        peak_ok = False
                    j += 1
                    loop = (j < nwinwidth) and peak_ok
            if peak_ok:
                if sx[i] - np.min(sx[i-nmed:(i+nmed+1)]) > deltaflux:
                    xpeaks.append(i)
                    i += nwinwidth - 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    ixpeaks = np.array(xpeaks)

    if debugplot:
        print('find_peaks_spectrum> number of peaks found:', len(ixpeaks))
        print(ixpeaks)

    return ixpeaks


def refine_peaks_spectrum(sx, ixpeaks, nwinwidth, method=None,
                          plots=False, title=None, pdf_output=None, pdf_only=False):
    """Refine line peaks in spectrum.

    Parameters
    ----------
    sx : numpy array
        Input array containing the spectrum.
    ixpeaks : numpy array
        Initial peak locations, in array coordinates (integers).
        These values can be the output from the function
        find_peaks_spectrum().
    nwinwidth : int
        Width of the window where each peak must be refined.
    method : string
        "poly2" : fit to a 2nd order polynomial
        "gaussian" : fit to a Gaussian
    plots : bool
        If True, display intermediate plots and results.
    title : str or None
            Plot title.
    pdf_output : str or None
        If not None, save plots in PDF file.
    pdf_only : bool
            If True, close the plot after generating the PDF file.

    Returns
    -------
    fxpeaks : numpy array
        Refined peak locations, in array coordinates (floats).
    sxpeaks : numpy array
        When fitting Gaussians, this array stores the fitted line
        widths (sigma). Otherwise, this array returns zeros.

    """

    nmed = nwinwidth//2

    numpeaks = len(ixpeaks)

    fxpeaks = np.zeros(numpeaks)
    sxpeaks = np.zeros(numpeaks)

    if plots:
        npprow = 4
        nrows = int(numpeaks / npprow)
        if numpeaks % npprow != 0:
            nrows += 1
        fig, axarr = plt.subplots(nrows=nrows, ncols=npprow,
                                  figsize=(15, 4*nrows))
        axarr = axarr.flatten()
        for ax in axarr:
            ax.axis('off')
    else:
        fig = None
        axarr = None

    for iline in range(numpeaks):
        jmax = ixpeaks[iline]
        x_fit = np.arange(-nmed, nmed+1, dtype=float)
        # prevent possible problem when fitting a line too near to any
        # of the borders of the spectrum
        j1 = jmax - nmed
        j2 = jmax + nmed + 1
        if j1 < 0:
            j1 = 0
            j2 = 2 * nmed + 1
            if j2 >= len(sx):
                raise ValueError("Unexpected j2=" + str(j2) +
                                 " value when len(sx)=" + str(len(sx)))
        if j2 >= len(sx):
            j2 = len(sx)
            j1 = j2 - (2 * nmed + 1)
            if j1 < 0:
                raise ValueError("Unexpected j1=" + str(j1) +
                                 " value when len(sx)=" + str(len(sx)))
        # it is important to create a copy in the next instruction in
        # order to avoid modifying the original array when normalizing
        # the data to be fitted
        y_fit = np.copy(sx[j1:j2].astype(float))
        sx_peak_flux = max(y_fit)
        if sx_peak_flux != 0:
            y_fit /= sx_peak_flux  # normalize to maximum value

        if method == "gaussian":
            # check that there are no negative or null values
            if min(y_fit) <= 0:
                if plots:
                    print("WARNING: negative or null value encountered" +
                          " in refine_peaks_spectrum with gaussian.")
                    print("         Using poly2 method instead.")
                final_method = "poly2"
            else:
                final_method = "gaussian"
        else:
            final_method = method

        if final_method == "poly2":
            poly_funct = Polynomial.fit(x_fit, y_fit, 2)
            poly_funct = Polynomial.cast(poly_funct)
            coef = poly_funct.coef
            if len(coef) == 3:
                if coef[2] != 0:
                    refined_peak = -coef[1]/(2.0*coef[2]) + jmax
                else:
                    refined_peak = float(jmax)
            else:
                refined_peak = float(jmax)
        elif final_method == "gaussian":
            poly_funct = Polynomial.fit(x_fit, np.log(y_fit), 2)
            poly_funct = Polynomial.cast(poly_funct)
            coef = poly_funct.coef
            if len(coef) == 3:
                if coef[2] != 0:
                    refined_peak = -coef[1]/(2.0*coef[2]) + jmax
                else:
                    refined_peak = 0.0 + jmax
                if coef[2] >= 0:
                    sxpeaks[iline] = None
                else:
                    sxpeaks[iline] = np.sqrt(-1 / (2.0 * coef[2]))
            else:
                refined_peak = 0.0 + jmax
                sxpeaks[iline] = None
        else:
            raise ValueError("Invalid method=" + str(final_method) + " value")

        fxpeaks[iline] = refined_peak

        if plots:
            ax = axarr[iline]
            ax.axis('on')
            xmin = min(x_fit)-1
            xmax = max(x_fit)+1
            ymin = 0
            ymax = max(y_fit)*1.10
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_xlabel('index around initial integer peak')
            ax.set_ylabel('Normalized number of counts')
            ax.set_title(f'Fit at array index {jmax}\n'
                         f'Refined peak location: {refined_peak:.2f}')
            ax.plot(x_fit, y_fit, "bo")
            x_plot = np.linspace(start=-nmed, stop=nmed, num=1000, dtype=float)
            if final_method == "poly2":
                y_plot = poly_funct(x_plot)
            elif final_method == "gaussian":
                amp = np.exp(coef[0] - coef[1] * coef[1] / (4 * coef[2]))
                x0 = -coef[1] / (2.0 * coef[2])
                sigma = np.sqrt(-1 / (2.0 * coef[2]))
                y_plot = amp * np.exp(-(x_plot - x0)**2 / (2 * sigma**2))
            else:
                raise ValueError("Invalid method=" + str(final_method) +
                                 " value")
            ax.plot(x_plot, y_plot, color="red")
            for ipix in range(-nmed, nmed+2):
                ax.axvline(ipix-0.5, linestyle='--', color='gray')
            ax.text((xmin+xmax)/2, ymin + (ymax-ymin)/15,
                    f'(method={method})',
                    ha='center', backgroundcolor='white')

    if plots:
        if title is not None:
            plt.suptitle(f'{title}\n', fontsize=16)
        plt.tight_layout()
        if pdf_output is None:
            if pdf_only:
                raise ValueError('Unexpected pdf_only=True when pdf_output=None')
            plt.show()
        else:
            parent = Path(pdf_output).parents[0]
            stem = Path(pdf_output).stem
            fname = parent / f'{stem}_fitpeak.pdf'
            print(f'--> Saving PDF file: {fname}')
            plt.savefig(fname)
            if pdf_only:
                plt.close(fig)
            else:
                plt.show()
    else:
        if pdf_output is not None:
            raise ValueError('You must set plots=True to make use of pdf_output')

    return fxpeaks, sxpeaks
