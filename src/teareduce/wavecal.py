# -*- coding: utf-8 -*-
#
# Copyright 2015-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u
from astropy.wcs import wcs
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy.polynomial import Polynomial
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tqdm.notebook import tqdm

from .ctext import ctext
from .imshow import imshow
from .peaks_spectrum import find_peaks_spectrum, refine_peaks_spectrum
from .polfit import polfit_residuals_with_sigma_rejection
from .sliceregion import SliceRegion1D


class TeaWaveCalibration:
    """Auxiliary class to compute and apply wavelength calibration.
    
    It is assumed that the wavelength and spatial directions correspond
    to the X (array columns) and Y (array rows) axes, respectively.
    
    Attributes
    ----------
    ns_window : int
        Number of spectra to be (median) averaged. It must be odd.
    threshold : float
        Minimum signal to search for peaks.
    sigma_smooth : int
        Standard deviation for Gaussian kernel to smooth median spectrum.
        A value of 0 means that no smoothing is performed.
    nx_window : int
        Number of pixels (spectral direction) of the window where the 
        peaks are sought. It must be odd.
    delta_flux : float
        Minimum difference between the flux at the line center and the
        flux at the borders of the window where the peaks are sought.
    method : str
        Function to be employed when fitting line peaks:
        - "poly2" : fit to a 2nd order polynomial
        - "gaussian" : fit to a Gaussian
    degree_cdistortion : int
        Polynomial degree to fit curvature of each arc line.
    degree_wavecalib : int
        Polynomial degree to fit wavelength variation with pixel number.
    peak_wavelengths : astropy.unit.Quantity or None
        Array providing the wavelengths of the line peaks, with units.

    """

    fits_keyword_list = ['ns_window', 'threshold', 'sigma_smooth',
                         'nx_window', 'delta_flux', 'method',
                         'degree_cdistortion', 'degree_wavecalib']

    def __init__(self,
                 ns_window=11,
                 threshold=0,
                 sigma_smooth=0,
                 nx_window=5,
                 delta_flux=0,
                 method='gaussian',
                 degree_cdistortion=1,
                 degree_wavecalib=1,
                 peak_wavelengths=None
                 ):
        if ns_window < 1:
            raise ValueError(f'ns_window={ns_window} must be >= 1')
        if ns_window % 2 != 1:
            raise ValueError(f'ns_window={ns_window} must be an odd number')
        if nx_window < 3:
            raise ValueError(f'nx_window={nx_window} must be >= 3')
        if nx_window % 2 != 1:
            raise ValueError(f'nx_window={nx_window} must be an odd number')
        if sigma_smooth < 0:
            raise ValueError(f'sigma_smooth={sigma_smooth} must be >= 0')

        self.ns_window = ns_window
        self.threshold = threshold
        self.sigma_smooth = sigma_smooth
        self.nx_window = nx_window
        self.delta_flux = delta_flux
        self.method = method
        self.degree_cdistortion = degree_cdistortion
        self.degree_wavecalib = degree_wavecalib
        self.peak_wavelengths = peak_wavelengths
        self.reset_image()

    def reset_image(self):
        self._nlines_reference = None
        self._naxis1 = None
        self._naxis2 = None
        self._xpeaks_all_lines_array = None
        self._valid_scans = None
        self._list_poly_cdistortion = None
        self._array_poly_wav = None
        self._array_poly_pix = None
        self._array_residual_std_wav = None
        self._array_residual_std_pix = None
        self._array_crval1_linear = None
        self._array_cdelt1_linear = None
        self._array_crmax1_linear = None

    @classmethod
    def read(cls, filename, plots=False, pdf_output=None, pdf_only=False, silent_mode=False):
        """Constructor from FITS file.

        Parameters
        ----------
        filename : str
            Input FITS file name.
        plots : bool
            If True, show intermediate plots.
        pdf_output : str or None
            If not None, save plots in PDF file.
        pdf_only : bool
            If True, close the plot after generating the PDF file.
        silent_mode : bool
            If True, do not show information messages.

        """

        # read primary header
        header = fits.getheader(filename)
        kwargs = dict()
        for keyword in cls.fits_keyword_list:
            key = keyword[:8].upper()
            kwargs[keyword] = header[key]
        wavecalib = cls(**kwargs)
        wavecalib._naxis1 = header['_NAXIS1']
        wavecalib._naxis2 = header['_NAXIS2']
        wavecalib._nlines_reference = header['_NLINES']
        wavecalib.peak_wavelengths = np.zeros(wavecalib._nlines_reference) * u.Unit(header['WPEAKUNI'])
        for i in range(wavecalib._nlines_reference):
            wavecalib.peak_wavelengths[i] = header[f'WPEAK{i+1:03d}'] * u.Unit(header['WPEAKUNI'])

        # read primary data
        data = fits.getdata(filename)
        naxis2, naxis1 = data.shape
        if naxis1 != wavecalib.degree_wavecalib + 1:
            raise ValueError(f'Number of coefficients: {naxis1} does not match '
                             f'expected value degree_wavecalib + 1: {wavecalib.degree_wavecalib+1}')
        if naxis2 != wavecalib._naxis2:
            raise ValueError(f'Unexpected inconsistency: NAXIS2: {naxis2} != _NAXIS2: {wavecalib._naxis2}')
        wavecalib._array_poly_wav = data

        # read INV_POLY extension
        wavecalib._array_poly_pix = fits.getdata(filename, extname='INV_POLY')

        # read CDISTOR extension
        cdistor_array = fits.getdata(filename, extname='CDISTOR')
        wavecalib._list_poly_cdistortion = []
        for i in range(wavecalib._nlines_reference):
            wavecalib._list_poly_cdistortion.append(Polynomial(cdistor_array[i, :]))

        # read COEFF extension
        tbl_header = fits.getheader(filename, extname='COEFF')
        tbl_data = fits.getdata(filename, extname='COEFF')
        wavecalib._array_residual_std_wav = tbl_data.residual_std_wav * u.Unit(tbl_header['TUNIT1'])
        wavecalib._array_residual_std_pix = tbl_data.residual_std_pix * u.Unit(tbl_header['TUNIT2'])
        wavecalib._array_crval1_linear = tbl_data.crval1_linear * u.Unit(tbl_header['TUNIT3'])
        wavecalib._array_cdelt1_linear = tbl_data.cdelt1_linear * u.Unit(tbl_header['TUNIT4'])
        wavecalib._array_crmax1_linear = tbl_data.crmax1_linear * u.Unit(tbl_header['TUNIT5'])

        if not silent_mode:
            print(f'>>> Reading file.........: {filename}')

        # estimate crval1, cdelt1
        wavecalib.estimate_crval1_cdelt1(silent_mode=silent_mode)

        if plots:
            wavecalib._plot_wavecoef(pdf_output=pdf_output, pdf_only=pdf_only)
        else:
            if pdf_output is not None:
                raise ValueError('You must set plots=True to make use of pdf_output')

        return wavecalib

    def __repr__(self):
        output = f'{self.__class__.__name__}('
        for i, item in enumerate(self.__dict__):
            if i != 0:
                output += ','
            output += f'\n    {item}={self.__dict__[item]!r}'
        output += '\n)'
        return output

    def _find_peaks_scan(self, data, ns1, ns2, plot_peaks, title=None,
                         pdf_output=None, pdf_only=False):
        """Compute location of line peaks in (median) averaged spectrum.

        Parameters
        ----------
        data : numpy array
            Array with 2D image. It is assumed that the spectral / spatial
            direction corrrespond to the X / Y axis.
        ns1 : int
            Initial row (spectrum number; following the FITS convention) to
            extract a (median) averaged spectrum.
        ns2 : int
            Final row (spectrum number; following the FITS convention) to
            extract a (median) averaged spectrum.
        plot_peaks : bool
            If True, display the fit of each individual peak.
        title : str or None
            Plot title.
        pdf_output : str or None
            If not None, save plots in PDF file.
        pdf_only : bool
            If True, close the plot after generating the PDF file.

        Returns
        -------
        xpeaks : numpy array
            Refined peak locations (float numbers), in array coordinates.
        ixpeaks : numpy array
            Initial peak locations (integer values), in array coordinates.
        sp_median_smooth : numpy array
            Smoothed spectrum where the peaks have been found.

        """

        sp_median = np.median(data[(ns1-1):ns2, :], axis=0)

        if self.sigma_smooth > 0:
            sp_median_smooth = gaussian_filter1d(input=sp_median, sigma=self.sigma_smooth)
        else:
            sp_median_smooth = sp_median

        ixpeaks = find_peaks_spectrum(
            sx=sp_median_smooth,
            nwinwidth=self.nx_window,
            deltaflux=self.delta_flux,
            threshold=self.threshold
        )

        if not plot_peaks:
            pdf_output_ = None
        else:
            pdf_output_ = pdf_output
        xpeaks, sxpeaks = refine_peaks_spectrum(
            sx=sp_median_smooth,
            ixpeaks=ixpeaks,
            nwinwidth=self.nx_window,
            method=self.method,
            plots=plot_peaks,
            title=title,
            pdf_output=pdf_output_,
            pdf_only=pdf_only
        )

        return xpeaks, ixpeaks, sp_median_smooth

    def compute_xpeaks_reference(self,
                                 data,
                                 ns_range=None,
                                 threshold=None,
                                 sigma_smooth=None,
                                 nx_window=None,
                                 delta_flux=None,
                                 method=None,
                                 plot_spectrum=False,
                                 plot_peaks=False,
                                 title=None,
                                 pdf_output=None,
                                 pdf_only=False):
        """Compute location of line peaks in reference spectrum.

        Parameters
        ----------
        data : numpy array
            Array with 2D image. It is assumed that the spectral / spatial
            direction corrrespond to the X / Y axis.
        ns_range : SliceRegion1D instance or None
            Region indicating the initial and final row
            (spectrum number) to extract a (median) averaged spectrum.
        threshold : float or None
            Minimum signal to search for peaks.
        sigma_smooth : int or None
            Standard deviation for Gaussian kernel to smooth median spectrum.
            A value of 0 means that no smoothing is performed.
        nx_window : int or None
            Number of pixels (spectral direction) of the window where the 
            peaks are sought. It must be odd.
        delta_flux : float
            Minimum difference between the flux at the line center and the
            flux at the borders of the window where the peaks are sought.
        method : str or None
            Function to be employed when fitting line peaks:
            - "poly2" : fit to a 2nd order polynomial
            - "gaussian" : fit to a Gaussian
        plot_spectrum : bool
            If True, plot median spectrum and peak location.
        plot_peaks : bool
            If True, display the fit of each individual peak.
        title : str or None
            Plot title.
        pdf_output : str or None
            If not None, save plots in PDF file.
        pdf_only : bool
            If True, close the plot after generating the PDF file.

        Returns
        -------
        xpeaks : numpy array
            Refined peak locations (float numbers), in array coordinates,
            with units of pixel.
        ixpeaks : numpy array
            Initial peak locations (integer values), in array coordinates.
        sp_median_smooth : numpy_array
            Median spectrum where the peaks have been sought.

        """

        if threshold is not None:
            self.threshold = threshold
        if sigma_smooth is not None:
            self.sigma_smooth = sigma_smooth
        if nx_window is not None:
            self.nx_window = nx_window
        if delta_flux is not None:
            self.delta_flux = delta_flux
        if method is not None:
            self.method = method
        
        naxis2, naxis1 = data.shape

        if self._naxis1 is None:
            self._naxis1 = naxis1
        else:
            if naxis1 != self._naxis1:
                raise ValueError(f'Unexpected naxis1: {naxis1}')
        
        if self._naxis2 is None:
            self._naxis2 = naxis2
        else:
            if naxis2 != self._naxis2:
                raise ValueError(f'Unexpected naxis1: {naxis2}')

        valid_ns_range = SliceRegion1D(np.s_[1:naxis2], mode='fits')
        if ns_range is None:
            ns_range = valid_ns_range

        if ns_range.within(valid_ns_range):
            pass
        else:
            raise ValueError(f'{ns_range} outside {valid_ns_range}]')

        ns1, ns2 = ns_range.fits.start, ns_range.fits.stop

        # initial median spectrum
        xpeaks, ixpeaks, sp_median_smooth = self._find_peaks_scan(
            data=data, 
            ns1=ns1,
            ns2=ns2,
            plot_peaks=plot_peaks,
            title=title,
            pdf_output=pdf_output,
            pdf_only=pdf_only
        )

        if plot_spectrum:
            sp_minimum = sp_median_smooth.min()
            if sp_minimum <= 0:
                yplot = sp_median_smooth + sp_minimum + 1
                ylabel = f'Number of counts (+{sp_minimum+1:.2f})'
            else:
                yplot = sp_median_smooth
                ylabel = 'Number of counts'
            fig, ax = plt.subplots(figsize=(15, 5))
            xplot = np.arange(naxis1)
            ax.plot(xplot, yplot, lw=1)
            ax.set_xlabel('X axis (array index)')
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            ax.set_title(f'Median spectrum (from scans {ns1} to {ns2})')
            if self.threshold > 0:
                ax.axhline(self.threshold, ls='--', color='C1')
            ax.plot(xpeaks, yplot[ixpeaks], 'ro')
            ymin, ymax = ax.get_ylim()
            dy = (np.log10(ymax) - np.log10(ymin)) / 40
            for ip, (xp, yp) in enumerate(zip(xpeaks, yplot[ixpeaks])):
                ax.text(xp, yp*10**dy, ip+1, ha='center')
            if title is not None:
                plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            if pdf_output is None:
                if pdf_only:
                    raise ValueError('Unexpected pdf_only=True when pdf_output=None')
                plt.show()
            else:
                print(f'--> Saving PDF file: {pdf_output}')
                plt.savefig(pdf_output)
                if pdf_only:
                    plt.close(fig)
                else:
                    plt.show()
        else:
            if pdf_output is not None:
                raise ValueError('You must set plot_spectrum=True to make use of pdf_output')

        return xpeaks * u.pixel, ixpeaks * u.pixel, sp_median_smooth

    @u.quantity_input(xpeaks_reference=u.pixel)
    def compute_xpeaks_image(self,
                             data,
                             xpeaks_reference=None,
                             ns_range=None,
                             direction='up',
                             ns_window = None,
                             threshold = None,
                             sigma_smooth = None,
                             nx_window = None,
                             delta_flux = None,
                             method = None,
                             plots=False,
                             title=None,
                             pdf_output=None,
                             pdf_only=False,
                             disable_tqdm=True):
        """Compute location of line peaks.

        Parameters
        ----------
        data : numpy array
            Array with 2D image. It is assumed that the spectral / spatial
            direction corrrespond to the X / Y axis.
        xpeaks_reference : numpy array
            Reference location of line peaks (array coordinates).
        ns_range : SliceRegion1D or None
            Initial and final row (spectrum number) defining the
            spectra where the line peaks are going to be sought.
        direction : str
            String indicating how the imgage is scanned:
            - 'up': from bottom to top
            - 'down': from top to bottom
        ns_window : int or None
            Number of spectra to be (median) averaged. It must be odd.
        threshold : float or None
            Minimum signal to search for peaks.
        sigma_smooth : int or None
            Standard deviation for Gaussian kernel to smooth median spectrum.
            A value of 0 means that no smoothing is performed.
        nx_window : int or None
            Number of pixels (spectral direction) of the window where the 
            peaks are sought. It must be odd.
        delta_flux : float
            Minimum difference between the flux at the line center and the
            flux at the borders of the window where the peaks are sought.
        method : str or None
            Function to be employed when fitting line peaks:
            - "poly2" : fit to a 2nd order polynomial
            - "gaussian" : fit to a Gaussian
        plots : bool
            If True, display intermediate plots.
        title : str or None
            Title to be displayed when plotting the 2D image with
            the location of the peaks.
        pdf_output : str or None
            If not None, save plots in PDF file.
        pdf_only : bool
            If True, close the plot after generating the PDF file.
        disable_tqdm : bool
            If True, disable tqdm progress bar.

        """

        if xpeaks_reference is None:
            raise ValueError('You must provide a valid xpeaks_reference array')

        if ns_window is not None:
            self.ns_window = ns_window
        if threshold is not None:
            self.threshold = threshold
        if sigma_smooth is not None:
            self.sigma_smooth = sigma_smooth
        if nx_window is not None:
            self.nx_window = nx_window
        if delta_flux is not None:
            self.delta_flux = delta_flux
        if method is not None:
            self.method = method

        naxis2, naxis1 = data.shape

        if self._nlines_reference is None:
            self._nlines_reference = len(xpeaks_reference)
            self._naxis1 = naxis1
            self._naxis2 = naxis2
            # array to store the location of peaks in all the spectra
            self._xpeaks_all_lines_array = np.zeros((self._naxis2, self._nlines_reference))
            self._valid_scans = np.array([False] * naxis2)
        else:
            if self._nlines_reference != len(xpeaks_reference):
                raise ValueError(f'Number of peaks in {xpeaks_reference} '
                                 f'is different than the expected value: {self._nlines_reference}')
            if self._naxis1 != naxis1:
                raise ValueError(f'Unexpected naxis1={naxis1} (expected value={self._naxis1}')
            if self._naxis2 != naxis2:
                raise ValueError(f'Unexpected naxis1={naxis2} (expected value={self._naxis2}')

        valid_ns_range = SliceRegion1D(np.s_[1:naxis2], mode='fits')
        if ns_range is None:
            ns_range = valid_ns_range

        if ns_range.within(valid_ns_range):
            pass
        else:
            raise ValueError(f'{ns_range} outside {valid_ns_range}]')

        ns_min_fits_, ns_max_fits_ = ns_range.fits.start, ns_range.fits.stop

        if direction in ['up', 'down']:
            if direction == 'up':
                ns_step = 1
                ns_min_fits = ns_min_fits_
                ns_max_fits = ns_max_fits_
            else:
                ns_step = -1
                ns_min_fits = ns_max_fits_
                ns_max_fits = ns_min_fits_
        else:
            raise ValueError(f'Unexpected direction value: {direction}')

        # display previously found peaks
        if plots:
            color_previous = ['blue', 'cyan']
            fig, ax = plt.subplots(figsize=(15, 15*naxis2/naxis1))
            vmin, vmax = np.percentile(data, [5, 95])
            img = imshow(fig, ax, data, vmin=vmin, vmax=vmax, cmap='gray', title=title, aspect='auto')
            # display previously identified lines
            yplot = np.arange(naxis2)[self._valid_scans]
            for i in range(self._nlines_reference):
                xplot = self._xpeaks_all_lines_array[self._valid_scans, i]
                ax.scatter(xplot, yplot, s=2, c=color_previous[i % 2], marker=',', alpha=0.2)
        else:
            fig = None
            ax = None
            img = None

        # search for peaks in the 2D image
        dict_xpeaks = dict()
        for ns in tqdm(range(ns_min_fits, ns_max_fits + ns_step, ns_step), 
                       desc='Finding peaks', disable=disable_tqdm):
            ns1 = ns - self.ns_window // 2
            ns1 = max([ns1, min(ns_min_fits, ns_max_fits)])
            ns2 = ns + self.ns_window // 2
            ns2 = min([ns2, max(ns_min_fits, ns_max_fits)])
            xpeaks, ixpeaks, sp_median_smooth = self._find_peaks_scan(
                data=data, 
                ns1=ns1,
                ns2=ns2,
                plot_peaks=False,
                pdf_output=None
            )
            dict_xpeaks[ns] = xpeaks

        # find peaks
        for ns in range(ns_min_fits, ns_max_fits + ns_step, ns_step):
            xpeaks = dict_xpeaks[ns]
            if ns == ns_min_fits:
                xpeaks_predicted = xpeaks_reference.value
            else:
                if ns_step == -1:
                    ns1 = ns + 1
                    ns2 = ns1 + self.ns_window - 1
                    ns2 = min([ns2, ns_min_fits])
                else:
                    ns2 = ns - 1
                    ns1 = ns2 - self.ns_window + 1
                    ns1 = max([ns1, ns_min_fits])
                xpeaks_predicted = np.median(self._xpeaks_all_lines_array[(ns1-1):ns2, :], axis=0)
            for i in range(self._nlines_reference):
                value = xpeaks_predicted[i]
                # if there is no peak near to the expected location 
                # (within a distance given by self.nx_window) the line
                # is probably weak and we need to avoid jumping into
                # another line
                if min(np.abs(xpeaks - value)) < self.nx_window:
                    # use new peak location
                    imin = (np.abs(xpeaks - value)).argmin()
                    self._xpeaks_all_lines_array[ns-1, i] = xpeaks[imin]
                else:
                    # use predicted location
                    self._xpeaks_all_lines_array[ns-1, i] = value
            self._valid_scans[ns-1] = True

        if plots:
            color_new = ['red', 'magenta']
            # display starting spectrum
            xplot = []
            yplot = []
            ccolor = []
            marker = {'up': '^', 'down': 'v'}
            ns = ns_min_fits
            for i in range(self._nlines_reference):
                xplot.append(self._xpeaks_all_lines_array[ns-1, i])
                yplot.append(ns-1)
                ccolor.append(color_new[i % 2])
            ax.scatter(xplot, yplot, s=200, c=ccolor, marker=marker[direction], alpha=1.0)
            # display all peaks found
            xplot = []
            yplot = []
            for ns in dict_xpeaks:
                xplot += dict_xpeaks[ns].tolist()
                nlines_found = len(dict_xpeaks[ns])
                yplot += [ns-1] * nlines_found
            ax.scatter(xplot, yplot, s=2, c='green', marker=',', alpha=0.2)
            # display new identified lines
            for i in range(self._nlines_reference):
                xplot = []
                yplot = []
                for ns in dict_xpeaks:
                    xplot.append(self._xpeaks_all_lines_array[ns-1, i])
                    yplot.append(ns-1)
                ax.scatter(xplot, yplot, s=2, c=color_new[i % 2], marker=',', alpha=0.2)
            if pdf_output is None:
                if pdf_only:
                    raise ValueError('Unexpected pdf_only=True when pdf_output=None')
                plt.show()
            else:
                print(f'--> Saving PDF file: {pdf_output}')
                plt.savefig(pdf_output)
                if pdf_only:
                    plt.close(fig)
                else:
                    plt.show()
        else:
            if pdf_output is not None:
                raise ValueError('You must set plots=True to make use of pdf_output')

    @u.quantity_input(xpeaks=u.pixel, wavelengths=u.m)
    def define_peak_wavelengths(self, xpeaks, wavelengths):
        """Define peak wavelengths with units.

        Parameters
        ----------
        xpeaks : numpy array
            Array with location of the peaks.
        wavelengths : numpy array
            Array with the wavelength of each peak.

        """

        if len(xpeaks) != len(wavelengths):
            raise ValueError(f'Number of peaks: {len(xpeaks)} != '
                             f'number of wavelengths: {len(wavelengths)}')
        self.peak_wavelengths = wavelengths

    @u.quantity_input(xpeaks=u.pixel)
    def overplot_identified_lines(self, xpeaks, spectrum, 
                                  title=None, fontsize_title=16, fontsize_wave=10,
                                  pdf_output=None, pdf_only=False):
        """Overplot identified lines

        Parameters
        ----------
        xpeaks : numpy array
            Array with location of the identified peaks.
        spectrum : numpy array
            Spectrum to be displayed.
        title : str or None
            Plot title.
        fontsize_title : int
            Font size for title
        fontsize_wave : int
            Font size for wavelength labels.
        pdf_output : str or None
            If not None, save plots in PDF file.
        pdf_only : bool
            If True, close the plot after generating the PDF file.

        """

        if self.peak_wavelengths is None:
            raise ValueError('You must execute define_peak_wavelengths first!')

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))

        # plot using linear scale
        xplot = np.arange(self._naxis1)
        ax1.plot(xplot, spectrum, lw=1)
        ixpeaks = np.asarray(xpeaks.value + 0.5, dtype=int)
        ax1.plot(xpeaks.value, spectrum[ixpeaks], 'o', color='C0')
        ymin, ymax = ax1.get_ylim()
        for ip, (xp, yp) in enumerate(zip(xpeaks.value, spectrum[ixpeaks])):
            ax1.text(xp, yp + (ymax - ymin)/50, ip + 1, color='C0', ha='center', fontsize=fontsize_wave)
        ax1.set_xlabel('X axis (array index)')
        ax1.set_ylabel('Number of counts')
        if title is not None:
            ax1.set_title(title, fontsize=fontsize_title)

        # plot using logarithmic scale
        sp_minimum = spectrum.min()
        if sp_minimum <= 0:
            yplot = spectrum + sp_minimum + 1
            ylabel = f'Number of counts (+{sp_minimum+1:.2f})'
        else:
            yplot = spectrum
            ylabel = 'Number of counts'
        ax2.plot(xplot, yplot, lw=1)
        ax2.set_xlabel('X axis (array index)')
        ax2.set_ylabel(ylabel)
        ixpeaks = np.asarray(xpeaks.value + 0.5, dtype=int)
        ax2.plot(xpeaks.value, yplot[ixpeaks], 'o', color='C0')
        ax2.set_yscale('log')
        # extra room for labels in the vertical direction
        ymin, ymax = ax2.get_ylim()
        dy = (np.log10(ymax) - np.log10(ymin)) / 6
        ymax *= 10**dy
        ax2.set_ylim(ymin, ymax)
        dy = (np.log10(ymax) - np.log10(ymin)) / 20
        # avoid colliding labels
        xmin, xmax = ax2.get_xlim()
        xp_last = 0
        yp_last = 0
        xfraction = 200
        yfraction = 10
        for ip, (xp, yp) in enumerate(zip(xpeaks.value, yplot[ixpeaks])):
            if ip > 0:
                if (xp - xp_last) / (xmax - xmin) < 1 / xfraction and (yp - yp_last) / (ymax - ymin) < 1 / yfraction:
                    xoffset = (xmax - xmin) / xfraction
                else:
                    xoffset = 0
            else:
                xoffset = 0
            ax2.plot([xp, xp+xoffset], [yp*10**(0.3*dy), yp*10**(0.7*dy)], '-', lw=1, color='gray')
            ax2.text(xp+xoffset, yp*10**dy, self.peak_wavelengths[ip].value,
                     ha='center', va='bottom', rotation=90, fontsize=fontsize_wave)
            ax2.text(xp, ymin*10**(0.3*dy), ip + 1, color='C0', ha='center', fontsize=fontsize_wave)
            xp_last = xp
            yp_last = yp
        plt.tight_layout()
        if pdf_output is None:
            if pdf_only:
                raise ValueError('Unexpected pdf_only=True when pdf_output=None')
            plt.show()
        else:
            print(f'--> Saving PDF file: {pdf_output}')
            plt.savefig(pdf_output)
            if pdf_only:
                plt.close(fig)
            else:
                plt.show()

    def fit_cdistortion(self, degree_cdistortion=None,
                        plots=False, title=None, pdf_output=None, pdf_only=False,
                        disable_tqdm=True):
        """Fit C distortion.

        Parameters
        ----------
        degree_cdistortion : int or None
            Polynomial degree to fit curvature of each arc line.
        plots : bool
            If True, display intermediate plots.
        title : str
            Plot title.
        pdf_output : str or None
            If not None, save plots in PDF file.
        pdf_only : bool
            If True, close the plot after generating the PDF file.
        disable_tqdm : bool
            If True, disable tqdm progress bar.

        """

        if self._valid_scans is None:
            raise ValueError('You must execute compute_xpeaks_image first')

        if degree_cdistortion is not None:
            self.degree_cdistortion = degree_cdistortion

        self._list_poly_cdistortion = []
        list_yfit = []
        list_reject = []
        xfit = np.arange(self._naxis2)[self._valid_scans]
        if len(xfit) <= self.degree_cdistortion:
            raise ValueError(f'Insufficient number of points to fit a polynomial of degree {self.degree_cdistortion}')
        for i in tqdm(range(self._nlines_reference), 
                      desc='Fitting C distortion', disable=disable_tqdm):
            yfit = self._xpeaks_all_lines_array[self._valid_scans, i]
            poly, yres, reject = polfit_residuals_with_sigma_rejection(
                x=xfit,
                y=yfit,
                deg=self.degree_cdistortion,
                times_sigma_reject=3.0
            )
            self._list_poly_cdistortion.append(poly)
            list_yfit.append(yfit)
            list_reject.append(reject)

        if plots:
            npprow = 4
            nrows = int(self._nlines_reference / npprow)
            if self._nlines_reference % npprow != 0:
                nrows += 1
            fig, axarr = plt.subplots(nrows=nrows, ncols=npprow, figsize=(15, 3.75*nrows))
            axarr = axarr.flatten()
            for ax in axarr:
                ax.axis('off')
            for i in range(self._nlines_reference):
                ax = axarr[i]
                ax.axis('on')
                yfit = list_yfit[i]
                reject = list_reject[i]
                poly = self._list_poly_cdistortion[i]
                ax.plot(yfit, xfit, 'o', label='peak location')
                ax.plot(yfit[reject], xfit[reject], 'rx', label='rejected points')
                ax.plot(poly(xfit), xfit, '-', label='polynomial fit')
                ax.set_xlabel('X axis (array index)')
                ax.set_ylabel('Y axis (array index)')
                ax.set_title(f'C distortion: line #{i+1}/{self._nlines_reference} '
                             f'(deg.: {self.degree_cdistortion})')
                xmin = yfit[np.logical_not(reject)].min()
                xmax = yfit[np.logical_not(reject)].max()
                if (xmax - xmin) < self.nx_window:
                    xmin_ = (xmin + xmax) / 2 - self.nx_window / 2
                    xmax_ = (xmin + xmax) / 2 + self.nx_window / 2
                    ax.set_xlim([xmin_, xmax_])
                ax.set_ylim([0, self._naxis2-1])
                ax.legend()
            if title is not None:
                fig.suptitle(f'{title}\n', fontsize=16)
            plt.tight_layout()
            if pdf_output is None:
                if pdf_only:
                    raise ValueError('Unexpected pdf_only=True when pdf_output=None')
                plt.show()
            else:
                print(f'--> Saving PDF file: {pdf_output}')
                plt.savefig(pdf_output)
                if pdf_only:
                    plt.close(fig)
                else:
                    plt.show()
        else:
            if pdf_output is not None:
                raise ValueError('You must set plots=True to make use of pdf_output')

    def plot_cdistortion(self, data, title=None, pdf_output=None, pdf_only=False):
        """Plot current C distortion fit.

        Parameters
        ----------
        data : numpy array
            Array with 2D image. It is assumed that the spectral / spatial
            direction corrrespond to the X / Y axis.
        title : str
            Plot title.
        pdf_output : str or None
            If not None, save plots in PDF file.
        pdf_only : bool
            If True, close the plot after generating the PDF file.

        """

        if self._nlines_reference is None:
            raise ValueError('You must execute compute_xpeaks_image first.')

        npprow = 4
        nrows = int(self._nlines_reference / npprow)
        if self._nlines_reference % npprow != 0:
            nrows += 1
        fig, axarr = plt.subplots(nrows=nrows, ncols=npprow, figsize=(15, 3.75 * nrows))
        axarr = axarr.flatten()
        for ax in axarr:
            ax.axis('off')

        xplot = np.arange(0, self._naxis2)
        for i in range(self._nlines_reference):
            ax = axarr[i]
            ax.axis('on')
            poly = self._list_poly_cdistortion[i]
            yplot = poly(xplot)
            xmin = yplot.min()
            xmax = yplot.max()
            if (xmax - xmin) < self.nx_window:
                xmin_ = (xmin + xmax) / 2 - self.nx_window / 2
                xmax_ = (xmin + xmax) / 2 + self.nx_window / 2
            else:
                xmin_, xmax_ = xmin, xmax
            ax.set_xlim([xmin_, xmax_])
            i1 = int(xmin_ + 0.5)
            i2 = int(xmax_ + 0.5)
            vmin, vmax = np.percentile(data[:, i1:(i2 + 1)], [5, 95])
            img = ax.imshow(data, vmin=vmin, vmax=vmax, origin='lower', aspect='auto', interpolation='nearest')
            ax.set_xlabel('X axis (array index)')
            ax.set_ylabel('Y axis (array index)')
            ax.set_title(f'C distortion: line #{i + 1}/{self._nlines_reference} '
                         f'(deg.: {self.degree_cdistortion})')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(img, cax=cax, label='Number of counts')
            colormarker = np.array(['gray'] * self._naxis2)
            colormarker[self._valid_scans] = 'red'
            sizemarker = np.array([1] * self._naxis2)
            sizemarker[self._valid_scans] = 3
            ax.scatter(yplot, xplot, marker='.', c=colormarker, s=sizemarker)
        if title is not None:
            fig.suptitle(f'{title}\n', fontsize=16)
        plt.tight_layout()
        if pdf_output is None:
            if pdf_only:
                raise ValueError('Unexpected pdf_only=True when pdf_output=None')
            plt.show()
        else:
            print(f'--> Saving PDF file: {pdf_output}')
            plt.savefig(pdf_output)
            if pdf_only:
                plt.close(fig)
            else:
                plt.show()

    def predict_cdistortion(self, ns_fits):
        """Predict peak locations of a particular spectrum.

        Parameters
        ----------
        ns_fits : int
            Spectrum number (following the FITS convention) whose
            peak locations are computed.

        Returns
        -------
        xpeaks_predicted : numpy array
            Peak locations (float numbers), in array coordinates, with
            units of pixel.
        """

        if self._list_poly_cdistortion is None:
            raise ValueError('You must execute fit_cdistortion first')

        xpeaks_predicted = np.zeros(self._nlines_reference)
        for i in range(self._nlines_reference):
            poly = self._list_poly_cdistortion[i]
            xpeaks_predicted[i] = poly(ns_fits-1)

        return xpeaks_predicted * u.pixel

    @u.quantity_input(xpeaks=u.pixel)
    def fit_xpeaks_wavelengths(self, xpeaks, degree_wavecalib=None,
                               plots=False, title=None,
                               pdf_output=None, pdf_only=False,
                               debug=False):
        """Compute wavelength calibration polynomial for particular xpeaks.

        Note that although the initial fit is a function of the form
        pixel(wavelength), the returned polynomial is inverted, i.e.,
        wavelength(pixel).

        Parameters
        ----------
        xpeaks : numpy array
            Location of line peaks (array coordinates), with units of
            pixel.
        degree_wavecalib : int or None
            Polynomial degree to fit wavelength variation with pixel number.
        plots : bool
            If True, show intermediate plots.
        title : str
            Plot title.
        pdf_output : str or None
            If not None, save plots in PDF file.
        pdf_only : bool
            If True, close the plot after generating the PDF file.
        debug : bool
            If True, display intermediate results.

        Returns
        -------
        poly_fits_yx : Polynomial instance
            Fitted polynomial wavelength(pixel), using the pixel coordinate
            following the FITS convention as argument.
        residual_std_yx : astropy.units.Quantity
            Float number providing the residual standard deviation of the
            previous fit.
        poly_fits_xy : Polynomial instance
            Fitted polynomial pixel(wavelength), using the pixel coordinate
            following the FITS convention as argument.
        residual_std_xy : astropy.units.Quantity
            Float number providing the residual standard deviation of the
            previous fit.
        crval1_linear : astropy.units.Quantity
            Float number providing CRVAL1 corresponding to a linear
            approximation.
        cdelt1_linear : astropy.units.Quantity
            Float number providing CDELT1 corresponding to a linear
            approximation.
        crmax1_linear : astropy.units.Quantity
            Float number providing the maximum wavelength (pixel NAXIS1)
            corresponding to a linear approximation.

        """

        if degree_wavecalib is not None:
            self.degree_wavecalib = degree_wavecalib

        if self.peak_wavelengths is None:
            raise ValueError('You must use define_peak_wavelengths first')

        # check wavelengths are sorted
        if np.all(np.diff(self.peak_wavelengths) >= 0):
            pass
        else:
            raise ValueError(f'peak_wavelengths array is not sorted: {self.peak_wavelengths}')

        # the number of wavelengths must match the number of peaks
        nwaves = len(self.peak_wavelengths)
        npeaks = len(xpeaks)
        if nwaves != npeaks:
            raise ValueError(f'Number of wavelengths: {nwaves} is different '
                             f'from the number of detected peaks: {npeaks}')

        xfit = xpeaks.value + 1  # FITS convention
        yfit = self.peak_wavelengths.value
        if len(xfit) <= self.degree_wavecalib:
            raise ValueError(f'Insufficient number of points to fit a polynomial of degree {self.degree_wavecalib}')
        # initial fit: pixel(wavelength)
        poly_fits_xy, stats_list_xy = Polynomial.fit(x=yfit, y=xfit, deg=self.degree_wavecalib, full=True)
        poly_fits_xy = Polynomial.cast(poly_fits_xy)
        # invert polynomial fit: wavelength(pixel)
        yfit_new = np.linspace(start=yfit.min(), stop=yfit.max(), num=self._naxis1)
        xfit_new = poly_fits_xy(yfit_new)
        poly_fits_yx, stats_list_yx = Polynomial.fit(x=xfit_new, y=yfit_new, deg=self.degree_wavecalib, full=True)
        poly_fits_yx = Polynomial.cast(poly_fits_yx)

        # obtain CRVAL1 and CDELT1 for a linear wavelength scale from the
        # last polynomial fit
        crpix1 = 1 * u.pixel
        crval1_linear = poly_fits_yx(crpix1.value)
        crval1_linear *= self.peak_wavelengths.unit
        cdelt1_linear = (poly_fits_yx(self._naxis1) - crval1_linear.value) / (self._naxis1 - 1)
        cdelt1_linear *= self.peak_wavelengths.unit / u.pixel
        crmax1_linear = crval1_linear.value + (self._naxis1 - 1) * cdelt1_linear.value
        crmax1_linear *= self.peak_wavelengths.unit
        if debug:
            print(f'>>> CRPIX1.............: {crpix1}')
            print(f'>>> CRVAL1 linear scale: {ctext(crval1_linear, rev=True)}')
            print(f'>>> CDELT1 linear scale: {ctext(cdelt1_linear, rev=True)}')
            print(f'>>> CRMAX1 linear scale: {crmax1_linear}')

        if npeaks > self.degree_wavecalib + 1:
            residual_std_xy = np.sqrt(stats_list_xy[0]/(npeaks - self.degree_wavecalib - 1))[0]
        else:
            residual_std_xy = 0.0
        residual_std_xy *= u.pixel
        # approximate residual_std_yx
        residual_std_yx = residual_std_xy * cdelt1_linear

        if debug:
            print('\n>>> Fitted coefficients pixel(wavelength):\n', poly_fits_xy.coef)
            print(f'>>> Residual std.........................: {residual_std_xy}')
            print('\n>>> Fitted coefficients wavelength(pixel):\n', poly_fits_yx.coef)
            print(f'>>> Residual std.........................: {residual_std_yx}')

        if plots:
            # polynomial fit pixel(wavelength)
            xpol = np.linspace(crval1_linear.value, crmax1_linear.value, self._naxis1)
            ypol = poly_fits_xy(xpol) - np.linspace(1, self._naxis1, self._naxis1)
            # arc lines
            xp = np.copy(yfit)
            yp = xfit - (crpix1.value + (xp - crval1_linear.value)/cdelt1_linear.value)
            yres = xfit - poly_fits_xy(xp)
            # residuals plot
            fig = plt.figure(figsize=(10, 10))
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.set_xlabel(f'Wavelength ({self.peak_wavelengths.unit})')
            ax2.set_ylabel('Residuals (pixel)')
            ax2.plot(xp, yres, 'o', color='C0')
            ax2.axhline(y=0.0, color="black", linestyle="dashed")
            ymin, ymax = ax2.get_ylim()
            for ip, (xtext, ytext) in enumerate(zip(xp, yres)):
                ax2.text(xtext, ytext + (ymax - ymin)/40, str(ip+1), ha='center', color='C0')

            # plot with differences between linear fit and fitted polynomial
            ax = fig.add_subplot(2, 1, 1, sharex=ax2)
            ax.set_xlabel(f'Wavelength ({self.peak_wavelengths.unit})')
            ax.set_ylabel('Differences with\nlinear solution (pixel)')
            ax.plot(xp, yp, 'o', color='C0', label="identified line")
            # polynomial fit
            ax.plot(xpol, ypol, '-', color='C1', label="polynomial fit")
            ymin = np.min(yp)
            ymax = np.max(yp)
            dy = ymax - ymin
            ymin -= dy / 10
            ymax += dy / 10
            ax.set_ylim(ymin, ymax)
            for ip, (xtext, ytext) in enumerate(zip(xp, yp)):
                ax.text(xtext, ytext + dy/30, str(ip+1), ha='center', color='C0')
            ax.legend()
            if title is not None:
                title_ = f'{title}\n'
            else:
                title_ = ''
            plt.title(f"{title_}Wavelength calibration (polynomial degree: {self.degree_wavecalib})")
            plt.tight_layout()
            if pdf_output is None:
                if pdf_only:
                    raise ValueError('Unexpected pdf_only=True when pdf_output=None')
                plt.show()
            else:
                print(f'--> Saving PDF file: {pdf_output}')
                plt.savefig(pdf_output)
                if pdf_only:
                    plt.close(fig)
                else:
                    plt.show()
        else:
            if pdf_output is not None:
                raise ValueError('You must set plots=True to make use of pdf_output')

        return poly_fits_yx, residual_std_yx, \
               poly_fits_xy, residual_std_xy, \
               crval1_linear, cdelt1_linear, crmax1_linear

    def fit_wavelengths(self, degree_wavecalib=None,
                        output_filename=None, history_list=None, 
                        plots=False, title=None,
                        pdf_output=None, pdf_only=False,
                        silent_mode=False, disable_tqdm=True):
        """Wavelength calibration of the whole image.

        Use the prediction of the fitted C-distortion to determine the
        location of each line.

        Parameters
        ----------
        degree_wavecalib : int or None
            Polynomial degree to fit wavelength variation with pixel number.
        output_filename : str or None
            Output file name. A multi-extension FITS file is generated
            using a primary HDU which contains the polynomial coefficients,
            a primary extension with the C-distortion polynomials, and a
            secondary extension with a binary table that stores the
            additional parameters.
        history_list : list
            List of strings to be saved as individual HISTORY lines.
        plots : bool
            If True, show intermediate plots.
        title : str
            Plot title.
        pdf_output : str or None
            If not None, save plots in PDF file.
        pdf_only : bool
            If True, close the plot after generating the PDF file.
        silent_mode : bool
            If True, do not show information messages.
        disable_tqdm : bool
            If True, disable tqdm progress bar.

        """

        if degree_wavecalib is not None:
            self.degree_wavecalib = degree_wavecalib

        if output_filename is None:
            raise ValueError(f'Invalid output_filename={output_filename}')

        self._array_poly_wav = np.zeros((self._naxis2, self.degree_wavecalib + 1))
        self._array_poly_pix = np.zeros((self._naxis2, self.degree_wavecalib + 1))
        self._array_residual_std_wav = np.zeros(self._naxis2) * self.peak_wavelengths.unit
        self._array_residual_std_pix = np.zeros(self._naxis2) * u.pixel
        self._array_crval1_linear = np.zeros(self._naxis2) * self.peak_wavelengths.unit
        self._array_cdelt1_linear = np.zeros(self._naxis2) * self.peak_wavelengths.unit / u.pixel
        self._array_crmax1_linear = np.zeros(self._naxis2) * self.peak_wavelengths.unit
        for k in tqdm(range(self._naxis2), desc='Computing wavelength calibration', disable=disable_tqdm):
            xpeaks = np.zeros(self._nlines_reference) * u.pixel
            for i in range(self._nlines_reference):
                poly_cdistortion = self._list_poly_cdistortion[i]
                xpeaks[i] = poly_cdistortion(k) * u.pixel
            poly1_fits, residual1_std, poly2_fits, residual2_std, crval1_linear, cdelt1_linear, crmax1_linear = \
                self.fit_xpeaks_wavelengths(xpeaks=xpeaks)
            self._array_poly_wav[k, :] = poly1_fits.coef
            self._array_poly_pix[k, :] = poly2_fits.coef
            self._array_residual_std_wav[k] = residual1_std
            self._array_residual_std_pix[k] = residual2_std
            self._array_crval1_linear[k] = crval1_linear
            self._array_cdelt1_linear[k] = cdelt1_linear
            self._array_crmax1_linear[k] = crmax1_linear

        # estimate crval1, cdelt1
        self.estimate_crval1_cdelt1(silent_mode=silent_mode)

        # return if output file name is not set
        if output_filename is None:
            return

        # generate output file
        header = fits.Header()
        if history_list is not None:
            for item in history_list:
                header.add_history(item)
        for keyword in self.__class__.fits_keyword_list:
            key = keyword[:8].upper()
            header[key] = (self.__dict__[keyword], keyword)
        header['_NAXIS1'] = (self._naxis1, 'original NAXIS1 in 2D spectroscopic image')
        header['_NAXIS2'] = (self._naxis2, 'original NAXIS2 in 2D spectroscopic image')
        header['_NLINES'] = (self._nlines_reference, 'number of lines employed for calibration')
        for i in range(self._nlines_reference):
            header[f'WPEAK{i+1:03d}'] = (
                self.peak_wavelengths[i].value,
                f'wavelength of line #{i+1} ({self.peak_wavelengths.unit})'
            )
        header['WPEAKUNI'] = self.peak_wavelengths.unit.to_string()
        header['LCRVAL1A'] = (
            np.min(self._array_crval1_linear.value),
            'minimum crval1_linear (Angstrom)'
        )
        header['LCRVAL1M'] = (
            np.median(self._array_crval1_linear.value),
            'median crval1_linear (Angstrom)'
        )
        header['LCRVAL1Z'] = (
            np.max(self._array_crval1_linear.value),
            'maximum crval1_linear (Angstrom)'
        )
        header['LCDELT1M'] = (
            np.median(self._array_cdelt1_linear.value),
            'median cdelt1_linear (Angstrom/pixel)'
        )
        header['FILENAME'] = f'{Path(output_filename).name}'

        primary_hdu = fits.PrimaryHDU(header=header, data=self._array_poly_wav)

        inv_poly_hdr = fits.Header()
        inv_poly_hdr['HISTORY'] = 'Coefficients of polynomial pixel(wavelength)'
        inv_poly_hdu = fits.ImageHDU(header=inv_poly_hdr, data=self._array_poly_pix, name='INV_POLY')

        cdistor_hdr = fits.Header()
        cdistor_hdr['HISTORY'] = 'C-distortion polynomial coefficients'
        cdistor_array = np.zeros((self._nlines_reference, self.degree_cdistortion + 1))
        for i in range(self._nlines_reference):
            cdistor_array[i, :] = self._list_poly_cdistortion[i].coef
        cdistor_hdu = fits.ImageHDU(header=cdistor_hdr, data=cdistor_array, name='CDISTOR')

        col1 = fits.Column(name='residual_std_wav',
                           format='D',
                           array=self._array_residual_std_wav.value,
                           unit=self._array_residual_std_wav.unit.to_string()
                           )
        col2 = fits.Column(name='residual_std_pix',
                           format='D',
                           array=self._array_residual_std_pix.value,
                           unit=self._array_residual_std_pix.unit.to_string()
                           )
        col3 = fits.Column(name='crval1_linear',
                           format='D',
                           array=self._array_crval1_linear.value,
                           unit=self._array_crval1_linear.unit.to_string()
                           )
        col4 = fits.Column(name='cdelt1_linear',
                           format='D',
                           array=self._array_cdelt1_linear.value,
                           unit=self._array_cdelt1_linear.unit.to_string()
                           )
        col5 = fits.Column(name='crmax1_linear',
                           format='D',
                           array=self._array_crmax1_linear.value,
                           unit=self._array_crmax1_linear.unit.to_string()
                           )
        coef_hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4, col5], name='COEFF')

        hdul = fits.HDUList([primary_hdu, inv_poly_hdu, cdistor_hdu, coef_hdu])
        hdul.writeto(output_filename, overwrite=True)

        print(f'--> Saving result in FITS file: {output_filename}')

        if plots:
            self._plot_wavecoef(title, pdf_output, pdf_only)
        else:
            if pdf_output is not None:
                raise ValueError('You must set plots=True to make use of pdf_output')

    def estimate_crval1_cdelt1(self, silent_mode=False):
        """Compute mean CDELT1 to cover min_crval1 and max_crmax1.

        Parameters
        ----------
        silent_mode : bool
            If True, do not show information messages.

        """

        min_crval1 = self._array_crval1_linear.min()
        max_crmax1 = self._array_crmax1_linear.max()
        cdelt1 = (max_crmax1 - min_crval1) / (self._naxis1 - 1)
        if not silent_mode:
            print(f'>>> minimum CRVAL1 linear: {min_crval1}')
            print(f'>>> maximum CRMAX1 linear: {max_crmax1}')
            print(f'>>> mean CDELT1..........: {cdelt1}')

    def _plot_wavecoef(self, title=None, pdf_output=None, pdf_only=False):
        """Plot wavelength calibration parameters and coefficients

        Parameters
        ----------
        title : str
            Plot title.
        pdf_output : str or None
            If not None, save plots in PDF file.
        pdf_only : bool
            If True, close the plot after generating the PDF file.

        """

        # crval1, cdelt1, crmax1, residual_std
        fig, axarr = plt.subplots(nrows=1, ncols=5, figsize=(15, 3))
        axarr = axarr.flatten()
        for i, item in enumerate(['_array_crval1_linear', 
                                  '_array_cdelt1_linear',
                                  '_array_crmax1_linear', 
                                  '_array_residual_std_wav',
                                  '_array_residual_std_pix']):
            ax = axarr[i]
            ax.plot(self.__dict__[item])
            ax.set_xlabel('Y axis (array index)')
            ax.set_ylabel(item)
        if title is not None:
            fig.suptitle(f'{title}', fontsize=16)
        plt.tight_layout()
        if pdf_output is None:
            if pdf_only:
                raise ValueError('Unexpected pdf_only=True when pdf_output=None')
            plt.show()
        else:
            parent = Path(pdf_output).parents[0]
            stem = Path(pdf_output).stem
            fname = parent / f'{stem}_p1.pdf'
            print(f'--> Saving PDF file: {fname}')
            plt.savefig(fname)
            if pdf_only:
                plt.close(fig)
            else:
                plt.show()
        # polynomial coefficients
        ncoeff = self.degree_wavecalib + 1
        if ncoeff < 4:
            npprow = ncoeff
            figwidth = 3.75 * npprow
        else:
            npprow = 4
            figwidth = 15
        nrows = int(ncoeff / npprow)
        if ncoeff % npprow != 0:
            nrows += 1
        fig, axarr = plt.subplots(nrows=nrows, ncols=npprow, 
                                  figsize=(figwidth, 3*nrows))
        axarr = axarr.flatten()
        for ax in axarr:
            ax.axis('off')
        for k in range(ncoeff):
            ax = axarr[k]
            ax.axis('on')
            ax.plot(self._array_poly_wav[:, k])
            ax.set_xlabel('Y axis (array index)')
            ax.set_ylabel(f'Coeff #{k}')
        if title is not None:
            fig.suptitle(f'{title}', fontsize=16)
        plt.tight_layout()
        if pdf_output is None:
            if pdf_only:
                raise ValueError('Unexpected pdf_only=True when pdf_output=None')
            plt.show()
        else:
            parent = Path(pdf_output).parents[0]
            stem = Path(pdf_output).stem
            fname = parent / f'{stem}_p2.pdf'
            print(f'--> Saving PDF file: {fname}')
            plt.savefig(fname)
            if pdf_only:
                plt.close(fig)
            else:
                plt.show()

    @u.quantity_input(crval1=u.Angstrom, cdelt1=u.Angstrom/u.pixel)
    def apply(self, data, crval1, cdelt1, silent_mode=False, disable_tqdm=True):
        """Apply wavelength calibration to data array.

        Parameters
        ----------
        data : numpy array
            Array with 2D image. It is assumed that the spectral / spatial
            direction corrrespond to the X / Y axis.
        crval1 : astropy.units.Quantity
            Float number providing the CRVAL1 value: wavelength at the
            center of the first pixel.
        cdelt1 : astropy.units.Quantity
            Float number providing CDELT1 value: wavelength increment
            per pixel.
        silent_mode : bool
            If True, do not show information messages.
        disable_tqdm : bool
            If True, disable tqdm progress bar.

        Returns
        -------
        data_wavecal : numpy array
            Wavelength calibrated array.

        """

        if self._array_poly_wav is None:
            raise ValueError('Wavelength calibration is not available yet')

        naxis2, naxis1 = data.shape
        if naxis1 != self._naxis1:
            raise ValueError(f'Unexpected NAXIS1: {naxis1} != _NAXIS1: {self._naxis1}')
        if naxis2 != self._naxis2:
            raise ValueError(f'Unexpected NAXIS2: {naxis2} != _NAXIS2: {self._naxis2}')

        if not silent_mode:
            print(f'>>> Using CDELT1: 1 {u.pixel}')
            print(f'>>> Using CRVAL1: {crval1}')
            print(f'>>> Using CDELT1: {cdelt1}')

        accum_flux = np.zeros((naxis2, naxis1 + 1))
        accum_flux[:, 1:] = np.cumsum(data, axis=1)

        data_wavecal = np.zeros_like(data)
        new_wl_borders = crval1 + cdelt1 * ((np.arange(naxis1 + 1) - 0.5) * u.pixel)

        old_x_borders_fits = np.arange(naxis1 + 1) + 0.5  # FITS convention

        for k in tqdm(range(naxis2), 
                      desc='Applying wavelength calibration',
                      disable=disable_tqdm):
            poly = Polynomial(self._array_poly_wav[k])
            old_wl_borders = poly(old_x_borders_fits) * self.peak_wavelengths.unit
            flux_borders = np.interp(
                x=new_wl_borders,
                xp=old_wl_borders,
                fp=accum_flux[k, :],
                left=0,
                right=accum_flux[k, -1]
            )
            data_wavecal[k, :] = flux_borders[1:] - flux_borders[:-1]

        return data_wavecal

    def plot_data_comparison(self, data_before, data_after,
                             crval1, cdelt1,
                             title=None,
                             semi_window=None):
        """Plot data array before and after correction.

        Parameters
        ----------
        data_before : numpy array
            Array with 2D image before C-distortion correction and
            wavelength calibration. It is assumed that the
            spectral / spatial direction corrrespond to the X / Y axis.
        data_after : numpy array
            Array with 2D image before C-distortion correction and
            wavelength calibration. It is assumed that the
            spectral / spatial direction corrrespond to the X / Y axis.
            It must have units of pixel.
        crval1 : astropy.units.Quantity
            Float number providing the CRVAL1 value: wavelength at the
            center of the first pixel, employed to generate 'data_after'.
        cdelt1 : astropy.units.Quantity
            Float number providing CDELT1 value: wavelength increment
            per pixel, employed to generate 'data_after'.
        title : str
            Plot title.
        semi_window : int
            Half width of the display window (in pixels).

        """

        if semi_window is None:
            semi_window = self.nx_window

        # approximate peak location (computed at the central spectrum)
        ns_center = self._naxis2 / 2
        xpeaks = [self._list_poly_cdistortion[k](ns_center-1) for k in range(self._nlines_reference)]

        # display region around each line (before / after wavelength calibration)
        for iline, xpeak in enumerate(xpeaks):
            fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            # limits before calibration
            i1 = int(xpeak + 0.5) - semi_window
            if i1 < 0:
                i1 = 0
            i2 = int(xpeak + 0.5) + semi_window
            if i2 > self._naxis1 - 1:
                i2 = self._naxis1 - 1
            # limits after calibration
            cdelt1_prior = self._array_cdelt1_linear[self._naxis2//2].value
            w1 = self.peak_wavelengths[iline].value - semi_window * cdelt1_prior
            w2 = self.peak_wavelengths[iline].value + semi_window * cdelt1_prior
            ii1 = int((w1 - crval1.value) / cdelt1.value + 0.5)
            if ii1 < 0:
                ii1 = 0
            ii2 = int((w2 - crval1.value) / cdelt1.value + 0.5)
            if ii2 > self._naxis1 - 1:
                ii2 = self._naxis1 - 1
            # display comparison
            vmin, vmax = np.percentile(data_before[:, i1:(i2 + 1)], [5, 95])
            for iplot, (data_, when) in enumerate(zip([data_before, data_after],
                                                      ['before', 'after'])):
                ax = axarr[iplot]
                img = ax.imshow(data_, vmin=vmin, vmax=vmax, origin='lower',
                                aspect='auto', interpolation='nearest')
                if iplot == 0:
                    ax.set_xlim(i1, i2)
                else:
                    ax.set_xlim(ii1, ii2)
                ax.set_xlabel('X axis (array index)')
                ax.set_ylabel('Y axis (array index)')
                ax.set_title(f'Line #{iline + 1}/{self._nlines_reference} '
                             f'({when} WL correction)\n'
                             f'Wavelength={self.peak_wavelengths[iline]}')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(img, cax=cax, label='Number of counts')
            if title is not None:
                fig.suptitle(f'{title}\n', fontsize=16)
            plt.tight_layout()
            plt.show()


def apply_wavecal_ccddata(infile, wcalibfile, outfile,
                          crval1, cdelt1,
                          ctype1_info=('AWAV', 'air wavelength'),
                          silent_mode=True,
                          plot_data_comparison=0,
                          title=None):
    """Apply wavelength calibration to FITS file storing CCDData.

    The FITS file must contain:
    - a primary HDU
    - extension1: MASK
    - extension2: UNCERT

    Parameters
    ----------
    infile : str
        Input file name of the image to be calibrated.
    wcalibfile : str
        Input file name with the wavelength calibration polynomials.
        This is the file generated by the method fit_wavelengths() of
        TeaWaveCalibration.
    outfile : str
        Output file name with the wavelength calibrated result.
    crval1 : astropy.units.Quantity
        Float number providing the CRVAL1 value: wavelength at the
        center of the first pixel.
    cdelt1 : astropy.units.Quantity
        Float number providing CDELT1 value: wavelength increment
        per pixel.
    plot_data_comparison : int
        Indicate whether to plot the comparison before / after
        the calibration. Options:
        0: no plot
        1: plot data array
        2: plot data array and also the arrays corresponding to
           the MASK and UNCERT extensions
    ctype1_info : tuple
        Value and description of the FITS keyword CTYPE1.
    silent_mode : bool
        If True, do not show information messages.
    title : str
        Plot title

    """

    if plot_data_comparison not in [0, 1, 2]:
        raise ValueError(f'Unexpected plot_data_comparison_value: {plot_data_comparison}')

    # read wavelength calibration from file
    wavecalib = TeaWaveCalibration.read(wcalibfile, silent_mode=silent_mode)

    # read image to be calibrated
    if not silent_mode:
        print(f'\n>>> Reading file.........: {infile}')
    ccdimage = CCDData.read(infile)

    # duplicate input CCDData object to store result
    ccdimage_wavecalib = ccdimage.copy()

    # apply C-distortion correction and wavelength calibration to PRIMARY extension
    if not silent_mode:
        print('\nApplying calibration to primary HDU')
    ccdimage_wavecalib.data = wavecalib.apply(
        data=ccdimage.data,
        crval1=crval1,
        cdelt1=cdelt1,
        silent_mode=silent_mode,
        disable_tqdm=silent_mode
    )

    # apply C-distortion correction and wavelength calibration to MASK extension
    if not silent_mode:
        print('\nApplying calibration to MASK extension')
    ccdimage_wavecalib.mask = wavecalib.apply(
        data=ccdimage.mask.astype(float),
        crval1=crval1,
        cdelt1=cdelt1,
        silent_mode=silent_mode,
        disable_tqdm=silent_mode
    ) > 0

    # apply C-distortion correction and wavelength calibration to UNCERT extension
    if not silent_mode:
        print('\nApplying calibration to UNCERT extension')
    ccdimage_wavecalib.uncertainty.array = wavecalib.apply(
        data=ccdimage.uncertainty.array,
        crval1=crval1,
        cdelt1=cdelt1,
        silent_mode=silent_mode,
        disable_tqdm=silent_mode
    )

    # include wavelength calibration parameters in FITS header
    wl_header = fits.Header()
    wl_header['CRPIX1'] = (1, f'{u.pixel}')
    wl_header['CRVAL1'] = (crval1.value, f'{crval1.unit}')
    wl_header['CDELT1'] = (cdelt1.value, f'{cdelt1.unit}')
    wl_header['CUNIT1'] = (f'{crval1.unit}', 'wavelength unit')
    wl_header['CTYPE1'] = ctype1_info
    ccdimage_wavecalib.wcs = wcs.WCS(wl_header)

    # update FILENAME keyword with output file name
    ccdimage_wavecalib.header['FILENAME'] = f'{Path(outfile).name}'
    # update HISTORY in header
    ccdimage_wavecalib.header['HISTORY'] = '-------------------'
    ccdimage_wavecalib.header['HISTORY'] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ccdimage_wavecalib.header['HISTORY'] = 'using tea_wavecal'
    ccdimage_wavecalib.header['HISTORY'] = f'calibration file: {Path(wcalibfile).name}'

    # save result
    ccdimage_wavecalib.write(outfile, overwrite='yes')
    if not silent_mode:
        print(f'--> Output FITS file.....: {outfile}')

    # plot comparison before / after the calibration
    if title is not None:
        title_ = f'{title}\n'
    else:
        title_ = ''
    if plot_data_comparison in [1, 2]:
        print('* Comparing primary HDU data before / after calibration')
        wavecalib.plot_data_comparison(
            data_before=ccdimage.data,
            data_after=ccdimage_wavecalib.data,
            crval1=crval1,
            cdelt1=cdelt1,
            title=f'{title_}(Primary HDU)'
        )
    if plot_data_comparison == 2:
        print('* Comparing MASK extension array before / after calibration')
        wavecalib.plot_data_comparison(
            data_before=ccdimage.mask.astype(float),
            data_after=ccdimage_wavecalib.mask.astype(float),
            crval1=crval1,
            cdelt1=cdelt1,
            title=f'{title_}(MASK extension)'
        )
        print('* Comparing UNCERT extension array before / after calibration')
        wavecalib.plot_data_comparison(
            data_before=ccdimage.uncertainty.array,
            data_after=ccdimage_wavecalib.uncertainty.array,
            crval1=crval1,
            cdelt1=cdelt1,
            title=f'{title_}(UNCERT extension)'
        )

