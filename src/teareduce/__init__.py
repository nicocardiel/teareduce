#
# Copyright 2023-2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from .avoid_astropy_warnings import avoid_astropy_warnings
from .cookbook.get_cookbook_file import get_cookbook_file
from .correct_pincushion_distortion import correct_pincushion_distortion
from .cosmicrays import cr2images, apply_cr2images_ccddata, crmedian
from .ctext import ctext
from .draw_rectangle import draw_rectangle
from .elapsed_time import elapsed_time
from .elapsed_time import elapsed_time_since
from .histogram1d import hist_step
from .histogram1d import plot_hist_step
from .imshow import imshow
from .imshow import imshowme
from .numsplines import AdaptiveLSQUnivariateSpline
from .peaks_spectrum import find_peaks_spectrum, refine_peaks_spectrum
from .polfit import polfit_residuals, polfit_residuals_with_sigma_rejection
from .robust_std import robust_std
from .sdistortion import fit_sdistortion
from .simulateccdexposure import SimulateCCDExposure
from .sliceregion import SliceRegion1D, SliceRegion2D, SliceRegion3D
from .statsummary import ifc_statsummary, statsummary
from .version import VERSION
from .wavecal import TeaWaveCalibration, apply_wavecal_ccddata
from .write_array_to_fits import write_array_to_fits
from .zscale import zscale

__version__ = VERSION
