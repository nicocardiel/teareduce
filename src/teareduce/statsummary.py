#
# Copyright 2022-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from astropy.io import fits
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

from .robust_std import robust_std
from .sliceregion import SliceRegion2D


def statsummary(x=None, rm_nan=False, show=True):
    """Compute basic statistical parameters.

    Parameters
    ----------
    x : numpy array or None
        Input array with values which statistical properties are 
        requested.
    rm_nan : bool
        If True, filter out NaN values before computing statistics.
    show : bool
        If True display computed values.

    Returns
    -------
    result : Python dictionary
        Number of points, minimum, percentile 25, percentile 50
        (median), mean, percentile 75, maximum, standard deviation,
        robust standard deviation, percentile 15.866 (equivalent
        to -1 sigma in a normal distribution) and percentile 84.134
        (+1 sigma). This result if returned only if return_output=True.

    """

    # protections
    if x is None:
        return ['npoints', 'minimum', 'maximum', 
                'mean', 'median', 'std', 'robust_std', 
                'percentile16', 'percentile25', 'percentile75', 'percentile84'] 

    if isinstance(x, np.ndarray):
        xx = np.copy(x.flatten())
    else:
        if isinstance(x, list):
            xx = np.array(x)
        else:
            raise ValueError('x=' + str(x) + ' must be a numpy.ndarray')

    # filter out NaN's
    if rm_nan:
        xx = xx[np.logical_not(np.isnan(xx))]

    # compute basic statistics
    npoints = len(xx)
    ok = npoints > 0
    result = {
        'npoints': npoints,
        'minimum': np.min(xx) if ok else 0,
        'percentile25': np.percentile(xx, 25) if ok else 0,
        'median': np.percentile(xx, 50) if ok else 0,
        'mean': np.mean(xx) if ok else 0,
        'percentile75': np.percentile(xx, 75) if ok else 0,
        'maximum': np.max(xx) if ok else 0,
        'std': np.std(xx) if ok else 0,
        'robust_std': robust_std(xx) if ok else 0,
        'percentile16': np.percentile(xx, 15.86553) if ok else 0,
        'percentile84': np.percentile(xx, 84.13447) if ok else 0
    }

    if show:
        print('>>> =============================================')
        print('>>> STATISTICAL SUMMARY:')
        print('>>> ---------------------------------------------')
        print('>>> Number of points.........:', result['npoints'])
        print('>>> Minimum..................:', result['minimum'])
        print('>>> 1st Quartile.............:', result['percentile25'])
        print('>>> Median...................:', result['median'])
        print('>>> Mean.....................:', result['mean'])
        print('>>> 3rd Quartile.............:', result['percentile75'])
        print('>>> Maximum..................:', result['maximum'])
        print('>>> ---------------------------------------------')
        print('>>> Standard deviation.......:', result['std'])
        print('>>> Robust standard deviation:', result['robust_std'])
        print('>>> 0.1586553 percentile.....:', result['percentile16'])
        print('>>> 0.8413447 percentile.....:', result['percentile84'])
        print('>>> =============================================')

    return result


def ifc_statsummary(ifc, directory, region=None):
    """Include statistical summary in ImageFileCollection object.

    Parameters
    ----------
    ifc : ImageFileCollection object
        Instance of ImageFileCollection.
    directory : Path object
        Directory where images are stored.
    region : SliceRegion2D instance or None
        Region where the statistical summary will be computed.
        If None the whole data array is employed.

    Returns
    -------
    summary : Astropy Table
        Updated table including the statistical measurements.

    """

    if region is not None:
        if not isinstance(region, SliceRegion2D):
            msg = f'region: {region} must be a SliceRegion2D instance'
            raise ValueError(msg)

    summary = ifc.summary.copy()

    for colname in statsummary():
        # if the column already exists, remove it
        if colname in summary.columns:
            summary.remove_column(colname)
        # create column (initialise to 0)
        if colname == 'npoints':
            summary[colname] = np.zeros(len(summary), dtype=int)
            summary[colname].info.format = 'd'
        else:
            summary[colname] = np.zeros(len(summary))
            summary[colname].info.format = '.3f'

    for i, filename in enumerate(tqdm(summary['file'])):
        data = fits.getdata(directory / Path(filename).name)
        naxis2, naxis1 = data.shape
        region_fullframe = SliceRegion2D(np.s_[0:naxis2, 0:naxis1],
                                         mode='python')
        if region is None:
            region = region_fullframe
        else:
            if not region.within(region_fullframe):
                msg = f'Region {region!r} outside full frame ' \
                      f'{region_fullframe!r}'
                raise ValueError(msg)
        result = statsummary(data[region.python], show=False)
        for key, value in result.items():
            summary[key][i] = value

    summary.region = region

    return summary
