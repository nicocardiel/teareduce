#
# Copyright 2022-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from astropy.nddata import CCDData
from datetime import datetime
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import ndimage

from .imshow import imshow
from .robust_std import robust_std
from .sliceregion import SliceRegion2D


def cr2images(data1, data2=None, ioffx=0, ioffy=0,
              tsigma_peak=10, tsigma_tail=3, maxsize=None,
              list_skipped_regions=None,
              image_region=None,
              median_size=None,
              return_masks=False,
              debug_level=0,
              zoom_region_imshow=None):
    """Remove cosmic rays from differences between 2 images.

    The input images must have the same shape. If only 1 image
    is available, this function computes a second image by
    applying a median filter to the first image.

    Parameters
    ----------
    data1 : numpy array
        First image
    data2 : numpy array
        Second image. If None, a median filtered version of 'data1'
        is employed. In this case, the parameter 'median_size' must
        be properly set.
    ioffx : int
        Integer offset (pixels) to place the second image on top of
        the first image in the horizontal direction (axis=1) in the
        numpy array.
    ioffy : int
        Integer offset (pixels) to place the second image on top of
        the first image in the vertical direction (axis=0) in the
        numpy array.
    tsigma_peak : float
        Times sigma to detect cosmic ray peaks
    tsigma_tail : float
        Times sigma to detect additional pixels affected by cosmic
        rays.
    maxsize : int
        Maximum number of pixels affected by a single cosmic ray.
        Above this number the detection is ignored.
    list_skipped_regions : list or None
        List of SliceRegion2D instances indicating image regions where
        detected cosmic rays will not be removed. The indices refer
        to the location of the regions in data1. This option is not
        compatible with image_region.
    image_region : SliceRegion2D instance or None
        Single instance of SliceRegion2D indicating the region where
        cosmic rays will be detected. The indices refer to the location
        of the region in data1. This option is not compatible
        with list_skipped_regions.
    median_size : tuple of integers
         Shape that is taken from the input array, at every element
         position, to compute the median filter. Note that the tuple
         order correspond to (Yaxis, Xaxis). This is only employed
         when 'data2' is None.
    return_masks : bool
        If True, return the masks with the replaced pixels flagged.
    debug_level : int
        If different from zero, print and display debugging information.
    zoom_region_imshow : SliceRegion2D instance or None
        If not None, display intermediate images, zooming in the
        indicated region.

    Returns
    -------
    data1c : numpy array
        Cleaned first image
    data2c : numpy.array
        Cleaned second image of median filtered version of 'data1' if
        'data2' is None.
    mask_data1c : numpy array
        Mask corresponding to replaced pixels in 'data1' (0: unmodified
        pixels, 1: replaced pixels). This array is returned only it the
        input parameter 'return_masks' is True.
    mask_data2c : numpy array
        Mask corresponding to replaced pixels in 'data2' (0: unmodified
        pixels, 1: replaced pixels). This array is returned only it the
        input parameter 'return_masks' is True.

    """

    if data2 is None:
        if (ioffx != 0) or (ioffy != 0):
            raise ValueError(f'ERROR: ioffx={ioffx} and ioffy={ioffy} must be zero!')
        if median_size is None:
            raise ValueError(f'ERROR: you must specify median_size when only one image is available')
        data2 = ndimage.median_filter(data1, size=median_size)
        single_mode = True
    else:
        single_mode = False

    if list_skipped_regions is not None and image_region is not None:
        raise ValueError('list_skipped_regions and useful_region are incompatible')

    shape1 = data1.shape
    shape2 = data2.shape

    if shape1 != shape2:
        raise ValueError('ERROR: the images have different shape')

    naxis2, naxis1 = shape1

    if abs(ioffx) > naxis1:
        raise ValueError(f'ERROR: ioffx={ioffx} is too large')
    if abs(ioffy) > naxis2:
        raise ValueError(f'ERROR: ioffy={ioffy} is too large')

    # compute intersection between the two images after accounting
    # for the relative offsets between them
    if ioffx <= 0:
        j1 = 0
        j2 = naxis1 + ioffx
        jj1 = -ioffx
        jj2 = naxis1
    else:
        j1 = ioffx
        j2 = naxis1
        jj1 = 0
        jj2 = naxis1 - ioffx

    if ioffy <= 0:
        i1 = 0
        i2 = naxis2 + ioffy
        ii1 = -ioffy
        ii2 = naxis2
    else:
        i1 = ioffy
        i2 = naxis2
        ii1 = 0
        ii2 = naxis2 - ioffy

    # define region where C.R. will be sought
    if list_skipped_regions is not None:
        mask_useful = np.ones((naxis2, naxis1), dtype=int)
        for region2d in list_skipped_regions:
            if isinstance(region2d, SliceRegion2D):
                mask_useful[region2d.python] = 0
            else:
                raise ValueError(f'Invalid item in list_skipped_regions: {region2d}')
    elif image_region is not None:
        mask_useful = np.zeros((naxis2, naxis1), dtype=int)
        if isinstance(image_region, SliceRegion2D):
            mask_useful[image_region.python] = 1
        else:
            raise ValueError(f'Invalid image_region: {image_region}')
    else:
        mask_useful = np.ones((naxis2, naxis1), dtype=int)

    if debug_level > 0:
        print('Computing overlaping rectangle:')
        print(f'ioffx: {ioffx}')
        print(f'ioffx: {ioffy}')
        print('data1:  i1,  i2,  j1,  j2: {}, {}, {}, {}'.format(i1, i2, j1, j2))
        print('data2: ii1, ii2, jj1, jj2: {}, {}, {}, {}'.format(ii1, ii2, jj1, jj2))

    # extract overlapping rectangular regions
    subdata1 = data1[i1:i2, j1:j2]
    subdata2 = data2[ii1:ii2, jj1:jj2]
    subuseful = mask_useful[i1:i2, j1:j2]

    shape1 = subdata1.shape
    shape2 = subdata2.shape
    if shape1 != shape2:
        raise ValueError('ERROR: overlapping regions have different shape')

    if maxsize is None:
        maxsize = shape1[0] * shape1[1]

    # difference between the two overlapping regions
    diff = subdata1 - subdata2
    if debug_level > 0:
        print(f'shape1: {shape1}, shape2: {shape2}, shape(diff): {diff.shape}')

    # statistical summary
    median = float(np.median(diff))
    std = robust_std(diff)
    if debug_level > 0:
        print('\n>>> Statistical summary of diff = overlapping data1 - data2:')
        print(f'>>> Median....: {median:.3f}')
        print(f'>>> Robust_std: {std:.3f}')

    # search for positive peaks (CR in data1)
    labels_pos_peak, no_cr_pos_peak = ndimage.label(diff > median + tsigma_peak * std)
    labels_pos_tail, no_cr_pos_tail = ndimage.label(diff > median + tsigma_tail * std)
    # set all CR peak pixels to 1
    mask_pos_peak = np.zeros_like(labels_pos_peak)
    mask_pos_peak[labels_pos_peak > 0] = 1
    # multiply peak and tail masks in order to find the intersection
    labels_pos_tail_in_peak = labels_pos_tail * mask_pos_peak * subuseful
    # define effective mask using tail mask of CR detected in peak mask
    mask_pos_clean = np.zeros_like(labels_pos_peak)
    for icr in np.unique(labels_pos_tail_in_peak):
        if icr > 0:
            npix_affected = np.sum(labels_pos_tail == icr)
            if npix_affected <= maxsize:
                mask_pos_clean[labels_pos_tail == icr] = 1
    # replace pixels affected by cosmic rays
    data1c = data1.copy()
    for item in np.argwhere(mask_pos_clean):
        data1c[item[0] + i1, item[1] + j1] = data2[item[0] + ii1, item[1] + jj1] + median

    if single_mode:
        data2c = data2.copy()
        mask_neg_clean = np.zeros_like(mask_pos_clean)
        labels_neg_peak = None
        labels_neg_tail = None
        labels_neg_tail_in_peak = None
        no_cr_neg_peak = 0
        no_cr_neg_tail = 0
    else:
        # search for negative peaks (CR in data2)
        labels_neg_peak, no_cr_neg_peak = ndimage.label(diff < median - tsigma_peak * std)
        labels_neg_tail, no_cr_neg_tail = ndimage.label(diff < median - tsigma_tail * std)
        # set all CR peak pixels to 1
        mask_neg_peak = np.zeros_like(labels_neg_peak)
        mask_neg_peak[labels_neg_peak > 0] = 1
        # multiply peak and tail masks in order to find the intersection
        labels_neg_tail_in_peak = labels_neg_tail * mask_neg_peak * subuseful
        # define effective mask using tail mask of CR detected in peak mask
        mask_neg_clean = np.zeros_like(labels_neg_peak)
        for icr in np.unique(labels_neg_tail_in_peak):
            if icr > 0:
                npix_affected = np.sum(labels_neg_tail == icr)
                if npix_affected <= maxsize:
                    mask_neg_clean[labels_neg_tail == icr] = 1
        # replace pixels affected by cosmic rays
        data2c = data2.copy()
        for item in np.argwhere(mask_neg_clean):
            data2c[item[0] + ii1, item[1] + jj1] = data1[item[0] + i1, item[1] + j1] - median

    # insert result in arrays with the original data shape
    mask_data1c = np.zeros((naxis2, naxis1), dtype=int)
    mask_data1c[i1:i2, j1:j2] = mask_pos_clean
    mask_data2c = np.zeros((naxis2, naxis1), dtype=int)
    mask_data2c[ii1:ii2, jj1:jj2] = mask_neg_clean

    # display intermediate results
    if debug_level > 0:
        if zoom_region_imshow is None:
            zoom_region_imshow = SliceRegion2D(np.s_[0:naxis2, 0:naxis1], mode='python')
        else:
            if isinstance(zoom_region_imshow, SliceRegion2D):
                pass
            else:
                raise ValueError(f'Object zoom_region_imshow={zoom_region_imshow} '
                                 f'of type {type(zoom_region_imshow)} is not a SliceRegion2D')
        if debug_level == 2:
            # display histogram
            hmin = max(min(diff.flatten()), median - 10*tsigma_peak*std)
            hmax = min(max(diff.flatten()), median + 10*tsigma_peak*std)
            bins = np.linspace(hmin, hmax, 100)
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 6))
            ax.hist(diff[zoom_region_imshow.python].flatten(), bins=bins)
            ax.set_xlabel('ADU')
            ax.set_ylabel('Number of pixels')
            ax.set_title('diff: overlapping data1 - data2')
            ax.set_yscale('log')
            plt.show()
            # display diff
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15*naxis2/naxis1))
            vmin = median - tsigma_peak * std
            vmax = median + tsigma_peak * std
            imshow(fig, ax, diff, vmin=vmin, vmax=vmax, cmap='seismic')
            ax.set_xlim([zoom_region_imshow.python[1].start, zoom_region_imshow.python[1].stop])
            ax.set_ylim([zoom_region_imshow.python[0].start, zoom_region_imshow.python[0].stop])
            ax.set_title('diff: overlapping data1 - data2')
            plt.tight_layout()
            plt.show()
        # display data and labels
        image_list1 = [
            data1, labels_pos_peak, labels_pos_tail, labels_pos_tail_in_peak, data1, data1c
        ]
        title_list1 = [
            'data1', 'labels_pos_peak', 'labels_pos_tail', 'labels_pos_tail_in_peak', 
            'data1 with C.R.', 'data1c'
        ]
        if debug_level == 1:
            del image_list1[1:4]
            del title_list1[1:4]
        if single_mode:
            image_list2 = None
            title_list2 = None
        else:
            image_list2 = [
                data2, labels_neg_peak, labels_neg_tail, labels_neg_tail_in_peak, data2, data2c
            ]
            title_list2 = [
                'data2', 'labels_neg_peak', 'labels_neg_tail', 'labels_neg_tail_in_peak', 
                'data2 with C.R.', 'data2c'
            ]
            if debug_level == 1:
                del image_list2[1:4]
                del title_list2[1:4]
        if single_mode:
            nblocks = 1
        else:
            nblocks = 2
        for iblock in range(nblocks):
            if iblock == 0:
                image_list = image_list1
                title_list = title_list1
                mask_datac = mask_data1c
                print(f'Number of CR in data1: {no_cr_pos_peak} peaks, {no_cr_pos_tail} tails')
            else:
                image_list = image_list2
                title_list = title_list2
                mask_datac = mask_data2c
                print(f'Number of CR in data2: {no_cr_neg_peak} peaks, {no_cr_neg_tail} tails')
            for iplot, (image, title) in enumerate(zip(image_list, title_list)):
                imgplot = image[zoom_region_imshow.python]
                naxis2_, naxis1_ = imgplot.shape
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 15*naxis2_/naxis1_))
                median_ = np.median(imgplot)
                std_ = robust_std(imgplot)
                if std_ == 0:
                    vmin = imgplot.min()
                    vmax = imgplot.max()
                    cmap = 'gist_ncar'
                else:
                    vmin = median_ - 2 * std_
                    vmax = median_ + 5 * std_
                    cmap = 'gray'
                imshow(fig, ax, image, vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_xlim([zoom_region_imshow.python[1].start, zoom_region_imshow.python[1].stop - 1])
                ax.set_ylim([zoom_region_imshow.python[0].start, zoom_region_imshow.python[0].stop - 1])
                ax.set_title(title)
                if title[-4:] == 'C.R.':
                    xyp = np.argwhere(mask_datac > 0)
                    xp = [item[1] for item in xyp]
                    yp = [item[0] for item in xyp]
                    ax.plot(xp, yp, 'r+')
                if 'data' in title:
                    if list_skipped_regions is not None:
                        for region2d in list_skipped_regions:
                            x1, x2 = region2d.python[1].start, region2d.python[1].stop - 1
                            y1, y2 = region2d.python[0].start, region2d.python[0].stop - 1
                            xwidth = x2 - x1
                            yheight = y2 - y1
                            if iblock == 1:
                                x1 = x1 - ioffx
                                y1 = y1 - ioffy
                            rect = patches.Rectangle((x1, y1), xwidth, yheight,
                                                     edgecolor='yellow', facecolor='none')
                            ax.add_patch(rect)
                    elif image_region is not None:
                        x1, x2 = image_region.python[1].start, image_region.python[1].stop - 1
                        y1, y2 = image_region.python[0].start, image_region.python[0].stop - 1
                        xwidth = x2 - x1
                        yheight = y2 - y1
                        if iblock == 1:
                            x1 = x1 - ioffx
                            y1 = y1 - ioffy
                        rect = patches.Rectangle((x1, y1), xwidth, yheight,
                                                 edgecolor='cyan', facecolor='none')
                        ax.add_patch(rect)
                plt.tight_layout()
                plt.show()

    if return_masks:
        return data1c, data2c, mask_data1c, mask_data2c
    else:
        return data1c, data2c


def apply_cr2images_ccddata(infile1, infile2=None, outfile1=None, outfile2=None,
                            ioffx=0, ioffy=0, tsigma_peak=10, tsigma_tail=3,
                            list_skipped_regions=None, image_region=None,
                            median_size=None, debug_level=0, zoom_region_imshow=None):
    """Apply cr2images() to FITS files storing CCDData.

    The FITS file must contain:
    - a primary HDU
    - extension1: MASK
    - extension2: UNCERT

    Parameters
    ----------
    infile1 : str
        Input file name corresponding to the first image.
    infile2 : str or None
        Input file name corresponding to the second image. If None,
        a median filtered version of the first image is employed.
        In this case, the parameter 'median_size' must be properly set.
    outfile1 : str
        Output file name of the cleaned version of the first image.
    outfile2 : str or None
        Output file name of the cleaned version of the second image
        (when a second input file is provided). Otherwise this parameter
        is ignored.
        ioffx : int
        Integer offset (pixels) to place the second image on top of
        the first image in the horizontal direction (axis=1) in the
        numpy array.
    ioffx : int
        Integer offset (pixels) to place the second image on top of
        the first image in the horizontal direction (axis=1) in the
        numpy array.
    ioffy : int
        Integer offset (pixels) to place the second image on top of
        the first image in the vertical direction (axis=0) in the
        numpy array.
    tsigma_peak : float
        Times sigma to detect cosmic ray peaks
    tsigma_tail : float
        Times sigma to detect additional pixels affected by cosmic
        rays.
    list_skipped_regions : list or None
        List of SliceRegion2D instances indicating image regions where
        detected cosmic rays will not be removed. The indices refer
        to the location of the regions in data1. This option is not
        compatible with image_region.
    image_region : SliceRegion2D instance or None
        Single instance of SliceRegion2D indicating the region where
        cosmic rays will be detected. The indices refer to the location
        of the region in data1. This option is not compatible
        with list_skipped_regions.
    median_size : tuple of integers
         Shape that is taken from the input array, at every element
         position, to compute the median filter. Note that the tuple
         order correspond to (Yaxis, Xaxis). This is only employed
         when 'data2' is None.
    debug_level : int
        If different from zero, print and display debugging information.
    zoom_region_imshow : SliceRegion2D instance or None
        If not None, display intermediate images, zooming in the
        indicated region.

    """

    if infile2 is None:
        if (ioffx != 0) or (ioffy != 0):
            raise ValueError(f'ERROR: ioffx={ioffx} and ioffy={ioffy} must be zero!')
        if median_size is None:
            raise ValueError(f'ERROR: you must specify median_size when only one image is available')

    history_list = ['using cr2images:']

    ccdimage1 = CCDData.read(infile1)
    ccdimage1_clean = ccdimage1.copy()

    if infile2 is not None:
        ccdimage2 = CCDData.read(infile2)
        ccdimage2_clean = ccdimage2.copy()
        ccdimage1_clean.data, ccdimage2_clean.data, mask_data1c, mask_data2c = cr2images(
            data1=ccdimage1.data,
            data2=ccdimage2.data,
            ioffx=ioffx,
            ioffy=ioffy,
            tsigma_peak=tsigma_peak,
            tsigma_tail=tsigma_tail,
            list_skipped_regions=list_skipped_regions,
            image_region=image_region,
            return_masks=True,
            debug_level=debug_level,
            zoom_region_imshow=zoom_region_imshow
        )
        ccdimage1_clean.mask[mask_data1c.astype(bool)] = True
        ccdimage2_clean.mask[mask_data2c.astype(bool)] = True
        ccdimage1_clean.uncertainty.array[ccdimage1_clean.mask] = \
            ccdimage2.uncertainty.array[ccdimage1_clean.mask]
        ccdimage2_clean.uncertainty.array[ccdimage2_clean.mask] = \
            ccdimage1.uncertainty.array[ccdimage2_clean.mask]
        history_list.append(f'- infile1: {Path(infile1).name}')
        history_list.append(f'- infile2: {Path(infile2).name}')
        history_list.append(f'- ioffx: {ioffx}')
        history_list.append(f'- ioffy: {ioffy}')
    else:
        ccdimage2_clean = None
        ccdimage1_clean.data, _, mask_data1c, _ = cr2images(
            data1=ccdimage1.data,
            median_size=median_size,
            tsigma_peak=tsigma_peak,
            tsigma_tail=tsigma_tail,
            list_skipped_regions=list_skipped_regions,
            image_region=image_region,
            return_masks=True,
            debug_level=debug_level,
            zoom_region_imshow=zoom_region_imshow
        )
        ccdimage1_clean.mask[mask_data1c.astype(bool)] = True
        ccdimage2_uncertainty_array = ndimage.median_filter(ccdimage1.uncertainty.array, size=median_size)
        ccdimage1_clean.uncertainty.array[ccdimage1_clean.mask] = \
            ccdimage2_uncertainty_array[ccdimage1_clean.mask]
        history_list.append(f'- infile1: {Path(infile1).name}')
        history_list.append('- infile2: None')
        history_list.append(f'- median_size: {median_size}')

    history_list.append(f'- tsigma_peak: {tsigma_peak}')
    history_list.append(f'- tsigma_tail: {tsigma_tail}')

    for outfile, ccdimage_clean in zip([outfile1, outfile2],
                                       [ccdimage1_clean, ccdimage2_clean]):
        if ccdimage_clean is not None:
            # update FILENAME keyword with output file name
            ccdimage_clean.header['FILENAME'] = Path(outfile).name

            # update HISTORY in header
            ccdimage_clean.header['HISTORY'] = '-------------------'
            ccdimage_clean.header['HISTORY'] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            for item in history_list:
                ccdimage_clean.header['HISTORY'] = item

            # save result
            ccdimage_clean.write(outfile, overwrite='yes')
            print(f'Output file name: {outfile}')


def crmedian(inputlist):
    """Remove cosmic rays by computing the median of a list of arrays.

    Parameters
    ----------
    inputlist : python list
        List of tuples with the necessary input data. Each tuple contains
        three items:
        - input numpy data array
        - offset in the X direction (integer value)
        - offset in the Y direction (integer value)

    Returns
    -------
    image2d : numpy masked array
        Median combination of the input arrays.

    """

    num_images = len(inputlist)

    # check number of images
    if num_images < 3:
        raise ValueError('input list must contain at least 3 images')

    # check array dimensions
    naxis2, naxis1 = inputlist[0][0].shape
    for k in range(1, num_images):
        naxis2_, naxis1_ = inputlist[k][0].shape
        if naxis2_ != naxis2 or naxis1_ != naxis1:
            raise ValueError(f'Image sizes are not identical: ({naxis1}, {naxis2}) != ({naxis1_}, {naxis2_})')

    # data cube to store all the data arrays
    image3d = np.ma.array(
        data=np.zeros((num_images, naxis2, naxis1), dtype=float),
        mask=np.ones((num_images, naxis2, naxis1), dtype=bool)
    )
    for k in range(num_images):
        data = inputlist[k][0]
        ioffx = inputlist[k][1]
        ioffy = inputlist[k][2]
        if ioffx <= 0:
            j1 = 0
            j2 = naxis1 + ioffx
            jj1 = -ioffx
            jj2 = naxis1
        else:
            j1 = ioffx
            j2 = naxis1
            jj1 = 0
            jj2 = naxis1 - ioffx

        if ioffy <= 0:
            i1 = 0
            i2 = naxis2 + ioffy
            ii1 = -ioffy
            ii2 = naxis2
        else:
            i1 = ioffy
            i2 = naxis2
            ii1 = 0
            ii2 = naxis2 - ioffy

        image3d.mask[k, i1:i2, j1:j2] = False
        image3d.data[k, i1:i2, j1:j2] = data[ii1:ii2, jj1:jj2]

    # compute median
    image2d = np.ma.median(image3d, axis=0)

    # return result
    return image2d
