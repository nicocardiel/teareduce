#
# Copyright 2022-2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#
"""Auxiliary classes to handle slicing regions in 1D, 2D, and 3D.

These classes provide a way to define and manipulate slices in a
consistent manner, following both FITS and Python conventions.
"""

import re

import numpy as np


class SliceRegion1D:
    """Store indices for slicing of 1D regions.

    The attributes .python and .fits provide the indices following
    the Python and the FITS convention, respectively.

    Attributes
    ----------
    fits : slice
        1D slice following the FITS convention.
    python : slice
        1D slice following the Python convention.
    mode : str
        Convention mode employed to define the slice.
        The two possible modes are 'fits' and 'python'.
    fits_section : str
        Resulting slice section in FITS convention: '[num1:num2]'.
        This string is defined after successfully initializing
        the SliceRegion1D instance.

    Methods
    -------
    within(other)
        Check if slice 'other' is within the parent slice.
    """

    def __init__(self, region, mode=None, naxis1=None):
        """Initialize SliceRegion1D.

        Parameters
        ----------
        region : slice or str
            Slice region. It can be provided as np.s_[num1:num2],
            as slice(num1, num2) or as a string '[num1:num2]'
        mode : str
            Convention mode employed to define the slice.
            The two possible modes are 'fits' and 'python'.
        naxis1 : int
            The axis 1 size (length) of the data being sliced.
            If provided, it is used to validate the slice region.
        """
        if isinstance(region, str):
            pattern = r'^\s*\[\s*\d+\s*:\s*\d+\s*\]\s*$'
            if not re.match(pattern, region):
                raise ValueError(f"Invalid {region!r}. It must match '[num:num]'")
            # extract numbers and generate np.s_[num:num]
            numbers_str = re.findall(r'\d+', region)
            numbers_int = list(map(int, numbers_str))
            region = np.s_[numbers_int[0]:numbers_int[1]]

        if isinstance(region, slice):
            for number in [region.start, region.stop]:
                if number is None:
                    raise ValueError(f'Invalid {region!r}: you must specify start:stop in slice by number')
        else:
            raise ValueError(f'Object {region} of type {type(region)} is not a slice')

        if region.step not in [1, None]:
            raise ValueError(f'This class {self.__class__.__name__} '
                             'does not handle step != 1')

        errmsg = f'Invalid mode={mode}. Only "FITS" or "Python" (case insensitive) are valid'
        if mode is None:
            raise ValueError(errmsg)
        self.mode = mode.lower()

        if self.mode == 'fits':
            if region.stop < region.start:
                raise ValueError(f'Invalid {region!r}: stop must be >= start')
            self.fits = region
            self.python = slice(region.start-1, region.stop)
        elif self.mode == 'python':
            if region.stop <= region.start:
                raise ValueError(f'Invalid {region!r}: stop must be > start')
            self.fits = slice(region.start+1, region.stop)
            self.python = region
        else:
            raise ValueError(errmsg)

        if naxis1 is not None:
            if not (1 <= self.fits.start <= naxis1):
                raise ValueError(f'Invalid start={self.fits.start} for naxis1={naxis1}')
            if not (1 <= self.fits.stop <= naxis1+1):
                raise ValueError(f'Invalid stop={self.fits.stop} for naxis1={naxis1}')

        s = self.fits
        self.fits_section = f'[{s.start}:{s.stop}]'

    def __eq__(self, other):
        return self.fits == other.fits and self.python == other.python

    def __repr__(self):
        if self.mode == 'fits':
            return (f'{self.__class__.__name__}('
                    f'{self.fits!r}, mode="fits")')
        else:
            return (f'{self.__class__.__name__}('
                    f'{self.python!r}, mode="python")')

    def within(self, other):
        """Determine if slice 'other' is within the parent slice.

        Parameters
        ----------
        other : SliceRegion1D
            New instance for which we want to determine
            if it is within the parent SliceRegion1D instance.

        Returns
        -------
        result : bool
            Return True if 'other' is within the parent slice.
            False otherwise.
        """
        if isinstance(other, self.__class__):
            pass
        else:
            raise ValueError(f'Object {other} of type {type(other)} is not a {self.__class__.__name__}')

        s = self.python
        s_other = other.python
        result = False
        if s.start < s_other.start:
            return result
        if s.stop > s_other.stop:
            return result
        result = True
        return result

    def length(self):
        """Return the length of the slice."""
        return self.python.stop - self.python.start


class SliceRegion2D:
    """Store indices for slicing of 2D regions.

    The attributes .python and .fits provide the indices following
    the Python and the FITS convention, respectively.

    Attributes
    ----------
    fits : slice
        2D slice following the FITS convention.
    python : slice
        2D slice following the Python convention.
    mode : str
        Convention mode employed to define the slice.
        The two possible modes are 'fits' and 'python'.
    fits_section : str
        Resulting slice section in FITS convention:
        '[num1:num2,num3:num4]'. This string is defined after
        successfully initializing the SliceRegion2D instance.

    Methods
    -------
    within(other)
        Check if slice 'other' is within the parent slice."""

    def __init__(self, region, mode=None, naxis1=None, naxis2=None):
        """Initialize SliceRegion2D.

        Parameters
        ----------
        region : slice or str
            Slice region. It can be provided as np.s_[num1:num2, num3:num4],
            as a tuple (slice(num1, num2), slice(num3, num4)),
            or as a string '[num1:num2, num3:num4]'
        mode : str
            Convention mode employed to define the slice.
            The two possible modes are 'fits' and 'python'.
        naxis1 : int
            The axis 1 size (length) of the data being sliced,
            assuming the FITS convention.
            If provided, it is used to validate the slice region.
        naxis2 : int
            The axis 2 size (length) of the data being sliced,
            assuming the FITS convention.
            If provided, it is used to validate the slice region.
        """
        if isinstance(region, str):
            pattern = r'^\s*\[\s*\d+\s*:\s*\d+\s*,\s*\d+\s*:\s*\d+\s*\]\s*$'
            if not re.match(pattern, region):
                raise ValueError(f"Invalid {region!r}. It must match '[num:num, num:num]'")
            # extract numbers and generate np.s_[num:num, num:num]
            numbers_str = re.findall(r'\d+', region)
            numbers_int = list(map(int, numbers_str))
            region = np.s_[numbers_int[0]:numbers_int[1], numbers_int[2]:numbers_int[3]]

        if isinstance(region, tuple) and len(region) == 2:
            s1, s2 = region
            for item in [s1, s2]:
                if isinstance(item, slice):
                    for number in [item.start, item.stop]:
                        if number is None:
                            raise ValueError(f'Invalid {item!r}: you must specify start:stop in slice by number')
                    if item.step not in [1, None]:
                        raise ValueError(f'This class {self.__class__.__name__} does not handle step != 1')
                else:
                    raise ValueError(f'Object {item} of type {type(item)} is not a slice')
        else:
            raise ValueError(f'This class {self.__class__.__name__} only handles 2D regions')

        errmsg = f'Invalid mode={mode}. Only "FITS" or "Python" (case insensitive) are valid'
        if mode is None:
            raise ValueError(errmsg)
        self.mode = mode.lower()

        if self.mode == 'fits':
            if s1.stop < s1.start:
                raise ValueError(f'Invalid {s1!r}: stop must be >= start')
            if s2.stop < s2.start:
                raise ValueError(f'Invalid {s2!r}: stop must be >= start')
            self.fits = region
            self.python = slice(s2.start-1, s2.stop), slice(s1.start-1, s1.stop)
        elif self.mode == 'python':
            if s1.stop <= s1.start:
                raise ValueError(f'Invalid {s1!r}: stop must be > start')
            if s2.stop <= s2.start:
                raise ValueError(f'Invalid {s2!r}: stop must be > start')
            self.fits = slice(s2.start+1, s2.stop), slice(s1.start+1, s1.stop)
            self.python = region
        else:
            raise ValueError(errmsg)

        s1, s2 = self.fits
        self.fits_section = f'[{s1.start}:{s1.stop},{s2.start}:{s2.stop}]'

        if naxis1 is not None:
            if not (1 <= s1.start <= naxis1):
                raise ValueError(f'Invalid start={s1.start} for naxis1={naxis1}')
            if not (1 <= s1.stop <= naxis1):
                raise ValueError(f'Invalid stop={s1.stop} for naxis1={naxis1}')
        if naxis2 is not None:
            if not (1 <= s2.start <= naxis2):
                raise ValueError(f'Invalid start={s2.start} for naxis2={naxis2}')
            if not (1 <= s2.stop <= naxis2):
                raise ValueError(f'Invalid stop={s2.stop} for naxis2={naxis2}')

    def __eq__(self, other):
        return self.fits == other.fits and self.python == other.python

    def __repr__(self):
        if self.mode == 'fits':
            return (f'{self.__class__.__name__}('
                    f'{self.fits!r}, mode="fits")')
        else:
            return (f'{self.__class__.__name__}('
                    f'{self.python!r}, mode="python")')

    def within(self, other):
        """Determine if slice 'other' is within the parent slice.

        Parameters
        ----------
        other : SliceRegion2D
            New instance for which we want to determine
            if it is within the parent SliceRegion2D instance.

        Returns
        -------
        result : bool
            Return True if 'other' is within the parent slice.
            False otherwise.
        """
        if isinstance(other, self.__class__):
            pass
        else:
            raise ValueError(f'Object {other} of type {type(other)} is not a {self.__class__.__name__}')

        s1, s2 = self.python
        s1_other, s2_other = other.python
        result = False
        if s1.start < s1_other.start:
            return result
        if s1.stop > s1_other.stop:
            return result
        if s2.start < s2_other.start:
            return result
        if s2.stop > s2_other.stop:
            return result
        result = True
        return result

    def area(self):
        """Return the area of the slice."""
        s1, s2 = self.python
        return (s1.stop - s1.start) * (s2.stop - s2.start)


class SliceRegion3D:
    """Store indices for slicing of 3D regions.

    The attributes .python and .fits provide the indices following
    the Python and the FITS convention, respectively.

    Attributes
    ----------
    fits : slice
        3D slice following the FITS convention.
    python : slice
        3D slice following the Python convention.
    mode : str
        Convention mode employed to define the slice.
        The two possible modes are 'fits' and 'python'.
    fits_section : str
        Resulting slice section in FITS convention:
        '[num1:num2,num3:num4,num5:num6]'. This string is defined after
        successfully initializing the SliceRegion3D instance.

    Methods
    -------
    within(other)
        Check if slice 'other' is within the parent slice."""

    def __init__(self, region, mode=None, naxis1=None, naxis2=None, naxis3=None):
        """Initialize SliceRegion3D.

        Parameters
        ----------
        region : slice or str
            Slice region. It can be provided as np.s_[num1:num2, num3:num4, num5:num6],
            as a tuple (slice(num1, num2), slice(num3, num4), slice(num5, num6)),
            or as a string '[num1:num2, num3:num4, num5:num6]'
        mode : str
            Convention mode employed to define the slice.
            The two possible modes are 'fits' and 'python'.
        naxis1 : int
            The axis 1 size (length) of the data being sliced,
            assuming the FITS convention.
            If provided, it is used to validate the slice region.
        naxis2 : int
            The axis 2 size (length) of the data being sliced,
            assuming the FITS convention.
            If provided, it is used to validate the slice region.
        naxis3 : int
            The axis 3 size (length) of the data being sliced,
            assuming the FITS convention.
            If provided, it is used to validate the slice region.
        """
        if isinstance(region, str):
            pattern = r'^\s*\[\s*\d+\s*:\s*\d+\s*,\s*\d+\s*:\s*\d+\s*,\s*\d+\s*:\s*\d+\s*\]\s*$'
            if not re.match(pattern, region):
                raise ValueError(f"Invalid {region!r}. It must match '[num:num, num:num, num:num]'")
            # extract numbers and generate np.s_[num:num, num:num, num:num]
            numbers_str = re.findall(r'\d+', region)
            numbers_int = list(map(int, numbers_str))
            region = np.s_[numbers_int[0]:numbers_int[1], numbers_int[2]:numbers_int[3], numbers_int[4]:numbers_int[5]]

        if isinstance(region, tuple) and len(region) == 3:
            s1, s2, s3 = region
            for item in [s1, s2, s3]:
                if isinstance(item, slice):
                    for number in [item.start, item.stop]:
                        if number is None:
                            raise ValueError(f'Invalid {item!r}: you must specify start:stop in slice by number')
                    if item.step not in [1, None]:
                        raise ValueError(f'This class {self.__class__.__name__} does not handle step != 1')
                else:
                    raise ValueError(f'Object {item} of type {type(item)} is not a slice')
        else:
            raise ValueError(f'This class {self.__class__.__name__} only handles 3D regions')

        errmsg = f'Invalid mode={mode}. Only "FITS" or "Python" (case insensitive) are valid'
        if mode is None:
            raise ValueError(errmsg)
        self.mode = mode.lower()

        if self.mode == 'fits':
            if s1.stop < s1.start:
                raise ValueError(f'Invalid {s1!r}: stop must be >= start')
            if s2.stop < s2.start:
                raise ValueError(f'Invalid {s2!r}: stop must be >= start')
            if s3.stop < s3.start:
                raise ValueError(f'Invalid {s3!r}: stop must be >= start')
            self.fits = region
            self.python = slice(s3.start-1, s3.stop), slice(s2.start-1, s2.stop), slice(s1.start-1, s1.stop)
        elif self.mode == 'python':
            if s1.stop <= s1.start:
                raise ValueError(f'Invalid {s1!r}: stop must be > start')
            if s2.stop <= s2.start:
                raise ValueError(f'Invalid {s2!r}: stop must be > start')
            if s3.stop <= s3.start:
                raise ValueError(f'Invalid {s3!r}: stop must be > start')
            self.fits = slice(s3.start+1, s3.stop), slice(s2.start+1, s2.stop), slice(s1.start+1, s1.stop)
            self.python = region
        else:
            raise ValueError(errmsg)

        s1, s2, s3 = self.fits
        self.fits_section = f'[{s1.start}:{s1.stop},{s2.start}:{s2.stop},{s3.start}:{s3.stop}]'

        if naxis1 is not None:
            if not (1 <= s1.start <= naxis1):
                raise ValueError(f'Invalid start={s1.start} for naxis1={naxis1}')
            if not (1 <= s1.stop <= naxis1):
                raise ValueError(f'Invalid stop={s1.stop} for naxis1={naxis1}')
        if naxis2 is not None:
            if not (1 <= s2.start <= naxis2):
                raise ValueError(f'Invalid start={s2.start} for naxis2={naxis2}')
            if not (1 <= s2.stop <= naxis2):
                raise ValueError(f'Invalid stop={s2.stop} for naxis2={naxis2}')
        if naxis3 is not None:
            if not (1 <= s3.start <= naxis3):
                raise ValueError(f'Invalid start={s3.start} for naxis3={naxis3}')
            if not (1 <= s3.stop <= naxis3):
                raise ValueError(f'Invalid stop={s3.stop} for naxis3={naxis3}')

    def __eq__(self, other):
        return self.fits == other.fits and self.python == other.python

    def __repr__(self):
        if self.mode == 'fits':
            return (f'{self.__class__.__name__}('
                    f'{self.fits!r}, mode="fits")')
        else:
            return (f'{self.__class__.__name__}('
                    f'{self.python!r}, mode="python")')

    def within(self, other):
        """Determine if slice 'other' is within the parent slice.

        Parameters
        ----------
        other : SliceRegion3D
            New instance for which we want to determine
            if it is within the parent SliceRegion3D instance.

        Returns
        -------
        result : bool
            Return True if 'other' is within the parent slice.
            False otherwise.
        """
        if isinstance(other, self.__class__):
            pass
        else:
            raise ValueError(f'Object {other} of type {type(other)} is not a {self.__class__.__name__}')

        s1, s2, s3 = self.python
        s1_other, s2_other, s3_other = other.python
        result = False
        if s1.start < s1_other.start:
            return result
        if s1.stop > s1_other.stop:
            return result
        if s2.start < s2_other.start:
            return result
        if s2.stop > s2_other.stop:
            return result
        if s3.start < s3_other.start:
            return result
        if s3.stop > s3_other.stop:
            return result
        result = True
        return result

    def volume(self):
        """Return the volume of the slice."""
        s1, s2, s3 = self.python
        return (s1.stop - s1.start) * (s2.stop - s2.start) * (s3.stop - s3.start)
