#
# Copyright 2022-2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import numpy as np
import re

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
    
    def __init__(self, region, mode=None):
        """Initialize SliceRegion1D.

        Parameters
        ----------
        region : slice or str
            Slice region. It can be provided as np.s_[num1:num2],
            as slice(num1, num2) or as a string '[num1:num2]'
        mode : str
            Convention mode employed to define the slice.
            The two possible modes are 'fits' and 'python'.
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
            for number in [slice.start, slice.stop]:
                if number is None:
                    raise ValueError(f'Invalid {slice!r}: you must specify start:stop in slice by number')
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
                raise ValueError(f'Invalid {region!r}')
            self.fits = region
            self.python = slice(region.start-1, region.stop)
        elif self.mode == 'python':
            if region.stop <= region.start:
                raise ValueError(f'Invalid {region!r}')
            self.fits = slice(region.start+1, region.stop)
            self.python = region
        else:
            raise ValueError(errmsg)

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
                     

class SliceRegion2D:
    """Store indices for slicing of 2D regions.

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
        Resulting slice section in FITS convention:
        '[num1:num2,num3:num4]'. This string is defined after
        successfully initializing the SliceRegion2D instance.

    Methods
    -------
    within(other)
        Check if slice 'other' is within the parent slice."""

    def __init__(self, region, mode=None):
        """Initialize SliceRegion1D.

        Parameters
        ----------
        region : slice or str
            Slice region. It can be provided as np.s_[num1:num2, num3:num4],
            as a tuple (slice(num1, num2), slice(num3, num4)),
            or as a string '[num1:num2, num3:num4]'
        mode : str
            Convention mode employed to define the slice.
            The two possible modes are 'fits' and 'python'.
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
                    for number in [s1.start, s1.stop, s2.start, s2.stop]:
                        if number is None:
                            raise ValueError(f'Invalid {item!r}: you must specify start:stop in slice by number')
                else:
                    raise ValueError(f'Object {item} of type {type(item)} is not a slice')
            if s1.step not in [1, None] or s2.step not in [1, None]:
                raise ValueError(f'This class {self.__class__.__name__} does not handle step != 1')
        else:
            raise ValueError(f'This class {self.__class__.__name__} only handles 2D regions')

        errmsg = f'Invalid mode={mode}. Only "FITS" or "Python" (case insensitive) are valid'
        if mode is None:
            raise ValueError(errmsg)
        self.mode = mode.lower()

        if self.mode == 'fits':
            if s1.stop < s1.start:
                raise ValueError(f'Invalid {s1!r}')
            if s2.stop < s2.start:
                raise ValueError(f'Invalid {s2!r}')
            self.fits = region
            self.python = slice(s2.start-1, s2.stop), slice(s1.start-1, s1.stop)
        elif self.mode == 'python':
            if s1.stop <= s1.start:
                raise ValueError(f'Invalid {s1!r}')
            if s2.stop <= s2.start:
                raise ValueError(f'Invalid {s2!r}')
            self.fits = slice(s2.start+1, s2.stop), slice(s1.start+1, s1.stop)
            self.python = region
        else:
            raise ValueError(errmsg)

        s1, s2 = self.fits
        self.fits_section = f'[{s1.start}:{s1.stop},{s2.start}:{s2.stop}]'

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
