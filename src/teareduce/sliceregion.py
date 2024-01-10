#
# Copyright 2022-2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


class SliceRegion1D:
    """Store indices for slicing of 1D regions.
    
    The attributes .python and .fits provide the indices following
    the Python and the FITS convention, respectively.
    
    """
    
    def __init__(self, region, mode=None):
        if isinstance(region, slice):
            pass
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
        if isinstance(other, self.__class__):
            pass
        else:
            raise ValueError(f'Object {other} of type {type(other)} is not a {self.__class__.__name__}')
        region = self.python
        region_other = other.python
        if region.start < region_other.start:
            return False
        if region.stop > region_other.stop:
            return False
        return True
                     

class SliceRegion2D:
    """Store indices for slicing of 2D regions.

    The attributes .python and .fits provide the indices following
    the Python and the FITS convention, respectively.

    """

    def __init__(self, region, mode=None):
        if len(region) != 2:
            raise ValueError(f'This class {self.__class__.__name__} '
                             'only handles 2D regions')

        s1, s2 = region
        for item in [s1, s2]:
            if isinstance(item, slice):
                pass
            else:
                raise ValueError(f'Object {item} of type {type(item)} is not a slice')

        if s1.step not in [1, None] or s2.step not in [1, None]:
            raise ValueError(f'This class {self.__class__.__name__} '
                             'does not handle step != 1')

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
        self.fits_section = f'[{s1.start}:{s1.stop}, {s2.start}:{s2.stop}]'

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
        if isinstance(other, self.__class__):
            pass
        else:
            raise ValueError(f'Object {other} of type {type(other)} is not a {self.__class__.__name__}')

        s1, s2 = self.python
        s1_other, s2_other = other.python
        if s1.start < s1_other.start:
            return False
        if s1.stop > s1_other.stop:
            return False
        if s2.start < s2_other.start:
            return False
        if s2.stop > s2_other.stop:
            return False
        return True
