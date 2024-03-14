# -*- coding: utf-8 -*-
#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import platform
import sys
from .version import version


def elapsed_time(time_ini, time_end, osinfo=True):
    """Display elapsed time and OS info

    Parameters
    ----------
    time_ini : datetime instance
        Initial time.
    time_end : datetime instance
        Final time.
    osinfo : bool
        If True, display OS info.

    """

    if osinfo:
        result = platform.uname()
        print(f'system...........: {result.system}')
        print(f'release..........: {result.release}')
        print(f'machine..........: {result.machine}')
        print(f'node.............: {result.node}')
        print(f'Python executable: {sys.executable}')

    print(f"teareduce version: {version}")
    print(f"Initial time.....: {time_ini}")
    print(f"Final time.......: {time_end}")
    print(f"Elapsed time.....: {time_end - time_ini}")
