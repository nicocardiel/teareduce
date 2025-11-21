#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt

import os
from pathlib import Path
import requests


def get_cookbook_file(file_path, verbose=True):
    """
    Retrieve the contents of an auxiliary file.

    The file is downloaded from the teareduce-cookbook GitHub repository

    Parameters
    ----------
    file_path : str
        The path to the auxiliary file.
    verbose : bool, optional
        If True, print a message when the file is downloaded.
        Default is True.
    """

    url = f"https://raw.githubusercontent.com/nicocardiel/teareduce-cookbook/main/{file_path}"
    response = requests.get(url)
    local_filename = os.path.basename(file_path)
    if Path(local_filename).is_file():
        if verbose:
            print(f"File {local_filename} already exists locally. Skipping download.")
        return
    with open(local_filename, "wb") as f:
        f.write(response.content)
    if verbose:
        print(f"File {local_filename} downloaded!")
