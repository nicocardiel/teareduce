#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Function to center a tk child window on its parent."""


def center_on_parent(child, parent, offset_x=0, offset_y=0):
    """Center child window on parent window.

    Parameters
    ----------
    child : tk.Toplevel or tk.Tk
        The child window to be centered.
    parent : tk.Toplevel or tk.Tk
        The parent window.
    offset_x : int, optional
        Horizontal offset from center position (default is 0).
    offset_y : int, optional
        Vertical offset from center position (default is 0).
    """
    # Update to get accurate dimensions
    child.update_idletasks()
    parent.update_idletasks()

    # Get parent position and size
    parent_x = parent.winfo_x()
    parent_y = parent.winfo_y()
    parent_width = parent.winfo_width()
    parent_height = parent.winfo_height()

    # Get child size
    child_width = child.winfo_width()
    child_height = child.winfo_height()

    # Calculate center position
    x = parent_x + (parent_width - child_width) // 2
    y = parent_y + (parent_height - child_height) // 2

    # Set child position
    child.geometry(f"+{x + offset_x}+{y + offset_y}")
