#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Tracked button widget for the cleanest module."""

import tkinter as tk
from rich.console import Console

console = Console()


class TrackedTkButton:
    """Class to create tracked Tkinter buttons with help text."""

    def __init__(self, root):
        """Initialize the TrackedTkButton instance.

        Initializes an empty list to store button information.

        Parameters
        ----------
        root : tk.Tk or tk.Frame
            The parent widget where the help information about
            the actions associated to each button will be displayed.
        """
        self.root = root
        self.buttons_info = []

    def new(self, parent, text, command, help_text, alttext=None, **kwargs):
        """Create a Tkinter button with tracking information.

        Parameters
        ----------
        parent : tk.Widget
            The parent widget where the button will be placed.
        text : str
            The text to display on the button.
        command : callable
            The function to call when the button is pressed.
        help_text : str
            The help text associated with the button.
        alttext : str, optional
            Alternative text for the button.
        **kwargs : dict
            Additional keyword arguments to pass to the Tkinter Button constructor.

        Returns
        -------
        button : tk.Button
            The created Tkinter button.
        """
        button = tk.Button(parent, text=text, command=command, **kwargs)
        self.buttons_info.append({"button": button, "text": text, "help_text": help_text, "alttext": alttext})
        return button

    def show_help(self):
        """Display help information for all tracked buttons."""
        console.rule("[bold red]Button Help Information[/bold red]")
        for info in self.buttons_info:
            if info["alttext"] is not None:
                text = info["alttext"]
            else:
                text = info["text"]
            # replace '[' by '\[' to avoid formatting issues
            text = text.replace("[", "\\[")
            console.print(f"[bold blue]{text}[/bold blue]: {info['help_text']}")
        console.rule()
