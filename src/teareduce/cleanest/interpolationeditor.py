#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Interpolation editor dialog for interpolation parameters."""

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from .definitions import VALID_CLEANING_METHODS


class InterpolationEditor:
    """Dialog to select interpolation cleaning parameters."""
    def __init__(self, root, last_dilation, last_npoints, last_degree, auxdata,
                 xmin, xmax, ymin, ymax, imgshape):
        """Initialize the interpolation editor dialog.

        Parameters
        ----------
        root : tk.Tk
            The root Tkinter window.
        last_dilation : int
            The last used dilation parameter.
        last_npoints : int
            The last used number of points for interpolation.
        last_degree : int
            The last used degree for interpolation.
        auxdata : array-like or None
            Auxiliary data for cleaning, if available.
        xmin : float
            Minimum x value of the data. From 1 to NAXIS1.
        xmax : float
            Maximum x value of the data. From 1 to NAXIS1.
        ymin : float
            Minimum y value of the data. From 1 to NAXIS2.
        ymax : float
            Maximum y value of the data. From 1 to NAXIS2.
        imgshape : tuple
            Shape of the image data (height, width).

        Methods
        -------
        create_widgets()
            Create the widgets for the dialog.
        on_ok()
            Handle the OK button click event.
        on_cancel()
            Handle the Cancel button click event.
        action_on_method_change()
            Handle changes in the selected cleaning method.
        check_interp_methods()
            Check that all interpolation methods are valid.

        Attributes
        ----------
        root : tk.Tk
            The root Tkinter window.
        last_dilation : int
            The last used dilation parameter.
        auxdata : array-like or None
            Auxiliary data for cleaning, if available.
        dict_interp_methods : dict
            Mapping of interpolation method names to their codes.
        cleaning_method : str or None
            The selected cleaning method code.
        npoints : int
            The number of points for interpolation.
        degree : int
            The degree for interpolation.
        xmin : float
            Minimum x value of the data. From 1 to NAXIS1.
        xmax : float
            Maximum x value of the data. From 1 to NAXIS1.
        ymin : float
            Minimum y value of the data. From 1 to NAXIS2.
        ymax : float
            Maximum y value of the data. From 1 to NAXIS2.
        imgshape : tuple
            Shape of the image data (height, width).
        """
        self.root = root
        self.root.title("Cleaning Parameters")
        self.last_dilation = last_dilation
        self.auxdata = auxdata
        self.dict_interp_methods = {
            "x interp.": "x",
            "y interp.": "y",
            "surface interp.": "a-plane",
            "median": "a-median",
            "mean": "a-mean",
            "lacosmic": "lacosmic",
            "auxdata": "auxdata"
        }
        self.check_interp_methods()
        # Initialize parameters
        self.cleaning_method = None
        self.npoints = last_npoints
        self.degree = last_degree
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.imgshape = imgshape
        # Dictionary to hold entry widgets for region parameters
        self.entries = {}
        # Create the form
        self.create_widgets()

    def create_widgets(self):
        """Create the widgets for the dialog."""
        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack()

        row = 0

        # Subtitle for cleaning method selection
        default_font = tk.font.nametofont("TkDefaultFont")
        bold_font = default_font.copy()
        bold_font.configure(weight="bold", size=default_font.cget("size") + 2)
        subtitle_label = tk.Label(main_frame, text="Select Cleaning Method", font=bold_font)
        subtitle_label.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        row += 1

        # Create labels and entry fields for each cleaning method
        row = 1
        self.cleaning_method_var = tk.StringVar(value="surface interp.")
        for interp_method in self.dict_interp_methods.keys():
            valid_method = True
            # Skip replace by L.A.Cosmic values if last dilation is not zero
            if interp_method == "lacosmic" and self.last_dilation != 0:
                valid_method = False
            # Skip auxdata method if auxdata is not available
            if interp_method == "auxdata" and self.auxdata is None:
                valid_method = False
            if valid_method:
                tk.Radiobutton(
                    main_frame,
                    text=interp_method,
                    variable=self.cleaning_method_var,
                    value=interp_method,
                    command=self.action_on_method_change
                ).grid(row=row, column=1, sticky='w')
                row += 1

        # Separator
        separator1 = ttk.Separator(main_frame, orient='horizontal')
        separator1.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(10, 10))
        row += 1

        # Subtitle for additional parameters
        subtitle_label = tk.Label(main_frame, text="Additional Parameters", font=bold_font)
        subtitle_label.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        row += 1

        # Create labels and entry fields for each additional parameter
        label = tk.Label(main_frame, text='Npoints:')
        label.grid(row=row, column=0, sticky='e', padx=(0, 10))
        self.entry_npoints = tk.Entry(main_frame, width=10)
        self.entry_npoints.insert(0, self.npoints)
        self.entry_npoints.grid(row=row, column=1, sticky='w')
        row += 1
        label = tk.Label(main_frame, text='Degree:')
        label.grid(row=row, column=0, sticky='e', padx=(0, 10))
        self.entry_degree = tk.Entry(main_frame, width=10)
        self.entry_degree.insert(0, self.degree)
        self.entry_degree.grid(row=row, column=1, sticky='w')
        row += 1

        # Separator
        separator2 = ttk.Separator(main_frame, orient='horizontal')
        separator2.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(10, 10))
        row += 1

        # Subtitle for region to be examined
        subtitle_label = tk.Label(main_frame, text="Region to be Examined", font=bold_font)
        subtitle_label.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        row += 1

        # Region to be examined label and entries
        for key in ['xmin', 'xmax', 'ymin', 'ymax']:
            # Parameter name label
            label = tk.Label(main_frame, text=f"{key}:", anchor='e', width=15)
            label.grid(row=row, column=0, sticky='w', pady=5)
            # Entry field
            entry = tk.Entry(main_frame, width=10)
            entry.insert(0, str(self.__dict__[key]))
            entry.grid(row=row, column=1, padx=10, pady=5)
            self.entries[key] = entry  # dictionary to hold entry widgets
            # Type label
            dumtext = "(int)"
            if key in ['xmax', 'ymax']:
                dumtext += f" --> [1, {self.imgshape[1]}]"
            else:
                dumtext += f" --> [1, {self.imgshape[0]}]"
            type_label = tk.Label(main_frame, text=dumtext, fg='gray', anchor='w', width=15)
            type_label.grid(row=row, column=2, sticky='w', pady=5)
            row += 1

        # Separator
        separator3 = ttk.Separator(main_frame, orient='horizontal')
        separator3.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(10, 10))
        row += 1

        # Button frame
        self.button_frame = tk.Frame(main_frame)
        self.button_frame.grid(row=row, column=0, columnspan=3, pady=(15, 0))

        # OK button
        self.ok_button = tk.Button(self.button_frame, text="OK", width=5, command=self.on_ok)
        self.ok_button.pack(side='left', padx=5)

        # Cancel button
        self.cancel_button = tk.Button(self.button_frame, text="Cancel", width=5, command=self.on_cancel)
        self.cancel_button.pack(side='left', padx=5)

        # Initial action depending on the default method
        self.action_on_method_change()

    def on_ok(self):
        """Handle the OK button click event."""
        self.cleaning_method = self.dict_interp_methods[self.cleaning_method_var.get()]
        try:
            self.npoints = int(self.entry_npoints.get())
        except ValueError:
            messagebox.showerror("Input Error", "Npoints must be a positive integer.")
            return
        if self.npoints < 1:
            messagebox.showerror("Input Error", "Npoints must be at least 1.")
            return

        try:
            self.degree = int(self.entry_degree.get())
        except ValueError:
            messagebox.showerror("Input Error", "Degree must be an integer.")
            return
        if self.degree < 0:
            messagebox.showerror("Input Error", "Degree must be non-negative.")
            return

        if self.cleaning_method in ['x', 'y'] and 2 * self.npoints <= self.degree:
            messagebox.showerror("Input Error", "2*Npoints must be greater than Degree for x and y interpolation.")
            return

        # Retrieve and validate region parameters
        try:
            xmin = int(self.entries['xmin'].get())
        except ValueError:
            messagebox.showerror("Input Error", "xmin must be an integer.")
            return
        try:
            xmax = int(self.entries['xmax'].get())
        except ValueError:
            messagebox.showerror("Input Error", "xmax must be an integer.")
            return
        if xmin >= xmax:
            messagebox.showerror("Input Error", "xmin must be less than xmax.")
            return
        try:
            ymin = int(self.entries['ymin'].get())
        except ValueError:
            messagebox.showerror("Input Error", "ymin must be an integer.")
            return
        try:
            ymax = int(self.entries['ymax'].get())
        except ValueError:
            messagebox.showerror("Input Error", "ymax must be an integer.")
            return
        if ymin >= ymax:
            messagebox.showerror("Input Error", "ymin must be less than ymax.")
            return
        for key, entry in self.entries.items():
            value = int(entry.get())
            if key in ['xmin', 'xmax']:
                if not (1 <= value <= self.imgshape[1]):
                    messagebox.showerror("Input Error", f"{key} must be in the range [1, {self.imgshape[1]}].")
                    return
            else:
                if not (1 <= value <= self.imgshape[0]):
                    messagebox.showerror("Input Error", f"{key} must be in the range [1, {self.imgshape[0]}].")
                    return
            self.__dict__[key] = value

        self.root.destroy()

    def on_cancel(self):
        """Close the dialog without saving selected parameters."""
        self.cleaning_method = None
        self.npoints = None
        self.degree = None
        self.root.destroy()

    def action_on_method_change(self):
        """Handle changes in the selected cleaning method."""
        selected_method = self.cleaning_method_var.get()
        print(f"Selected cleaning method: {selected_method}")
        if selected_method in ['x interp.', 'y interp.']:
            self.entry_npoints.config(state='normal')
            self.entry_degree.config(state='normal')
        elif selected_method == 'surface interp.':
            self.entry_npoints.config(state='normal')
            self.entry_degree.config(state='disabled')
        elif selected_method == 'median':
            self.entry_npoints.config(state='normal')
            self.entry_degree.config(state='disabled')
        elif selected_method == 'mean':
            self.entry_npoints.config(state='normal')
            self.entry_degree.config(state='disabled')
        elif selected_method == 'lacosmic':
            self.entry_npoints.config(state='disabled')
            self.entry_degree.config(state='disabled')
        elif selected_method == 'auxdata':
            self.entry_npoints.config(state='disabled')
            self.entry_degree.config(state='disabled')

    def check_interp_methods(self):
        """Check that all interpolation methods are valid."""
        for method in self.dict_interp_methods.keys():
            if method not in VALID_CLEANING_METHODS:
                raise ValueError(f"Invalid interpolation method: {method}")
        for method in VALID_CLEANING_METHODS:
            if method not in self.dict_interp_methods.keys():
                raise ValueError(f"Interpolation method not mapped: {method}")
