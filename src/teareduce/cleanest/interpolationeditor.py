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

from .definitions import VALID_CLEANING_METHODS


class InterpolationEditor:
    def __init__(self, root, last_dilation, last_npoints, last_degree, auxdata):
        self.root = root
        self.root.title("Cleaning Parameters")
        self.last_dilation = last_dilation
        self.auxdata = auxdata
        self.dict_interp_methods = {
            "x interp.": "x",
            "y interp.": "y",
            "surface interp.": "a-plane",
            "median": "a-median",
            "lacosmic": "lacosmic",
            "auxdata": "auxdata"
        }
        self.check_interp_methods()
        # Initialize parameters
        self.cleaning_method = None
        self.npoints = last_npoints
        self.degree = last_degree
        # Create the form
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack()

        # Title
        title_label = tk.Label(main_frame, text="Select Cleaning Method", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))

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
                ).grid(row=row, column=0, sticky='w')
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

        # Button frame
        self.button_frame = tk.Frame(main_frame)
        self.button_frame.grid(row=row, column=0, columnspan=2, pady=(15, 0))

        # OK button
        self.ok_button = tk.Button(self.button_frame, text="OK", width=5, command=self.on_ok)
        self.ok_button.pack(side='left', padx=5)

        # Cancel button
        self.cancel_button = tk.Button(self.button_frame, text="Cancel", width=5, command=self.on_cancel)
        self.cancel_button.pack(side='left', padx=5)

        # Initial action depending on the default method
        self.action_on_method_change()

    def on_ok(self):
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
