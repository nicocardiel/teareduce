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

from .definitions import VALID_CLEANING_METHODS


class InterpolationEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Cleaning Parameters")
        self.dict_interp_methods = {
            "x interp.": "x",
            "y interp.": "y",
            "surface interp.": "a-plane",
            "median": "a-median",
            "lacosmic": "lacosmic"
        }
        self.check_interp_methods()
        # Initialize parameters
        self.cleaning_method = None
        self.npoints = None
        self.degree = None
        self.dilation = None
        # Create the form
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack()

        # Title
        title_label = tk.Label(main_frame, text="Edit Cleaning Parameters", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))

        # Create labels and entry fields for each cleaning method
        row = 1
        self.cleaning_method_var = tk.StringVar(value="surface interp.")
        for interp_method in self.dict_interp_methods.keys():
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
        self.entry_npoints.insert(0, 2)
        self.entry_npoints.grid(row=row, column=1, sticky='w')
        row += 1
        label = tk.Label(main_frame, text='Degree:')
        label.grid(row=row, column=0, sticky='e', padx=(0, 10))
        self.entry_degree = tk.Entry(main_frame, width=10)
        self.entry_degree.insert(0, 1)
        self.entry_degree.grid(row=row, column=1, sticky='w')
        row += 1
        label = tk.Label(main_frame, text='Dilation:')
        label.grid(row=row, column=0, sticky='e', padx=(0, 10))
        self.entry_dilation = tk.Entry(main_frame, width=10)
        self.entry_dilation.insert(0, 0)
        self.entry_dilation.grid(row=row, column=1, sticky='w')
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
        self.npoints = int(self.entry_npoints.get())
        self.degree = int(self.entry_degree.get())
        self.dilation = int(self.entry_dilation.get())
        self.root.destroy()

    def on_cancel(self):
        """Close the dialog without saving selected parameters."""
        self.cleaning_method = None
        self.npoints = None
        self.degree = None
        self.dilation = None
        self.root.destroy()

    def action_on_method_change(self):
        """Handle changes in the selected cleaning method."""
        selected_method = self.cleaning_method_var.get()
        print(f"Selected cleaning method: {selected_method}")
        if selected_method in ['x interp.', 'y interp.']:
            self.entry_npoints.config(state='normal')
            self.entry_degree.config(state='normal')
            self.entry_dilation.config(state='normal')
        elif selected_method == 'surface interp.':
            self.entry_npoints.config(state='normal')
            self.entry_degree.config(state='disabled')
            self.entry_dilation.config(state='normal')
        elif selected_method == 'median':
            self.entry_npoints.config(state='normal')
            self.entry_degree.config(state='disabled')
            self.entry_dilation.config(state='normal')
        elif selected_method == 'lacosmic':
            self.entry_npoints.config(state='disabled')
            self.entry_degree.config(state='disabled')
            self.entry_dilation.config(state='disabled')

    def check_interp_methods(self):
        """Check that all interpolation methods are valid."""
        for method in self.dict_interp_methods.keys():
            if method not in VALID_CLEANING_METHODS:
                raise ValueError(f"Invalid interpolation method: {method}")
        for method in VALID_CLEANING_METHODS:
            if method not in self.dict_interp_methods.keys():
                raise ValueError(f"Interpolation method not mapped: {method}")
