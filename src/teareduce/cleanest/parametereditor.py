#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Parameter editor dialog for L.A.Cosmic parameters."""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


class ParameterEditor:
    def __init__(self, root, param_dict, window_title):
        self.root = root
        self.root.title(window_title)
        self.param_dict = param_dict
        self.entries = {}  # dictionary to hold entry widgets
        self.result_dict = None

        # Create the form
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack()

        row = 0

        # Subtitle for L.A.Cosmic parameters
        subtitle_label = tk.Label(main_frame, text="L.A.Cosmic Parameters", font=("Arial", 14, "bold"))
        subtitle_label.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        row += 1

        # Create labels and entry fields for each parameter
        for key, info in self.param_dict.items():
            if key.lower() != 'dilation':
                # Parameter name label
                label = tk.Label(main_frame, text=f"{key}:", anchor='e', width=15)
                label.grid(row=row, column=0, sticky='w', pady=5)
                # Entry field
                entry = tk.Entry(main_frame, width=10)
                entry.insert(0, str(info['value']))
                entry.grid(row=row, column=1, padx=10, pady=5)
                self.entries[key] = entry  # dictionary to hold entry widgets
                # Type label
                type_label = tk.Label(main_frame, text=f"({info['type'].__name__})", fg='gray', anchor='w', width=10)
                type_label.grid(row=row, column=2, sticky='w', pady=5)
                row += 1

        # Separator
        separator1 = ttk.Separator(main_frame, orient='horizontal')
        separator1.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(10, 10))
        row += 1

        # Subtitle for additional parameters
        subtitle_label = tk.Label(main_frame, text="Additional Parameters", font=("Arial", 14, "bold"))
        subtitle_label.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        row += 1

        # Dilation label and entry
        label = tk.Label(main_frame, text="Dilation:", anchor='e', width=15)
        label.grid(row=row, column=0, sticky='w', pady=5)
        entry = tk.Entry(main_frame, width=10)
        entry.insert(0, str(self.param_dict['dilation']['value']))
        entry.grid(row=row, column=1, padx=10, pady=5)
        self.entries['dilation'] = entry
        type_label = tk.Label(main_frame, text=f"({self.param_dict['dilation']['type'].__name__})",
                              fg='gray', anchor='w', width=10)
        type_label.grid(row=row, column=2, sticky='w', pady=5)
        row += 1

        # Separator
        separator2 = ttk.Separator(main_frame, orient='horizontal')
        separator2.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(10, 10))
        row += 1

        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=(15, 0))

        # OK button
        ok_button = tk.Button(button_frame, text="OK", width=5, command=self.on_ok)
        ok_button.pack(side='left', padx=5)

        # Cancel button
        cancel_button = tk.Button(button_frame, text="Cancel", width=5, command=self.on_cancel)
        cancel_button.pack(side='left', padx=5)

        # Reset button
        reset_button = tk.Button(button_frame, text="Reset", width=5, command=self.on_reset)
        reset_button.pack(side='left', padx=5)

    def on_ok(self):
        """Validate and save the updated values"""
        try:
            updated_dict = {}

            for key, info in self.param_dict.items():
                entry_value = self.entries[key].get()
                value_type = info['type']

                # Convert string to appropriate type
                if value_type == bool:
                    # Handle boolean conversion
                    if entry_value.lower() in ['true', '1', 'yes']:
                        converted_value = True
                    elif entry_value.lower() in ['false', '0', 'no']:
                        converted_value = False
                    else:
                        raise ValueError(f"Invalid boolean value for {key}")
                else:
                    converted_value = value_type(entry_value)
                    if 'positive' in info and info['positive'] and converted_value < 0:
                        raise ValueError(f"Value for {key} must be positive")

                updated_dict[key] = {
                    'value': converted_value,
                    'type': value_type
                }

            self.result_dict = updated_dict
            self.root.destroy()

        except ValueError as e:
            messagebox.showerror("Invalid Input",
                                 f"Error converting values:\n{str(e)}\n\n"
                                 "Please check your inputs.")

    def on_cancel(self):
        """Close without saving"""
        self.result_dict = None
        self.root.destroy()

    def on_reset(self):
        """Reset all fields to original values"""
        for key, info in self.param_dict.items():
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, str(info['value']))

    def get_result(self):
        """Return the updated dictionary"""
        return self.result_dict
