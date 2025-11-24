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
from tkinter import messagebox


class ParameterEditor:
    def __init__(self, root, param_dict, window_title):
        self.root = root
        self.root.title(window_title)
        self.param_dict = param_dict
        self.entries = {}
        self.result_dict = None

        # Create the form
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack()

        # Title
        title_label = tk.Label(main_frame, text="Edit Parameters",
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15))

        # Create labels and entry fields for each parameter
        row = 1
        for key, info in self.param_dict.items():
            # Parameter name label
            label = tk.Label(main_frame, text=f"{key}:", anchor='w', width=15)
            label.grid(row=row, column=0, sticky='w', pady=5)

            # Entry field
            entry = tk.Entry(main_frame, width=15)
            entry.insert(0, str(info['value']))
            entry.grid(row=row, column=1, padx=10, pady=5)
            self.entries[key] = entry

            # Type label
            type_label = tk.Label(main_frame, text=f"({info['type'].__name__})",
                                  fg='gray', anchor='w', width=10)
            type_label.grid(row=row, column=2, sticky='w', pady=5)

            row += 1

        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=(15, 0))

        # OK button
        ok_button = tk.Button(button_frame, text="OK", width=10,
                              command=self.on_ok)
        ok_button.pack(side='left', padx=5)

        # Cancel button
        cancel_button = tk.Button(button_frame, text="Cancel", width=10,
                                  command=self.on_cancel)
        cancel_button.pack(side='left', padx=5)

        # Reset button
        reset_button = tk.Button(button_frame, text="Reset", width=10,
                                 command=self.on_reset)
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
