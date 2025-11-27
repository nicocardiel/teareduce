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

from .definitions import lacosmic_default_dict


class ParameterEditor:
    def __init__(self, root, param_dict, window_title, xmin, xmax, ymin, ymax, imgshape):
        self.root = root
        self.root.title(window_title)
        self.param_dict = param_dict
        # Set default region values
        self.param_dict['xmin']['value'] = xmin
        self.param_dict['xmax']['value'] = xmax
        self.param_dict['ymin']['value'] = ymin
        self.param_dict['ymax']['value'] = ymax
        self.imgshape = imgshape
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
            if key.lower() not in ['dilation', 'xmin', 'xmax', 'ymin', 'ymax']:
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

        # Subtitle for region to be examined
        subtitle_label = tk.Label(main_frame, text="Region to be Examined", font=("Arial", 14, "bold"))
        subtitle_label.grid(row=row, column=0, columnspan=3, pady=(0, 15))
        row += 1

        # Region to be examined label and entries
        for key, info in self.param_dict.items():
            if key.lower() in ['xmin', 'xmax', 'ymin', 'ymax']:
                # Parameter name label
                label = tk.Label(main_frame, text=f"{key}:", anchor='e', width=15)
                label.grid(row=row, column=0, sticky='w', pady=5)
                # Entry field
                entry = tk.Entry(main_frame, width=10)
                entry.insert(0, str(info['value']))
                entry.grid(row=row, column=1, padx=10, pady=5)
                self.entries[key] = entry  # dictionary to hold entry widgets
                # Type label
                dumtext = f"({info['type'].__name__})"
                if key.lower() in ['xmax', 'ymax']:
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
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=(5, 0))

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

            # Additional validation for region limits
            try:
                if updated_dict['xmax']['value'] <= updated_dict['xmin']['value']:
                    raise ValueError("xmax must be greater than xmin")
                if updated_dict['ymax']['value'] <= updated_dict['ymin']['value']:
                    raise ValueError("ymax must be greater than ymin")
                self.result_dict = updated_dict
                self.root.destroy()
            except ValueError as e:
                messagebox.showerror("Invalid Inputs",
                                     "Error in region limits:\n"
                                     f"{str(e)}\n\nPlease check your inputs.")

        except ValueError as e:
            messagebox.showerror("Invalid Inputs",
                                 f"Error converting value for {key}:\n{str(e)}\n\n"
                                 "Please check your inputs.")

    def on_cancel(self):
        """Close without saving"""
        self.result_dict = None
        self.root.destroy()

    def on_reset(self):
        """Reset all fields to original values"""
        self.param_dict = lacosmic_default_dict.copy()
        self.param_dict['xmin']['value'] = 1
        self.param_dict['xmax']['value'] = self.imgshape[1]
        self.param_dict['ymin']['value'] = 1
        self.param_dict['ymax']['value'] = self.imgshape[0]
        for key, info in self.param_dict.items():
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, str(info['value']))

    def get_result(self):
        """Return the updated dictionary"""
        return self.result_dict
