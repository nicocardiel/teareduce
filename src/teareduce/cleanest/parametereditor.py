#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Parameter editor dialog for L.A.Cosmic parameters."""

from astropy.io import fits
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from .askextension import ask_extension_input_image
from .centerchildparent import center_on_parent
from .definitions import lacosmic_default_dict


class ParameterEditor:
    """A dialog to edit L.A.Cosmic parameters."""

    def __init__(
        self,
        root,
        param_dict,
        window_title,
        xmin,
        xmax,
        ymin,
        ymax,
        imgshape,
        inbkg=None,
        extnum_inbkg=None,
        invar=None,
        extnum_invar=None,
    ):
        """Initialize the parameter editor dialog.

        Parameters
        ----------
        root : tk.Tk
            The root Tkinter window.
        param_dict : dict
            Dictionary with L.A.Cosmic parameters.
        window_title : str
            Title of the dialog window.
        xmin : int
            Minimum x-coordinate of the region to be examined.
            From 1 to NAXIS1.
        xmax : int
            Maximum x-coordinate of the region to be examined.
            From 1 to NAXIS1.
        ymin : int
            Minimum y-coordinate of the region to be examined.
            From 1 to NAXIS2.
        ymax : int
            Maximum y-coordinate of the region to be examined.
            From 1 to NAXIS2.
        imgshape : tuple
            Shape of the image (height, width).
        inbkg : str or None
            Path to the input background image FITS file.
        extnum_inbkg : int or None
            FITS extension number for the input background image.
        invar : str or None
            Path to the input variance image FITS file.
        extnum_invar : int or None
            FITS extension number for the input variance image.

        Methods
        -------
        create_widgets()
            Create the widgets for the dialog.
        on_ok()
            Validate and save the updated values.
        on_cancel()
            Close the dialog without saving.
        on_reset()
            Reset all fields to original values.
        get_result()
            Return the updated dictionary.

        Attributes
        ----------
        root : tk.Tk
            The root Tkinter window.
        param_dict : dict
            Dictionary with L.A.Cosmic parameters.
        imgshape : tuple
            Shape of the image (height, width).
        entries : dict
            Dictionary to hold entry widgets.
        result_dict : dict or None
            The updated dictionary of parameters or None if cancelled.
        """
        self.root = root
        self.root.title(window_title)
        self.param_dict = param_dict
        # Set default region values
        self.param_dict["xmin"]["value"] = xmin
        self.param_dict["xmax"]["value"] = xmax
        self.param_dict["ymin"]["value"] = ymin
        self.param_dict["ymax"]["value"] = ymax
        self.imgshape = imgshape
        self.entries = {"run1": {}, "run2": {}}  # dictionary to hold entry widgets
        self.inbkg = inbkg
        self.extnum_inbkg = extnum_inbkg
        self.invar = invar
        self.extnum_invar = extnum_invar
        self.result_dict = {}

        # Create the form
        self.create_widgets()
        center_on_parent(child=self.root, parent=self.root.master)

    def create_widgets(self):
        """Create the widgets for the dialog."""
        # Define different styles for different conditions
        style = ttk.Style()
        style.configure("Normal.TCombobox", foreground="black", background="white")
        style.configure("Changed.TCombobox", foreground="red", background="white")

        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack()

        row = 0

        # Subtitle for L.A.Cosmic parameters
        default_font = tk.font.nametofont("TkDefaultFont")
        bold_font = default_font.copy()
        bold_font.configure(weight="bold", size=default_font.cget("size") + 2)
        subtitle_label = tk.Label(main_frame, text="L.A.Cosmic Parameters", font=bold_font)
        subtitle_label.grid(row=row, column=0, columnspan=9, pady=(0, 10))
        row += 1

        # Count number of parameters for run1 and run2
        nparams_run1 = sum(1 for key in self.param_dict.keys() if key.startswith("run1_"))
        nparams_run2 = sum(1 for key in self.param_dict.keys() if key.startswith("run2_"))
        if nparams_run1 != nparams_run2:
            raise ValueError("Number of parameters for run1 and run2 do not match.")
        nparams_input_images = len(["inbkg", "invar"])
        nparams_total = nparams_input_images + nparams_run1
        max_num_params_in_columns = nparams_total // 2 + nparams_total % 2

        # Create labels and entry fields for each parameter.
        bold_font_subheader = default_font.copy()
        bold_font_subheader.configure(weight="bold", size=default_font.cget("size") + 1)
        for subtable in range(2):
            if subtable == 0:
                coloff = 0
            else:
                coloff = 5
            label = tk.Label(main_frame, text="Parameter", font=bold_font_subheader, anchor="w", fg="gray")
            label.grid(row=row, column=0 + coloff, sticky="e", pady=0)
            label = tk.Label(main_frame, text="Run 1", font=bold_font_subheader, anchor="w", fg="gray", width=10)
            label.grid(row=row, column=1 + coloff, sticky="w", padx=10, pady=0)
            label = tk.Label(main_frame, text="Run 2", font=bold_font_subheader, anchor="w", fg="gray", width=10)
            label.grid(row=row, column=2 + coloff, sticky="w", padx=10, pady=0)
            label = tk.Label(main_frame, text="Type", font=bold_font_subheader, anchor="w", fg="gray", width=10)
            label.grid(row=row, column=3 + coloff, sticky="w", pady=0)
        row += 1

        # Note: here we are using entry_vars to trace changes in the entries
        # so that we can update the color of run2 entries if they differ from run1.
        self.entry_vars = {}
        row_subtable = 0
        coloff = 0
        for key, info in self.param_dict.items():
            if not key.startswith("run1_"):
                continue
            # Parameter name label
            label = tk.Label(main_frame, text=f"{key[5:]}:", anchor="e", width=15)
            label.grid(row=row, column=coloff, sticky="w", pady=5)
            # Entry field for run1
            self.entry_vars[key] = tk.StringVar()
            if key[5:] in ["cleantype", "fsmode", "psfmodel"]:
                entry = ttk.Combobox(
                    main_frame,
                    textvariable=self.entry_vars[key],
                    width=8,
                    state="readonly",
                    values=info["valid_values"],
                    style="Normal.TCombobox",
                )
                self.entry_vars[key].set(str(info["value"]))
                entry.bind("<<ComboboxSelected>>", lambda e: self.update_colour_param_run1_run2())
            else:
                self.entry_vars[key].trace_add("write", lambda *args: self.update_colour_param_run1_run2())
                entry = tk.Entry(main_frame, textvariable=self.entry_vars[key], width=10)
                entry.insert(0, str(info["value"]))
            entry.grid(row=row, column=1 + coloff, padx=10, pady=5)
            self.entries[key] = entry  # dictionary to hold entry widgets
            # Entry field for run2
            key2 = "run2_" + key[5:]
            self.entry_vars[key2] = tk.StringVar()
            if key[5:] in ["cleantype", "fsmode", "psfmodel"]:
                entry = ttk.Combobox(
                    main_frame,
                    textvariable=self.entry_vars[key2],
                    width=8,
                    state="readonly",
                    values=info["valid_values"],
                    style="Normal.TCombobox",
                )
                self.entry_vars[key2].set(str(self.param_dict[key2]["value"]))
                entry.bind("<<ComboboxSelected>>", lambda e: self.update_colour_param_run1_run2())
            else:
                self.entry_vars[key2].trace_add("write", lambda *args: self.update_colour_param_run1_run2())
                entry = tk.Entry(main_frame, textvariable=self.entry_vars[key2], width=10)
                entry.insert(0, str(self.param_dict[key2]["value"]))
            entry.grid(row=row, column=2 + coloff, padx=10, pady=5)
            self.entries["run2_" + key[5:]] = entry  # dictionary to hold entry widgets
            # Type label
            infotext = info["type"].__name__
            if infotext == "int":
                if "intmode" in info:
                    if info["intmode"] == "odd":
                        infotext += ", odd"
                    elif info["intmode"] == "even":
                        infotext += ", even"
            type_label = tk.Label(main_frame, text=f"({infotext})", fg="gray", anchor="w", width=10)
            type_label.grid(row=row, column=3 + coloff, sticky="w", pady=5)
            row_subtable += 1
            if row_subtable == max_num_params_in_columns:
                coloff = 5
                row -= max_num_params_in_columns
            row += 1

        # Auxiliary images
        label = tk.Label(main_frame, text="inbkg:", anchor="e", width=15)
        label.grid(row=row, column=coloff, sticky="w", pady=5)
        if self.inbkg is None:
            self.filename_inbkg = tk.StringVar(value="None")
        else:
            self.filename_inbkg = tk.StringVar(value=str(Path(self.inbkg).name + f"[{self.extnum_inbkg}]"))
        file_inbkg_label = tk.Label(
            main_frame, textvariable=self.filename_inbkg, fg="blue", bg="white", cursor="hand2", anchor="w", width=40
        )
        file_inbkg_label.grid(row=row, column=coloff + 1, columnspan=4, sticky="w", padx=10, pady=5)
        file_inbkg_label.bind("<Button-1>", lambda e: self.define_inbkg())
        row += 1

        label = tk.Label(main_frame, text="invar:", anchor="e", width=15)
        label.grid(row=row, column=coloff, sticky="w", pady=5)
        if self.invar is None:
            self.filename_invar = tk.StringVar(value="None")
        else:
            self.filename_invar = tk.StringVar(value=str(Path(self.invar).name + f"[{self.extnum_invar}]"))
        file_invar_label = tk.Label(
            main_frame, textvariable=self.filename_invar, fg="blue", bg="white", cursor="hand2", anchor="w", width=40
        )
        file_invar_label.grid(row=row, column=coloff + 1, columnspan=4, sticky="w", padx=10, pady=5)
        file_invar_label.bind("<Button-1>", lambda e: self.define_invar())
        row += 1

        # Adjust row if odd number of parameters
        if nparams_total % 2 != 0:
            row += nparams_total % 2

        # Vertical separator between splitted table
        separatorv1 = ttk.Separator(main_frame, orient="vertical")
        separatorv1.grid(
            row=row - max_num_params_in_columns, column=4, rowspan=max_num_params_in_columns, sticky="ns", padx=10
        )

        # Separator
        separator1 = ttk.Separator(main_frame, orient="horizontal")
        separator1.grid(row=row, column=0, columnspan=9, sticky="ew", pady=(10, 10))
        row += 1

        # Subtitle for additional parameters
        subtitle_label = tk.Label(main_frame, text="Additional Parameters", font=bold_font)
        subtitle_label.grid(row=row, column=0, columnspan=9, pady=(0, 10))
        row += 1

        # Dilation label and entry
        label = tk.Label(main_frame, text="Dilation:", anchor="e", width=15)
        label.grid(row=row, column=0, sticky="w", pady=5)
        entry = tk.Entry(main_frame, width=10)
        entry.insert(0, str(self.param_dict["dilation"]["value"]))
        entry.grid(row=row, column=1, padx=10, pady=5)
        self.entries["dilation"] = entry
        type_label = tk.Label(
            main_frame, text=f"({self.param_dict['dilation']['type'].__name__})", fg="gray", anchor="w", width=10
        )
        type_label.grid(row=row, column=2, sticky="w", pady=5)

        label = tk.Label(main_frame, text="Border Padding:", anchor="e", width=15)
        label.grid(row=row, column=5, sticky="w", pady=5)
        entry = tk.Entry(main_frame, width=10)
        entry.insert(0, str(self.param_dict["borderpadd"]["value"]))
        entry.grid(row=row, column=6, padx=10, pady=5)
        self.entries["borderpadd"] = entry
        type_label = tk.Label(
            main_frame, text=f"({self.param_dict['borderpadd']['type'].__name__})", fg="gray", anchor="w", width=10
        )
        type_label.grid(row=row, column=7, sticky="w", pady=5)
        row += 1

        # Vertical separator
        separatorv2 = ttk.Separator(main_frame, orient="vertical")
        separatorv2.grid(row=row - 1, column=4, rowspan=1, sticky="ns", padx=10)

        # Separator
        separator2 = ttk.Separator(main_frame, orient="horizontal")
        separator2.grid(row=row, column=0, columnspan=9, sticky="ew", pady=(10, 10))
        row += 1

        # Subtitle for region to be examined
        subtitle_label = tk.Label(main_frame, text="Region to be Examined", font=bold_font)
        subtitle_label.grid(row=row, column=0, columnspan=9, pady=(0, 10))
        row += 1

        # Region to be examined label and entries
        for key, info in self.param_dict.items():
            if key.lower() in ["xmin", "xmax", "ymin", "ymax"]:
                # Parameter name label
                label = tk.Label(main_frame, text=f"{key}:", anchor="e", width=15)
                if key.lower() in ["xmin", "xmax"]:
                    coloff = 0
                else:
                    coloff = 5
                label.grid(row=row, column=coloff, sticky="w", pady=5)
                # Entry field
                entry = tk.Entry(main_frame, width=10)
                entry.insert(0, str(info["value"]))
                entry.grid(row=row, column=coloff + 1, padx=10, pady=5)
                self.entries[key] = entry  # dictionary to hold entry widgets
                # Type label
                dumtext = f"({info['type'].__name__})"
                if key.lower() in ["xmin", "xmax"]:
                    dumtext += f" --> [1, {self.imgshape[1]}]"
                else:
                    dumtext += f" --> [1, {self.imgshape[0]}]"
                type_label = tk.Label(main_frame, text=dumtext, fg="gray", anchor="w", width=15)
                type_label.grid(row=row, column=coloff + 2, sticky="w", pady=5)
                if key.lower() == "xmax":
                    row -= 1
                else:
                    row += 1

        # Vertical separator
        separatorv3 = ttk.Separator(main_frame, orient="vertical")
        separatorv3.grid(row=row - 2, column=4, rowspan=2, sticky="ns", padx=10)

        # Separator
        separator3 = ttk.Separator(main_frame, orient="horizontal")
        separator3.grid(row=row, column=0, columnspan=9, sticky="ew", pady=(10, 10))
        row += 1

        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=9, pady=(5, 0))

        # OK button
        self.ok_button = ttk.Button(button_frame, text="OK", takefocus=True, command=self.on_ok)
        self.ok_button.pack(side="left", padx=5)

        # Cancel button
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.on_cancel)
        self.cancel_button.pack(side="left", padx=5)

        # Reset button
        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.on_reset)
        self.reset_button.pack(side="left", padx=5)

        # Set focus to OK button
        self.ok_button.focus_set()

    def define_inbkg(self):
        """Define the input background image."""
        self.inbkg = filedialog.askopenfilename(
            parent=self.root,
            title="Select FITS file to be used as input background image",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")],
        )
        if not self.inbkg:
            self.inbkg = None
        if isinstance(self.inbkg, str):
            self.inbkg = self.inbkg.strip()

        if self.inbkg in ["", None]:
            self.inbkg = None
            self.filename_inbkg.set("None")
            self.extnum_inbkg = None
            return
        else:
            self.extnum_inbkg = ask_extension_input_image(self.inbkg, self.imgshape)
            if self.extnum_inbkg is None:
                self.inbkg = None
            else:
                self.filename_inbkg.set(str(Path(self.inbkg).name + f"[{self.extnum_inbkg}]"))

    def define_invar(self):
        """Define the input variance image."""
        self.invar = filedialog.askopenfilename(
            parent=self.root,
            title="Select FITS file to be used as input variance image",
            filetypes=[("FITS files", "*.fits *.fit *.fts"), ("All files", "*.*")],
        )
        if not self.invar:
            self.invar = None
        if isinstance(self.invar, str):
            self.invar = self.invar.strip()

        if self.invar in ["", None]:
            self.invar = None
            self.filename_invar.set("None")
            self.extnum_invar = None
            return
        else:
            self.extnum_invar = ask_extension_input_image(self.invar, self.imgshape)
            if self.extnum_invar is None:
                self.invar = None
            else:
                self.filename_invar.set(str(Path(self.invar).name + f"[{self.extnum_invar}]"))

    def on_ok(self):
        """Validate and save the updated values"""
        try:
            updated_dict = {
                "inbkg": {"value": self.inbkg, "type": str},
                "extnum_inbkg": {"value": self.extnum_inbkg, "type": int},
                "invar": {"value": self.invar, "type": str},
                "extnum_invar": {"value": self.extnum_invar, "type": int},
            }

            for key, info in self.param_dict.items():
                if key in ["nruns", "inbkg", "extnum_inbkg", "invar", "extnum_invar"]:
                    continue
                entry_value = self.entries[key].get()
                value_type = info["type"]

                # Convert string to appropriate type
                if value_type == bool:
                    # Handle boolean conversion
                    if entry_value.lower() in ["true", "1", "yes"]:
                        converted_value = True
                    elif entry_value.lower() in ["false", "0", "no"]:
                        converted_value = False
                    else:
                        raise ValueError(f"Invalid boolean value for {key}")
                elif value_type == str:
                    converted_value = entry_value
                    if "valid_values" in info and entry_value not in info["valid_values"]:
                        raise ValueError(f"Invalid value for {key}. Valid values are: {info['valid_values']}")
                else:
                    converted_value = value_type(entry_value)
                    if "positive" in info and info["positive"] and converted_value < 0:
                        raise ValueError(f"Value for {key} must be positive")
                    if "intmode" in info:
                        if info["intmode"] == "odd" and converted_value % 2 == 0:
                            raise ValueError(f"Value for {key} must be an odd integer")
                        elif info["intmode"] == "even" and converted_value % 2 != 0:
                            raise ValueError(f"Value for {key} must be an even integer")

                # Duplicate the parameter info and update only the value
                # (preserving other metadata)
                updated_dict[key] = self.param_dict[key].copy()
                updated_dict[key]["value"] = converted_value

            # Check whether any run1 and run2 parameters differ
            nruns = 1
            for key in self.param_dict.keys():
                if key.startswith("run1_"):
                    parname = key[5:]
                    key2 = "run2_" + parname
                    if updated_dict[key]["value"] != updated_dict[key2]["value"]:
                        nruns = 2
                        print(
                            f"Parameter '{parname}' differs between run1 and run2: "
                            f"{updated_dict[key]['value']} (run1) vs {updated_dict[key2]['value']} (run2)"
                        )

            # Additional validation for region limits
            try:
                if updated_dict["xmax"]["value"] <= updated_dict["xmin"]["value"]:
                    raise ValueError("xmax must be greater than xmin")
                if updated_dict["ymax"]["value"] <= updated_dict["ymin"]["value"]:
                    raise ValueError("ymax must be greater than ymin")
                self.result_dict = updated_dict
                self.result_dict["nruns"] = {"value": nruns, "type": int, "positive": True}
                if nruns not in [1, 2]:
                    raise ValueError("nruns must be 1 or 2")
                self.root.destroy()
            except ValueError as e:
                messagebox.showerror(
                    "Invalid Inputs", "Error in region limits:\n" f"{str(e)}\n\nPlease check your inputs."
                )

        except ValueError as e:
            messagebox.showerror(
                "Invalid Inputs", f"Error converting value for {key}:\n{str(e)}\n\n" "Please check your inputs."
            )

    def on_cancel(self):
        """Close without saving"""
        self.result_dict = None
        self.root.destroy()

    def on_reset(self):
        """Reset all fields to original values"""
        self.param_dict = lacosmic_default_dict.copy()
        self.param_dict["xmin"]["value"] = 1
        self.param_dict["xmax"]["value"] = self.imgshape[1]
        self.param_dict["ymin"]["value"] = 1
        self.param_dict["ymax"]["value"] = self.imgshape[0]
        self.inbkg = None
        self.extnum_inbkg = None
        self.filename_inbkg.set("None")
        self.invar = None
        self.extnum_invar = None
        self.filename_invar.set("None")
        for key, info in self.param_dict.items():
            if key in ["nruns", "inbkg", "extnum_inbkg", "invar", "extnum_invar"]:
                continue
            parname = key[5:]
            if parname in ["cleantype", "fsmode", "psfmodel"]:
                self.entries[key].set(str(info["value"]))
            else:
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, str(info["value"]))

    def get_result(self):
        """Return the updated dictionary"""
        return self.result_dict

    def update_colour_param_run1_run2(self):
        """Update the foreground color of run1 and run2 entries."""
        # Highlight run2 parameter if different from run1
        for key in self.param_dict.keys():
            if key.startswith("run1_"):
                parname = key[5:]
                if key in self.entries and "run2_" + parname in self.entries:
                    if self.entries[key].get() != self.entries["run2_" + parname].get():
                        if parname in ["cleantype", "fsmode", "psfmodel"]:
                            self.entries["run2_" + parname].configure(style="Changed.TCombobox")
                        else:
                            self.entries["run2_" + parname].config(fg="red")
                    else:
                        if parname in ["cleantype", "fsmode", "psfmodel"]:
                            self.entries["run2_" + parname].configure(style="Normal.TCombobox")
                        else:
                            self.entries["run2_" + parname].config(fg="black")
                # Remove the highlight after choosing an option from the dropdown
                # (to see the color change immediately)
                if parname in ["cleantype", "fsmode", "psfmodel"]:
                    if "run_1_" + parname in self.entries:
                        self.entries["run1_" + parname].selection_clear()
                    if "run2_" + parname in self.entries:
                        self.entries["run2_" + parname].selection_clear()
