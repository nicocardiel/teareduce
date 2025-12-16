#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Define the ReviewCosmicRay class."""

import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog

try:
    from maskfill import maskfill
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The 'teareduce.cleanest' module requires the 'ccdproc' and 'maskfill' packages. "
        "Please install teareduce with the 'cleanest' extra dependencies: "
        "`pip install teareduce[cleanest]`."
    ) from e
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from rich import print

from .centerchildparent import center_on_parent
from .definitions import DEFAULT_TK_WINDOW_SIZE_X
from .definitions import DEFAULT_TK_WINDOW_SIZE_Y
from .definitions import MAX_PIXEL_DISTANCE_TO_CR
from .imagedisplay import ImageDisplay
from .interpolation_a import interpolation_a
from .interpolation_x import interpolation_x
from .interpolation_y import interpolation_y
from .trackedbutton import TrackedTkButton

from ..imshow import imshow
from ..sliceregion import SliceRegion2D
from ..zscale import zscale
from ..version import VERSION

import matplotlib

matplotlib.use("TkAgg")


class ReviewCosmicRay(ImageDisplay):
    """Class to review suspected cosmic ray pixels."""

    def __init__(
        self,
        root,
        root_width,
        root_height,
        data,
        auxdata,
        cleandata_lacosmic,
        cr_labels,
        num_features,
        first_cr_index=1,
        single_cr=False,
        last_dilation=None,
        last_npoints=None,
        last_degree=None,
        last_maskfill_size=None,
        last_maskfill_operator=None,
        last_maskfill_smooth=None,
        last_maskfill_verbose=None,
    ):
        """Initialize the review window.

        Parameters
        ----------
        root : tk.Toplevel
            The parent Tkinter root window.
        root_width : int
            The width of the root window. The review window is scaled accordingly.
        root_height : int
            The height of the root window. The review window is scaled accordingly.
        data : 2D numpy array
            The original image data.
        auxdata : 2D numpy array or None
            The auxiliary image data.
        cleandata_lacosmic: 2D numpy array or None
            The cleaned image data from L.A.Cosmic.
        cr_labels : 2D numpy array
            Labels of connected cosmic ray pixel groups.
        num_features : int
            Number of connected cosmic ray pixel groups.
        first_cr_index : int, optional
            The index of the first cosmic ray to review (default is 1).
        single_cr : bool, optional
            Whether to review a single cosmic ray (default is False).
            If True, the review window will close after reviewing the
            selected first cosmic ray.
        last_dilation : int or None, optional
            The last used dilation parameter employed after L.A.Cosmic
            detection. If > 0, the replacement by the L.A.Cosmic cleaned
            data will not be allowed.
        last_npoints : int or None, optional
            The last used number of points parameter for interpolation.
        last_degree : int or None, optional
            The last used degree parameter for interpolation.
        last_maskfill_size : int or None, optional
            The last used maskfill size parameter.
        last_maskfill_operator : str or None, optional
            The last used maskfill operator parameter.
        last_maskfill_smooth : bool or None, optional
            The last used maskfill smooth parameter.
        last_maskfill_verbose : bool or None, optional
            The last used maskfill verbose parameter.

        Methods
        -------
        create_widgets()
            Create the GUI widgets for the review window.
        update_display(cleaned=False)
            Update the display to show the current cosmic ray.
        set_ndeg()
            Set the Npoints and Degree parameters for interpolation.
        interp_x()
            Perform X-interpolation for the current cosmic ray.
        interp_y()
            Perform Y-interpolation for the current cosmic ray.
        interp_a(method)
            Perform interpolation using the specified method for the current cosmic ray.
        use_lacosmic()
            Replace cosmic ray pixels with L.A.Cosmic cleaned data.
        use_auxdata()
            Replace cosmic ray pixels with auxiliary data.
        remove_crosses()
            Remove all pixels of the current cosmic ray from the review.
        restore_cr()
            Restore all pixels of the current cosmic ray to their original values.
        continue_cr()
            Move to the next cosmic ray for review.
        exit_review()
            Close the review window.
        on_key(event)
            Handle key press events for shortcuts.
        on_click(event)
            Handle mouse click events to mark/unmark pixels as cosmic rays.

        Attributes
        ----------
        root : tk.Toplevel
            The parent Tkinter root window.
        factor_width : float
            The scaling factor for the width of the review window.
        factor_height : float
            The scaling factor for the height of the review window.
        data : 2D numpy array
            The original image data.
        auxdata : 2D numpy array or None
            The auxiliary image data.
        cleandata_lacosmic: 2D numpy array or None
            The cleaned image data from L.A.Cosmic.
        cr_labels : 2D numpy array
            Labels of connected cosmic ray pixel groups.
        num_features : int
            Number of connected cosmic ray pixel groups.
        num_cr_cleaned : int
            Number of cosmic rays cleaned during the review.
        mask_fixed : 2D numpy array
            Mask of pixels fixed during the review.
        first_plot : bool
            Flag to indicate if it's the first plot.
        degree : int
            Degree parameter for interpolation.
        npoints : int
            Number of points parameter for interpolation.
        maskfill_size : int
            The last used maskfill size parameter.
        maskfill_operator : str
            The last used maskfill operator parameter.
        maskfill_smooth : bool
            The last used maskfill smooth parameter.
        maskfill_verbose : bool
            The last used maskfill verbose parameter.
        last_dilation : int or None
            The last used dilation parameter employed after L.A.Cosmic
            detection.
        """
        self.root = root
        self.root.title(f"Review Cosmic Rays (TEA version {VERSION})")
        self.factor_width = root_width / DEFAULT_TK_WINDOW_SIZE_X
        self.factor_height = root_height / DEFAULT_TK_WINDOW_SIZE_Y
        self.auxdata = auxdata
        if self.auxdata is not None:
            # self.root.geometry("1000x700+100+100")  # This does not work in Fedora
            window_width = int(1000 * self.factor_width + 0.5)
            window_height = int(700 * self.factor_height + 0.5)
            self.root.minsize(window_width, window_height)
        else:
            # self.root.geometry("900x700+100+100")  # This does not work in Fedora
            window_width = int(900 * self.factor_width + 0.5)
            window_height = int(700 * self.factor_height + 0.5)
            self.root.minsize(window_width, window_height)
        self.root.update_idletasks()
        self.root.geometry("+100+100")
        self.data = data
        self.cleandata_lacosmic = cleandata_lacosmic
        self.data_original = data.copy()
        self.cr_labels = cr_labels
        self.num_features = num_features
        self.num_cr_cleaned = 0
        self.mask_fixed = np.zeros(self.data.shape, dtype=bool)  # Mask of pixels fixed during review
        self.first_plot = True
        self.degree = last_degree if last_degree is not None else 1
        self.npoints = last_npoints if last_npoints is not None else 2
        self.maskfill_size = last_maskfill_size if last_maskfill_size is not None else 3
        self.maskfill_operator = last_maskfill_operator if last_maskfill_operator is not None else "median"
        self.maskfill_smooth = last_maskfill_smooth if last_maskfill_smooth is not None else True
        self.maskfill_verbose = last_maskfill_verbose if last_maskfill_verbose is not None else False
        self.last_dilation = last_dilation
        # Make a copy of the original labels to allow pixel re-marking
        self.cr_labels_original = self.cr_labels.copy()
        sdum = str(np.sum(self.cr_labels > 0))
        print(f"Number of cosmic ray pixels detected..: {sdum}")
        print(f"Number of cosmic rays (grouped pixels): {self.num_features:>{len(sdum)}}")
        if self.num_features == 0:
            print("No CR hits found!")
        else:
            self.cr_index = first_cr_index
            self.single_cr = single_cr
            self.create_widgets()
            center_on_parent(child=self.root, parent=self.root.master, offset_x=50, offset_y=50)

    def create_widgets(self):
        """Create the GUI widgets for the review window."""
        # Define instance of TrackedTkButton, that facilitates to show help information
        # for each button displayed in the current application window.
        tkbutton = TrackedTkButton(self.root)

        # Row 1 of buttons
        self.button_frame1 = tk.Frame(self.root)
        self.button_frame1.pack(pady=5)
        self.ndeg_label = tkbutton.new(
            self.button_frame1,
            text=f"Npoints={self.npoints}, Degree={self.degree}",
            command=self.set_ndeg,
            help_text="Set the Npoints and Degree parameters for interpolation.",
            alttext="Npoints=?, Degree=?",
        )
        self.ndeg_label.pack(side=tk.LEFT, padx=5)
        self.maskfill_button = tkbutton.new(
            self.button_frame1,
            text="Maskfill params.",
            command=self.set_maskfill_params,
            help_text="Set the parameters for the maskfill method.",
        )
        self.maskfill_button.pack(side=tk.LEFT, padx=5)
        self.remove_crosses_button = tkbutton.new(
            self.button_frame1,
            text="remove all x's",
            command=self.remove_crosses,
            help_text="Remove all cross marks from the image.",
        )
        self.remove_crosses_button.pack(side=tk.LEFT, padx=5)
        self.restore_cr_button = tkbutton.new(
            self.button_frame1,
            text="restore CR",
            command=self.restore_cr,
            help_text="Restore current cosmic ray pixels to their original values.",
        )
        self.restore_cr_button.pack(side=tk.LEFT, padx=5)
        self.restore_cr_button.config(state=tk.DISABLED)
        self.next_button = tkbutton.new(
            self.button_frame1,
            text="[c]ontinue",
            command=self.continue_cr,
            help_text="Continue to the next cosmic ray.",
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.exit_button = tkbutton.new(
            self.button_frame1,
            text="[e]xit review",
            command=self.exit_review,
            help_text="Exit the cosmic ray review process.",
        )
        self.exit_button.pack(side=tk.LEFT, padx=5)

        # Row 2 of buttons
        self.button_frame2 = tk.Frame(self.root)
        self.button_frame2.pack(pady=5)
        self.interp_x_button = tkbutton.new(
            self.button_frame2,
            text="[x] interp.",
            command=self.interp_x,
            help_text="Perform X-interpolation for the current cosmic ray.",
        )
        self.interp_x_button.pack(side=tk.LEFT, padx=5)
        self.interp_y_button = tkbutton.new(
            self.button_frame2,
            text="[y] interp.",
            command=self.interp_y,
            help_text="Perform Y-interpolation for the current cosmic ray.",
        )
        self.interp_y_button.pack(side=tk.LEFT, padx=5)
        # it is important to use lambda here to pass the method argument correctly
        # (avoiding the execution of the function at button creation time, which would happen
        # if we didn't use lambda; in that case, the function would be called immediately and
        # its return value (None) would be assigned to the command parameter; furthermore,
        # the function is trying to deactivate the buttons before they are created, which
        # would lead to an error; in addition, since I have two buttons calling the same function
        # with different arguments, using lambda allows to differentiate them)
        self.interp_s_button = tkbutton.new(
            self.button_frame2,
            text="[s]urface interp.",
            command=lambda: self.interp_a("surface"),
            help_text="Perform surface interpolation for the current cosmic ray.",
        )
        self.interp_s_button.pack(side=tk.LEFT, padx=5)
        self.interp_d_button = tkbutton.new(
            self.button_frame2,
            text="me[d]ian",
            command=lambda: self.interp_a("median"),
            help_text="Perform median interpolation for the current cosmic ray.",
        )
        self.interp_d_button.pack(side=tk.LEFT, padx=5)
        self.interp_m_button = tkbutton.new(
            self.button_frame2,
            text="[m]ean",
            command=lambda: self.interp_a("mean"),
            help_text="Perform mean interpolation for the current cosmic ray.",
        )
        self.interp_m_button.pack(side=tk.LEFT, padx=5)
        self.interp_l_button = tkbutton.new(
            self.button_frame2,
            text="[l]acosmic",
            command=self.use_lacosmic,
            help_text="Use L.A.Cosmic interpolation for the current cosmic ray.",
        )
        self.interp_l_button.pack(side=tk.LEFT, padx=5)
        if self.last_dilation is not None and self.last_dilation > 0:
            self.interp_l_button.config(state=tk.DISABLED)
        if self.cleandata_lacosmic is None:
            self.interp_l_button.config(state=tk.DISABLED)
        self.interp_maskfill_button = tkbutton.new(
            self.button_frame2,
            text="mask[f]ill",
            command=self.use_maskfill,
            help_text="Perform maskfill interpolation for the current cosmic ray.",
        )
        self.interp_maskfill_button.pack(side=tk.LEFT, padx=5)
        self.interp_aux_button = tkbutton.new(
            self.button_frame2,
            text="[a]ux. data",
            command=self.use_auxdata,
            help_text="Use auxiliary data for interpolation of the current cosmic ray.",
        )
        self.interp_aux_button.pack(side=tk.LEFT, padx=5)
        if self.auxdata is None:
            self.interp_aux_button.config(state=tk.DISABLED)

        # Row 3 of buttons
        self.button_frame3 = tk.Frame(self.root)
        self.button_frame3.pack(pady=5)
        vmin, vmax = zscale(self.data)
        self.vmin_button = tkbutton.new(
            self.button_frame3,
            text=f"vmin: {vmin:.2f}",
            command=self.set_vmin,
            help_text="Set the minimum value for the display scale.",
            alttext="vmin: ??",
        )
        self.vmin_button.pack(side=tk.LEFT, padx=5)
        self.vmax_button = tkbutton.new(
            self.button_frame3,
            text=f"vmax: {vmax:.2f}",
            command=self.set_vmax,
            help_text="Set the maximum value for the display scale.",
            alttext="vmax: ??",
        )
        self.vmax_button.pack(side=tk.LEFT, padx=5)
        self.set_minmax_button = tkbutton.new(
            self.button_frame3,
            text="minmax [,]",
            command=self.set_minmax,
            help_text="Set the display scale to the minimum and maximum data values.",
        )
        self.set_minmax_button.pack(side=tk.LEFT, padx=5)
        self.set_zscale_button = tkbutton.new(
            self.button_frame3,
            text="zscale [/]",
            command=self.set_zscale,
            help_text="Set the display scale using zscale.",
        )
        self.set_zscale_button.pack(side=tk.LEFT, padx=5)
        self.help_button = tkbutton.new(
            self.button_frame3,
            text="Help",
            command=tkbutton.show_help,
            help_text="Show help information for all buttons.",
        )
        self.help_button.pack(side=tk.LEFT, padx=5)

        # Figure
        if self.auxdata is not None:
            self.fig, (self.ax, self.ax_aux) = plt.subplots(
                ncols=2, figsize=(11 * self.factor_width, 5.5 * self.factor_height), constrained_layout=True
            )
        else:
            self.fig, self.ax = plt.subplots(
                figsize=(9 * self.factor_width, 5.5 * self.factor_height), constrained_layout=True
            )
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(padx=5, pady=5)
        # The next two instructions prevent a segmentation fault when pressing "q"
        self.canvas.mpl_disconnect(self.canvas.mpl_connect("key_press_event", key_press_handler))
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas_widget = self.canvas.get_tk_widget()
        # self.canvas_widget.pack(fill=tk.BOTH, expand=True)  # This does not work in Fedora
        self.canvas_widget.pack(expand=True)

        # Matplotlib toolbar
        self.toolbar_frame = tk.Frame(self.root)
        self.toolbar_frame.pack(fill=tk.X, expand=False, pady=5)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.update_display()

    def update_display(self, cleaned=False):
        """Update the display to show the current cosmic ray.

        Parameters
        ----------
        cleaned : bool, optional
            Whether the cosmic ray has been cleaned (default is False).
            If True, the cosmic ray pixels will be marked differently.
        """
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        ycr_list_original, xcr_list_original = np.where(self.cr_labels_original == self.cr_index)
        if self.first_plot:
            print(
                f"Cosmic ray {self.cr_index}: "
                f"Number of pixels = {len(xcr_list)}, "
                f"Centroid = ({np.mean(xcr_list)+1:.2f}, {np.mean(ycr_list)+1:.2f})"
            )
        # Use original positions to define the region to display in order
        # to avoid image shifts when some pixels are unmarked or new ones are marked
        i0 = int(np.mean(ycr_list_original) + 0.5)
        j0 = int(np.mean(xcr_list_original) + 0.5)
        max_distance_from_center = np.max(
            [np.max(np.abs(ycr_list_original - i0)), np.max(np.abs(xcr_list_original - j0))]
        )
        semiwidth = int(np.max([max_distance_from_center, MAX_PIXEL_DISTANCE_TO_CR]))
        jmin = j0 - semiwidth if j0 - semiwidth >= 0 else 0
        jmax = j0 + semiwidth if j0 + semiwidth < self.data.shape[1] else self.data.shape[1] - 1
        imin = i0 - semiwidth if i0 - semiwidth >= 0 else 0
        imax = i0 + semiwidth if i0 + semiwidth < self.data.shape[0] else self.data.shape[0] - 1
        # Force the region to be of size (2*semiwidth + 1) x (2*semiwidth + 1)
        if jmin == 0:
            jmax = np.min([2 * semiwidth, self.data.shape[1] - 1])
        elif jmax == self.data.shape[1] - 1:
            jmin = np.max([0, self.data.shape[1] - 1 - 2 * semiwidth])
        if imin == 0:
            imax = np.min([2 * semiwidth, self.data.shape[0] - 1])
        elif imax == self.data.shape[0] - 1:
            imin = np.max([0, self.data.shape[0] - 1 - 2 * semiwidth])
        self.region = SliceRegion2D(f"[{jmin+1}:{jmax+1}, {imin+1}:{imax+1}]", mode="fits").python
        self.ax.clear()
        vmin = self.get_vmin()
        vmax = self.get_vmax()
        xlabel = "X pixel (from 1 to NAXIS1)"
        ylabel = "Y pixel (from 1 to NAXIS2)"
        self.image, _, _ = imshow(
            self.fig,
            self.ax,
            self.data[self.region],
            colorbar=False,
            xlabel=xlabel,
            ylabel=ylabel,
            vmin=vmin,
            vmax=vmax,
        )
        self.image.set_extent([jmin + 0.5, jmax + 1.5, imin + 0.5, imax + 1.5])
        if self.auxdata is not None:
            self.ax_aux.clear()
            self.image_aux, _, _ = imshow(
                self.fig,
                self.ax_aux,
                self.auxdata[self.region],
                colorbar=False,
                xlabel=xlabel,
                ylabel=ylabel,
                vmin=vmin,
                vmax=vmax,
            )
            self.image_aux.set_extent([jmin + 0.5, jmax + 1.5, imin + 0.5, imax + 1.5])
            self.ax_aux.set_title("Auxiliary data")
        # Overplot cosmic ray pixels
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        for xcr, ycr in zip(xcr_list, ycr_list):
            xcr += 1  # from index to pixel
            ycr += 1  # from index to pixel
            if cleaned:
                self.ax.plot(xcr, ycr, "C1o", markersize=4)
            else:
                self.ax.plot([xcr - 0.5, xcr + 0.5], [ycr + 0.5, ycr - 0.5], "r-")
                self.ax.plot([xcr - 0.5, xcr + 0.5], [ycr - 0.5, ycr + 0.5], "r-")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_title(f"Cosmic ray #{self.cr_index}/{self.num_features}")
        if self.first_plot:
            self.first_plot = False
        self.canvas.draw_idle()

    def set_ndeg(self):
        """Set the number of points and degree for interpolation."""
        new_npoints = simpledialog.askinteger("Set Npoints", "Enter Npoints:", initialvalue=self.npoints, minvalue=1)
        if new_npoints is None:
            return
        new_degree = simpledialog.askinteger(
            "Set degree", "Enter Degree (min=0):", initialvalue=self.degree, minvalue=0
        )
        if new_degree is None:
            return
        self.degree = new_degree
        self.npoints = new_npoints
        self.ndeg_label.config(text=f"Npoints={self.npoints}, Degree={self.degree}")

    def set_maskfill_params(self):
        """Set the maskfill parameters."""
        new_size = simpledialog.askinteger(
            "Set Maskfill Size", "Enter Maskfill Size (odd integer >=1):", initialvalue=self.maskfill_size, minvalue=1
        )
        if new_size is None:
            return
        if new_size % 2 == 0:
            messagebox.showerror("Input Error", "Maskfill size must be an odd integer.")
            return
        new_operator = simpledialog.askstring(
            "Set Maskfill Operator",
            "Enter Maskfill Operator ('median' or 'mean'):",
            initialvalue=self.maskfill_operator,
        )
        if new_operator is None:
            return
        if new_operator not in ["median", "mean"]:
            messagebox.showerror("Input Error", "Maskfill operator must be 'median' or 'mean'.")
            return
        smooth_str = simpledialog.askstring(
            "Set Maskfill Smooth",
            "Enter Maskfill Smooth ('True' or 'False'):",
            initialvalue=str(self.maskfill_smooth),
        )
        if smooth_str is None:
            return
        smooth_str = smooth_str.strip().lower()
        if smooth_str == "true":
            new_smooth = True
        elif smooth_str == "false":
            new_smooth = False
        else:
            messagebox.showerror("Input Error", "Maskfill Smooth must be 'True' or 'False'.")
            return
        verbose_str = simpledialog.askstring(
            "Set Maskfill Verbose",
            "Enter Maskfill Verbose ('True' or 'False'):",
            initialvalue=str(self.maskfill_verbose),
        )
        if verbose_str is None:
            return
        verbose_str = verbose_str.strip().lower()
        if verbose_str == "true":
            new_verbose = True
        elif verbose_str == "false":
            new_verbose = False
        else:
            messagebox.showerror("Input Error", "Maskfill Verbose must be 'True' or 'False'.")
            return
        self.maskfill_size = new_size
        self.maskfill_operator = new_operator
        self.maskfill_smooth = new_smooth
        self.maskfill_verbose = new_verbose

    def set_buttons_after_cleaning_cr(self):
        """Set the state of buttons after cleaning a cosmic ray."""
        self.disable_interpolation_buttons()
        self.restore_cr_button.config(state=tk.NORMAL)
        self.remove_crosses_button.config(state=tk.DISABLED)

    def interp_x(self):
        """Perform x-direction interpolation to clean a cosmic ray."""
        if 2 * self.npoints <= self.degree:
            messagebox.showerror("Input Error", "2*Npoints must be greater than Degree for x interpolation.")
            return
        print(f"X-interpolation of cosmic ray {self.cr_index}")
        print(f"Interpolation parameters: Npoints={self.npoints}, Degree={self.degree}")
        interpolation_performed, xfit_all, yfit_all = interpolation_x(
            data=self.data,
            mask_fixed=self.mask_fixed,
            cr_labels=self.cr_labels,
            cr_index=self.cr_index,
            npoints=self.npoints,
            degree=self.degree,
        )
        if interpolation_performed:
            self.num_cr_cleaned += 1
        self.set_buttons_after_cleaning_cr()
        self.update_display(cleaned=interpolation_performed)
        if len(xfit_all) > 0:
            self.ax.plot(np.array(xfit_all) + 1, np.array(yfit_all) + 1, "mo", markersize=4)  # +1: from index to pixel
            self.canvas.draw_idle()

    def interp_y(self):
        """Perform y-direction interpolation to clean a cosmic ray."""
        if 2 * self.npoints <= self.degree:
            messagebox.showerror("Input Error", "2*Npoints must be greater than Degree for y interpolation.")
            return
        print(f"Y-interpolation of cosmic ray {self.cr_index}")
        print(f"Interpolation parameters: Npoints={self.npoints}, Degree={self.degree}")
        interpolation_performed, xfit_all, yfit_all = interpolation_y(
            data=self.data,
            mask_fixed=self.mask_fixed,
            cr_labels=self.cr_labels,
            cr_index=self.cr_index,
            npoints=self.npoints,
            degree=self.degree,
        )
        if interpolation_performed:
            self.num_cr_cleaned += 1
        self.set_buttons_after_cleaning_cr()
        self.update_display(cleaned=interpolation_performed)
        if len(xfit_all) > 0:
            self.ax.plot(np.array(xfit_all) + 1, np.array(yfit_all) + 1, "mo", markersize=4)  # +1: from index to pixel
            self.canvas.draw_idle()

    def interp_a(self, method):
        """Perform interpolation using the specified method to clean a cosmic ray.

        Parameters
        ----------
        method : str
            The interpolation method to use ('surface', 'median' or 'mean').
        """
        print(f"{method} interpolation of cosmic ray {self.cr_index}")
        print(f'Interpolation parameters: Npoints={self.npoints}, method="{method}"')
        interpolation_performed, xfit_all, yfit_all = interpolation_a(
            data=self.data,
            mask_fixed=self.mask_fixed,
            cr_labels=self.cr_labels,
            cr_index=self.cr_index,
            npoints=self.npoints,
            method=method,
        )
        if interpolation_performed:
            self.num_cr_cleaned += 1
        self.set_buttons_after_cleaning_cr()
        self.update_display(cleaned=interpolation_performed)
        if len(xfit_all) > 0:
            self.ax.plot(np.array(xfit_all) + 1, np.array(yfit_all) + 1, "mo", markersize=4)  # +1: from index to pixel
            self.canvas.draw_idle()

    def use_lacosmic(self):
        """Use L.A.Cosmic cleaned data to clean a cosmic ray."""
        if self.cleandata_lacosmic is None:
            print("L.A.Cosmic cleaned data not available.")
            return
        print(f"L.A.Cosmic interpolation of cosmic ray {self.cr_index}")
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        for iy, ix in zip(ycr_list, xcr_list):
            self.data[iy, ix] = self.cleandata_lacosmic[iy, ix]
            self.mask_fixed[iy, ix] = True
        self.num_cr_cleaned += 1
        self.set_buttons_after_cleaning_cr()
        self.update_display(cleaned=True)

    def use_maskfill(self):
        """Use maskfill cleaned data to clean a cosmic ray."""
        print(f"Maskfill interpolation of cosmic ray {self.cr_index}")
        print(
            f"Maskfill parameters: size={self.maskfill_size}, "
            f"operator={self.maskfill_operator}, "
            f"smooth={self.maskfill_smooth}, "
            f"verbose={self.maskfill_verbose}"
        )
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        mask = np.zeros(self.data.shape, dtype=bool)
        for iy, ix in zip(ycr_list, xcr_list):
            mask[iy, ix] = True
        smoothed_output, _ = maskfill(
            input_image=self.data,
            mask=mask,
            size=self.maskfill_size,
            operator=self.maskfill_operator,
            smooth=self.maskfill_smooth,
            verbose=self.maskfill_verbose,
        )
        for iy, ix in zip(ycr_list, xcr_list):
            self.data[iy, ix] = smoothed_output[iy, ix]
            self.mask_fixed[iy, ix] = True
        self.num_cr_cleaned += 1
        self.set_buttons_after_cleaning_cr()
        self.update_display(cleaned=True)

    def use_auxdata(self):
        """Use auxiliary data to clean a cosmic ray."""
        if self.auxdata is None:
            print("Auxiliary data not available.")
            return
        print(f"Auxiliary data interpolation of cosmic ray {self.cr_index}")
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        for iy, ix in zip(ycr_list, xcr_list):
            self.data[iy, ix] = self.auxdata[iy, ix]
            self.mask_fixed[iy, ix] = True
        self.num_cr_cleaned += 1
        self.set_buttons_after_cleaning_cr()
        self.update_display(cleaned=True)

    def remove_crosses(self):
        """Remove all pixels of the current cosmic ray from the review."""
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        for iy, ix in zip(ycr_list, xcr_list):
            self.cr_labels[iy, ix] = 0
        print(f"Removed all pixels of cosmic ray {self.cr_index}")
        self.disable_interpolation_buttons()
        self.update_display()

    def restore_cr(self):
        """Restore all pixels of the current cosmic ray to their original values."""
        ycr_list, xcr_list = np.where(self.cr_labels == self.cr_index)
        if len(xcr_list) == 0:
            print(f"No pixels to restore for cosmic ray {self.cr_index}")
            return
        for iy, ix in zip(ycr_list, xcr_list):
            self.data[iy, ix] = self.data_original[iy, ix]
            self.mask_fixed[iy, ix] = False
        print(f"Restored all pixels of cosmic ray {self.cr_index}")
        self.num_cr_cleaned -= 1
        self.enable_interpolation_buttons()
        self.remove_crosses_button.config(state=tk.NORMAL)
        self.restore_cr_button.config(state=tk.DISABLED)
        self.update_display()

    def continue_cr(self):
        """Move to the next cosmic ray for review."""
        if self.single_cr:
            self.exit_review()
            return  # important: do not remove (to avoid errors)
        self.cr_index += 1
        if self.cr_index > self.num_features:
            self.exit_review()
            return  # important: do not remove (to avoid errors)
        self.first_plot = True
        self.restore_cr_button.config(state=tk.DISABLED)
        self.enable_interpolation_buttons()
        self.remove_crosses_button.config(state=tk.NORMAL)
        self.update_display()

    def exit_review(self):
        """Close the review window."""
        self.root.destroy()

    def on_key(self, event):
        """Handle key press events."""
        if event.key == "q":
            pass  # Ignore the "q" key to prevent closing the window
        elif event.key == "r":
            if self.restore_cr_button.cget("state") != "disabled":
                self.restore_cr()
        elif event.key == "x":
            if self.interp_x_button.cget("state") != "disabled":
                self.interp_x()
        elif event.key == "y":
            if self.interp_y_button.cget("state") != "disabled":
                self.interp_y()
        elif event.key == "s":
            if self.interp_s_button.cget("state") != "disabled":
                self.interp_a("surface")
        elif event.key == "d":
            if self.interp_d_button.cget("state") != "disabled":
                self.interp_a("median")
        elif event.key == "m":
            if self.interp_m_button.cget("state") != "disabled":
                self.interp_a("mean")
        elif event.key == "l":
            if self.interp_l_button.cget("state") != "disabled":
                self.use_lacosmic()
        elif event.key == "k":
            if self.interp_maskfill_button.cget("state") != "disabled":
                self.use_maskfill()
        elif event.key == "a":
            if self.interp_aux_button.cget("state") != "disabled":
                self.use_auxdata()
        elif event.key == "right" or event.key == "c":
            self.continue_cr()
        elif event.key == ",":
            self.set_minmax()
        elif event.key == "/":
            self.set_zscale()
        elif event.key == "e":
            self.exit_review()
            return  # important: do not remove (to avoid errors)
        else:
            print(f"Key pressed: {event.key}")

    def on_click(self, event):
        """Handle mouse click events on the image."""
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            ix = int(x + 0.5) - 1  # from pixel to index
            iy = int(y + 0.5) - 1  # from pixel to index
            if int(self.cr_labels[iy, ix]) == self.cr_index:
                self.cr_labels[iy, ix] = 0
                print(f"Pixel ({ix+1}, {iy+1}) unmarked as cosmic ray.")
            else:
                self.cr_labels[iy, ix] = self.cr_index
                print(f"Pixel ({ix+1}, {iy+1}), with signal {self.data[iy, ix]}, marked as cosmic ray.")
            xcr_list, ycr_list = np.where(self.cr_labels == self.cr_index)
            if len(xcr_list) == 0:
                self.disable_interpolation_buttons()
                self.remove_crosses_button.config(state=tk.DISABLED)
            else:
                self.enable_interpolation_buttons()
                self.remove_crosses_button.config(state=tk.NORMAL)
            # Update the display to reflect the change
            self.update_display()

    def disable_interpolation_buttons(self):
        """Disable all interpolation buttons."""
        self.interp_x_button.config(state=tk.DISABLED)
        self.interp_y_button.config(state=tk.DISABLED)
        self.interp_s_button.config(state=tk.DISABLED)
        self.interp_d_button.config(state=tk.DISABLED)
        self.interp_m_button.config(state=tk.DISABLED)
        self.interp_l_button.config(state=tk.DISABLED)
        self.interp_maskfill_button.config(state=tk.DISABLED)
        self.interp_aux_button.config(state=tk.DISABLED)

    def enable_interpolation_buttons(self):
        """Enable all interpolation buttons."""
        self.interp_x_button.config(state=tk.NORMAL)
        self.interp_y_button.config(state=tk.NORMAL)
        self.interp_s_button.config(state=tk.NORMAL)
        self.interp_d_button.config(state=tk.NORMAL)
        self.interp_m_button.config(state=tk.NORMAL)
        if self.cleandata_lacosmic is not None:
            if self.last_dilation is None or self.last_dilation == 0:
                self.interp_l_button.config(state=tk.NORMAL)
        self.interp_maskfill_button.config(state=tk.NORMAL)
        if self.auxdata is not None:
            self.interp_aux_button.config(state=tk.NORMAL)
