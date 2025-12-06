#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Module defining a progress bar widget for Tkinter."""

import tkinter as tk
from tkinter import ttk
import time


class ModalProgressBar:
    def __init__(
        self, parent, iterable=None, total=None, desc="Processing", completion_msg="Processing completed successfully!"
    ):
        self.parent = parent
        self.iterable = iterable
        self.total = total if total is not None else (len(iterable) if iterable is not None else 100)
        self.current = 0
        self.start_time = time.time()
        self.window = None
        self.desc = desc
        self.completion_msg = completion_msg
        self.continue_clicked = False

    def __enter__(self):
        # Create the modal window when entering context
        self.window = tk.Toplevel(self.parent)
        self.window.title("Progress")

        # Make it modal
        self.window.transient(self.parent)
        self.window.grab_set()
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)

        # Set geometry and force it
        minwinsize_x = 400
        minwinsize_y = 120
        self.window.minsize(minwinsize_x, minwinsize_y)
        self.window.update_idletasks()
        self.window.update()

        # Center on parent
        self._center_on_parent()

        # UI elements
        default_font = tk.font.nametofont("TkDefaultFont")
        bold_font = default_font.copy()
        bold_font.configure(weight="bold", size=default_font.cget("size") + 2)
        self.desc_label = tk.Label(self.window, text=self.desc, font=bold_font)
        self.desc_label.pack(padx=10, pady=5)

        self.progress = ttk.Progressbar(self.window, length=minwinsize_x - 20, mode="determinate", maximum=self.total)
        self.progress.pack(padx=10, pady=10)

        self.status_label = tk.Label(self.window, text=f"0/{self.total} (0.0%)")
        self.status_label.pack(padx=10, pady=2)

        self.time_label = tk.Label(self.window, text="Elapsed: 0s | ETA: --")
        self.time_label.pack(padx=10, pady=2)

        # Continue button (to close the dialog after completion; hidden until done)
        self.continue_button = tk.Button(self.window, text="Continue", command=self._on_continue)
        self.continue_button.pack(padx=10, pady=10)
        self.continue_button.pack_forget()  # Hide initially

        # Force another update
        self.window.update_idletasks()
        self.window.update()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.window:
            # Show completion screen instead of closing
            self._show_completion()

            # Wait for user to click Continue
            self.window.wait_variable(self._continue_var)

            # Now close
            self._destroy()
        return False

    def __iter__(self):
        """Allow iteration like tqdm"""
        if self.iterable is None:
            raise ValueError("No iterable provided for iteration")

        for item in self.iterable:
            yield item
            self.update(1)

    def _center_on_parent(self):
        self.window.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - (self.window.winfo_width() // 2)
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")

    def update(self, n=1):
        self.current += n
        self.progress["value"] = self.current
        percentage = (self.current / self.total) * 100

        elapsed = time.time() - self.start_time

        if self.current > 0:
            rate = self.current / elapsed
            remaining = self.total - self.current
            eta_seconds = remaining / rate if rate > 0 else 0

            elapsed_str = self._format_time(elapsed)
            eta_str = self._format_time(eta_seconds)
            total_str = self._format_time(elapsed + eta_seconds)
            rate_str = f"{rate:.2f} CR/s" if rate >= 1 else f"{1/rate:.2f} s/CR"

            self.status_label.config(text=f"{self.current}/{self.total} ({percentage:.1f}%) | {rate_str}")
            self.time_label.config(text=f"Expected Total: {total_str} | Elapsed: {elapsed_str} | ETA: {eta_str}")
        self.window.update_idletasks()
        self.window.update()

    def _format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f} s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs} s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes} m"

    def _show_completion(self):
        """Transform the window into a completion dialog"""
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)

        # Update title
        self.window.title("Completed")

        # Update description to show completion message
        default_font = tk.font.nametofont("TkDefaultFont")
        bold_font = default_font.copy()
        bold_font.configure(weight="bold", size=default_font.cget("size") + 2)
        self.desc_label.config(text=self.completion_msg, fg="green", font=bold_font)

        # Hide progress bar
        self.progress.pack_forget()

        # Update status to show final stats
        avg_rate = self.current / elapsed if elapsed > 0 else 0
        rate_str = f"{avg_rate:.2f} CR/s" if avg_rate >= 1 else f"{1/avg_rate:.2f} s/CR"
        self.status_label.config(text=f"Processed {self.current} CRs | {rate_str}")

        # Update time label
        self.time_label.config(text=f"Total time: {elapsed_str}")

        # Show the Continue button
        self.continue_button.pack(padx=10, pady=15)

        # Create a variable to track when Continue is clicked
        self._continue_var = tk.BooleanVar(value=False)

        # Re-enable close button to work like Continue
        self.window.protocol("WM_DELETE_WINDOW", self._on_continue)

        self.window.update()
        self._center_on_parent()

    def _on_continue(self):
        """Called when Continue button is clicked"""
        self._continue_var.set(True)

    def _destroy(self):
        self.window.grab_release()
        self.window.destroy()
