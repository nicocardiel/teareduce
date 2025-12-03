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
    def __init__(self, parent, iterable=None, total=None, desc="Processing"):
        self.parent = parent
        self.iterable = iterable
        self.total = total if total is not None else (len(iterable) if iterable is not None else 100)
        self.current = 0
        self.start_time = time.time()
        self.window = None
        self.desc = desc

    def __enter__(self):
        # Create the modal window when entering context
        self.window = tk.Toplevel(self.parent)
        self.window.title("Progress")
        self.window.geometry("500x150")

        # Make it modal
        self.window.transient(self.parent)
        self.window.grab_set()
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)

        # Center on parent
        self._center_on_parent()

        # UI elements
        self.desc_label = tk.Label(self.window, text=self.desc, font=('Arial', 10, 'bold'))
        self.desc_label.pack(pady=5)

        self.progress = ttk.Progressbar(self.window, length=400, mode='determinate', maximum=self.total)
        self.progress.pack(pady=10)

        self.status_label = tk.Label(self.window, text=f"0/{self.total} (0.0%)")
        self.status_label.pack(pady=2)

        self.time_label = tk.Label(self.window, text="Elapsed: 0s | ETA: --")
        self.time_label.pack(pady=2)

        self.window.update()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.window:
            elapsed = time.time() - self.start_time
            elapsed_str = self._format_time(elapsed)
            self.time_label.config(text=f"Total time: {elapsed_str}")
            self.window.after(1000, self._destroy)
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
        self.progress['value'] = self.current
        percentage = (self.current / self.total) * 100

        elapsed = time.time() - self.start_time

        if self.current > 0:
            rate = self.current / elapsed
            remaining = self.total - self.current
            eta_seconds = remaining / rate if rate > 0 else 0

            elapsed_str = self._format_time(elapsed)
            eta_str = self._format_time(eta_seconds)
            rate_str = f"{rate:.2f} it/s" if rate >= 1 else f"{1/rate:.2f} s/it"

            self.status_label.config(text=f"{self.current}/{self.total} ({percentage:.1f}%) | {rate_str}")
            self.time_label.config(text=f"Elapsed: {elapsed_str} | ETA: {eta_str}")

        self.window.update_idletasks()
        self.window.update()

    def _format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _destroy(self):
        self.window.grab_release()
        self.window.destroy()
