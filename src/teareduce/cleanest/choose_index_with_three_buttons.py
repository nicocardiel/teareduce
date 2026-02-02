#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of teareduce
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import tkinter as tk
from tkinter import simpledialog, ttk


class ThreeButtonListDialog(simpledialog.Dialog):
    """
    A modal dialog that shows a list of strings and returns:
      - 1..N : the highlighted item index (1-based) when pressing OK (or double-click / Enter)
      - 0    : when pressing Cancel or closing the window (Esc or window X)
      - N+1  : when pressing New (N = len(items))
    """

    def __init__(self, parent, title, prompt, items):
        self.prompt = prompt
        self.items = list(items)
        self.result = None
        self._force_new = False  # used to bypass selection validation for "New"
        super().__init__(parent, title)

    # ---------- Core layout ----------
    def body(self, master):
        ttk.Label(master, text=self.prompt).grid(row=0, column=0, columnspan=1, sticky="w", padx=8, pady=(8, 4))

        self.listbox = tk.Listbox(master, height=min(12, max(4, len(self.items))), activestyle="dotbox")
        for s in self.items:
            self.listbox.insert(tk.END, s)
        self.listbox.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=4)

        scrollbar = ttk.Scrollbar(master, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=4, padx=(0, 8))

        # Keyboard bindings
        self.listbox.bind("<Double-Button-1>", lambda e: self.ok())  # OK on double-click
        self.bind("<Return>", lambda e: self.ok())  # Enter = OK
        self.bind("<Escape>", lambda e: self.cancel())  # Esc = Cancel (→ 0)

        # Resize behavior
        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(1, weight=1)

        if self.items:
            self.listbox.selection_set(0)
            self.listbox.activate(0)

        return self.listbox  # initial focus on listbox

    # ---------- Buttons (OK, Cancel, New) ----------
    def buttonbox(self):
        box = ttk.Frame(self)

        self.ok_button = ttk.Button(box, text="OK", width=10, command=self.ok)
        self.ok_button.grid(row=0, column=0, padx=5, pady=8)

        self.cancel_button = ttk.Button(box, text="Cancel", width=10, command=self.cancel)
        self.cancel_button.grid(row=0, column=1, padx=5, pady=8)

        self.new_button = ttk.Button(box, text="New", width=10, command=self._on_new)
        self.new_button.grid(row=0, column=2, padx=5, pady=8)

        # Bindings for Return/Escape already set in body()
        self.bind("<Return>", lambda e: self.ok())
        self.bind("<Escape>", lambda e: self.cancel())

        box.pack(side="bottom", fill="x")

    # ---------- Dialog flow ----------
    def validate(self):
        """
        Called by OK to decide whether dialog can close.
        - If 'New' was pressed, allow closing without selection.
        - Otherwise, ensure there is a highlighted item.
        """
        if self._force_new:
            return True
        cur = self.listbox.curselection()
        if not cur:
            self.bell()
            return False
        self._selected_index = cur[0]
        return True

    def apply(self):
        """
        Called after validate(). Set the final result here.
        """
        if self._force_new:
            self.result = len(self.items) + 1  # N+1 (New)
        else:
            self.result = self._selected_index + 1  # 1..N (index of selection)

    def _on_new(self):
        """
        Trigger New: returns N+1.
        This must bypass selection validation.
        """
        self._force_new = True
        self.ok()

    # Important: Don't overwrite OK/New results when closing via Dialog.ok().
    # Dialog.ok() calls validate() -> apply() -> cancel().
    # Only set 0 in cancel() if result wasn't already set.
    def cancel(self, event=None):
        if self.result is None:
            self.result = 0
        super().cancel(event)


def choose_index_with_three_buttons(parent, title, prompt, items):
    """
    Show the dialog and return:
      - 1..N : highlighted item index (OK)
      - 0    : Cancel/close
      - N+1  : New
    """
    dlg = ThreeButtonListDialog(parent, title, prompt, items)
    return dlg.result
