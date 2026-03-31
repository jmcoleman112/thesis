from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def sanitize_filename(name: str) -> str:
    """
    Replace characters that are invalid in Windows filenames.
    """
    # Windows reserved characters: <>:"/\|?*
    return re.sub(r"[<>:\"/\\|?*]", "_", name).strip()


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "produced_images"


def prompt_save_figure(
    fig,
    *,
    default_name: Optional[str] = None,
    out_dir: Optional[Path] = None,
    dpi: int = 200,
    bbox_inches: str = "tight",
) -> Optional[Path]:
    """
    Ask the user (via a small popup) whether to save a figure.

    If user chooses Yes, ask for a filename and save to out_dir (default: Figures/produced_images).
    Returns the Path to the saved file, or None if not saved/cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox, simpledialog
    except Exception:
        print("Tkinter not available; cannot show save dialog.")
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        save = messagebox.askyesno("Save figure", "Save this figure?", parent=root)
        if not save:
            return None

        while True:
            name = simpledialog.askstring(
                "Figure name",
                "Enter filename (without extension):",
                initialvalue=default_name or "",
                parent=root,
            )
            if name is None:
                return None
            name = sanitize_filename(name)
            if name:
                break
            messagebox.showwarning("Invalid name", "Please enter a valid filename.", parent=root)

        if not name.lower().endswith(".png"):
            name = f"{name}.png"

        out_path_dir = out_dir if out_dir is not None else _default_output_dir()
        out_path_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_path_dir / name

        fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
        return out_path
    finally:
        try:
            root.destroy()
        except Exception:
            pass
