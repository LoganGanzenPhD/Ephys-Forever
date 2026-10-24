"""Package-level invariants."""

from __future__ import annotations

import subprocess
import sys

import ephys_forever


def test_version_is_set() -> None:
    assert ephys_forever.__version__


def test_core_imports_no_gui_modules() -> None:
    code = (
        "import sys, pkgutil, importlib, ephys_forever.core as c\n"
        "for m in pkgutil.walk_packages(c.__path__, 'ephys_forever.core.'):\n"
        "    importlib.import_module(m.name)\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "{'PySide6', 'PyQt5', 'PyQt6', 'pyqtgraph', 'ephys_forever'} and "
        "(m.startswith(('PySide6', 'PyQt', 'pyqtgraph')) or m.startswith('ephys_forever.gui'))]\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
