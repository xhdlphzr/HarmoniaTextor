# -*- mode: python ; coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""PyInstaller spec for the HarmoniaTextor desktop application.

The spec is cross-platform: it selects the matching icon on Windows and macOS
and builds a macOS ``.app`` bundle on Darwin.  The optional desktop extras must
be installed before building, e.g.::

    uv sync --extra desktop --group dev
    uv run --no-sync pyinstaller HarmoniaTextor.spec
"""

import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

datas = [
    ("app/templates", "app/templates"),
    ("app/static", "app/static"),
    ("assets", "assets"),
    # Interface message catalogues (en.yaml / zh.yaml).
    ("i18n", "i18n"),
]

# pywebview is imported dynamically (importlib.import_module("webview")), so
# PyInstaller cannot see it.  Its platform backends load their JavaScript and
# native WebView2 libraries from package data at runtime, so bundle both the
# package data and every importable submodule explicitly.
datas += collect_data_files("webview")
binaries = []
hiddenimports = ["app.desktop", "webview"]
hiddenimports += collect_submodules("webview", on_error="ignore")

# Bundle music21 code and its non-corpus data.  The upstream corpus manifest
# references a few files that are not shipped, so missing entries are dropped.
_music21_datas = collect_data_files("music21", excludes=["corpus/**"])
datas += [(src, dest) for src, dest in _music21_datas if os.path.exists(src)]
hiddenimports += collect_submodules("music21")

icon = None
if sys.platform.startswith("win"):
    icon = "assets/Franx.ico"
elif sys.platform == "darwin":
    icon = "assets/Franx.icns"

a = Analysis(
    ["packaging/entry.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HarmoniaTextor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HarmoniaTextor",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="HarmoniaTextor.app",
        icon=icon,
        bundle_identifier="com.xhdlphzr.harmoniatextor",
        info_plist={
            "NSHighResolutionCapable": True,
            "LSApplicationCategoryType": "public.app-category.music",
        },
    )
