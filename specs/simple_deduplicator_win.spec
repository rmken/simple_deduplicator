# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

proj_root = os.getcwd()
script = os.path.join(proj_root, "simple-deduplicator.py")

datas = []
hiddenimports = []
binaries = []
hookspath = []
hooksconfig = {}
runtime_hooks = []
excludes = [
    "PySide6.scripts", "PySide6.Qt3D*", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtNetwork", "PySide6.QtQml",
    "PySide6.QtQuick", "PySide6.QtTest", "PySide6.QtWeb*", "PySide6.QtBluetooth",
    "PySide6.QtNfc", "PySide6.QtPositioning", "PySide6.QtSerialPort", "PySide6.QtSql",
    "PySide6.QtSvg", "PySide6.QtXml",
]

datas         += collect_data_files("PySide6.QtCore")
datas         += collect_data_files("PySide6.QtGui")
datas         += collect_data_files("PySide6.QtWidgets")
hiddenimports += collect_submodules("PySide6.QtCore")
hiddenimports += collect_submodules("PySide6.QtGui")
hiddenimports += collect_submodules("PySide6.QtWidgets")

hiddenimports += ["blake3"]
binaries      += collect_dynamic_libs("blake3")

hiddenimports += ["xxhash"]
binaries      += collect_dynamic_libs("xxhash")

a = Analysis(
    [script],
    pathex=[proj_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=hookspath,
    hooksconfig=hooksconfig,
    runtime_hooks=runtime_hooks,
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="SimpleDeduplicator",
    console=False,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SimpleDeduplicator",
)
