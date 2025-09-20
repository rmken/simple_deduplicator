#!/usr/bin/env python3
"""
Build script using PyInstaller for cross-platform packaging
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Use local config dir to avoid permission issues on macOS sandbox
PYINSTALLER_CONFIG_DIR = Path.cwd() / ".pyinstaller"
os.environ.setdefault("PYINSTALLER_CONFIG_DIR", str(PYINSTALLER_CONFIG_DIR))
PYINSTALLER_CONFIG_DIR.mkdir(exist_ok=True)

def install_dependencies():
    """Install required dependencies for building."""
    print("Installing dependencies...")
    packages = [
        "pyinstaller>=5.0",
        "PySide6>=6.5.0",
        "xxhash>=3.0.0",
        "blake3>=0.3.0",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

    if sys.platform == "win32":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "winshell>=0.6"])

def build_executable():
    """Build the executable using PyInstaller."""
    print(f"Building executable for {sys.platform}...")

    for folder in ["build", "dist"]:
        if Path(folder).exists():
            shutil.rmtree(folder)

    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name", "SimpleDeduplicator",
        "--icon", "icon.ico" if sys.platform == "win32" else "icon.icns",
        "--collect-all", "PySide6",
        "--hidden-import", "xxhash",
        "--hidden-import", "blake3",
    ]

    if sys.platform == "win32":
        cmd.extend(["--hidden-import", "winshell"])

    cmd.append("simple-deduplicator.py")

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"❌ Build failed: {exc}")
        return False

    print("✅ Build successful! Executable created in 'dist' folder")

    exe_name = "SimpleDeduplicator.exe" if sys.platform == "win32" else "SimpleDeduplicator"
    exe_path = Path("dist") / exe_name
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"Executable size: {size_mb:.1f} MB")
        print(f"Location: {exe_path.resolve()}")
    else:
        print("⚠️ Expected output not found; check PyInstaller logs.")

    return True

def create_spec_file():
    """Create a custom PyInstaller spec file for advanced configuration."""
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['simple-deduplicator.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'xxhash',
        'blake3',
        {'winshell,' if sys.platform == "win32" else ''}
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'PySide6.QtGui',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
        'PIL',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SimpleDeduplicator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='{"icon.ico" if sys.platform == "win32" else "icon.icns"}',
)
'''
    with open("SimpleDeduplicator.spec", "w", encoding="utf-8") as spec_file:
        spec_file.write(spec_content)

    print("✅ Spec file created: SimpleDeduplicator.spec")

def build_with_spec():
    """Build using the custom spec file."""
    print("Building with custom spec file...")
    try:
        subprocess.check_call(["pyinstaller", "SimpleDeduplicator.spec"])
        return True
    except subprocess.CalledProcessError as exc:
        print(f"❌ Build with spec failed: {exc}")
        return False

def main() -> int:
    """Main build function."""
    print("🚀 Simple Deduplicator - Build Script")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")

    install_dependencies()
    create_spec_file()

    if build_with_spec():
        print("\n✅ Build completed successfully!")
        print("\n📦 Distribution files:")
        dist_path = Path("dist")
        if dist_path.exists():
            for file in dist_path.iterdir():
                print(f"  - {file.name}")

        print("\n🎯 To distribute your app:")
        if sys.platform == "win32":
            print("  - Windows: Share the .exe file")
        elif sys.platform == "darwin":
            print("  - macOS: Share the app bundle or create a DMG")
        else:
            print("  - Linux: Share the executable or create an AppImage")
        return 0

    print("\n❌ Build failed!")
    return 1

if __name__ == "__main__":
    sys.exit(main())
