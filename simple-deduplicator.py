#!/usr/bin/env python3
"""
Simple Deduplicator Desktop Application
A cross-platform PySide6 app for finding and managing duplicate files.
"""

import sys
import os
import hashlib
import logging
import threading
import time
import locale
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
import json
import sqlite3
import tempfile
import shutil

def _ensure_utf8_locale():
    """Force a UTF-8 locale so Qt has the expected environment."""
    target = "C.UTF-8"
    current = os.environ.get("LC_ALL")

    if current in (None, "", "C"):
        try:
            locale.setlocale(locale.LC_ALL, target)
            os.environ["LC_ALL"] = target
            os.environ.setdefault("LANG", target)
        except locale.Error:
            pass  # Leave locale untouched if target doesn't exist


_ensure_utf8_locale()

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QTextEdit, QSplitter, QFileDialog, QMessageBox,
    QComboBox, QLabel, QCheckBox, QSpinBox, QGroupBox, QFrame,
    QAbstractItemView, QStyle, QListView, QTreeView, QSizePolicy,
    QMenu, QLayout
)
from PySide6.QtCore import (
    QThread, QObject, Signal, QTimer, Qt, QSize, QSettings, QUrl
)
from PySide6.QtGui import QFont, QPalette, QColor, QIcon

# Third-party imports for hashing algorithms
try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Data class for file information."""
    path: Path
    size: int
    hash_value: str
    is_duplicate: bool = False
    error: Optional[str] = None


@dataclass
class ScanProgress:
    """Data class for scan progress information."""
    files_scanned: int = 0
    total_files: int = 0
    current_file: str = ""
    duplicates_found: int = 0
    errors_count: int = 0
    bytes_processed: int = 0


class HashWorker(QObject):
    """Worker thread for file hashing operations."""
    
    # Signals
    progress_updated = Signal(ScanProgress)
    file_processed = Signal(FileInfo)
    scan_completed = Signal(dict)  # {hash: [FileInfo]}
    error_occurred = Signal(str)
    duplicates_updated = Signal(list)  # List[FileInfo]
    
    def __init__(self):
        super().__init__()
        self.folders: List[Path] = []
        self.hash_algorithm = "md5"
        self.chunk_size = 8192
        self.is_paused = False
        self.is_cancelled = False
        self.executor: Optional[ThreadPoolExecutor] = None
        
    def set_parameters(self, folders: List[Path], algorithm: str, chunk_size: int):
        """Set scanning parameters."""
        self.folders = folders
        self.hash_algorithm = algorithm.lower()
        self.chunk_size = chunk_size
        
    def pause(self):
        """Pause the scanning process."""
        self.is_paused = True
        
    def resume(self):
        """Resume the scanning process."""
        self.is_paused = False
        
    def cancel(self):
        """Cancel the scanning process."""
        self.is_cancelled = True
        if self.executor:
            self.executor.shutdown(wait=False)
    
    def _get_hasher(self):
        """Get the appropriate hash function."""
        if self.hash_algorithm == "md5":
            return hashlib.md5()
        elif self.hash_algorithm == "sha256":
            return hashlib.sha256()
        elif self.hash_algorithm == "crc32":
            import zlib
            return None  # Special case for CRC32
        elif self.hash_algorithm == "blake3" and HAS_BLAKE3:
            return blake3.blake3()
        elif self.hash_algorithm == "xxhash" and HAS_XXHASH:
            return xxhash.xxh64()
        else:
            return hashlib.md5()  # Fallback
    
    def _hash_file(self, file_path: Path) -> str:
        """Calculate hash for a single file."""
        try:
            if self.hash_algorithm == "crc32":
                import zlib
                crc = 0
                with open(file_path, 'rb') as f:
                    while chunk := f.read(self.chunk_size):
                        crc = zlib.crc32(chunk, crc)
                return f"{crc:08x}"
            
            hasher = self._get_hasher()
            with open(file_path, 'rb') as f:
                while chunk := f.read(self.chunk_size):
                    if self.is_cancelled:
                        return ""
                    hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except (IOError, OSError, PermissionError) as e:
            logger.warning(f"Cannot read file {file_path}: {e}")
            raise
    
    def _collect_files(self) -> List[Path]:
        """Collect all files from selected folders."""
        all_files = []
        for folder in self.folders:
            try:
                for file_path in folder.rglob("*"):
                    if file_path.is_file() and not file_path.is_symlink():
                        all_files.append(file_path)
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot access folder {folder}: {e}")
                self.error_occurred.emit(f"Cannot access folder {folder}: {e}")
        return all_files
    
    def run_scan(self):
        """Main scanning method."""
        logger.info("Starting file scan")
        start_time = time.time()
        
        # Collect all files
        all_files = self._collect_files()
        if not all_files:
            self.error_occurred.emit("No files found in selected folders")
            return
            
        progress = ScanProgress(total_files=len(all_files))
        
        # Group files by size first (optimization)
        size_groups: Dict[int, List[Path]] = {}
        for file_path in all_files:
            try:
                size = file_path.stat().st_size
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(file_path)
            except (OSError, IOError):
                progress.errors_count += 1
        
        # Only hash files with potential duplicates (same size)
        files_to_hash = []
        for size, paths in size_groups.items():
            if len(paths) > 1:  # Only hash if multiple files have same size
                files_to_hash.extend(paths)
        
        # Hash files and group by hash
        hash_groups: Dict[str, List[FileInfo]] = {}

        # Update total files based on actual hashing workload
        progress.total_files = len(files_to_hash) if files_to_hash else len(all_files)
        if progress.total_files == 0:
            progress.total_files = len(all_files)
        self.progress_updated.emit(progress)

        with ThreadPoolExecutor(max_workers=4) as executor:
            self.executor = executor
            future_to_path = {}
            
            for file_path in files_to_hash:
                if self.is_cancelled:
                    break
                future = executor.submit(self._hash_file, file_path)
                future_to_path[future] = file_path
            
            for future in future_to_path:
                if self.is_cancelled:
                    break
                    
                while self.is_paused and not self.is_cancelled:
                    time.sleep(0.1)
                
                file_path = future_to_path[future]
                progress.files_scanned += 1
                progress.current_file = str(file_path)
                
                try:
                    file_size = file_path.stat().st_size
                    hash_value = future.result()
                    
                    if hash_value:  # Successfully hashed
                        file_info = FileInfo(
                            path=file_path,
                            size=file_size,
                            hash_value=hash_value
                        )
                        
                        if hash_value not in hash_groups:
                            hash_groups[hash_value] = []
                        hash_groups[hash_value].append(file_info)
                        
                        progress.bytes_processed += file_size
                        self.file_processed.emit(file_info)

                        if len(hash_groups[hash_value]) >= 2:
                            for info in hash_groups[hash_value]:
                                info.is_duplicate = True
                            self.duplicates_updated.emit(hash_groups[hash_value][:])
                    
                except Exception as e:
                    progress.errors_count += 1
                    logger.warning(f"Error processing {file_path}: {e}")
                
                self.progress_updated.emit(progress)
        
        # Mark duplicates
        duplicates_count = 0
        for hash_value, files in hash_groups.items():
            if len(files) > 1:
                for file_info in files:
                    file_info.is_duplicate = True
                    duplicates_count += 1
        
        progress.duplicates_found = duplicates_count
        progress.files_scanned = progress.total_files
        progress.current_file = ""
        self.progress_updated.emit(progress)
        
        duration = time.time() - start_time
        logger.info(f"Scan completed in {duration:.2f} seconds")
        logger.info(f"Found {duplicates_count} duplicate files")
        
        self.scan_completed.emit(hash_groups)


class SystemMessagesWidget(QWidget):
    """Widget for displaying system messages."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Title
        title = QLabel("System Messages")
        title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(title)

        # Messages text area
        self.messages = QTextEdit()
        self.messages.setReadOnly(True)
        self.messages.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.messages, 1)

        # Clear button
        clear_btn = QPushButton("Clear Messages")
        clear_btn.clicked.connect(self.clear_messages)
        layout.addWidget(clear_btn)
    
    def add_message(self, message: str):
        """Add a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.messages.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.messages.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_messages(self):
        """Clear all messages."""
        self.messages.clear()


class SimpleDeduplicatorApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings("SimpleDeduplicator", "Settings")
        self.hash_worker = HashWorker()
        self.worker_thread: Optional[QThread] = None
        self.file_data: Dict[str, List[FileInfo]] = {}
        self.scan_start_time: Optional[float] = None
        
        self.init_ui()
        self.setup_connections()
        self.apply_theme()
        self.setup_cleanup_timer()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Simple Deduplicator")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (main controls and table)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel (system messages)
        self.system_messages = SystemMessagesWidget()
        splitter.addWidget(self.system_messages)
        
        # Set splitter proportions (80% left, 20% right)
        splitter.setSizes([800, 200])
        
        self._tune_layout_spacing()

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_left_panel(self) -> QWidget:
        """Create the left panel with controls and table."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Controls section
        controls_frame = self.create_controls_section()
        layout.addWidget(controls_frame)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results table
        self.create_results_table()
        layout.addWidget(self.results_table)
        
        # Bottom controls
        bottom_controls = self.create_bottom_controls()
        layout.addWidget(bottom_controls)
        
        return panel

    def _tune_layout_spacing(self):
        """Ensure layouts have comfortable spacing in both themes."""
        for layout in self.findChildren(QLayout):
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

    def setup_cleanup_timer(self):
        """Periodically remove rows for files deleted outside the app."""
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.setInterval(10000)  # 10 seconds
        self.cleanup_timer.timeout.connect(self.prune_missing_files)
        self.cleanup_timer.start()

    def prune_missing_files(self):
        """Check for rows referencing files no longer on disk and drop them."""
        rows_to_remove: List[Tuple[int, FileInfo]] = []

        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if not item:
                continue

            file_info: Optional[FileInfo] = item.data(Qt.ItemDataRole.UserRole)
            if not file_info:
                continue

            if not file_info.path.exists():
                rows_to_remove.append((row, file_info))

        if not rows_to_remove:
            return

        for row, file_info in reversed(rows_to_remove):
            self.results_table.removeRow(row)
            self._remove_file_from_data(file_info)

        self.update_selection_info()
        removed_names = ", ".join(file_info.path.name for _, file_info in rows_to_remove[:5])
        summary = removed_names if len(rows_to_remove) <= 5 else f"{len(rows_to_remove)} entries"
        self.system_messages.add_message(
            f"Removed missing files from results: {summary}"
        )

    def _remove_file_from_data(self, file_info: FileInfo):
        """Update cached duplicate groups after a file disappears."""
        files = self.file_data.get(file_info.hash_value)
        if not files:
            return

        updated = [info for info in files if info.path != file_info.path]
        if updated:
            self.file_data[file_info.hash_value] = updated
        else:
            self.file_data.pop(file_info.hash_value, None)

    def _row_for_path(self, path: Path) -> Optional[int]:
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if not item:
                continue
            file_info: Optional[FileInfo] = item.data(Qt.ItemDataRole.UserRole)
            if file_info and file_info.path == path:
                return row
        return None

    def _refresh_row(self, row: int, file_info: FileInfo):
        filename_item = self.results_table.item(row, 0)
        path_item = self.results_table.item(row, 1)
        size_item = self.results_table.item(row, 2)
        hash_item = self.results_table.item(row, 3)
        status_item = self.results_table.item(row, 4)

        if filename_item:
            filename_item.setText(file_info.path.name)
            filename_item.setToolTip(str(file_info.path))
            filename_item.setData(Qt.ItemDataRole.UserRole, file_info)

        if path_item:
            path_item.setText(str(file_info.path.parent))
            path_item.setToolTip(str(file_info.path))

        if size_item:
            size_item.setText(self.format_size(file_info.size))
            size_item.setData(Qt.ItemDataRole.UserRole, file_info.size)

        if hash_item:
            hash_item.setText(file_info.hash_value[:16] + "...")
            hash_item.setToolTip(file_info.hash_value)

        if status_item:
            status_item.setText("Duplicate")
            status_item.setBackground(QColor(255, 200, 200))

    def update_duplicates_view(self, files: List[FileInfo]):
        if not files:
            return

        hash_value = files[0].hash_value

        existing_files = [info for info in files if info.path.exists()]
        if not existing_files:
            self.file_data.pop(hash_value, None)
            return

        self.file_data[hash_value] = existing_files

        for file_info in existing_files:
            row = self._row_for_path(file_info.path)
            if row is None:
                self.add_file_to_table(file_info)
            else:
                self._refresh_row(row, file_info)

        self.update_selection_info()

    def create_controls_section(self) -> QFrame:
        """Create the main controls section."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(frame)
        
        # Folder selection
        folder_layout = QHBoxLayout()
        self.select_folders_btn = QPushButton("Select Folders")
        self.select_folders_btn.clicked.connect(self.select_folders)
        folder_layout.addWidget(self.select_folders_btn)
        
        self.selected_folders_label = QLabel("No folders selected")
        folder_layout.addWidget(self.selected_folders_label)
        folder_layout.addStretch()
        layout.addLayout(folder_layout)
        
        # Algorithm and settings
        settings_layout = QHBoxLayout()
        
        # Hash algorithm selection
        settings_layout.addWidget(QLabel("Hash Algorithm:"))
        self.algorithm_combo = QComboBox()
        algorithms = ["MD5", "SHA256", "CRC32"]
        if HAS_BLAKE3:
            algorithms.append("BLAKE3")
        if HAS_XXHASH:
            algorithms.append("XXHash")
        self.algorithm_combo.addItems(algorithms)
        self.algorithm_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        widest_option = max(len(option) for option in algorithms)
        self.algorithm_combo.setMinimumContentsLength(widest_option + 2)
        self.algorithm_combo.view().setMinimumWidth(self.algorithm_combo.sizeHint().width() + 24)
        settings_layout.addWidget(self.algorithm_combo)
        
        # Chunk size
        settings_layout.addWidget(QLabel("Chunk Size (KB):"))
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(1, 1024)
        self.chunk_size_spin.setValue(8)
        self.chunk_size_spin.setSuffix(" KB")
        settings_layout.addWidget(self.chunk_size_spin)
        
        settings_layout.addStretch()
        layout.addLayout(settings_layout)
        
        # Scan controls
        scan_layout = QHBoxLayout()
        self.start_scan_btn = QPushButton("Start Scan")
        self.start_scan_btn.clicked.connect(self.start_scan)
        scan_layout.addWidget(self.start_scan_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_scan)
        self.pause_btn.setEnabled(False)
        scan_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_scan)
        self.stop_btn.setEnabled(False)
        scan_layout.addWidget(self.stop_btn)
        
        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.clicked.connect(self.clear_results)
        scan_layout.addWidget(self.clear_btn)
        
        scan_layout.addStretch()
        layout.addLayout(scan_layout)
        
        return frame
    
    def create_results_table(self):
        """Create the results table."""
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Filename", "Path", "Size", "Hash", "Status", "Action"
        ])
        
        # Configure table
        header = self.results_table.horizontalHeader()
        for column in range(self.results_table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)

        # Provide sensible starting widths while keeping columns resizable
        self.results_table.setColumnWidth(0, 200)  # Filename
        self.results_table.setColumnWidth(1, 420)  # Path
        self.results_table.setColumnWidth(2, 120)  # Size
        self.results_table.setColumnWidth(3, 180)  # Hash
        self.results_table.setColumnWidth(4, 120)  # Status
        self.results_table.setColumnWidth(5, 120)  # Action

        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSortingEnabled(True)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self.show_results_context_menu)
        self.results_table.verticalHeader().setDefaultSectionSize(30)
    
    def create_bottom_controls(self) -> QWidget:
        """Create bottom control buttons."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Selection info
        self.selection_info_label = QLabel("0 files selected")
        layout.addWidget(self.selection_info_label)
        
        # Bulk delete
        self.delete_selected_btn = QPushButton("Delete Selected")
        self.delete_selected_btn.clicked.connect(self.delete_selected_files)
        self.delete_selected_btn.setEnabled(False)
        layout.addWidget(self.delete_selected_btn)
        
        layout.addStretch()
        
        # Theme toggle
        self.theme_btn = QPushButton("🌙")  # Moon for dark theme
        self.theme_btn.setToolTip("Toggle Dark/Light Theme")
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.theme_btn.setMaximumSize(QSize(40, 40))
        layout.addWidget(self.theme_btn)
        
        return widget
    
    def setup_connections(self):
        """Set up signal connections."""
        # Worker signals
        self.hash_worker.progress_updated.connect(self.update_progress)
        self.hash_worker.file_processed.connect(self.file_processed)
        self.hash_worker.scan_completed.connect(self.scan_completed)
        self.hash_worker.error_occurred.connect(self.handle_error)
        self.hash_worker.duplicates_updated.connect(self.update_duplicates_view)
        
        # Table selection changes
        self.results_table.itemSelectionChanged.connect(self.update_selection_info)
    
    def select_folders(self):
        """Open folder selection dialog that supports multiple directories."""
        dialog = QFileDialog(self, "Select Folders to Scan")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontResolveSymlinks, True)

        # Native dialogs on macOS do not allow multi-select directories
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

        # Help users reach mounted network volumes quickly
        sidebar_urls = [QUrl.fromLocalFile(str(Path.home()))]
        volumes_path = Path("/Volumes")
        if volumes_path.exists():
            sidebar_urls.append(QUrl.fromLocalFile(str(volumes_path)))
            dialog.setDirectory(str(volumes_path))
        else:
            dialog.setDirectory(str(Path.home()))
        dialog.setSidebarUrls(sidebar_urls)

        # Enable multi-selection in the internal views
        for view_class in (QListView, QTreeView):
            for view in dialog.findChildren(view_class):
                view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        if dialog.exec():
            folders = [Path(path) for path in dialog.selectedFiles() if Path(path).is_dir()]

            if folders:
                self.selected_folders = folders

                display_text = str(folders[0])
                if len(folders) > 1:
                    display_text = f"{display_text} (+{len(folders) - 1} more)"
                self.selected_folders_label.setText(f"Selected: {display_text}")
                self.selected_folders_label.setToolTip("\n".join(str(folder) for folder in folders))

                total_files = 0
                for folder in folders:
                    total_files += sum(1 for _ in folder.rglob("*") if _.is_file())

                folder_word = "folder" if len(folders) == 1 else "folders"
                self.system_messages.add_message(
                    f"{len(folders)} {folder_word} selected: {total_files} files found"
                )
    
    def start_scan(self):
        """Start the file scanning process."""
        if not hasattr(self, 'selected_folders') or not self.selected_folders:
            self.system_messages.add_message("ERROR: No folders selected for scanning")
            QMessageBox.warning(self, "Warning", "Please select folders to scan first.")
            return
        
        # Prepare worker
        algorithm = self.algorithm_combo.currentText().lower()
        chunk_size = self.chunk_size_spin.value() * 1024  # Convert KB to bytes
        
        self.hash_worker.set_parameters(self.selected_folders, algorithm, chunk_size)
        
        # Set up worker thread
        self.worker_thread = QThread()
        self.hash_worker.moveToThread(self.worker_thread)
        
        # Connect thread signals
        self.worker_thread.started.connect(self.hash_worker.run_scan)
        self.worker_thread.finished.connect(self.scan_finished)
        
        # Update UI state
        self.start_scan_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusBar().showMessage("Preparing scan...")

        self.scan_start_time = time.time()
        self.system_messages.add_message(
            f"Scan started using {algorithm.upper()} hashing"
        )
        
        # Start the thread
        self.worker_thread.start()
    
    def pause_scan(self):
        """Pause or resume the scanning process."""
        if self.hash_worker.is_paused:
            self.hash_worker.resume()
            self.pause_btn.setText("Pause")
            self.system_messages.add_message("Scan resumed")
        else:
            self.hash_worker.pause()
            self.pause_btn.setText("Resume")
            self.system_messages.add_message("Scan paused")
    
    def stop_scan(self):
        """Stop the scanning process."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.hash_worker.cancel()
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.system_messages.add_message("Scan aborted by user")
            self.statusBar().showMessage("Scan aborted")
            self.progress_bar.setVisible(False)
    
    def update_progress(self, progress: ScanProgress):
        """Update the progress bar and status."""
        if progress.total_files > 0:
            percentage = int((progress.files_scanned / progress.total_files) * 100)
        else:
            percentage = 0

        self.progress_bar.setValue(percentage)

        status = f"Scanning: {progress.files_scanned}/{progress.total_files} files"
        if progress.current_file:
            status += f" - {Path(progress.current_file).name}"
        self.statusBar().showMessage(status)
    
    def file_processed(self, file_info: FileInfo):
        """Handle a processed file."""
        # Add to results table if it's a duplicate or we're showing all files
        if file_info.is_duplicate:
            self.add_file_to_table(file_info)
    
    def scan_completed(self, hash_groups: Dict[str, List[FileInfo]]):
        """Handle scan completion."""
        self.progress_bar.setValue(100)
        self.file_data = hash_groups
        
        # Populate the results table with duplicates only
        self.results_table.setRowCount(0)
        for hash_value, files in hash_groups.items():
            if len(files) > 1:  # Only show duplicates
                for file_info in files:
                    self.add_file_to_table(file_info)
        
        # Calculate statistics
        total_duplicates = sum(len(files) for files in hash_groups.values() if len(files) > 1)
        duplicate_groups = sum(1 for files in hash_groups.values() if len(files) > 1)
        
        duration = time.time() - self.scan_start_time if self.scan_start_time else 0
        
        self.system_messages.add_message(f"Scan completed in {duration:.1f} seconds")
        self.system_messages.add_message(f"Found {total_duplicates} duplicate files in {duplicate_groups} groups")
    
    def scan_finished(self):
        """Handle scan thread completion."""
        # Clean up
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None
        
        # Reset UI state
        self.start_scan_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        
        self.statusBar().showMessage("Scan completed")
    
    def add_file_to_table(self, file_info: FileInfo):
        """Add a file to the results table."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Filename
        filename_item = QTableWidgetItem(file_info.path.name)
        filename_item.setToolTip(str(file_info.path))
        self.results_table.setItem(row, 0, filename_item)

        # Path
        path_item = QTableWidgetItem(str(file_info.path.parent))
        path_item.setToolTip(str(file_info.path))
        self.results_table.setItem(row, 1, path_item)
        
        # Size
        size_item = QTableWidgetItem(self.format_size(file_info.size))
        size_item.setData(Qt.ItemDataRole.UserRole, file_info.size)  # Store raw size for sorting
        self.results_table.setItem(row, 2, size_item)
        
        # Hash
        hash_item = QTableWidgetItem(file_info.hash_value[:16] + "...")  # Truncated for display
        hash_item.setToolTip(file_info.hash_value)  # Full hash in tooltip
        self.results_table.setItem(row, 3, hash_item)
        
        # Status
        status_text = "Duplicate" if file_info.is_duplicate else "Unique"
        status_item = QTableWidgetItem(status_text)
        if file_info.is_duplicate:
            status_item.setBackground(QColor(255, 200, 200))  # Light red background
        self.results_table.setItem(row, 4, status_item)
        
        # Delete button
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(lambda: self.delete_single_file(row))
        self.results_table.setCellWidget(row, 5, delete_btn)
        
        # Store file info for later use
        filename_item.setData(Qt.ItemDataRole.UserRole, file_info)

    def show_results_context_menu(self, position):
        """Display context menu for results table rows."""
        index = self.results_table.indexAt(position)
        if not index.isValid():
            return

        row = index.row()
        filename_item = self.results_table.item(row, 0)
        if not filename_item:
            return

        file_info: Optional[FileInfo] = filename_item.data(Qt.ItemDataRole.UserRole)
        if not file_info:
            return

        menu = QMenu(self)
        if sys.platform == "darwin":
            label = "Show in Finder"
        elif os.name == "nt":
            label = "Show in Explorer"
        else:
            label = "Show in File Manager"
        reveal_action = menu.addAction(label)
        selected_action = menu.exec(self.results_table.viewport().mapToGlobal(position))

        if selected_action == reveal_action:
            self.reveal_in_file_explorer(file_info.path)

    def reveal_in_file_explorer(self, path: Path):
        """Open the given path in the OS file explorer."""
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", "-R", str(path)], check=False)
            elif os.name == "nt":
                subprocess.run(["explorer", "/select,", str(path)], check=False)
            else:
                target = path if path.is_dir() else path.parent
                subprocess.run(["xdg-open", str(target)], check=False)
        except Exception as exc:
            QMessageBox.warning(self, "Open Location Failed", f"Could not open file location:\n{exc}")

    def delete_single_file(self, row: int):
        """Delete a single file."""
        filename_item = self.results_table.item(row, 0)
        if not filename_item:
            return
            
        file_info: FileInfo = filename_item.data(Qt.ItemDataRole.UserRole)
        if not file_info:
            return
        
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete:\n{file_info.path}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.move_to_trash(file_info.path)
                self.results_table.removeRow(row)
                self._remove_file_from_data(file_info)
                self.system_messages.add_message(f"Deleted: {file_info.path.name}")
                self.update_selection_info()
            except Exception as e:
                QMessageBox.critical(self, "Delete Error", f"Could not delete file:\n{e}")
    
    def delete_selected_files(self):
        """Delete all selected files."""
        selected_rows = set()
        for item in self.results_table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select files to delete.")
            return
        
        # Calculate total size
        total_size = 0
        files_to_delete = []
        
        for row in selected_rows:
            filename_item = self.results_table.item(row, 0)
            if filename_item:
                file_info: FileInfo = filename_item.data(Qt.ItemDataRole.UserRole)
                if file_info:
                    files_to_delete.append((row, file_info))
                    total_size += file_info.size
        
        if not files_to_delete:
            return
        
        reply = QMessageBox.question(
            self, "Confirm Bulk Delete",
            f"Are you sure you want to delete {len(files_to_delete)} files?\n"
            f"Total size: {self.format_size(total_size)}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            deleted_count = 0
            failed_count = 0
            
            # Sort by row in descending order to avoid index issues when removing
            files_to_delete.sort(key=lambda x: x[0], reverse=True)
            
            for row, file_info in files_to_delete:
                try:
                    self.move_to_trash(file_info.path)
                    self.results_table.removeRow(row)
                    self._remove_file_from_data(file_info)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_info.path}: {e}")
                    failed_count += 1
            
            self.system_messages.add_message(
                f"Bulk delete completed: {deleted_count} deleted, {failed_count} failed"
            )
            self.update_selection_info()

    def move_to_trash(self, file_path: Path):
        """Move file to trash/recycle bin."""
        try:
            # Try using system trash first
            if sys.platform == "win32":
                import winshell
                winshell.delete_file(str(file_path))
            elif sys.platform == "darwin":  # macOS
                import subprocess
                subprocess.run(["osascript", "-e", f'tell app "Finder" to delete POSIX file "{file_path}"'])
            else:  # Linux and others
                # Try using gio trash command
                import subprocess
                result = subprocess.run(["gio", "trash", str(file_path)], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    # Fallback to direct deletion if trash is not available
                    file_path.unlink()
        except ImportError:
            # Fallback to direct deletion
            file_path.unlink()
    
    def clear_results(self):
        """Clear all results."""
        self.results_table.setRowCount(0)
        self.file_data.clear()
        self.system_messages.add_message("Results cleared")
    
    def update_selection_info(self):
        """Update selection information."""
        selected_count = len(self.results_table.selectedItems()) // self.results_table.columnCount()
        self.selection_info_label.setText(f"{selected_count} files selected")
        self.delete_selected_btn.setEnabled(selected_count > 0)
    
    def handle_error(self, error_message: str):
        """Handle errors from the worker thread."""
        self.system_messages.add_message(f"ERROR: {error_message}")
        logger.error(error_message)
    
    def toggle_theme(self):
        """Toggle between light and dark themes."""
        current_theme = self.settings.value("theme", "light")
        new_theme = "dark" if current_theme == "light" else "light"
        self.settings.setValue("theme", new_theme)
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the current theme."""
        theme = self.settings.value("theme", "light")
        
        if theme == "dark":
            # Dark theme
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QTableWidget {
                    background-color: #3c3c3c;
                    alternate-background-color: #454545;
                    gridline-color: #555555;
                    selection-background-color: #4a90e2;
                }
                QTableWidget::item {
                    padding: 5px;
                }
                QHeaderView::section {
                    background-color: #404040;
                    color: #ffffff;
                    border: 1px solid #555555;
                    padding: 5px;
                }
                QPushButton {
                    background-color: #404040;
                    border: 1px solid #555555;
                    color: #ffffff;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                }
                QPushButton:pressed {
                    background-color: #353535;
                }
                QPushButton:disabled {
                    background-color: #2a2a2a;
                    color: #666666;
                }
                QComboBox, QSpinBox {
                    background-color: #404040;
                    border: 1px solid #555555;
                    color: #ffffff;
                    padding: 3px;
                    min-height: 28px;
                }
                QComboBox QAbstractItemView {
                    background-color: #404040;
                    color: #ffffff;
                    selection-background-color: #4a90e2;
                    selection-color: #ffffff;
                }
                QTextEdit {
                    background-color: #3c3c3c;
                    border: 1px solid #555555;
                    color: #ffffff;
                }
                QProgressBar {
                    border: 1px solid #555555;
                    border-radius: 3px;
                    background-color: #3c3c3c;
                }
                QProgressBar::chunk {
                    background-color: #4a90e2;
                    border-radius: 3px;
                }
                QFrame {
                    border: 1px solid #555555;
                    background-color: #2b2b2b;
                }
                QLabel {
                    color: #ffffff;
                }
            """)
            self.theme_btn.setText("☀️")  # Sun for light theme
        else:
            # Light theme (default)
            self.setStyleSheet("")
            self.theme_btn.setText("🌙")  # Moon for dark theme
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Stop any running scan
        if self.worker_thread and self.worker_thread.isRunning():
            self.hash_worker.cancel()
            self.worker_thread.quit()
            self.worker_thread.wait(3000)  # Wait up to 3 seconds
        
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Simple Deduplicator")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("SimpleDeduplicator")
    
    # Set application icon if available
    try:
        app.setWindowIcon(app.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
    except:
        pass
    
    # Create and show main window
    window = SimpleDeduplicatorApp()
    window.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
