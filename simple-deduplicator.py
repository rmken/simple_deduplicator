#!/usr/bin/env python3
"""
Simple Deduplicator Desktop Application
A cross-platform PySide6 app for finding and managing duplicate files.
Optimized version with performance improvements and bug fixes.
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
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from datetime import datetime
import json
import sqlite3
import tempfile
import shutil
import weakref

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
    QComboBox, QLabel, QSpinBox, QFrame,
    QAbstractItemView, QStyle, QListView, QTreeView, QSizePolicy,
    QMenu, QLayout, QTabWidget, QCheckBox, QStatusBar, QListWidget, QListWidgetItem
)
from PySide6.QtCore import (
    QThread, QObject, Signal, QTimer, Qt, QSize, QSettings, QUrl,
    QMutex, QMutexLocker, QEvent
)
from PySide6.QtGui import QFont, QPalette, QColor, QIcon, QAction

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
    """Data class for file information with optimization for memory usage."""
    path: Path
    size: int
    hash_value: str
    is_duplicate: bool = False
    error: Optional[str] = None
    status: str = "Unknown"
    missing_locations: List[str] = field(default_factory=list)

    def __hash__(self) -> int:
        """Enable FileInfo to be used in sets for deduplication."""
        return hash((str(self.path), self.size, self.hash_value))

    def __eq__(self, other) -> bool:
        """Enable FileInfo comparison."""
        if not isinstance(other, FileInfo):
            return False
        return (self.path == other.path and 
                self.size == other.size and 
                self.hash_value == other.hash_value)


@dataclass
class ScanProgress:
    """Data class for scan progress information."""
    files_scanned: int = 0
    total_files: int = 0
    current_file: str = ""
    duplicates_found: int = 0
    errors_count: int = 0
    bytes_processed: int = 0
    scan_speed: float = 0.0  # Files per second


class HashWorker(QObject):
    """Worker thread for file hashing operations with performance optimizations."""
    
    # Signals
    progress_updated = Signal(ScanProgress)
    file_processed = Signal(FileInfo)
    scan_completed = Signal(dict)  # {hash: [FileInfo]}
    error_occurred = Signal(str)
    duplicates_updated = Signal(list)  # List[FileInfo]
    missing_found = Signal(FileInfo)
    
    def __init__(self):
        super().__init__()
        self.folders: List[Path] = []
        self.hash_algorithm = "md5"
        self.chunk_size = 65536  # Increased default chunk size for better performance
        self.is_paused = False
        self.is_cancelled = False
        self.executor: Optional[ThreadPoolExecutor] = None
        self.mode = "duplicates"
        self.mutex = QMutex()  # Thread safety
        self._start_time = 0
        self.skip_small_threshold = 0
        self.missing_sources: List[Path] = []
        self.missing_destinations: List[Path] = []

    def set_parameters(
        self,
        folders: List[Path],
        algorithm: str,
        chunk_size: int,
        mode: str = "duplicates",
        skip_small_threshold: int = 0,
        sources: Optional[List[Path]] = None,
        destinations: Optional[List[Path]] = None,
    ):
        """Set scanning parameters with validation."""
        with QMutexLocker(self.mutex):
            self.folders = folders
            self.hash_algorithm = algorithm.lower()
            self.chunk_size = max(1024, chunk_size)  # Minimum 1KB chunks
            self.mode = mode
            # Reset control flags for a fresh run
            self.is_paused = False
            self.is_cancelled = False
            self.skip_small_threshold = max(0, skip_small_threshold)
            if mode == "missing":
                self.missing_sources = [Path(p) for p in (sources or [])]
                self.missing_destinations = [Path(p) for p in (destinations or [])]
            else:
                self.missing_sources = []
                self.missing_destinations = []
        
    def pause(self):
        """Pause the scanning process."""
        with QMutexLocker(self.mutex):
            self.is_paused = True
        
    def resume(self):
        """Resume the scanning process."""
        with QMutexLocker(self.mutex):
            self.is_paused = False
        
    def cancel(self):
        """Cancel the scanning process."""
        with QMutexLocker(self.mutex):
            self.is_cancelled = True
        if self.executor:
            self.executor.shutdown(wait=False)
    
    def _get_hasher(self):
        """Get the appropriate hash function with optimized selection."""
        algorithm_map = {
            "md5": hashlib.md5,
            "sha256": hashlib.sha256,
            "sha1": hashlib.sha1,  # Added SHA1 for completeness
        }
        
        if self.hash_algorithm == "crc32":
            return None  # Special case for CRC32
        elif self.hash_algorithm == "blake3" and HAS_BLAKE3:
            return blake3.blake3
        elif self.hash_algorithm == "xxhash" and HAS_XXHASH:
            return xxhash.xxh64
        else:
            return algorithm_map.get(self.hash_algorithm, hashlib.md5)
    
    def _hash_file(self, file_path: Path) -> str:
        """Calculate hash for a single file with improved error handling."""
        try:
            if self.hash_algorithm == "crc32":
                import zlib
                crc = 0
                with open(file_path, 'rb') as f:
                    while True:
                        if self.is_cancelled:
                            return ""
                        chunk = f.read(self.chunk_size)
                        if not chunk:
                            break
                        crc = zlib.crc32(chunk, crc)
                return f"{crc & 0xffffffff:08x}"  # Ensure positive 32-bit value
            
            hasher = self._get_hasher()()
            with open(file_path, 'rb') as f:
                while True:
                    if self.is_cancelled:
                        return ""
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except (IOError, OSError, PermissionError) as e:
            logger.warning(f"Cannot read file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error hashing {file_path}: {e}")
            raise
    
    def _collect_files(self) -> List[Path]:
        """Collect all files from selected folders with improved performance."""
        all_files = []
        total_size = 0
        
        for folder in self.folders:
            try:
                folder_files = []
                for file_path in folder.rglob("*"):
                    if self.is_cancelled:
                        break
                    if file_path.is_file() and not file_path.is_symlink():
                        try:
                            # Pre-filter by size to avoid tiny files unless needed
                            stat_info = file_path.stat()
                            if self.skip_small_threshold and stat_info.st_size < self.skip_small_threshold:
                                continue
                            folder_files.append(file_path)
                            total_size += stat_info.st_size
                        except (OSError, PermissionError):
                            continue
                
                all_files.extend(folder_files)
                logger.info(f"Found {len(folder_files)} files in {folder}")
                
            except (PermissionError, OSError) as e:
                error_msg = f"Cannot access folder {folder}: {e}"
                logger.warning(error_msg)
                self.error_occurred.emit(error_msg)
        
        logger.info(f"Total files collected: {len(all_files)}, Total size: {total_size / (1024*1024):.1f} MB")
        return all_files
    
    def run_scan(self):
        """Main scanning method with performance optimizations."""
        logger.info(f"Starting {self.mode} scan")
        self._start_time = time.time()

        if self.mode == "missing":
            self.run_missing_scan()
            return

        # Collect all files
        all_files = self._collect_files()
        if not all_files:
            self.error_occurred.emit("No files found in selected folders")
            return
            
        progress = ScanProgress(total_files=len(all_files))
        
        # Group files by size first (major optimization)
        logger.info("Grouping files by size...")
        size_groups: Dict[int, List[Path]] = {}
        for file_path in all_files:
            if self.is_cancelled:
                break
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
        
        logger.info(f"Files requiring hashing: {len(files_to_hash)} out of {len(all_files)}")
        
        # Hash files and group by hash
        hash_groups: Dict[str, List[FileInfo]] = {}
        
        # Update total files based on actual hashing workload
        progress.total_files = len(files_to_hash) if files_to_hash else len(all_files)
        if progress.total_files == 0:
            progress.total_files = len(all_files)
        self.progress_updated.emit(progress)

        # Use optimal number of worker threads
        max_workers = min(8, os.cpu_count() or 4)  # Cap at 8 threads
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.executor = executor
            
            # Submit all hashing tasks
            future_to_path = {
                executor.submit(self._hash_file, file_path): file_path
                for file_path in files_to_hash
                if not self.is_cancelled
            }
            
            # Process completed futures as they finish
            for future in as_completed(future_to_path):
                if self.is_cancelled:
                    break
                    
                while self.is_paused and not self.is_cancelled:
                    time.sleep(0.1)
                
                file_path = future_to_path[future]
                progress.files_scanned += 1
                progress.current_file = str(file_path)
                
                # Calculate scan speed
                elapsed = time.time() - self._start_time
                if elapsed > 0:
                    progress.scan_speed = progress.files_scanned / elapsed
                
                try:
                    file_size = file_path.stat().st_size
                    hash_value = future.result()
                    
                    if hash_value:  # Successfully hashed
                        file_info = FileInfo(
                            path=file_path,
                            size=file_size,
                            hash_value=hash_value,
                            status="Unique"
                        )
                        
                        if hash_value not in hash_groups:
                            hash_groups[hash_value] = []
                        hash_groups[hash_value].append(file_info)
                        
                        progress.bytes_processed += file_size
                        self.file_processed.emit(file_info)

                        # Mark as duplicate if we now have multiple files with same hash
                        if len(hash_groups[hash_value]) >= 2:
                            for info in hash_groups[hash_value]:
                                info.is_duplicate = True
                                info.status = "Duplicate"
                            self.duplicates_updated.emit(hash_groups[hash_value][:])
                    
                except Exception as e:
                    progress.errors_count += 1
                    logger.warning(f"Error processing {file_path}: {e}")
                
                # Emit progress updates every 10 files or every second
                if progress.files_scanned % 10 == 0 or time.time() - getattr(self, '_last_progress_time', 0) > 1:
                    self.progress_updated.emit(progress)
                    self._last_progress_time = time.time()
        
        # Final statistics
        duplicates_count = sum(len(files) for files in hash_groups.values() if len(files) > 1)
        progress.duplicates_found = duplicates_count
        progress.files_scanned = progress.total_files
        progress.current_file = ""
        self.progress_updated.emit(progress)
        
        duration = time.time() - self._start_time
        logger.info(f"Scan completed in {duration:.2f} seconds")
        logger.info(f"Found {duplicates_count} duplicate files")
        logger.info(f"Scan speed: {progress.total_files / duration:.1f} files/second")
        
        self.scan_completed.emit(hash_groups)

    def run_missing_scan(self):
        """Identify files missing from destination folders compared to all sources."""
        sources = self.missing_sources if self.missing_sources else self.folders[:1]
        destinations = self.missing_destinations if self.missing_destinations else self.folders[1:]

        if not sources or not destinations:
            self.error_occurred.emit("Select at least one source and one destination folder")
            return

        # Collect all source files
        try:
            source_files: List[Path] = []
            for folder in sources:
                for file_path in folder.rglob("*"):
                    if self.is_cancelled:
                        return
                    if not file_path.is_file() or file_path.is_symlink():
                        continue
                    try:
                        size = file_path.stat().st_size
                    except (OSError, PermissionError):
                        continue
                    if self.skip_small_threshold and size < self.skip_small_threshold:
                        continue
                    source_files.append(file_path)
        except (OSError, PermissionError) as exc:
            self.error_occurred.emit(f"Cannot access source folder: {exc}")
            return

        progress = ScanProgress(total_files=len(source_files))
        self.progress_updated.emit(progress)

        # Build destination index (union of all destination folders)
        destination_indexes: List[Tuple[Path, Set[Tuple[str, int]]]] = []
        aggregate_index: Set[Tuple[str, int]] = set()
        for folder in destinations:
            if self.is_cancelled:
                return
            key_set: Set[Tuple[str, int]] = set()
            try:
                for file_path in folder.rglob("*"):
                    if self.is_cancelled:
                        return
                    if file_path.is_file() and not file_path.is_symlink():
                        try:
                            size = file_path.stat().st_size
                        except (OSError, PermissionError):
                            continue
                        if self.skip_small_threshold and size < self.skip_small_threshold:
                            continue
                        key = (file_path.name.lower(), size)
                        key_set.add(key)
                        aggregate_index.add(key)
            except (OSError, PermissionError) as exc:
                self.error_occurred.emit(f"Cannot access destination folder {folder}: {exc}")
            destination_indexes.append((folder, key_set))
            logger.info(f"Indexed {len(key_set)} files in destination folder {folder}")

        if not aggregate_index:
            self.error_occurred.emit("No files found in destination folders")
            return

        missing_count = 0
        for file_path in source_files:
            if self.is_cancelled:
                return

            while self.is_paused and not self.is_cancelled:
                time.sleep(0.1)

            try:
                size = file_path.stat().st_size
            except (OSError, PermissionError):
                continue

            if self.skip_small_threshold and size < self.skip_small_threshold:
                continue

            progress.files_scanned += 1
            progress.current_file = str(file_path)

            if progress.files_scanned % 100 == 0:
                self.progress_updated.emit(progress)

            key = (file_path.name.lower(), size)
            if key not in aggregate_index:
                missing_count += 1
                info = FileInfo(
                    path=file_path,
                    size=size,
                    hash_value=f"missing::{file_path.resolve()}",
                    status="Missing",
                    missing_locations=[str(folder) for folder, _ in destination_indexes],
                )
                self.missing_found.emit(info)

        progress.files_scanned = progress.total_files
        progress.current_file = ""
        progress.duplicates_found = missing_count
        self.progress_updated.emit(progress)

        duration = time.time() - self._start_time
        logger.info(f"Missing scan completed in {duration:.2f} seconds, found {missing_count} missing files")

        self.scan_completed.emit({})
        
        # Important: The thread should exit normally here, which will trigger scan_finished


class SystemMessagesWidget(QWidget):
    """Widget for displaying system messages with improved functionality."""
    
    def __init__(self):
        super().__init__()
        self.max_messages = 1000  # Limit message history to prevent memory bloat
        self.message_count = 0
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Title with message count
        self.title = QLabel("System Messages (0)")
        self.title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.title)

        # Messages text area
        self.messages = QTextEdit()
        self.messages.setReadOnly(True)
        # Note: PySide6 doesn't have setMaximumBlockCount, we'll manage message count manually
        self.messages.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.messages, 1)

        # Controls
        controls_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_messages)
        controls_layout.addWidget(clear_btn)
        
        # Export messages button
        export_btn = QPushButton("Export")
        export_btn.setToolTip("Export messages to file")
        export_btn.clicked.connect(self.export_messages)
        controls_layout.addWidget(export_btn)
        
        controls_layout.addStretch()
        layout.addWidget(QWidget())  # Spacer
        layout.addLayout(controls_layout)
    
    def add_message(self, message: str, level: str = "INFO"):
        """Add a message with timestamp and level, managing message count manually."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"
        
        # Add the message
        self.messages.append(formatted_message)
        
        self.message_count += 1
        self.title.setText(f"System Messages ({self.message_count})")
        
        # Manually limit message count since PySide6 doesn't have setMaximumBlockCount
        if self.message_count > self.max_messages:
            # Get all text and keep only the last max_messages lines
            all_text = self.messages.toPlainText()
            lines = all_text.split('\n')
            if len(lines) > self.max_messages:
                # Keep only the last max_messages lines
                kept_lines = lines[-self.max_messages:]
                self.messages.clear()
                self.messages.setPlainText('\n'.join(kept_lines))
                self.message_count = len(kept_lines)
                self.title.setText(f"System Messages ({self.message_count})")
        
        # Auto-scroll to bottom
        scrollbar = self.messages.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def add_error(self, message: str):
        """Add an error message."""
        self.add_message(message, "ERROR")
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.add_message(message, "WARN")
    
    def clear_messages(self):
        """Clear all messages."""
        self.messages.clear()
        self.message_count = 0
        self.title.setText("System Messages (0)")
    
    def export_messages(self):
        """Export messages to a text file."""
        if self.message_count == 0:
            QMessageBox.information(self, "No Messages", "No messages to export.")
            return
        
        filename = f"deduplicator_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Messages", filename, "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.messages.toPlainText())
                self.add_message(f"Messages exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Could not export messages:\n{e}")


class SimpleDeduplicatorApp(QMainWindow):
    """Main application window with performance optimizations."""
    
    scan_cleanup_requested = Signal()

    def __init__(self):
        super().__init__()
        self.settings = QSettings("SimpleDeduplicator", "Settings")
        self.hash_worker = HashWorker()
        self.worker_thread: Optional[QThread] = None
        self.file_data: Dict[str, List[FileInfo]] = {}
        self.scan_start_time: Optional[float] = None
        self.current_mode = "duplicates"
        self.missing_found_counter = 0
        self.missing_source: List[Path] = []
        self.missing_destinations: List[Path] = []
        self._cleanup_status_message = "Ready"
        
        # Performance tracking
        self.last_update_time = 0
        self.update_throttle = 0.1  # Minimum time between UI updates (seconds)
        self._cleanup_requested = False
        self._cleanup_in_progress = False
        self.shared_owner = "duplicates"

        self.init_ui()
        self.setup_connections()
        self.apply_theme()
        self.setup_cleanup_timer()
        self.restore_settings()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Simple Deduplicator v1.1")
        self.setGeometry(100, 100, 1400, 900)  # Slightly larger default size
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        splitter.addWidget(left_container)

        # Tab widget for different modes
        tabs = QTabWidget()
        self.tabs = tabs
        tabs.currentChanged.connect(self.on_tab_changed)
        left_layout.addWidget(tabs)

        # Shared widgets
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(True)
        self.progress_bar.setTextVisible(True)  # Show percentage
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%")
        
        self.results_table = QTableWidget()
        self.configure_results_table()
        self.bottom_controls = self.create_bottom_controls()

        # Duplicates tab
        duplicates_tab = QWidget()
        duplicates_layout = QVBoxLayout(duplicates_tab)
        self.duplicates_layout = duplicates_layout
        duplicates_controls = self.create_duplicates_controls()
        duplicates_layout.addWidget(duplicates_controls)
        duplicates_layout.addWidget(self.progress_bar)
        duplicates_layout.addWidget(self.results_table)
        duplicates_layout.addWidget(self.bottom_controls)
        tabs.addTab(duplicates_tab, "Find Duplicates")

        # Missing files tab
        missing_tab = QWidget()
        missing_layout = QVBoxLayout(missing_tab)
        self.missing_layout = None  # Missing tab manages shared widgets via splitter
        missing_splitter = QSplitter(Qt.Orientation.Vertical)
        self.missing_splitter = missing_splitter
        missing_layout.addWidget(missing_splitter)

        missing_controls = self.create_missing_panel_controls()
        missing_controls_container = QWidget()
        controls_layout = QVBoxLayout(missing_controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(missing_controls)
        controls_layout.addStretch()
        missing_splitter.addWidget(missing_controls_container)

        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_layout.setContentsMargins(0, 0, 0, 0)
        self.missing_results_placeholder = QWidget()
        self.missing_results_placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        results_layout.addWidget(self.missing_results_placeholder)
        missing_splitter.addWidget(results_container)

        self.missing_results_layout = results_layout

        missing_splitter.setStretchFactor(0, 1)
        missing_splitter.setStretchFactor(1, 2)
        missing_splitter.setSizes([360, 520])
        tabs.addTab(missing_tab, "Find Missing")

        self.shared_widgets = [self.progress_bar, self.results_table, self.bottom_controls]
        self.shared_owner = "duplicates"

        # Right panel (system messages)
        self.system_messages = SystemMessagesWidget()
        splitter.addWidget(self.system_messages)

        # Set splitter proportions
        splitter.setSizes([1000, 400])
        
        self._tune_layout_spacing()

        # Enhanced status bar
        self.statusBar().showMessage("Ready - Select folders to begin")
        self.status_label = QLabel("Ready")
        self.statusBar().addPermanentWidget(self.status_label)

    def create_duplicates_controls(self) -> QWidget:
        """Create controls for duplicate finding tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls_frame = self.create_controls_section()
        layout.addWidget(controls_frame)

        return widget

    def _tune_layout_spacing(self):
        """Ensure layouts have comfortable spacing in both themes."""
        for layout in self.findChildren(QLayout):
            if layout:
                layout.setContentsMargins(8, 8, 8, 8)
                layout.setSpacing(6)

    def setup_cleanup_timer(self):
        """Periodically remove rows for files deleted outside the app."""
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.setInterval(15000)  # 15 seconds (less frequent for better performance)
        self.cleanup_timer.timeout.connect(self.prune_missing_files)
        self.cleanup_timer.start()

    def prune_missing_files(self):
        """Check for rows referencing files no longer on disk and drop them."""
        if self.results_table.rowCount() == 0:
            return
            
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

        # Remove rows in reverse order to maintain indices
        for row, file_info in reversed(rows_to_remove):
            self.results_table.removeRow(row)
            self._remove_file_from_data(file_info)

        self.update_selection_info()
        
        if len(rows_to_remove) <= 5:
            removed_names = ", ".join(file_info.path.name for _, file_info in rows_to_remove)
            self.system_messages.add_message(f"Removed missing files: {removed_names}")
        else:
            self.system_messages.add_message(f"Removed {len(rows_to_remove)} missing files from results")

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
        self._cleanup_empty_hash(file_info.hash_value)

    def _cleanup_empty_hash(self, hash_value: str):
        """Clean up hash groups that no longer have duplicates."""
        files = self.file_data.get(hash_value)
        if not files:
            self.file_data.pop(hash_value, None)
            return

        if self.current_mode != "duplicates":
            return

        if len(files) <= 1:
            # No longer duplicates, remove from display
            if files:
                survivor = files[0]
                row = self._row_for_path(survivor.path)
                if row is not None:
                    self.results_table.removeRow(row)
            self.file_data.pop(hash_value, None)

    def _set_status_item(self, status_item: QTableWidgetItem, file_info: FileInfo):
        """Set status item with appropriate styling."""
        if not status_item:
            return

        status = getattr(file_info, "status", "Unknown")

        if file_info.error == "Missing":
            status_item.setText("File Missing")
            status_item.setBackground(QColor(220, 220, 220))
        elif status == "Missing":
            status_item.setText("Missing from Backup")
            status_item.setBackground(QColor(255, 230, 150))
            if file_info.missing_locations:
                tooltip = "Missing from:\n" + "\n".join(file_info.missing_locations)
                status_item.setToolTip(tooltip)
        elif status == "Duplicate" or file_info.is_duplicate:
            status_item.setText("Duplicate")
            status_item.setBackground(QColor(255, 200, 200))
        elif status == "Unique":
            status_item.setText("Unique")
            status_item.setBackground(QColor(240, 255, 240))
        else:
            status_item.setText(status)
            status_item.setBackground(QColor(245, 245, 245))

    def _row_for_path(self, path: Path) -> Optional[int]:
        """Find the row index for a given file path."""
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if not item:
                continue
            file_info: Optional[FileInfo] = item.data(Qt.ItemDataRole.UserRole)
            if file_info and file_info.path == path:
                return row
        return None

    def _refresh_row(self, row: int, file_info: FileInfo):
        """Refresh a table row with updated file information."""
        items = [
            self.results_table.item(row, col) 
            for col in range(self.results_table.columnCount())
        ]
        
        if items[0]:  # Filename
            items[0].setText(file_info.path.name)
            items[0].setToolTip(str(file_info.path))
            items[0].setData(Qt.ItemDataRole.UserRole, file_info)

        if items[1]:  # Path
            items[1].setText(str(file_info.path.parent))
            items[1].setToolTip(str(file_info.path))

        if items[2]:  # Size
            items[2].setText(self.format_size(file_info.size))
            items[2].setData(Qt.ItemDataRole.UserRole, file_info.size)

        if items[3]:  # Hash
            if getattr(file_info, "status", "") == "Missing":
                items[3].setText("Missing")
            else:
                items[3].setText(file_info.hash_value[:16] + "...")
            items[3].setToolTip(file_info.hash_value)

        if items[4]:  # Status
            self._set_status_item(items[4], file_info)

    def update_duplicates_view(self, files: List[FileInfo]):
        """Update the duplicates view with new duplicate files."""
        if not files:
            return

        hash_value = files[0].hash_value

        # Filter out files that no longer exist
        existing_files = [info for info in files if info.path.exists()]
        if not existing_files:
            self.file_data.pop(hash_value, None)
            return

        # Update file status
        for info in existing_files:
            info.is_duplicate = True
            info.status = "Duplicate"

        self.file_data[hash_value] = existing_files

        # Update or add files to table
        for file_info in existing_files:
            row = self._row_for_path(file_info.path)
            if row is None:
                self.add_file_to_table(file_info)
            else:
                self._refresh_row(row, file_info)

        self.update_selection_info()

    def handle_missing_found(self, file_info: FileInfo):
        """Handle a missing file found during comparison."""
        if self.current_mode != "missing":
            return

        key = file_info.hash_value
        file_info.status = "Missing"

        # Check if we already have this file in our data
        existing = self.file_data.get(key, [])
        for existing_info in existing:
            if existing_info.path == file_info.path:
                existing_info.missing_locations = file_info.missing_locations
                existing_info.status = "Missing"
                row = self._row_for_path(existing_info.path)
                if row is not None:
                    self._refresh_row(row, existing_info)
                break
        else:
            self.file_data[key] = [file_info]
            self.add_file_to_table(file_info)

        self.missing_found_counter += 1
        self.update_selection_info()

    def create_controls_section(self) -> QFrame:
        """Create the main controls section with improved layout."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(frame)
        
        # Folder selection
        folder_layout = QHBoxLayout()
        self.select_folders_btn = QPushButton("Select Folders")
        self.select_folders_btn.clicked.connect(self.select_folders)
        folder_layout.addWidget(self.select_folders_btn)
        
        self.selected_folders_label = QLabel("No folders selected")
        self.selected_folders_label.setWordWrap(True)
        folder_layout.addWidget(self.selected_folders_label, 1)
        layout.addLayout(folder_layout)
        
        # Algorithm and settings
        settings_layout = QHBoxLayout()
        
        # Hash algorithm selection
        settings_layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        algorithms = ["MD5", "SHA256", "SHA1", "CRC32"]
        if HAS_BLAKE3:
            algorithms.append("BLAKE3")
        if HAS_XXHASH:
            algorithms.append("XXHash")
        self.algorithm_combo.addItems(algorithms)
        self.algorithm_combo.setCurrentText("MD5")  # Default to MD5
        self.algorithm_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        settings_layout.addWidget(self.algorithm_combo)
        
        # Chunk size
        settings_layout.addWidget(QLabel("Chunk (KB):"))
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(1, 1024)
        self.chunk_size_spin.setValue(64)  # Increased default for better performance
        self.chunk_size_spin.setSuffix(" KB")
        settings_layout.addWidget(self.chunk_size_spin)
        
        # Skip small files option
        self.skip_small_cb = QCheckBox("Skip files < 1KB")
        self.skip_small_cb.setChecked(False)
        self.skip_small_cb.setToolTip("Skip very small files that are unlikely to be meaningful duplicates")
        settings_layout.addWidget(self.skip_small_cb)
        
        settings_layout.addStretch()
        layout.addLayout(settings_layout)
        
        # Scan controls
        scan_layout = QHBoxLayout()
        self.start_scan_btn = QPushButton("Start Scan")
        self.start_scan_btn.clicked.connect(self.start_scan)
        self.start_scan_btn.setMinimumHeight(32)
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
        
        # Save/Load controls
        file_controls = QHBoxLayout()
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setToolTip("Save scan results to JSON file")
        file_controls.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load Results")
        self.load_btn.clicked.connect(self.open_results)
        self.load_btn.setToolTip("Load previously saved scan results")
        file_controls.addWidget(self.load_btn)
        
        file_controls.addStretch()
        
        scan_layout.addLayout(file_controls)
        layout.addLayout(scan_layout)

        return frame

    def create_missing_panel_controls(self) -> QWidget:
        """Create controls for the missing files panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Source selection
        source_group = QVBoxLayout()
        source_button_row = QHBoxLayout()
        self.missing_source_btn = QPushButton("Add Source Folders")
        self.missing_source_btn.clicked.connect(self.select_missing_source)
        source_button_row.addWidget(self.missing_source_btn)

        self.remove_source_btn = QPushButton("Remove Selected")
        self.remove_source_btn.clicked.connect(self.remove_selected_missing_source)
        self.remove_source_btn.setEnabled(False)
        source_button_row.addWidget(self.remove_source_btn)
        source_button_row.addStretch()
        source_group.addLayout(source_button_row)

        self.missing_source_list = QListWidget()
        self.missing_source_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.missing_source_list.itemSelectionChanged.connect(
            lambda: self.remove_source_btn.setEnabled(bool(self.missing_source_list.selectedItems()))
        )
        self.missing_source_list.installEventFilter(self)
        source_group.addWidget(self.missing_source_list)
        layout.addLayout(source_group)

        # Destination selection
        dest_group = QVBoxLayout()
        dest_button_row = QHBoxLayout()
        self.missing_dest_btn = QPushButton("Add Destination Folders")
        self.missing_dest_btn.clicked.connect(self.select_missing_destinations)
        dest_button_row.addWidget(self.missing_dest_btn)

        self.remove_dest_btn = QPushButton("Remove Selected")
        self.remove_dest_btn.clicked.connect(self.remove_selected_missing_destinations)
        self.remove_dest_btn.setEnabled(False)
        dest_button_row.addWidget(self.remove_dest_btn)
        dest_button_row.addStretch()
        dest_group.addLayout(dest_button_row)

        self.missing_dest_list = QListWidget()
        self.missing_dest_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.missing_dest_list.itemSelectionChanged.connect(
            lambda: self.remove_dest_btn.setEnabled(bool(self.missing_dest_list.selectedItems()))
        )
        self.missing_dest_list.installEventFilter(self)
        dest_group.addWidget(self.missing_dest_list)
        layout.addLayout(dest_group)

        # Scan and control buttons
        button_layout = QHBoxLayout()
        self.missing_scan_btn = QPushButton("Find Missing Files")
        self.missing_scan_btn.setToolTip("Compare source folders against destinations")
        self.missing_scan_btn.clicked.connect(self.start_missing_scan)
        self.missing_scan_btn.setMinimumHeight(32)
        button_layout.addWidget(self.missing_scan_btn)
        
        # Add pause, stop, and clear buttons to missing tab too
        self.missing_pause_btn = QPushButton("Pause")
        self.missing_pause_btn.clicked.connect(self.pause_scan)
        self.missing_pause_btn.setEnabled(False)
        button_layout.addWidget(self.missing_pause_btn)
        
        self.missing_stop_btn = QPushButton("Stop")
        self.missing_stop_btn.clicked.connect(self.stop_scan)
        self.missing_stop_btn.setEnabled(False)
        button_layout.addWidget(self.missing_stop_btn)
        
        self.missing_clear_btn = QPushButton("Clear Results")
        self.missing_clear_btn.clicked.connect(self.clear_results)
        button_layout.addWidget(self.missing_clear_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)

        layout.addStretch()

        # Initial population (handles restoring selections)
        self._refresh_missing_list(self.missing_source_list, self.missing_source)
        self._refresh_missing_list(self.missing_dest_list, self.missing_destinations)
        return panel

    def select_missing_source(self):
        """Select source folders for missing file comparison."""
        folders = self._select_multiple_folders("Select Source Folders")
        if folders:
            new_paths = [Path(folder) for folder in folders if Path(folder).is_dir()]
            added = 0
            for path in new_paths:
                if path not in self.missing_source:
                    self.missing_source.append(path)
                    added += 1
            if added:
                self.system_messages.add_message(f"Added {added} source folder(s)")
                self._refresh_missing_list(self.missing_source_list, self.missing_source)
            else:
                self.system_messages.add_warning("Selected source folders are already in the list")

    def select_missing_destinations(self):
        """Select destination folders for missing file comparison."""
        folders = self._select_multiple_folders("Select Destination Folders")
        if folders:
            new_paths = [Path(folder) for folder in folders if Path(folder).is_dir()]
            added = 0
            for path in new_paths:
                if path not in self.missing_destinations:
                    self.missing_destinations.append(path)
                    added += 1
            if added:
                self.system_messages.add_message(f"Added {added} destination folder(s)")
                self._refresh_missing_list(self.missing_dest_list, self.missing_destinations)
            else:
                self.system_messages.add_warning("Selected destination folders are already in the list")

    def _select_multiple_folders(self, title: str) -> List[Path]:
        """Common method for selecting multiple folders."""
        dialog = QFileDialog(self, title)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontResolveSymlinks, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

        # Enable multi-selection
        for view_class in (QListView, QTreeView):
            for view in dialog.findChildren(view_class):
                view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        if dialog.exec():
            return [Path(path) for path in dialog.selectedFiles() if Path(path).is_dir()]
        return []

    def _refresh_missing_list(self, widget: QListWidget, folders: List[Path]):
        widget.blockSignals(True)
        widget.clear()
        for path in folders:
            item = QListWidgetItem(str(path))
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            widget.addItem(item)
        widget.blockSignals(False)
        if widget is getattr(self, 'missing_source_list', None):
            self.remove_source_btn.setEnabled(False)
        if widget is getattr(self, 'missing_dest_list', None):
            self.remove_dest_btn.setEnabled(False)

    def remove_selected_missing_source(self):
        """Remove selected source folders from the list."""
        selected_items = self.missing_source_list.selectedItems()
        if not selected_items:
            return
        removed = 0
        for item in selected_items:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path in self.missing_source:
                self.missing_source.remove(path)
                removed += 1
        if removed:
            self.system_messages.add_message(f"Removed {removed} source folder(s)")
            self._refresh_missing_list(self.missing_source_list, self.missing_source)
        self.remove_source_btn.setEnabled(False)

    def remove_selected_missing_destinations(self):
        """Remove selected destination folders from the list."""
        selected_items = self.missing_dest_list.selectedItems()
        if not selected_items:
            return
        removed = 0
        for item in selected_items:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path in self.missing_destinations:
                self.missing_destinations.remove(path)
                removed += 1
        if removed:
            self.system_messages.add_message(f"Removed {removed} destination folder(s)")
            self._refresh_missing_list(self.missing_dest_list, self.missing_destinations)
        self.remove_dest_btn.setEnabled(False)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if obj is getattr(self, 'missing_source_list', None):
                self.remove_selected_missing_source()
                return True
            if obj is getattr(self, 'missing_dest_list', None):
                self.remove_selected_missing_destinations()
                return True
        return super().eventFilter(obj, event)

    def _format_folder_list(self, folders: List[Path]) -> str:
        """Format a list of folders for display."""
        if not folders:
            return "No folders selected"
        if len(folders) == 1:
            return str(folders[0])
        return f"{folders[0]} (+{len(folders) - 1} more)"
    
    def configure_results_table(self):
        """Configure the results table with improved performance."""
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Filename", "Path", "Size", "Hash", "Status", "Action"
        ])
        
        # Configure table performance
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setSortingEnabled(True)
        self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self.show_results_context_menu)
        
        # Set default column widths
        header = self.results_table.horizontalHeader()
        header.setStretchLastSection(False)
        self.results_table.setColumnWidth(0, 200)  # Filename
        self.results_table.setColumnWidth(1, 350)  # Path  
        self.results_table.setColumnWidth(2, 100)  # Size
        self.results_table.setColumnWidth(3, 120)  # Hash
        self.results_table.setColumnWidth(4, 120)  # Status
        self.results_table.setColumnWidth(5, 100)  # Action

        # Make columns resizable
        for i in range(6):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
        
        # Optimize row height
        self.results_table.verticalHeader().setDefaultSectionSize(28)
        self.results_table.verticalHeader().hide()

    def move_shared_widgets(self, target_layout: QVBoxLayout):
        """Move shared widgets between tabs."""
        for widget in self.shared_widgets:
            widget.setParent(None)
            target_layout.addWidget(widget)

    def on_tab_changed(self, index: int):
        """Handle tab change to move shared widgets."""
        if index == 0:  # Duplicates tab
            self.current_mode = "duplicates"
            if self.shared_owner != "duplicates":
                self.move_shared_widgets(self.duplicates_layout)
                self.shared_owner = "duplicates"
                # Restore placeholder in missing layout so splitter retains space
                if hasattr(self, 'missing_results_layout') and hasattr(self, 'missing_results_placeholder'):
                    if self.missing_results_placeholder.parent() is None:
                        self.missing_results_layout.addWidget(self.missing_results_placeholder)
        else:  # Missing tab
            self.current_mode = "missing"
            if self.shared_owner != "missing":
                if hasattr(self, 'missing_results_layout') and self.missing_results_layout is not None:
                    if hasattr(self, 'missing_results_placeholder') and self.missing_results_placeholder.parent() is not None:
                        self.missing_results_layout.removeWidget(self.missing_results_placeholder)
                        self.missing_results_placeholder.setParent(None)
                    self.move_shared_widgets(self.missing_results_layout)
            self.shared_owner = "missing"

    def create_bottom_controls(self) -> QWidget:
        """Create bottom control buttons with improved layout."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Selection info
        self.selection_info_label = QLabel("0 files selected")
        layout.addWidget(self.selection_info_label)
        
        # Bulk operations
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_files)
        layout.addWidget(self.select_all_btn)
        
        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self.select_no_files)
        layout.addWidget(self.select_none_btn)
        
        # Delete button
        self.delete_selected_btn = QPushButton("Delete Selected")
        self.delete_selected_btn.clicked.connect(self.delete_selected_files)
        self.delete_selected_btn.setEnabled(False)
        layout.addWidget(self.delete_selected_btn)
        
        layout.addStretch()
        
        # Theme toggle
        self.theme_btn = QPushButton("🌙")
        self.theme_btn.setToolTip("Toggle Dark/Light Theme")
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.theme_btn.setMaximumSize(QSize(40, 40))
        layout.addWidget(self.theme_btn)
        
        return widget
    
    def select_all_files(self):
        """Select all files in the results table."""
        self.results_table.selectAll()
    
    def select_no_files(self):
        """Deselect all files in the results table."""
        self.results_table.clearSelection()
    
    def setup_connections(self):
        """Set up signal connections."""
        # Worker signals
        self.hash_worker.progress_updated.connect(self.update_progress)
        self.hash_worker.file_processed.connect(self.file_processed)
        self.hash_worker.scan_completed.connect(self.scan_completed)
        self.hash_worker.error_occurred.connect(self.handle_error)
        self.hash_worker.duplicates_updated.connect(self.update_duplicates_view)
        self.hash_worker.missing_found.connect(self.handle_missing_found)

        # UI connections
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.results_table.itemSelectionChanged.connect(self.update_selection_info)
    
    def select_folders(self):
        """Select folders for duplicate scanning with improved dialog."""
        folders = self._select_multiple_folders("Select Folders to Scan for Duplicates")
        
        if folders:
            self.selected_folders = folders
            display_text = self._format_folder_list(folders)
            self.selected_folders_label.setText(f"Selected: {display_text}")
            self.selected_folders_label.setToolTip("\n".join(str(folder) for folder in folders))

            # Count files asynchronously to avoid UI blocking
            QTimer.singleShot(100, lambda: self._count_files_async(folders))

    def _count_files_async(self, folders: List[Path]):
        """Count files in selected folders without blocking UI."""
        try:
            total_files = 0
            for folder in folders:
                folder_files = sum(1 for f in folder.rglob("*") if f.is_file())
                total_files += folder_files
                
            folder_word = "folder" if len(folders) == 1 else "folders"
            self.system_messages.add_message(
                f"{len(folders)} {folder_word} selected: {total_files:,} files found"
            )
        except Exception as e:
            self.system_messages.add_error(f"Error counting files: {e}")
    
    def start_scan(self):
        """Start the file scanning process with improved validation."""
        if self.current_mode != "duplicates":
            self.tabs.setCurrentIndex(0)
            
        if not hasattr(self, 'selected_folders') or not self.selected_folders:
            self.system_messages.add_error("No folders selected for scanning")
            QMessageBox.warning(self, "Warning", "Please select folders to scan first.")
            return

        # Clear previous results
        self.clear_results()

        # Prepare worker
        algorithm = self.algorithm_combo.currentText().lower()
        chunk_size = self.chunk_size_spin.value() * 1024
        
        self.current_mode = "duplicates"
        skip_threshold = 1024 if self.skip_small_cb.isChecked() else 0
        self.hash_worker.set_parameters(
            self.selected_folders,
            algorithm,
            chunk_size,
            mode="duplicates",
            skip_small_threshold=skip_threshold,
        )

        # Reset cleanup state for the new run
        self._cleanup_requested = False
        self._cleanup_in_progress = False
        
        # Set up worker thread
        self.worker_thread = QThread()
        self.hash_worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker_thread.started.connect(self.hash_worker.run_scan)
        self.worker_thread.finished.connect(self._on_worker_thread_finished)
        try:
            self.scan_cleanup_requested.disconnect(self._on_scan_cleanup_requested)
        except (TypeError, RuntimeError):
            pass
        self.scan_cleanup_requested.connect(self._on_scan_cleanup_requested)
        
        # Update UI state
        self._set_scan_ui_state(True)
        
        self.scan_start_time = time.time()
        self.system_messages.add_message(
            f"Duplicate scan started using {algorithm.upper()} algorithm"
        )
        
        # Start the thread
        self.worker_thread.start()

    def start_missing_scan(self):
        """Start missing file comparison with improved validation."""
        if self.current_mode != "missing":
            self.tabs.setCurrentIndex(1)
            
        if not self.missing_source:
            QMessageBox.warning(self, "Missing Source", "Select one or more source folders.")
            return
        if not self.missing_destinations:
            QMessageBox.warning(self, "Missing Destination", "Select one or more destination folders.")
            return

        # Clear previous results
        self.clear_results()
        
        # Combine folders for worker
        self.selected_folders = list(self.missing_source) + list(self.missing_destinations)
        chunk_size = self.chunk_size_spin.value() * 1024

        self.current_mode = "missing"
        self.missing_found_counter = 0
        skip_threshold = 1024 if self.skip_small_cb.isChecked() else 0
        self.hash_worker.set_parameters(
            self.selected_folders,
            "md5",
            chunk_size,
            mode="missing",
            skip_small_threshold=skip_threshold,
            sources=self.missing_source,
            destinations=self.missing_destinations,
        )

        # Reset cleanup state for the new run
        self._cleanup_requested = False
        self._cleanup_in_progress = False

        # Set up worker thread
        self.worker_thread = QThread()
        self.hash_worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.hash_worker.run_scan)
        self.worker_thread.finished.connect(self._on_worker_thread_finished)
        try:
            self.scan_cleanup_requested.disconnect(self._on_scan_cleanup_requested)
        except (TypeError, RuntimeError):
            pass
        self.scan_cleanup_requested.connect(self._on_scan_cleanup_requested)

        # Update UI
        self._set_scan_ui_state(True)
        
        self.scan_start_time = time.time()
        self.system_messages.add_message("Missing file comparison started")
        self.worker_thread.start()

    def _set_scan_ui_state(self, scanning: bool):
        """Set UI state for scanning or idle - handles both tabs."""
        # Duplicates tab buttons
        self.start_scan_btn.setEnabled(not scanning)
        self.pause_btn.setEnabled(scanning)
        self.stop_btn.setEnabled(scanning)
        
        # Missing tab buttons
        self.missing_scan_btn.setEnabled(not scanning)
        if hasattr(self, 'missing_pause_btn'):
            self.missing_pause_btn.setEnabled(scanning)
        if hasattr(self, 'missing_stop_btn'):
            self.missing_stop_btn.setEnabled(scanning)
        
        if scanning:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0% (0/0)")
            self.statusBar().showMessage("Scanning...")
        else:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0%")
            self.statusBar().showMessage(self._cleanup_status_message or "Ready")

    def pause_scan(self):
        """Pause or resume the scanning process with synchronized button text."""
        if self.hash_worker.is_paused:
            self.hash_worker.resume()
            # Update both pause buttons
            self.pause_btn.setText("Pause")
            if hasattr(self, 'missing_pause_btn'):
                self.missing_pause_btn.setText("Pause")
            self.system_messages.add_message("Scan resumed")
            self.statusBar().showMessage("Scanning...")
        else:
            self.hash_worker.pause()
            # Update both pause buttons
            self.pause_btn.setText("Resume")
            if hasattr(self, 'missing_pause_btn'):
                self.missing_pause_btn.setText("Resume")
            self.system_messages.add_message("Scan paused")
            self.statusBar().showMessage("Paused")
    
    def stop_scan(self):
        """Stop the scanning process."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.hash_worker.cancel()
            self.worker_thread.quit()
            self.worker_thread.wait(5000)  # Wait up to 5 seconds
            self.system_messages.add_message("Scan stopped by user")
            self._cleanup_status_message = "Scan stopped"
            self.statusBar().showMessage(self._cleanup_status_message)
            self._request_scan_cleanup()
    
    def update_progress(self, progress: ScanProgress):
        """Update progress with throttling for better performance."""
        current_time = time.time()
        
        # Throttle UI updates for better performance
        if current_time - self.last_update_time < self.update_throttle:
            return
        
        self.last_update_time = current_time

        if progress.total_files > 0:
            percentage = min(100, int((progress.files_scanned / progress.total_files) * 100))
            self.progress_bar.setValue(percentage)
            
            # Enhanced progress text
            progress_text = f"{percentage}% ({progress.files_scanned:,}/{progress.total_files:,})"
            if progress.scan_speed > 0:
                progress_text += f" - {progress.scan_speed:.1f} files/sec"
            self.progress_bar.setFormat(progress_text)

        # Status bar with current file info
        if progress.current_file:
            filename = Path(progress.current_file).name
            status = f"Scanning: {filename}"
            if progress.duplicates_found > 0:
                status += f" | {progress.duplicates_found} duplicates found"
        else:
            status = f"Processed {progress.files_scanned:,} files"
            
        self.statusBar().showMessage(status)
    
    def file_processed(self, file_info: FileInfo):
        """Handle a processed file - only add duplicates to save memory."""
        if file_info.is_duplicate:
            # Only add to table when we know it's a duplicate
            pass  # Will be handled by update_duplicates_view
    
    def scan_completed(self, hash_groups: Dict[str, List[FileInfo]]):
        """Handle scan completion with improved statistics and cleanup."""
        self.progress_bar.setValue(100)
        duration = time.time() - self.scan_start_time if self.scan_start_time else 0

        if self.current_mode == "duplicates":
            self.file_data = hash_groups

            # Clear table and repopulate with only duplicates
            self.results_table.setRowCount(0)
            duplicate_files = 0
            
            for hash_value, files in hash_groups.items():
                if len(files) > 1:  # Only show duplicates
                    for file_info in files:
                        file_info.status = "Duplicate"
                        file_info.is_duplicate = True
                        self.add_file_to_table(file_info)
                        duplicate_files += 1

            duplicate_groups = sum(1 for files in hash_groups.values() if len(files) > 1)
            total_size = sum(
                sum(f.size for f in files[1:])  # Size of duplicates (keep first, delete rest)
                for files in hash_groups.values() if len(files) > 1
            )

            self.system_messages.add_message(f"Duplicate scan completed in {duration:.1f}s")
            self.system_messages.add_message(
                f"Found {duplicate_files:,} duplicate files in {duplicate_groups:,} groups"
            )
            self.system_messages.add_message(
                f"Potential space savings: {self.format_size(total_size)}"
            )
            self._cleanup_status_message = "Duplicate scan completed"
        else:
            # Missing scan completed
            self.system_messages.add_message(f"Missing file comparison completed in {duration:.1f}s")
            self.system_messages.add_message(f"Found {self.missing_found_counter:,} missing files")
            self._cleanup_status_message = "Missing comparison completed"

        self.update_selection_info()
        
        # Request cleanup; let the worker thread finishing drive teardown
        self._request_scan_cleanup()

    def _request_scan_cleanup(self):
        """Mark that cleanup is needed and trigger it when safe."""
        if self._cleanup_requested:
            return
        self._cleanup_requested = True
        if self.worker_thread:
            if self.worker_thread.isRunning():
                # Ask the worker thread's event loop to exit; finished signal will trigger cleanup
                self.worker_thread.quit()
                return
        # Either there is no worker thread or it is already stopped – perform cleanup immediately
        self.scan_cleanup_requested.emit()

    def _on_scan_cleanup_requested(self):
        self._perform_scan_cleanup()

    def _on_worker_thread_finished(self):
        self.scan_cleanup_requested.emit()

    def _perform_scan_cleanup(self):
        """Handle scan thread completion with proper cleanup."""
        if self._cleanup_in_progress:
            return
        if not self._cleanup_requested:
            self._cleanup_requested = True

        self._cleanup_in_progress = True

        # Disconnect signals defensively to prevent duplicate cleanups
        if self.worker_thread:
            try:
                self.worker_thread.started.disconnect(self.hash_worker.run_scan)
            except (TypeError, RuntimeError):
                pass
            try:
                self.worker_thread.finished.disconnect(self._on_worker_thread_finished)
            except (TypeError, RuntimeError):
                pass
        try:
            self.scan_cleanup_requested.disconnect(self._on_scan_cleanup_requested)
        except (TypeError, RuntimeError):
            pass

        # Move worker back to main thread and clean up
        if hasattr(self, 'hash_worker'):
            self.hash_worker.moveToThread(self.thread())  # Move back to main thread
        
        # Clean up worker thread completely
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait(1000)  # Wait up to 1 second
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.worker_thread.wait(500)
            self.worker_thread.deleteLater()
            self.worker_thread = None
        
        # Reset UI state completely
        self._set_scan_ui_state(False)
        self.pause_btn.setText("Pause")
        if hasattr(self, 'missing_pause_btn'):
            self.missing_pause_btn.setText("Pause")

        if hasattr(self, 'hash_worker'):
            self.hash_worker.is_paused = False
            self.hash_worker.is_cancelled = False

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%")
        status_msg = getattr(self, '_cleanup_status_message', "Ready")
        self.statusBar().showMessage(status_msg)
        self._cleanup_status_message = "Ready"

        self._cleanup_requested = False
        self._cleanup_in_progress = False
    
    def add_file_to_table(self, file_info: FileInfo):
        """Add a file to the results table with optimized rendering."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Create items with proper data
        items = [
            QTableWidgetItem(file_info.path.name),  # Filename
            QTableWidgetItem(str(file_info.path.parent)),  # Path
            QTableWidgetItem(self.format_size(file_info.size)),  # Size
            QTableWidgetItem("Missing" if file_info.status == "Missing" else f"{file_info.hash_value[:16]}..."),  # Hash
            QTableWidgetItem(),  # Status (will be set by _set_status_item)
        ]
        
        # Set tooltips and user data
        items[0].setToolTip(str(file_info.path))
        items[0].setData(Qt.ItemDataRole.UserRole, file_info)
        items[1].setToolTip(str(file_info.path))
        items[2].setData(Qt.ItemDataRole.UserRole, file_info.size)
        items[3].setToolTip(file_info.hash_value)
        
        # Add items to table
        for col, item in enumerate(items):
            self.results_table.setItem(row, col, item)
        
        # Set status with styling
        self._set_status_item(items[4], file_info)
        
        # Add delete button
        delete_btn = QPushButton("Delete")
        delete_btn.setMaximumHeight(24)
        delete_btn.clicked.connect(lambda checked=False, btn=delete_btn: self.delete_single_file_by_button(btn))
        self.results_table.setCellWidget(row, 5, delete_btn)

    def show_results_context_menu(self, position):
        """Show context menu for results table."""
        index = self.results_table.indexAt(position)
        if not index.isValid():
            return

        file_info = self._get_file_info_from_row(index.row())
        if not file_info:
            return

        menu = QMenu(self)
        
        # Show in file manager
        if sys.platform == "darwin":
            reveal_text = "Show in Finder"
        elif os.name == "nt":
            reveal_text = "Show in Explorer"
        else:
            reveal_text = "Show in File Manager"
        
        reveal_action = menu.addAction(reveal_text)
        
        # Copy path
        copy_action = menu.addAction("Copy Path")
        
        # Show file properties
        props_action = menu.addAction("Properties")
        
        selected_action = menu.exec(self.results_table.viewport().mapToGlobal(position))

        if selected_action == reveal_action:
            self.reveal_in_file_explorer(file_info.path)
        elif selected_action == copy_action:
            QApplication.clipboard().setText(str(file_info.path))
            self.system_messages.add_message(f"Copied path: {file_info.path}")
        elif selected_action == props_action:
            self.show_file_properties(file_info)

    def show_file_properties(self, file_info: FileInfo):
        """Show file properties dialog."""
        try:
            stat = file_info.path.stat()
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            props = f"""File: {file_info.path.name}
Path: {file_info.path.parent}
Size: {self.format_size(file_info.size)} ({file_info.size:,} bytes)
Modified: {mod_time}
Hash: {file_info.hash_value}
Status: {file_info.status}"""
            
            if file_info.missing_locations:
                props += f"\nMissing from: {', '.join(file_info.missing_locations)}"
            
            QMessageBox.information(self, "File Properties", props)
        except Exception as e:
            QMessageBox.warning(self, "Properties Error", f"Could not read file properties:\n{e}")

    def reveal_in_file_explorer(self, path: Path):
        """Open file location in the system file explorer."""
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", "-R", str(path)], check=False)
            elif os.name == "nt":
                subprocess.run(["explorer", "/select,", str(path)], check=False)
            else:
                # Linux - open parent directory
                parent = path.parent if path.is_file() else path
                subprocess.run(["xdg-open", str(parent)], check=False)
        except Exception as e:
            QMessageBox.warning(self, "Open Failed", f"Could not open file location:\n{e}")

    def _get_file_info_from_row(self, row: int) -> Optional[FileInfo]:
        """Get FileInfo object from table row."""
        if row < 0 or row >= self.results_table.rowCount():
            return None
        item = self.results_table.item(row, 0)
        if not item:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def delete_single_file_by_button(self, button: QPushButton):
        """Resolve row from delete button and remove the file."""
        cell_widget = button.parentWidget()
        if not cell_widget:
            return
        index = self.results_table.indexAt(cell_widget.pos())
        if not index.isValid():
            return
        self.delete_single_file(index.row())

    def delete_single_file(self, row: int):
        """Delete a single file with improved error handling."""
        file_info = self._get_file_info_from_row(row)
        if not file_info:
            return

        # Confirm deletion
        if file_info.path.exists():
            message = f"Move to trash:\n{file_info.path}\n\nSize: {self.format_size(file_info.size)}"
            reply = QMessageBox.question(
                self, "Confirm Delete",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        else:
            reply = QMessageBox.question(
                self, "File Missing",
                f"This file no longer exists on disk:\n{file_info.path}\n\nRemove it from the list?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            if file_info.path.exists():
                self.move_to_trash(file_info.path)
                self.system_messages.add_message(f"Moved to trash: {file_info.path}")
            else:
                self.system_messages.add_warning(f"File already missing: {file_info.path}")
            self.results_table.removeRow(row)
            self._remove_file_from_data(file_info)
            self._cleanup_empty_hash(file_info.hash_value)
            self.update_selection_info()
        except Exception as e:
            self.system_messages.add_error(f"Failed to delete {file_info.path.name}: {e}")
            QMessageBox.critical(self, "Delete Error", f"Could not delete file:\n{e}")
    
    def delete_selected_files(self):
        """Delete all selected files with improved confirmation and progress."""
        selected_rows = {index.row() for index in self.results_table.selectionModel().selectedRows()}
        
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select files to delete.")
            return
        
        # Get file info for selected files
        files_to_delete = []
        total_size = 0
        
        for row in selected_rows:
            file_info = self._get_file_info_from_row(row)
            if file_info:
                files_to_delete.append((row, file_info))
                total_size += file_info.size
        
        if not files_to_delete:
            return
        
        # Enhanced confirmation dialog
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Confirm Bulk Delete")
        msg.setText(f"Move {len(files_to_delete)} files to trash?")
        preview_lines = "\n".join(f"• {info.path}" for _, info in files_to_delete[:10])
        if len(files_to_delete) > 10:
            preview_lines += f"\n... and {len(files_to_delete) - 10} more"
        msg.setInformativeText("Files will be moved to the system trash.")
        msg.setDetailedText(f"Total size: {self.format_size(total_size)}\n\n{preview_lines}")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        
        if msg.exec() != QMessageBox.StandardButton.Yes:
            return
        
        # Delete files with progress
        self.progress_bar.setMaximum(len(files_to_delete))
        self.progress_bar.setValue(0)
        
        deleted_count = 0
        failed_count = 0
        
        # Sort by row in descending order to avoid index issues
        files_to_delete.sort(key=lambda x: x[0], reverse=True)
        
        for i, (row, file_info) in enumerate(files_to_delete):
            try:
                self.move_to_trash(file_info.path)
                self.results_table.removeRow(row)
                self._remove_file_from_data(file_info)
                self._cleanup_empty_hash(file_info.hash_value)
                deleted_count += 1
                self.system_messages.add_message(f"Moved to trash: {file_info.path}")
            except Exception as e:
                self.system_messages.add_error(f"Failed to delete {file_info.path.name}: {e}")
                failed_count += 1
            
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()  # Keep UI responsive
        
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0%")
        self.progress_bar.setMaximum(100)

        # Report results
        if failed_count == 0:
            self.system_messages.add_message(f"Successfully deleted {deleted_count} files")
        else:
            self.system_messages.add_message(
                f"Bulk delete completed: {deleted_count} deleted, {failed_count} failed"
            )
        
        self.update_selection_info()

    def move_to_trash(self, file_path: Path):
        """Move file to trash with improved cross-platform support."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            if sys.platform == "win32":
                # Windows - use winshell if available, otherwise use PowerShell
                try:
                    import winshell
                    winshell.delete_file(str(file_path))
                except ImportError:
                    # Fallback to PowerShell
                    subprocess.run([
                        "powershell", "-Command", 
                        f"Add-Type -AssemblyName Microsoft.VisualBasic; "
                        f"[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile('{file_path}', "
                        f"'OnlyErrorDialogs', 'SendToRecycleBin')"
                    ], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            elif sys.platform == "darwin":  # macOS
                subprocess.run([
                    "osascript", "-e", 
                    f'tell app "Finder" to delete POSIX file "{file_path}"'
                ], check=True)
            else:  # Linux and others
                # Try gio first, then fall back to trash-cli
                try:
                    subprocess.run(["gio", "trash", str(file_path)], check=True, 
                                 capture_output=True, text=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    try:
                        subprocess.run(["trash", str(file_path)], check=True)
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # Last resort - direct deletion with confirmation
                        reply = QMessageBox.question(
                            self, "Trash Not Available",
                            f"System trash is not available. Permanently delete file?\n{file_path}",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            file_path.unlink()
                        else:
                            raise RuntimeError("User cancelled permanent deletion")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to move to trash: {e}")
        except Exception as e:
            raise RuntimeError(f"Trash operation failed: {e}")
    
    def clear_results(self):
        """Clear all results with confirmation if there are many results."""
        row_count = self.results_table.rowCount()
        if row_count > 100:
            reply = QMessageBox.question(
                self, "Clear Results",
                f"Clear {row_count} results?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        self.results_table.setRowCount(0)
        self.file_data.clear()
        self.missing_found_counter = 0
        self.system_messages.add_message("Results cleared")
        self.update_selection_info()

    def save_results(self):
        """Save results to JSON with improved metadata."""
        if not self.file_data:
            QMessageBox.information(self, "No Results", "No scan results to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "missing" if self.current_mode == "missing" else "duplicates"
        default_name = f"deduplicator_{mode}_{timestamp}.json"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Scan Results", default_name, 
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        # Prepare data with enhanced metadata
        data = {
            "version": "1.1",
            "app_version": "1.1",
            "saved_at": datetime.now().isoformat(),
            "mode": self.current_mode,
            "algorithm": self.algorithm_combo.currentText(),
            "chunk_size": self.chunk_size_spin.value(),
            "folders": [str(folder) for folder in getattr(self, "selected_folders", [])],
            "statistics": {
                "total_groups": len(self.file_data),
                "total_files": sum(len(files) for files in self.file_data.values()),
                "duplicate_files": sum(len(files) for files in self.file_data.values() if len(files) > 1),
                "duplicate_groups": sum(1 for files in self.file_data.values() if len(files) > 1),
            },
            "entries": [],
        }

        # Add file entries
        for hash_value, files in self.file_data.items():
            entry = {
                "hash": hash_value,
                "algorithm": self.algorithm_combo.currentText().lower(),
                "files": [
                    {
                        "path": str(file.path),
                        "size": file.size,
                        "duplicate": file.is_duplicate,
                        "status": getattr(file, "status", "Unknown"),
                        "missing_locations": getattr(file, "missing_locations", []),
                    }
                    for file in files
                ],
            }
            data["entries"].append(entry)

        try:
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            
            file_size = Path(file_path).stat().st_size
            self.system_messages.add_message(
                f"Results saved to {Path(file_path).name} ({self.format_size(file_size)})"
            )
        except Exception as e:
            self.system_messages.add_error(f"Failed to save results: {e}")
            QMessageBox.critical(self, "Save Failed", f"Could not save results:\n{e}")

    def open_results(self):
        """Load results from JSON with improved validation."""
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Scan Running", "Please stop the current scan before loading results.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Scan Results", "", 
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Could not load results:\n{e}")
            return

        # Validate data structure
        if not isinstance(data, dict) or "entries" not in data:
            QMessageBox.critical(self, "Invalid File", "The selected file is not a valid results file.")
            return

        entries = data.get("entries", [])
        if not entries:
            QMessageBox.information(self, "No Data", "The selected file contains no results.")
            return

        # Clear current results
        self.clear_results()

        # Restore folders if available
        folders = data.get("folders", [])
        if folders:
            self.selected_folders = [Path(folder) for folder in folders]
            display_text = self._format_folder_list(self.selected_folders)
            self.selected_folders_label.setText(f"Loaded: {display_text}")

        # Load entries
        loaded_missing = False
        missing_count = 0
        duplicate_count = 0

        for entry in entries:
            hash_value = entry.get("hash")
            file_items = entry.get("files", [])
            if not hash_value or not file_items:
                continue

            files: List[FileInfo] = []
            for file_data in file_items:
                path_str = file_data.get("path")
                if not path_str:
                    continue

                file_path = Path(path_str)
                size = file_data.get("size", 0)
                is_duplicate = file_data.get("duplicate", False)
                status = file_data.get("status", "Unknown")
                missing_locations = file_data.get("missing_locations", [])

                file_info = FileInfo(
                    path=file_path,
                    size=size,
                    hash_value=hash_value,
                    is_duplicate=is_duplicate,
                    status=status,
                    missing_locations=missing_locations
                )

                if missing_locations:
                    loaded_missing = True
                    missing_count += 1

                if not file_path.exists():
                    file_info.error = "Missing"

                files.append(file_info)
                
                if is_duplicate:
                    duplicate_count += 1

            if files:
                self.file_data[hash_value] = files
                for file_info in files:
                    self.add_file_to_table(file_info)

        # Set mode based on loaded data
        self.current_mode = "missing" if loaded_missing else "duplicates"
        self.missing_found_counter = missing_count
        
        # Switch to appropriate tab
        self.tabs.setCurrentIndex(1 if loaded_missing else 0)
        
        # Show statistics
        stats = data.get("statistics", {})
        total_files = stats.get("total_files", len([f for files in self.file_data.values() for f in files]))
        
        self.system_messages.add_message(f"Loaded {len(entries)} groups with {total_files} files")
        if loaded_missing:
            self.system_messages.add_message(f"Loaded {missing_count} missing files")
        else:
            self.system_messages.add_message(f"Loaded {duplicate_count} duplicate files")
        
        self.update_selection_info()

    def update_selection_info(self):
        """Update selection information with enhanced statistics."""
        selected_rows = self.results_table.selectionModel().selectedRows()
        selected_count = len(selected_rows)
        
        if selected_count == 0:
            self.selection_info_label.setText("No files selected")
        else:
            # Calculate total size of selected files
            total_size = 0
            for index in selected_rows:
                file_info = self._get_file_info_from_row(index.row())
                if file_info:
                    total_size += file_info.size
            
            size_text = f" ({self.format_size(total_size)})" if total_size > 0 else ""
            self.selection_info_label.setText(f"{selected_count:,} files selected{size_text}")
        
        self.delete_selected_btn.setEnabled(selected_count > 0)
        self.select_all_btn.setEnabled(self.results_table.rowCount() > selected_count)
        self.select_none_btn.setEnabled(selected_count > 0)
    
    def handle_error(self, error_message: str):
        """Handle errors from worker thread."""
        self.system_messages.add_error(error_message)
        logger.error(error_message)
    
    def toggle_theme(self):
        """Toggle between light and dark themes."""
        current_theme = self.settings.value("theme", "light")
        new_theme = "dark" if current_theme == "light" else "light"
        self.settings.setValue("theme", new_theme)
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the selected theme with improved styling."""
        theme = self.settings.value("theme", "light")
        
        if theme == "dark":
            # Enhanced dark theme
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2d2d2d;
                    color: #ffffff;
                }
                QWidget {
                    background-color: #2d2d2d;
                    color: #ffffff;
                }
                QTableWidget {
                    background-color: #3a3a3a;
                    alternate-background-color: #464646;
                    gridline-color: #555555;
                    selection-background-color: #4a90e2;
                    selection-color: #ffffff;
                }
                QTableWidget::item {
                    padding: 6px;
                    border: none;
                }
                QHeaderView::section {
                    background-color: #424242;
                    color: #ffffff;
                    border: 1px solid #555555;
                    padding: 8px;
                    font-weight: bold;
                }
                QPushButton {
                    background-color: #424242;
                    border: 1px solid #555555;
                    color: #ffffff;
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-weight: normal;
                }
                QPushButton:hover {
                    background-color: #4a4a4a;
                    border-color: #666666;
                }
                QPushButton:pressed {
                    background-color: #363636;
                }
                QPushButton:disabled {
                    background-color: #2a2a2a;
                    color: #666666;
                    border-color: #444444;
                }
                QComboBox, QSpinBox {
                    background-color: #424242;
                    border: 1px solid #555555;
                    color: #ffffff;
                    padding: 6px;
                    border-radius: 3px;
                }
                QComboBox::drop-down {
                    border: none;
                }
                QComboBox::down-arrow {
                    image: none;
                    border-left: 4px solid transparent;
                    border-right: 4px solid transparent;
                    border-top: 4px solid #ffffff;
                }
                QComboBox QAbstractItemView {
                    background-color: #424242;
                    color: #ffffff;
                    selection-background-color: #4a90e2;
                    selection-color: #ffffff;
                    border: 1px solid #555555;
                }
                QTextEdit {
                    background-color: #3a3a3a;
                    border: 1px solid #555555;
                    color: #ffffff;
                    border-radius: 3px;
                }
                QProgressBar {
                    border: 1px solid #555555;
                    border-radius: 4px;
                    background-color: #3a3a3a;
                    text-align: center;
                    color: #ffffff;
                }
                QProgressBar::chunk {
                    background-color: #4a90e2;
                    border-radius: 3px;
                }
                QFrame {
                    border: 1px solid #555555;
                    background-color: #2d2d2d;
                    border-radius: 4px;
                }
                QLabel {
                    color: #ffffff;
                }
                QTabWidget::pane {
                    border: 1px solid #555555;
                    background-color: #2d2d2d;
                }
                QTabBar::tab {
                    background-color: #424242;
                    color: #ffffff;
                    padding: 8px 16px;
                    margin-right: 2px;
                    border-radius: 4px 4px 0 0;
                }
                QTabBar::tab:selected {
                    background-color: #4a90e2;
                }
                QTabBar::tab:hover {
                    background-color: #4a4a4a;
                }
                QCheckBox {
                    color: #ffffff;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                }
                QCheckBox::indicator:unchecked {
                    background-color: #424242;
                    border: 1px solid #555555;
                }
                QCheckBox::indicator:checked {
                    background-color: #4a90e2;
                    border: 1px solid #4a90e2;
                }
            """)
            self.theme_btn.setText("☀️")
        else:
            # Light theme (default)
            self.setStyleSheet("")
            self.theme_btn.setText("🌙")
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size with improved precision."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 1)
        
        # Show more precision for smaller sizes
        if i <= 1:  # Bytes and KB
            s = int(s)
        
        return f"{s:,} {size_names[i]}"
    
    def restore_settings(self):
        """Restore application settings."""
        # Restore window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore algorithm selection
        algorithm = self.settings.value("algorithm", "MD5")
        index = self.algorithm_combo.findText(algorithm)
        if index >= 0:
            self.algorithm_combo.setCurrentIndex(index)
        
        # Restore chunk size
        chunk_size = self.settings.value("chunk_size", 64)
        self.chunk_size_spin.setValue(int(chunk_size))
    
    def save_settings(self):
        """Save application settings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("algorithm", self.algorithm_combo.currentText())
        self.settings.setValue("chunk_size", self.chunk_size_spin.value())
    
    def closeEvent(self, event):
        """Handle application close event with proper thread checking."""
        # Check if we actually have a running scan - be more specific about the check
        has_running_scan = (self.worker_thread is not None and 
                           not self.worker_thread.isFinished())
        
        if has_running_scan:
            reply = QMessageBox.question(
                self, "Scan Running",
                "A scan is currently running. Stop it and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.hash_worker.cancel()
                if self.worker_thread:
                    # Disconnect signals
                    try:
                        self.worker_thread.started.disconnect()
                        self.worker_thread.finished.disconnect()
                    except:
                        pass
                    # Move worker back
                    self.hash_worker.moveToThread(self.thread())
                    # Terminate thread
                    self.worker_thread.quit()
                    self.worker_thread.wait(3000)
                    if self.worker_thread.isRunning():
                        self.worker_thread.terminate()
                        self.worker_thread.wait(1000)
            else:
                event.ignore()
                return
        
        # Save settings
        self.save_settings()
        
        # Clean up any remaining thread
        if self.worker_thread:
            self.worker_thread.deleteLater()
            self.worker_thread = None
        
        event.accept()


def main():
    """Main application entry point with improved error handling."""
    app = QApplication(sys.argv)
    app.setApplicationName("Simple Deduplicator")
    app.setApplicationVersion("1.1")
    app.setOrganizationName("SimpleDeduplicator")
    
    # Set application icon
    try:
        app.setWindowIcon(app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
    except:
        pass
    
    try:
        # Create and show main window
        window = SimpleDeduplicatorApp()
        window.show()
        
        # Run the application
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Application error: {e}")
        QMessageBox.critical(None, "Application Error", f"Failed to start application:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
