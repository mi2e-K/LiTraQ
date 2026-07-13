#!/usr/bin/env python
"""PyQt6 GUI for LiTraQ calibration and movement analysis."""

from __future__ import annotations

import contextlib
import io
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Callable

import cv2
import matplotlib
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

matplotlib.use("Agg", force=True)

import litraq as motion

try:
    import av
except ImportError:
    av = None

KNOWN_PX_PER_CM_DEFAULT = motion.DEFAULT_KNOWN_PX_PER_CM


def bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width, channels = rgb.shape
    qimage = QImage(
        rgb.data,
        width,
        height,
        channels * width,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(qimage.copy())


def center_widget(widget: QWidget, parent: QWidget | None = None) -> None:
    frame = widget.frameGeometry()
    if parent is not None and parent.isVisible():
        center = parent.frameGeometry().center()
    else:
        screen = widget.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        center = screen.availableGeometry().center()
    frame.moveCenter(center)
    widget.move(frame.topLeft())


def analysis_preview_paths(args: Namespace) -> dict[str, object]:
    prefix = args.prefix or motion.infer_prefix(Path(args.tracking))
    return motion.analysis_output_paths(
        Path(args.out_dir),
        prefix,
        args.density_cmap,
        args.bin_seconds,
    )


def make_control_scroll_area(content: QWidget, width: int) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    scroll.setMinimumWidth(width)
    scroll.setMaximumWidth(width + 30)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return scroll


class VideoFrameReader:
    """Small random-access video reader with optional PyAV decoding."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.backend = "OpenCV"
        self._container = None
        self._stream = None
        self._cap = None
        self._fps = 0.0
        self._frame_count = 0

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._cap = cap

        if av is not None:
            try:
                self._container = av.open(self.path)
                self._stream = self._container.streams.video[0]
                self._stream.thread_type = "AUTO"
                self.backend = "PyAV"
                if self._frame_count <= 0 and self._stream.frames:
                    self._frame_count = int(self._stream.frames)
                if self._fps <= 0 and self._stream.average_rate:
                    self._fps = float(self._stream.average_rate)
            except Exception:
                self._container = None
                self._stream = None
                self.backend = "OpenCV"

    @property
    def frame_count(self) -> int:
        return max(1, self._frame_count)

    @property
    def fps(self) -> float:
        return self._fps

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_frame(self, frame_index: int) -> np.ndarray:
        frame_index = max(0, min(self.frame_count - 1, int(frame_index)))
        if self._container is not None and self._stream is not None and self._fps > 0:
            frame = self._read_frame_pyav(frame_index)
            if frame is not None:
                return frame
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.path)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_index} from {self.path}")
        return frame

    def _read_frame_pyav(self, frame_index: int) -> np.ndarray | None:
        assert self._container is not None
        assert self._stream is not None
        try:
            seconds = frame_index / self._fps
            seek_ts = int(seconds / float(self._stream.time_base))
            self._container.seek(seek_ts, any_frame=False, backward=True, stream=self._stream)
            best_frame = None
            best_delta = float("inf")
            for decoded in self._container.decode(self._stream):
                if decoded.time is None:
                    continue
                decoded_index = int(round(decoded.time * self._fps))
                delta = abs(decoded_index - frame_index)
                if delta < best_delta:
                    best_delta = delta
                    best_frame = decoded
                if decoded_index >= frame_index:
                    break
            if best_frame is None:
                return None
            rgb = best_frame.to_ndarray(format="rgb24")
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None


class PlotImagePreview(QLabel):
    def __init__(self, empty_text: str) -> None:
        super().__init__(empty_text)
        self._pixmap: QPixmap | None = None
        self._empty_text = empty_text
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(720, 320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setWordWrap(True)
        self.setStyleSheet("background: #0f131b; border: 1px solid #30384a; border-radius: 6px;")

    def set_plot_path(self, path: str | Path | None) -> None:
        if not path:
            self.set_message(self._empty_text)
            return
        plot_path = Path(path)
        if not plot_path.exists():
            self.set_message(f"Plot not found:\n{plot_path}")
            return
        pixmap = QPixmap(str(plot_path))
        if pixmap.isNull():
            self.set_message(f"Could not display plot:\n{plot_path}")
            return
        self._pixmap = pixmap
        self.setToolTip("")
        self.setStatusTip("")
        self._update_scaled()

    def set_message(self, text: str) -> None:
        self._pixmap = None
        self.setToolTip("")
        self.setStatusTip("")
        self.clear()
        self.setText(text)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class ArenaCornerDialog(QDialog):
    def __init__(self, video_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick arena corners")
        self.resize(1180, 760)
        self.reader = VideoFrameReader(video_path)
        self.selected_frame = 0
        self.selected_image: np.ndarray | None = None
        self.selected_points: np.ndarray | None = None
        self._updating = False

        layout = QVBoxLayout(self)
        self.canvas = ImageCanvas("arena")
        self.canvas.setMinimumSize(860, 520)
        self.canvas.selectionComplete.connect(self.on_selection_complete)
        layout.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.reader.frame_count - 1)
        self.slider.setTracking(True)
        self.frame_spin = NoWheelSpinBox()
        self.frame_spin.setRange(0, self.reader.frame_count - 1)
        self.frame_label = QLabel()
        clear_button = QPushButton("Clear Points")
        clear_button.clicked.connect(self.restart_selection)
        controls.addWidget(QLabel("Frame"))
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.frame_spin)
        controls.addWidget(self.frame_label)
        controls.addWidget(clear_button)
        layout.addLayout(controls)

        hint = QLabel(
            "Click four arena corners directly on this frame. Mouse wheel zooms; middle-drag or Shift+drag pans; right-click undoes."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.slider.valueChanged.connect(self.on_slider_changed)
        self.frame_spin.valueChanged.connect(self.on_spin_changed)
        self.update_frame(0)
        center_widget(self, parent)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.reader.close()
        super().closeEvent(event)

    def on_slider_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.frame_spin.setValue(value)
        self._updating = False
        self.update_frame(value)

    def on_spin_changed(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self.slider.setValue(value)
        self._updating = False
        self.update_frame(value)

    def update_frame(self, frame_index: int) -> None:
        image = self.reader.read_frame(frame_index)
        self.selected_frame = int(frame_index)
        self.selected_image = image
        self.selected_points = None
        self.canvas.set_image(image)
        self.canvas.start_selection("arena_corners", "points", 4)
        fps_text = f"{self.reader.fps:.3f} fps" if self.reader.fps else "unknown fps"
        self.frame_label.setText(
            f"{self.selected_frame + 1}/{self.reader.frame_count} | {fps_text} | {self.reader.backend}"
        )

    def restart_selection(self) -> None:
        if self.selected_image is None:
            return
        self.selected_points = None
        self.canvas.start_selection("arena_corners", "points", 4)

    def on_selection_complete(self, selection_name: str, points: object) -> None:
        if selection_name != "arena_corners":
            return
        self.selected_points = np.asarray(points, dtype=np.float32)
        self.accept()


class ImageCanvas(QWidget):
    selectionComplete = pyqtSignal(str, object)
    selectionChanged = pyqtSignal(str, object)

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.name = name
        self.image_bgr: np.ndarray | None = None
        self.pixmap: QPixmap | None = None
        self.points: list[tuple[float, float]] = []
        self.selection_name: str | None = None
        self.selection_kind = "points"
        self.max_points = 0
        self.hover_point: tuple[float, float] | None = None
        self.image_rect = QRectF()
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._panning = False
        self._last_pan_pos = QPointF()
        self.setMinimumSize(520, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, image_bgr: np.ndarray | None) -> None:
        self.image_bgr = image_bgr
        self.pixmap = bgr_to_pixmap(image_bgr) if image_bgr is not None else None
        self.points.clear()
        self.selection_name = None
        self.hover_point = None
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._panning = False
        self.update()

    def start_selection(self, selection_name: str, kind: str, max_points: int) -> None:
        if self.image_bgr is None:
            raise ValueError(f"Select a frame before selecting {selection_name}.")
        self.selection_name = selection_name
        self.selection_kind = kind
        self.max_points = max_points
        self.points.clear()
        self.hover_point = None
        self.setFocus()
        self.update()

    def clear_selection(self) -> None:
        self.points.clear()
        self.hover_point = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111318"))

        if self.pixmap is None:
            painter.setPen(QColor("#6f7785"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame selected")
            return

        self.image_rect = self._calculate_image_rect()
        painter.drawPixmap(self.image_rect.toRect(), self.pixmap)

        painter.setPen(QPen(QColor("#2b3038"), 1))
        painter.drawRect(self.image_rect)
        self._draw_overlay(painter)

    def _calculate_image_rect(self) -> QRectF:
        if self.pixmap is None:
            return QRectF()
        available = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        scale = min(
            available.width() / self.pixmap.width(),
            available.height() / self.pixmap.height(),
        ) * self.zoom
        draw_w = self.pixmap.width() * scale
        draw_h = self.pixmap.height() * scale
        x = available.x() + (available.width() - draw_w) / 2 + self.pan.x()
        y = available.y() + (available.height() - draw_h) / 2 + self.pan.y()
        return QRectF(x, y, draw_w, draw_h)

    def _draw_overlay(self, painter: QPainter) -> None:
        if not self.points and not (self.selection_kind == "roi" and self.hover_point):
            return

        point_pen = QPen(QColor("#00f28f"), 2)
        line_pen = QPen(QColor("#00f28f"), 3)
        roi_pen = QPen(QColor("#55d0ff"), 2)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))

        widget_points = [self._image_to_widget(point) for point in self.points]

        if self.selection_kind == "roi" and self.points:
            end = self._image_to_widget(self.hover_point or self.points[-1])
            rect = QRectF(widget_points[0], end).normalized()
            painter.setPen(roi_pen)
            painter.setBrush(QBrush(QColor(85, 208, 255, 35)))
            painter.drawRect(rect)
        elif len(widget_points) > 1:
            painter.setPen(line_pen)
            for first, second in zip(widget_points[:-1], widget_points[1:]):
                painter.drawLine(first, second)

        painter.setPen(point_pen)
        painter.setBrush(QBrush(QColor("#00f28f")))
        for idx, point in enumerate(widget_points, start=1):
            painter.drawEllipse(point, 5, 5)
            painter.drawText(point + QPointF(8, -8), str(idx))

        if self.selection_name:
            painter.setPen(QColor("#c9d1d9"))
            status = f"{self.selection_name}: {len(self.points)}/{self.max_points}"
            painter.drawText(self.image_rect.adjusted(8, 8, -8, -8), status)

    def _widget_to_image(self, point: QPointF) -> tuple[float, float] | None:
        self.image_rect = self._calculate_image_rect()
        if self.pixmap is None or not self.image_rect.contains(point):
            return None
        x = (point.x() - self.image_rect.x()) / self.image_rect.width() * self.pixmap.width()
        y = (point.y() - self.image_rect.y()) / self.image_rect.height() * self.pixmap.height()
        return (float(x), float(y))

    def _image_to_widget(self, point: tuple[float, float]) -> QPointF:
        if self.pixmap is None:
            return QPointF()
        self.image_rect = self._calculate_image_rect()
        x = self.image_rect.x() + point[0] / self.pixmap.width() * self.image_rect.width()
        y = self.image_rect.y() + point[1] / self.pixmap.height() * self.image_rect.height()
        return QPointF(x, y)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.pixmap is not None and (
            event.button() == Qt.MouseButton.MiddleButton
            or (
                event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            )
            or (self.selection_name is None and event.button() == Qt.MouseButton.LeftButton)
        ):
            self._panning = True
            self._last_pan_pos = event.position()
            self.setFocus()
            return
        if self.selection_name is None:
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self.points:
                self.points.pop()
                self.selectionChanged.emit(self.selection_name, list(self.points))
                self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._widget_to_image(event.position())
        if point is None:
            return
        if len(self.points) < self.max_points:
            self.points.append(point)
            self.selectionChanged.emit(self.selection_name, list(self.points))
            if len(self.points) >= self.max_points:
                selection_name = self.selection_name
                payload = list(self.points)
                self.selection_name = None
                self.hover_point = None
                self.selectionComplete.emit(selection_name, payload)
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            delta = event.position() - self._last_pan_pos
            self.pan += delta
            self._last_pan_pos = event.position()
            self.update()
            return
        if self.selection_name and self.selection_kind == "roi":
            self.hover_point = self._widget_to_image(event.position())
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._panning = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self.pixmap is not None:
            self.zoom = 1.0
            self.pan = QPointF(0, 0)
            self.update()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.pixmap is None:
            event.ignore()
            return
        before = self._widget_to_image(event.position())
        old_zoom = self.zoom
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.zoom = max(1.0, min(12.0, self.zoom * factor))
        if before is not None and self.zoom != old_zoom:
            self.image_rect = self._calculate_image_rect()
            after = self._image_to_widget(before)
            self.pan += event.position() - after
        self.update()
        event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete, Qt.Key.Key_U):
            if self.points:
                self.points.pop()
                self.selectionChanged.emit(self.selection_name or "", list(self.points))
                self.update()
        elif key == Qt.Key.Key_R:
            self.points.clear()
            self.selectionChanged.emit(self.selection_name or "", list(self.points))
            self.update()
        else:
            super().keyPressEvent(event)


class AnalysisWorker(QThread):
    finishedOk = pyqtSignal(str, object)
    failed = pyqtSignal(str)

    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args

    def run(self) -> None:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                motion.command_analyze(self.args)
            paths = {} if self.args.no_plots else analysis_preview_paths(self.args)
            self.finishedOk.emit(buffer.getvalue(), paths)
        except Exception:
            self.failed.emit(buffer.getvalue() + "\n" + traceback.format_exc())


class BatchAnalysisWorker(QThread):
    progress = pyqtSignal(str)
    progressValue = pyqtSignal(int, int)
    finishedOk = pyqtSignal(str, object)

    def __init__(self, jobs: list[Namespace]) -> None:
        super().__init__()
        self.jobs = jobs

    def run(self) -> None:
        log_parts: list[str] = []
        last_paths: dict[str, object] = {}
        failures = 0
        for index, args in enumerate(self.jobs, start=1):
            tracking_name = Path(args.tracking).name
            self.progress.emit(f"[{index}/{len(self.jobs)}] Running {tracking_name}")
            self.progressValue.emit(index - 1, len(self.jobs))
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    motion.command_analyze(args)
                log_parts.append(f"[{index}/{len(self.jobs)}] {tracking_name}\n{buffer.getvalue()}")
                if not args.no_plots:
                    last_paths = analysis_preview_paths(args)
            except Exception:
                failures += 1
                log_parts.append(
                    f"[{index}/{len(self.jobs)}] FAILED {tracking_name}\n"
                    + buffer.getvalue()
                    + "\n"
                    + traceback.format_exc()
                )
            self.progressValue.emit(index, len(self.jobs))
        summary = f"Batch completed: {len(self.jobs) - failures} succeeded, {failures} failed."
        log_parts.append(summary)
        self.finishedOk.emit("\n\n".join(log_parts), last_paths)


class PathRow(QWidget):
    pathChanged = pyqtSignal(str)

    def __init__(
        self,
        label: str,
        mode: str,
        parent: QWidget | None = None,
        directory: bool = False,
    ) -> None:
        super().__init__(parent)
        self.mode = mode
        self.directory = directory
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(label)
        button = QPushButton("Browse")
        button.clicked.connect(self.browse)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, path: str) -> None:
        if self.edit.text() == path:
            return
        self.edit.setText(path)
        self.pathChanged.emit(path)

    def browse(self) -> None:
        settings = QSettings("LiTraQ", "LiTraQ")
        current = self.path()
        if current:
            current_path = Path(current)
            start_path = current_path if self.directory else current_path.parent
        else:
            start_path = Path(str(settings.value("last_directory", str(Path.cwd()))))
        start = str(start_path if start_path.exists() else Path.cwd())
        if self.directory:
            path = QFileDialog.getExistingDirectory(self, "Choose directory", start)
        elif self.mode == "save":
            path, _ = QFileDialog.getSaveFileName(self, "Choose output file", start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose file", start)
        if path:
            self.edit.setText(path)
            chosen = Path(path)
            settings.setValue(
                "last_directory",
                str(chosen if self.directory else chosen.parent),
            )
            self.pathChanged.emit(path)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LiTraQ")
        width, height = 1420, 760
        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, max(1100, int(available.width() * 0.92)))
            height = min(height, max(560, int(available.height() * 0.78)))
        self.resize(width, height)

        self.reference_frame_bgr: np.ndarray | None = None
        self.reference_display_bgr: np.ndarray | None = None
        self.reference_source: dict | None = None
        self.arena_frame_bgr: np.ndarray | None = None
        self.crop: dict | None = None
        self.arena_points_clicked: np.ndarray | None = None
        self.arena_points_ordered: np.ndarray | None = None
        self.reference_points_raw: np.ndarray | None = None
        self.reference_segment: np.ndarray | None = None
        self.reference_roi: tuple[int, int, int, int] | None = None
        self.reference_detection: dict | None = None
        self.worker: AnalysisWorker | BatchAnalysisWorker | None = None
        self.reference_frame_index = 123
        self.arena_frame_index = 0
        self.workflow_step: str | None = None
        self.batch_jobs: list[Namespace] = []
        self.batch_rows: list[dict[str, Path | str | None]] = []

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_calibration_tab(), "Arena Setup")
        self.tabs.addTab(self._build_analysis_tab(), "Analysis")
        self.tabs.addTab(self._build_batch_tab(), "Batch Analysis")
        self.setCentralWidget(self.tabs)
        self._create_actions()

    def center_on_screen(self) -> None:
        center_widget(self)

    def _create_actions(self) -> None:
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

    def _build_calibration_tab(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        controls = QWidget()
        controls.setMinimumWidth(430)
        controls.setMaximumWidth(520)
        control_layout = QVBoxLayout(controls)
        control_layout.setSpacing(12)

        source_group = QGroupBox("Arena setup")
        source_form = QFormLayout(source_group)
        self.processed_video_row = PathRow("Processed video matching DLC coordinates", "open")
        self.arena_tracking_row = PathRow("Matching DLC _filtered.csv or .h5", "open")
        self.output_calibration_edit = QLineEdit()
        self.output_calibration_edit.setReadOnly(True)
        source_form.addRow("Processed video", self.processed_video_row)
        source_form.addRow("Tracking", self.arena_tracking_row)
        source_form.addRow("Calibration JSON", self.output_calibration_edit)
        control_layout.addWidget(source_group)

        workflow_group = QGroupBox("Arena corners")
        workflow_layout = QVBoxLayout(workflow_group)
        self.start_workflow_button = QPushButton("Pick Arena Corners")
        self.start_workflow_button.clicked.connect(self.start_calibration_workflow)
        self.go_to_analysis_button = QPushButton("Go To Analysis")
        self.go_to_analysis_button.setEnabled(False)
        self.go_to_analysis_button.clicked.connect(self.go_to_analysis_tab)
        workflow_layout.addWidget(self.start_workflow_button)
        workflow_layout.addWidget(self.go_to_analysis_button)
        workflow_hint = QLabel(
            "Select a processed video, then click the four arena corners on a frame. The 19 px/cm scale is used automatically."
        )
        workflow_hint.setWordWrap(True)
        workflow_layout.addWidget(workflow_hint)
        control_layout.addWidget(workflow_group)

        params_group = QGroupBox("Scale")
        params_form = QFormLayout(params_group)
        self.known_px_per_cm_spin = self._double_spinbox(0.001, 10000.0, KNOWN_PX_PER_CM_DEFAULT, 6, 0.1)
        params_form.addRow("Known scale (px/cm)", self.known_px_per_cm_spin)
        control_layout.addWidget(params_group)

        self.status_label = QLabel("Choose a processed video, then pick arena corners if a calibration JSON is not available yet.")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("StatusLabel")
        control_layout.addWidget(self.status_label)
        self.calibration_log = QPlainTextEdit()
        self.calibration_log.setReadOnly(True)
        self.calibration_log.setMaximumHeight(150)
        control_layout.addWidget(self.calibration_log)
        control_layout.addStretch(1)
        splitter.addWidget(make_control_scroll_area(controls, 430))

        self.reference_canvas = ImageCanvas("reference")
        self.arena_canvas = ImageCanvas("arena")
        splitter.addWidget(self.arena_canvas)
        splitter.setSizes([450, 950])
        self.processed_video_row.pathChanged.connect(self.on_processed_video_changed)
        self.arena_tracking_row.pathChanged.connect(self.on_arena_tracking_changed)
        return root

    def _build_analysis_tab(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)

        controls = QWidget()
        controls.setMinimumWidth(500)
        controls.setMaximumWidth(540)
        form_layout = QVBoxLayout(controls)

        files_group = QGroupBox("Files")
        files_form = QFormLayout(files_group)
        self.tracking_row = PathRow("DLC CSV or H5", "open")
        self.analysis_video_row = PathRow("Processed video for FPS", "open")
        self.analysis_calibration_row = PathRow("Calibration JSON", "open")
        self.analysis_out_dir_row = PathRow("Analysis output directory", "open", directory=True)
        files_form.addRow("Tracking", self.tracking_row)
        files_form.addRow("Video", self.analysis_video_row)
        files_form.addRow("Calibration", self.analysis_calibration_row)
        files_form.addRow("Output dir", self.analysis_out_dir_row)
        form_layout.addWidget(files_group)

        options_group = QGroupBox("Movement definition")
        options_form = QFormLayout(options_group)
        self.bodypart_edit = QLineEdit("bodycenter")
        self.likelihood_spin = self._double_spinbox(0.0, 1.0, 0.90, 3, 0.01)
        self.speed_spin = self._double_spinbox(0.0, 100.0, 1.0, 2, 0.1)
        self.max_gap_spin = self._double_spinbox(0.0, 10.0, 0.50, 2, 0.05)
        self.smooth_spin = self._double_spinbox(0.0, 10.0, 0.20, 2, 0.05)
        self.stop_gap_spin = self._double_spinbox(0.0, 10.0, 0.20, 2, 0.05)
        self.min_bout_spin = self._double_spinbox(0.0, 10.0, 0.20, 2, 0.05)
        self.min_disp_spin = self._double_spinbox(0.0, 100.0, 0.50, 2, 0.1)
        self.arena_margin_spin = self._double_spinbox(0.0, 100.0, 2.0, 2, 0.5)
        self.bin_seconds_edit = QLineEdit("60 120")
        self.speed_color_center_spin = self._double_spinbox(0.1, 500.0, 20.0, 2, 1.0)
        self.density_cmap_combo = QComboBox()
        self.density_cmap_combo.addItems(["coolwarm", "CMRmap", "rocket", "magma", "viridis", "RdBu"])
        self.density_bins_spin = self._spinbox(10, 500, 90)
        self.no_plots_check = QCheckBox("Skip plot PNGs")
        options_form.addRow("Bodypart", self.bodypart_edit)
        options_form.addRow("Likelihood threshold", self.likelihood_spin)
        options_form.addRow("Speed threshold (cm/s)", self.speed_spin)
        options_form.addRow("Max gap (s)", self.max_gap_spin)
        options_form.addRow("Smooth window (s)", self.smooth_spin)
        options_form.addRow("Stop gap (s)", self.stop_gap_spin)
        options_form.addRow("Min bout (s)", self.min_bout_spin)
        options_form.addRow("Min bout displacement (cm)", self.min_disp_spin)
        options_form.addRow("Arena margin (cm)", self.arena_margin_spin)
        options_form.addRow("Bin seconds", self.bin_seconds_edit)
        options_form.addRow("Speed color midpoint (cm/s)", self.speed_color_center_spin)
        options_form.addRow("Density colormap", self.density_cmap_combo)
        options_form.addRow("Density bins", self.density_bins_spin)
        options_form.addRow("", self.no_plots_check)
        form_layout.addWidget(options_group)

        event_group = QGroupBox("Edge events and QC")
        event_form = QFormLayout(event_group)
        self.edge_width_spin = self._double_spinbox(0.1, 100.0, motion.DEFAULT_EDGE_WIDTH_CM, 2, 0.5)
        self.transit_edge_tolerance_spin = self._double_spinbox(
            0.0,
            5.0,
            motion.DEFAULT_TRANSIT_EDGE_TOLERANCE_CM,
            2,
            0.1,
        )
        self.straight_efficiency_spin = self._double_spinbox(
            0.0,
            1.0,
            motion.DEFAULT_STRAIGHT_PATH_EFFICIENCY_THRESHOLD,
            3,
            0.01,
        )
        self.straight_deviation_spin = self._double_spinbox(0.0, 100.0, 6.0, 2, 0.5)
        self.event_stop_speed_spin = self._double_spinbox(0.0, 100.0, 1.0, 2, 0.1)
        self.straight_stop_min_spin = self._double_spinbox(
            0.0,
            5.0,
            motion.DEFAULT_STRAIGHT_STOP_MIN_SEC,
            2,
            0.05,
        )
        self.straight_low_speed_fraction_spin = self._double_spinbox(
            0.0,
            1.0,
            motion.DEFAULT_STRAIGHT_LOW_SPEED_FRACTION_THRESHOLD,
            3,
            0.01,
        )
        self.straight_max_step_jump_spin = self._double_spinbox(0.1, 100.0, 8.0, 2, 0.5)
        self.straight_min_valid_fraction_spin = self._double_spinbox(0.0, 1.0, 0.95, 3, 0.01)
        self.wall_posture_check = QCheckBox("Reject sustained wall posture")
        self.wall_posture_check.setChecked(motion.DEFAULT_WALL_POSTURE_CHECK)
        self.posture_wall_distance_spin = self._double_spinbox(
            0.0,
            10.0,
            motion.DEFAULT_POSTURE_WALL_DISTANCE_CM,
            2,
            0.1,
        )
        self.posture_center_away_spin = self._double_spinbox(
            0.0,
            10.0,
            motion.DEFAULT_POSTURE_CENTER_AWAY_CM,
            2,
            0.1,
        )
        self.posture_min_fraction_spin = self._double_spinbox(
            0.0,
            1.0,
            motion.DEFAULT_POSTURE_MIN_FRACTION,
            2,
            0.05,
        )
        self.posture_borderline_fraction_spin = self._double_spinbox(
            0.0,
            1.0,
            motion.DEFAULT_POSTURE_BORDERLINE_FRACTION,
            2,
            0.05,
        )
        self.posture_min_valid_fraction_spin = self._double_spinbox(
            0.0,
            1.0,
            motion.DEFAULT_POSTURE_MIN_VALID_FRACTION,
            2,
            0.05,
        )
        self.posture_interruption_speed_spin = self._double_spinbox(
            0.0,
            100.0,
            motion.DEFAULT_POSTURE_INTERRUPTION_MAX_SPEED_CM_S,
            2,
            0.5,
        )
        self.posture_interruption_min_sec_spin = self._double_spinbox(
            0.0,
            5.0,
            motion.DEFAULT_POSTURE_INTERRUPTION_MIN_SEC,
            2,
            0.05,
        )
        self.back_forth_window_spin = self._double_spinbox(1.0, 600.0, 30.0, 1, 1.0)
        self.back_forth_min_round_trips_spin = self._spinbox(1, 20, 1)
        self.qc_video_check = QCheckBox("Create full QC video")
        self.qc_event_clips_check = QCheckBox("Create event QC clips")
        event_form.addRow("Edge width (cm)", self.edge_width_spin)
        event_form.addRow(
            "Transit edge tolerance (cm)",
            self.transit_edge_tolerance_spin,
        )
        event_form.addRow("Straight path efficiency", self.straight_efficiency_spin)
        event_form.addRow("Straight max deviation (cm)", self.straight_deviation_spin)
        event_form.addRow("Stop speed threshold (cm/s)", self.event_stop_speed_spin)
        event_form.addRow("Reject stop >= (s)", self.straight_stop_min_spin)
        event_form.addRow(
            "Max low-speed fraction (1.0 = off)",
            self.straight_low_speed_fraction_spin,
        )
        event_form.addRow("Max tracking jump (cm)", self.straight_max_step_jump_spin)
        event_form.addRow("Min valid fraction", self.straight_min_valid_fraction_spin)
        event_form.addRow("", self.wall_posture_check)
        event_form.addRow("Nose-wall distance (cm)", self.posture_wall_distance_spin)
        event_form.addRow("Center-away distance (cm)", self.posture_center_away_spin)
        event_form.addRow("Reject wall-posture fraction", self.posture_min_fraction_spin)
        event_form.addRow(
            "Borderline wall-posture fraction",
            self.posture_borderline_fraction_spin,
        )
        event_form.addRow(
            "Posture min valid fraction",
            self.posture_min_valid_fraction_spin,
        )
        event_form.addRow(
            "Posture interruption max speed (cm/s)",
            self.posture_interruption_speed_spin,
        )
        event_form.addRow(
            "Posture interruption min duration (s)",
            self.posture_interruption_min_sec_spin,
        )
        event_form.addRow("Back-and-forth window (s)", self.back_forth_window_spin)
        event_form.addRow("Min round trips", self.back_forth_min_round_trips_spin)
        event_form.addRow("", self.qc_video_check)
        event_form.addRow("", self.qc_event_clips_check)
        form_layout.addWidget(event_group)

        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.clicked.connect(self.run_analysis)
        run_buttons = QHBoxLayout()
        run_buttons.addWidget(self.run_analysis_button)
        form_layout.addLayout(run_buttons)
        form_layout.addStretch(1)

        self.path_plot_preview = PlotImagePreview("Run analysis to display the speed-colored path plot.")
        self.density_plot_preview = PlotImagePreview("Run analysis to display the bodycenter density plot.")
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.path_plot_preview)
        plot_layout.addWidget(self.density_plot_preview)

        self.analysis_log = QPlainTextEdit()
        self.analysis_log.setReadOnly(True)
        self.analysis_progress = QProgressBar()
        self.analysis_progress.setVisible(False)
        self.analysis_progress.setTextVisible(True)

        output_splitter = QSplitter(Qt.Orientation.Vertical)
        output_splitter.addWidget(plot_panel)
        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(self.analysis_progress)
        log_layout.addWidget(self.analysis_log)
        output_splitter.addWidget(log_panel)
        output_splitter.setSizes([560, 260])

        layout.addWidget(make_control_scroll_area(controls, 520))
        layout.addWidget(output_splitter, 1)
        self.analysis_video_row.pathChanged.connect(self.on_analysis_video_changed)
        self.tracking_row.pathChanged.connect(self.on_tracking_changed)
        return root

    def _build_batch_tab(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)

        controls = QWidget()
        controls.setMinimumWidth(430)
        controls.setMaximumWidth(500)
        control_layout = QVBoxLayout(controls)

        files_group = QGroupBox("Batch folder")
        files_form = QFormLayout(files_group)
        self.batch_folder_row = PathRow("Folder containing videos and tracking CSV files", "open", directory=True)
        self.batch_out_dir_row = PathRow("Batch output directory", "open", directory=True)
        self.batch_video_pattern_edit = QLineEdit("*_processed.mp4")
        files_form.addRow("Folder", self.batch_folder_row)
        files_form.addRow("Output root", self.batch_out_dir_row)
        files_form.addRow("Video pattern", self.batch_video_pattern_edit)
        control_layout.addWidget(files_group)

        button_group = QGroupBox("Batch workflow")
        button_layout = QVBoxLayout(button_group)
        self.batch_shared_arena_check = QCheckBox("Use one arena calibration for all videos")
        self.batch_shared_arena_check.setToolTip(
            "Use this when the camera, crop, resolution, and arena position are identical for every video."
        )
        shared_arena_hint = QLabel(
            "Shared mode asks for four corners only once and reuses those coordinates for every batch job."
        )
        shared_arena_hint.setWordWrap(True)
        self.batch_scan_button = QPushButton("Scan Folder")
        self.batch_pick_corners_button = QPushButton("Pick Missing Arena Corners")
        self.batch_run_button = QPushButton("Run Batch Analysis")
        self.batch_pick_corners_button.setEnabled(False)
        self.batch_run_button.setEnabled(False)
        self.batch_scan_button.clicked.connect(self.scan_batch_folder)
        self.batch_pick_corners_button.clicked.connect(self.pick_missing_batch_corners)
        self.batch_run_button.clicked.connect(self.run_batch_analysis)
        self.batch_shared_arena_check.toggled.connect(self.on_batch_shared_arena_toggled)
        button_layout.addWidget(self.batch_shared_arena_check)
        button_layout.addWidget(shared_arena_hint)
        button_layout.addWidget(self.batch_scan_button)
        button_layout.addWidget(self.batch_pick_corners_button)
        button_layout.addWidget(self.batch_run_button)
        control_layout.addWidget(button_group)
        control_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.batch_table = QTableWidget(0, 5)
        self.batch_table.setHorizontalHeaderLabels(
            ["Status", "Video", "Tracking", "Calibration", "Output"]
        )
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.batch_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        self.batch_log = QPlainTextEdit()
        self.batch_log.setReadOnly(True)
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        self.batch_progress.setTextVisible(True)
        batch_log_panel = QWidget()
        batch_log_layout = QVBoxLayout(batch_log_panel)
        batch_log_layout.setContentsMargins(0, 0, 0, 0)
        batch_log_layout.addWidget(self.batch_progress)
        batch_log_layout.addWidget(self.batch_log)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self.batch_table)
        right_splitter.addWidget(batch_log_panel)
        right_splitter.setSizes([560, 180])
        right_layout.addWidget(right_splitter)

        layout.addWidget(make_control_scroll_area(controls, 450))
        layout.addWidget(right, 1)

        self.batch_folder_row.pathChanged.connect(self.on_batch_folder_changed)
        return root

    def _spinbox(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = NoWheelSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _double_spinbox(
        self,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
        step: float = 0.1,
    ) -> QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def append_calibration_log(self, text: str) -> None:
        self.calibration_log.appendPlainText(text)

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def sample_base_from_video(self, video_path: str | Path) -> str:
        stem = Path(video_path).stem
        return stem[: -len("_processed")] if stem.endswith("_processed") else stem

    def default_calibration_path(self, video_path: str | Path) -> Path:
        path = Path(video_path)
        return path.parent / f"{self.sample_base_from_video(path)}_calibration.json"

    def default_output_dir(self, video_path: str | Path) -> Path:
        path = Path(video_path)
        return path.parent / "motion_outputs" / self.sample_base_from_video(path)

    def find_tracking_for_video(self, video_path: str | Path) -> Path | None:
        path = Path(video_path)
        folder = path.parent
        stem = path.stem
        base = self.sample_base_from_video(path)
        patterns = [
            f"{stem}*filtered.csv",
            f"{base}*filtered.csv",
            f"{stem}*filtered.h5",
            f"{base}*filtered.h5",
        ]
        for pattern in patterns:
            matches = sorted(folder.glob(pattern))
            if matches:
                return matches[0]
        return None

    def find_video_for_tracking(self, tracking_path: str | Path) -> Path | None:
        path = Path(tracking_path)
        prefix = motion.infer_prefix(path)
        base = prefix[: -len("_processed")] if prefix.endswith("_processed") else prefix
        candidates = [
            path.with_name(f"{prefix}.mp4"),
            path.with_name(f"{base}_processed.mp4"),
            path.with_name(f"{base}.mp4"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        matches = sorted(path.parent.glob(f"{base}*_processed.mp4"))
        return matches[0] if matches else None

    def on_processed_video_changed(self, path: str) -> None:
        if not path:
            return
        calibration_path = self.default_calibration_path(path)
        self.output_calibration_edit.setText(str(calibration_path))
        self.analysis_video_row.set_path(path)
        self.analysis_calibration_row.set_path(str(calibration_path))
        self.analysis_out_dir_row.set_path(str(self.default_output_dir(path)))
        tracking = self.find_tracking_for_video(path)
        if tracking is not None:
            self.arena_tracking_row.set_path(str(tracking))
            self.tracking_row.set_path(str(tracking))
        self.update_arena_ready_state()

    def on_analysis_video_changed(self, path: str) -> None:
        if not path:
            return
        calibration_path = self.default_calibration_path(path)
        self.analysis_calibration_row.set_path(str(calibration_path))
        if not self.analysis_out_dir_row.path():
            self.analysis_out_dir_row.set_path(str(self.default_output_dir(path)))
        tracking = self.find_tracking_for_video(path)
        if tracking is not None and not self.tracking_row.path():
            self.tracking_row.set_path(str(tracking))

    def on_tracking_changed(self, path: str) -> None:
        if not path:
            return
        if not self.analysis_video_row.path():
            video = self.find_video_for_tracking(path)
            if video is not None:
                self.analysis_video_row.set_path(str(video))
        if not self.analysis_out_dir_row.path():
            prefix = motion.infer_prefix(Path(path))
            base = prefix[: -len("_processed")] if prefix.endswith("_processed") else prefix
            self.analysis_out_dir_row.set_path(str(Path(path).parent / "motion_outputs" / base))
        if not self.arena_tracking_row.path():
            self.arena_tracking_row.set_path(path)
        self.update_arena_ready_state()

    def on_arena_tracking_changed(self, path: str) -> None:
        if path:
            self.tracking_row.set_path(path)
        self.update_arena_ready_state()

    def on_batch_folder_changed(self, path: str) -> None:
        if not path:
            return
        out_dir = Path(path) / "motion_batch_outputs"
        if not self.batch_out_dir_row.path():
            self.batch_out_dir_row.set_path(str(out_dir))
        self.scan_batch_folder()

    def shared_batch_calibration_path(self, video_path: str | Path) -> Path:
        folder_text = self.batch_folder_row.path()
        if not folder_text:
            raise ValueError("Choose a batch folder first.")
        out_root = Path(
            self.batch_out_dir_row.path() or (Path(folder_text) / "motion_batch_outputs")
        )
        base = self.safe_dir_name(self.sample_base_from_video(video_path))
        return out_root / "shared_arena_calibrations" / f"{base}_calibration.json"

    def on_batch_shared_arena_toggled(self, checked: bool) -> None:
        self.batch_pick_corners_button.setText(
            "Pick Shared Arena Corners" if checked else "Pick Missing Arena Corners"
        )
        rows = getattr(self, "batch_rows", [])
        if rows:
            for row in rows:
                video = row.get("video")
                if not isinstance(video, Path):
                    continue
                row["calibration"] = (
                    self.shared_batch_calibration_path(video)
                    if checked
                    else self.default_calibration_path(video)
                )
            self.render_batch_table()
        mode = "shared" if checked else "per-video"
        if hasattr(self, "batch_log"):
            self.batch_log.appendPlainText(f"Arena calibration mode: {mode}.")

    def update_arena_ready_state(self) -> None:
        video = self.processed_video_row.path() or self.analysis_video_row.path()
        tracking = self.arena_tracking_row.path() or self.tracking_row.path()
        calibration_text = self.output_calibration_edit.text().strip() or self.analysis_calibration_row.path()
        ready = bool(video and tracking and calibration_text and Path(calibration_text).exists())
        if hasattr(self, "go_to_analysis_button"):
            self.go_to_analysis_button.setEnabled(ready)

    def go_to_analysis_tab(self) -> None:
        self.tabs.setCurrentIndex(1)

    def start_calibration_workflow(self) -> None:
        try:
            processed_video = self.processed_video_row.path()
            if not processed_video:
                raise ValueError("Choose a processed video first.")
            self.reset_calibration_state(clear_log=True)
            output_json = self.default_calibration_path(processed_video)
            self.output_calibration_edit.setText(str(output_json))
            self.append_calibration_log(
                f"Using known scale: {self.known_px_per_cm_spin.value():.6f} px/cm"
            )
            dialog = ArenaCornerDialog(processed_video, self)
            result = dialog.exec()
            if result != QDialog.DialogCode.Accepted or dialog.selected_points is None:
                self.status_label.setText("Arena corner selection cancelled.")
                return
            self.arena_frame_index = dialog.selected_frame
            self.arena_frame_bgr = dialog.selected_image.copy() if dialog.selected_image is not None else None
            self.arena_points_clicked = dialog.selected_points
            self.arena_points_ordered = motion.order_arena_corners(dialog.selected_points)
            self.arena_canvas.set_image(self.arena_frame_bgr)
            self.arena_canvas.points = [tuple(point) for point in self.arena_points_ordered]
            self.arena_canvas.update()
            self.append_calibration_log(
                "Arena corners selected and ordered as TL,TR,BR,BL: "
                f"{self.arena_points_ordered.tolist()}"
            )
            self.complete_calibration_workflow()
        except Exception as exc:
            self.workflow_step = None
            self.show_error("Calibration workflow failed", str(exc))
        finally:
            if "dialog" in locals():
                dialog.reader.close()

    def reset_calibration_state(self, clear_log: bool = False) -> None:
        self.reference_frame_bgr = None
        self.reference_display_bgr = None
        self.reference_source = None
        self.arena_frame_bgr = None
        self.crop = None
        self.arena_points_clicked = None
        self.arena_points_ordered = None
        self.reference_points_raw = None
        self.reference_segment = None
        self.reference_roi = None
        self.reference_detection = None
        self.workflow_step = None
        self.reference_canvas.set_image(None)
        self.arena_canvas.set_image(None)
        if clear_log:
            self.calibration_log.clear()

    def complete_calibration_workflow(self) -> None:
        try:
            if not self.save_calibration():
                return
            self.workflow_step = None
            self.status_label.setText("Arena setup completed.")
        except Exception as exc:
            self.workflow_step = None
            self.show_error("Calibration workflow failed", str(exc))

    def save_calibration(self) -> bool:
        try:
            if self.arena_points_ordered is None:
                raise ValueError("Pick arena corners first.")
            processed_video = self.processed_video_row.path() or self.analysis_video_row.path()
            if not processed_video:
                raise ValueError("Choose a processed video first.")
            output_text = self.output_calibration_edit.text().strip() or str(
                self.default_calibration_path(processed_video)
            )
            output_path = Path(output_text)
            video_path = Path(processed_video)
            video_size = None
            if video_path:
                meta = motion.video_metadata(video_path)
                video_size = (meta["width"], meta["height"])
            source_image = self.arena_frame_bgr
            if source_image is None:
                raise ValueError("Select a processed video frame first.")
            image_h, image_w = source_image.shape[:2]
            reference_source = {
                "type": "known_scale",
                "known_px_per_cm": float(self.known_px_per_cm_spin.value()),
            }
            payload = motion.write_calibration(
                out_path=output_path,
                reference_source=reference_source,
                video_path=video_path,
                image_size=(image_w, image_h),
                video_size=video_size,
                crop=self.crop,
                arena_corners_px=self.arena_points_ordered,
                reference_segment_px=None,
                reference_length_cm=9.0,
                arena_corners_clicked_px=self.arena_points_clicked,
                arena_source={
                    "type": "video",
                    "video": str(video_path) if video_path else None,
                    "frame": int(self.arena_frame_index),
                },
                reference_detection=None,
                known_px_per_cm=float(self.known_px_per_cm_spin.value()),
            )
            preview_path = output_path.with_suffix(".preview.png")
            motion.save_calibration_preview(
                self.arena_frame_bgr,
                self.arena_points_ordered,
                None,
                preview_path,
            )
            self.output_calibration_edit.setText(str(output_path))
            self.analysis_calibration_row.set_path(str(output_path))
            self.append_calibration_log(f"Wrote calibration: {output_path}")
            self.append_calibration_log(f"Wrote preview: {preview_path}")
            self.append_calibration_log(
                "Scale after rectification: "
                f"{payload['px_per_cm_rectified']:.3f} px/cm "
                f"({payload['cm_per_px_rectified']:.5f} cm/px)"
            )
            self.status_label.setText("Calibration saved.")
            self.update_arena_ready_state()
            return True
        except Exception as exc:
            self.show_error("Save failed", str(exc))
            return False

    def parse_bin_seconds(self) -> list[int]:
        bins = [int(item) for item in self.bin_seconds_edit.text().replace(",", " ").split()]
        if not bins:
            raise ValueError("Enter at least one bin duration.")
        return bins

    def build_analysis_args(
        self,
        tracking: str | Path,
        video: str | Path,
        calibration: str | Path,
        out_dir: str | Path,
        prefix: str | None = None,
    ) -> Namespace:
        return Namespace(
            tracking=str(tracking),
            calibration=str(calibration),
            video=str(video),
            fps=None,
            bodypart=self.bodypart_edit.text().strip() or "bodycenter",
            likelihood_threshold=self.likelihood_spin.value(),
            speed_threshold_cm_s=self.speed_spin.value(),
            max_gap_sec=self.max_gap_spin.value(),
            smooth_window_sec=self.smooth_spin.value(),
            stop_gap_sec=self.stop_gap_spin.value(),
            min_bout_sec=self.min_bout_spin.value(),
            min_bout_displacement_cm=self.min_disp_spin.value(),
            arena_margin_cm=self.arena_margin_spin.value(),
            edge_width_cm=self.edge_width_spin.value(),
            transit_edge_tolerance_cm=self.transit_edge_tolerance_spin.value(),
            straight_path_efficiency_threshold=self.straight_efficiency_spin.value(),
            straight_max_deviation_cm=self.straight_deviation_spin.value(),
            event_stop_speed_threshold_cm_s=self.event_stop_speed_spin.value(),
            straight_stop_min_sec=self.straight_stop_min_spin.value(),
            straight_low_speed_fraction_threshold=self.straight_low_speed_fraction_spin.value(),
            straight_max_step_jump_cm=self.straight_max_step_jump_spin.value(),
            straight_min_valid_fraction=self.straight_min_valid_fraction_spin.value(),
            straight_borderline_margin=0.10,
            wall_posture_check=self.wall_posture_check.isChecked(),
            posture_nose_bodypart="nose",
            posture_center_bodypart="bodycenter",
            posture_tailbase_bodypart="tailbase",
            posture_wall_distance_cm=self.posture_wall_distance_spin.value(),
            posture_center_away_cm=self.posture_center_away_spin.value(),
            posture_min_fraction=self.posture_min_fraction_spin.value(),
            posture_borderline_fraction=self.posture_borderline_fraction_spin.value(),
            posture_min_valid_fraction=self.posture_min_valid_fraction_spin.value(),
            posture_interruption_max_speed_cm_s=self.posture_interruption_speed_spin.value(),
            posture_interruption_min_sec=self.posture_interruption_min_sec_spin.value(),
            back_forth_time_window_sec=self.back_forth_window_spin.value(),
            back_forth_min_round_trips=self.back_forth_min_round_trips_spin.value(),
            bin_seconds=self.parse_bin_seconds(),
            density_cmap=self.density_cmap_combo.currentText(),
            density_bins=self.density_bins_spin.value(),
            speed_color_center_cm_s=self.speed_color_center_spin.value(),
            out_dir=str(out_dir),
            prefix=prefix,
            no_plots=self.no_plots_check.isChecked(),
            qc_video=self.qc_video_check.isChecked(),
            qc_event_clips=self.qc_event_clips_check.isChecked(),
            qc_clip_padding_sec=1.0,
        )

    def set_analysis_busy(self, busy: bool) -> None:
        self.run_analysis_button.setEnabled(not busy)
        if hasattr(self, "batch_run_button"):
            self.batch_run_button.setEnabled((not busy) and self.batch_ready())
        if hasattr(self, "batch_pick_corners_button"):
            self.batch_pick_corners_button.setEnabled((not busy) and bool(self.batch_rows))

    def clear_analysis_previews(self, message: str) -> None:
        self.path_plot_preview.set_message(message)
        self.density_plot_preview.set_message(message)

    def show_analysis_previews(self, paths: dict[str, object]) -> None:
        if not paths:
            self.clear_analysis_previews("Plot previews are unavailable because plot PNGs were skipped.")
            return
        self.path_plot_preview.set_plot_path(paths.get("path_speed_png"))
        self.density_plot_preview.set_plot_path(paths.get("density_png"))

    def ensure_calibration_for_analysis(self) -> bool:
        calibration_path = Path(self.analysis_calibration_row.path()) if self.analysis_calibration_row.path() else None
        if calibration_path is not None and calibration_path.exists():
            return True
        video_path = self.analysis_video_row.path()
        if not video_path:
            raise ValueError("Choose a processed video before creating calibration.")
        self.processed_video_row.set_path(video_path)
        self.status_label.setText("Calibration JSON was not found. Pick arena corners to create it.")
        self.start_calibration_workflow()
        calibration_path = Path(self.analysis_calibration_row.path()) if self.analysis_calibration_row.path() else None
        return bool(calibration_path is not None and calibration_path.exists())

    def run_analysis(self) -> None:
        try:
            if not self.ensure_calibration_for_analysis():
                self.analysis_log.appendPlainText("Analysis cancelled because calibration was not created.")
                return
            args = self.build_analysis_args(
                tracking=self.tracking_row.path(),
                video=self.analysis_video_row.path(),
                calibration=self.analysis_calibration_row.path(),
                out_dir=self.analysis_out_dir_row.path(),
            )
            self.analysis_log.clear()
            self.clear_analysis_previews("Analysis is running.")
            self.analysis_log.appendPlainText("Running analysis...")
            self.analysis_progress.setVisible(True)
            self.analysis_progress.setRange(0, 0)
            self.analysis_progress.setFormat("Running analysis...")
            self.set_analysis_busy(True)
            self.worker = AnalysisWorker(args)
            self.worker.finishedOk.connect(self.on_analysis_finished)
            self.worker.failed.connect(self.on_analysis_failed)
            self.worker.start()
        except Exception as exc:
            self.show_error("Analysis setup failed", str(exc))

    def run_batch_analysis(self) -> None:
        try:
            self.scan_batch_folder()
            jobs = self.discover_batch_jobs()
            if not jobs:
                raise ValueError("No ready batch jobs were found.")
            self.batch_log.clear()
            self.batch_log.appendPlainText(f"Running batch analysis for {len(jobs)} file(s)...")
            self.batch_progress.setVisible(True)
            self.batch_progress.setRange(0, len(jobs))
            self.batch_progress.setValue(0)
            self.batch_progress.setFormat(f"%v / {len(jobs)} files")
            self.set_analysis_busy(True)
            self.worker = BatchAnalysisWorker(jobs)
            self.worker.progress.connect(self.batch_log.appendPlainText)
            self.worker.progressValue.connect(self.on_batch_progress)
            self.worker.finishedOk.connect(self.on_batch_finished)
            self.worker.start()
        except Exception as exc:
            self.show_error("Batch setup failed", str(exc))

    def scan_batch_folder(self) -> None:
        folder_text = self.batch_folder_row.path()
        if not folder_text:
            self.batch_log.appendPlainText("Choose a batch folder first.")
            return
        folder = Path(folder_text)
        if not folder.exists():
            self.show_error("Batch folder not found", str(folder))
            return
        out_root = Path(self.batch_out_dir_row.path() or (folder / "motion_batch_outputs"))
        self.batch_out_dir_row.set_path(str(out_root))
        patterns = [
            item.strip()
            for item in self.batch_video_pattern_edit.text().replace(",", ";").split(";")
            if item.strip()
        ] or ["*_processed.mp4"]
        video_paths: list[Path] = []
        for pattern in patterns:
            video_paths.extend(path for path in sorted(folder.glob(pattern)) if path.is_file())
        if not video_paths:
            video_paths = [path for path in sorted(folder.glob("*.mp4")) if path.is_file()]
        seen: set[Path] = set()
        rows: list[dict[str, Path | str | None]] = []
        shared_mode = self.batch_shared_arena_check.isChecked()
        for video_path in video_paths:
            resolved = video_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            base = self.sample_base_from_video(video_path)
            rows.append(
                {
                    "video": video_path,
                    "tracking": self.find_tracking_for_video(video_path),
                    "calibration": (
                        self.shared_batch_calibration_path(video_path)
                        if shared_mode
                        else self.default_calibration_path(video_path)
                    ),
                    "output": out_root / self.safe_dir_name(base),
                    "status": "",
                }
            )
        self.batch_rows = rows
        self.render_batch_table()
        self.batch_log.appendPlainText(f"Found {len(rows)} video file(s) in {folder}.")

    def render_batch_table(self) -> None:
        rows = getattr(self, "batch_rows", [])
        self.batch_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            status = self.batch_row_status(row)
            row["status"] = status
            values = [
                status,
                str(row.get("video") or ""),
                str(row.get("tracking") or "missing"),
                str(row.get("calibration") or ""),
                str(row.get("output") or ""),
            ]
            for column, value in enumerate(values):
                self.batch_table.setItem(row_index, column, QTableWidgetItem(value))
        self.batch_table.resizeColumnsToContents()
        self.batch_pick_corners_button.setEnabled(bool(rows))
        self.batch_run_button.setEnabled(self.batch_ready())

    def batch_row_status(self, row: dict[str, Path | str | None]) -> str:
        tracking = row.get("tracking")
        calibration = row.get("calibration")
        tracking_ok = isinstance(tracking, Path) and tracking.exists()
        calibration_ok = isinstance(calibration, Path) and calibration.exists()
        if tracking_ok and calibration_ok:
            return "ready"
        missing = []
        if not tracking_ok:
            missing.append("tracking")
        if not calibration_ok:
            missing.append("corners")
        return "missing " + ", ".join(missing)

    def batch_ready(self) -> bool:
        rows = getattr(self, "batch_rows", [])
        return bool(rows) and all(self.batch_row_status(row) == "ready" for row in rows)

    def pick_missing_batch_corners(self) -> None:
        if not getattr(self, "batch_rows", []):
            self.scan_batch_folder()
        if self.batch_shared_arena_check.isChecked():
            self.pick_shared_batch_corners()
            return
        for row in getattr(self, "batch_rows", []):
            calibration = row.get("calibration")
            if isinstance(calibration, Path) and calibration.exists():
                continue
            video = row.get("video")
            if not isinstance(video, Path):
                continue
            try:
                row["calibration"] = self.create_known_scale_calibration_for_video(
                    video,
                    calibration if isinstance(calibration, Path) else self.default_calibration_path(video),
                )
                self.batch_log.appendPlainText(f"Saved calibration for {video.name}")
            except RuntimeError:
                self.batch_log.appendPlainText(f"Corner selection cancelled for {video.name}")
                break
            except Exception as exc:
                self.show_error("Batch arena setup failed", f"{video}\n\n{exc}")
                break
        self.render_batch_table()

    def pick_shared_batch_corners(self) -> None:
        rows = getattr(self, "batch_rows", [])
        videos = [row.get("video") for row in rows if isinstance(row.get("video"), Path)]
        if not videos:
            self.show_error("Batch arena setup failed", "No batch videos were found.")
            return
        source_video = videos[0]
        dialog = ArenaCornerDialog(str(source_video), self)
        try:
            source_meta = motion.video_metadata(source_video)
            source_size = (int(source_meta["width"]), int(source_meta["height"]))
            mismatched: list[str] = []
            for video in videos[1:]:
                meta = motion.video_metadata(video)
                size = (int(meta["width"]), int(meta["height"]))
                if size != source_size:
                    mismatched.append(f"{video.name}: {size[0]}x{size[1]}")
            if mismatched:
                details = "\n".join(mismatched[:10])
                if len(mismatched) > 10:
                    details += f"\n... and {len(mismatched) - 10} more"
                raise ValueError(
                    "Shared arena calibration requires every video to have the same frame size.\n"
                    f"Reference: {source_video.name} ({source_size[0]}x{source_size[1]})\n"
                    f"Mismatched videos:\n{details}"
                )

            result = dialog.exec()
            if result != QDialog.DialogCode.Accepted or dialog.selected_points is None:
                raise RuntimeError("corner selection cancelled")
            if dialog.selected_image is None:
                raise ValueError("No selected frame was available.")

            frame = dialog.selected_image.copy()
            clicked = np.asarray(dialog.selected_points, dtype=np.float32)
            ordered = motion.order_arena_corners(clicked)
            for row in rows:
                video = row.get("video")
                if not isinstance(video, Path):
                    continue
                output_path = self.shared_batch_calibration_path(video)
                self.write_known_scale_calibration(
                    video_path=video,
                    output_path=output_path,
                    frame=frame,
                    ordered_points=ordered,
                    clicked_points=clicked,
                    arena_source_video=source_video,
                    arena_frame_index=int(dialog.selected_frame),
                )
                row["calibration"] = output_path

            self.batch_log.appendPlainText(
                f"Selected shared arena corners once from {source_video.name}."
            )
            self.batch_log.appendPlainText(
                f"Created {len(rows)} compatible calibration JSON file(s) using the same corners."
            )
        except RuntimeError:
            self.batch_log.appendPlainText(
                f"Shared corner selection cancelled for {source_video.name}"
            )
        except Exception as exc:
            self.show_error("Shared batch arena setup failed", str(exc))
        finally:
            dialog.reader.close()
        self.render_batch_table()

    def write_known_scale_calibration(
        self,
        video_path: Path,
        output_path: Path,
        frame: np.ndarray,
        ordered_points: np.ndarray,
        clicked_points: np.ndarray,
        arena_source_video: Path,
        arena_frame_index: int,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta = motion.video_metadata(video_path)
        video_size = (meta["width"], meta["height"])
        motion.write_calibration(
            out_path=output_path,
            reference_source={
                "type": "known_scale",
                "known_px_per_cm": float(self.known_px_per_cm_spin.value()),
            },
            video_path=video_path,
            image_size=(frame.shape[1], frame.shape[0]),
            video_size=video_size,
            crop=None,
            arena_corners_px=ordered_points,
            reference_segment_px=None,
            reference_length_cm=9.0,
            arena_corners_clicked_px=clicked_points,
            arena_source={
                "type": "video",
                "video": str(arena_source_video),
                "frame": int(arena_frame_index),
                "shared_across_batch": arena_source_video != video_path,
            },
            reference_detection=None,
            known_px_per_cm=float(self.known_px_per_cm_spin.value()),
        )
        motion.save_calibration_preview(
            frame,
            ordered_points,
            None,
            output_path.with_suffix(".preview.png"),
        )
        return output_path

    def create_known_scale_calibration_for_video(self, video_path: Path, output_path: Path) -> Path:
        dialog = ArenaCornerDialog(str(video_path), self)
        try:
            result = dialog.exec()
            if result != QDialog.DialogCode.Accepted or dialog.selected_points is None:
                raise RuntimeError("corner selection cancelled")
            if dialog.selected_image is None:
                raise ValueError("No selected frame was available.")
            frame = dialog.selected_image.copy()
            clicked = np.asarray(dialog.selected_points, dtype=np.float32)
            ordered = motion.order_arena_corners(clicked)
            return self.write_known_scale_calibration(
                video_path=video_path,
                output_path=output_path,
                frame=frame,
                ordered_points=ordered,
                clicked_points=clicked,
                arena_source_video=video_path,
                arena_frame_index=int(dialog.selected_frame),
            )
        finally:
            dialog.reader.close()

    def discover_batch_jobs(self) -> list[Namespace]:
        jobs: list[Namespace] = []
        for row in getattr(self, "batch_rows", []):
            if self.batch_row_status(row) != "ready":
                continue
            tracking_path = row.get("tracking")
            video_path = row.get("video")
            calibration_path = row.get("calibration")
            out_dir = row.get("output")
            if not all(isinstance(path, Path) for path in (tracking_path, video_path, calibration_path, out_dir)):
                continue
            prefix = motion.infer_prefix(tracking_path)
            jobs.append(
                self.build_analysis_args(
                    tracking=tracking_path,
                    video=video_path,
                    calibration=calibration_path,
                    out_dir=out_dir,
                    prefix=prefix,
                )
            )
        return jobs

    def safe_dir_name(self, name: str) -> str:
        invalid = '<>:"/\\|?*'
        return "".join("_" if char in invalid else char for char in name).strip() or "sample"

    def on_batch_finished(self, text: str, paths: object) -> None:
        del paths
        self.set_analysis_busy(False)
        self.batch_progress.setVisible(False)
        self.batch_log.appendPlainText(text or "Batch analysis completed.")

    def on_analysis_finished(self, text: str, paths: object) -> None:
        self.set_analysis_busy(False)
        self.analysis_progress.setRange(0, 1)
        self.analysis_progress.setValue(1)
        self.analysis_progress.setVisible(False)
        self.analysis_log.appendPlainText(text or "Analysis completed.")
        self.show_analysis_previews(paths if isinstance(paths, dict) else {})

    def on_analysis_failed(self, text: str) -> None:
        self.set_analysis_busy(False)
        self.analysis_progress.setRange(0, 1)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setVisible(False)
        self.analysis_log.appendPlainText(text)
        self.show_error("Analysis failed", "See the analysis log for details.")

    def on_batch_progress(self, value: int, total: int) -> None:
        self.batch_progress.setVisible(True)
        self.batch_progress.setRange(0, max(1, total))
        self.batch_progress.setValue(max(0, min(value, total)))
        self.batch_progress.setFormat(f"%v / {max(1, total)} files")


def dark_stylesheet() -> str:
    return """
    QWidget {
        background: #151821;
        color: #d7dde8;
        font-family: Segoe UI, Yu Gothic UI, Arial;
        font-size: 10pt;
    }
    QMainWindow, QTabWidget::pane {
        background: #151821;
        border: 1px solid #2a3040;
    }
    QTabBar::tab {
        background: #1d2230;
        color: #9ea7b8;
        padding: 9px 16px;
        border: 1px solid #2a3040;
        border-bottom: none;
    }
    QTabBar::tab:selected {
        background: #252c3b;
        color: #ffffff;
    }
    QGroupBox {
        border: 1px solid #30384a;
        border-radius: 8px;
        margin-top: 12px;
        padding: 12px 8px 8px 8px;
        background: #191e29;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: #9fc5ff;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QComboBox {
        background: #0f131b;
        border: 1px solid #343d50;
        border-radius: 6px;
        padding: 6px;
        color: #e6edf7;
        selection-background-color: #2f81f7;
    }
    QComboBox QAbstractItemView {
        background: #0f131b;
        color: #e6edf7;
        border: 1px solid #343d50;
        selection-background-color: #2f81f7;
    }
    QPushButton {
        background: #263247;
        color: #edf3ff;
        border: 1px solid #3c4961;
        border-radius: 6px;
        padding: 7px 10px;
    }
    QPushButton:hover {
        background: #31405b;
        border-color: #5686d6;
    }
    QPushButton:pressed {
        background: #1f6feb;
    }
    QPushButton:disabled {
        color: #6f7785;
        background: #1a1f2a;
        border-color: #2b3038;
    }
    QSplitter::handle {
        background: #242a38;
    }
    QLabel#StatusLabel {
        color: #b8c7dc;
        background: #10151f;
        border: 1px solid #30384a;
        border-radius: 6px;
        padding: 8px;
    }
    QCheckBox {
        spacing: 8px;
    }
    """


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    icon_path = next(
        (
            candidate
            for candidate in (
                script_dir / "assets" / "LiTraQ.ico",
                script_dir.parent / "assets" / "LiTraQ.ico",
            )
            if candidate.exists()
        ),
        script_dir / "assets" / "LiTraQ.ico",
    )

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "LiTraQ"
        )

    app = QApplication(sys.argv)
    app.setStyleSheet(dark_stylesheet())

    app_icon = QIcon(str(icon_path)) if icon_path.exists() else QIcon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    window = MainWindow()
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)

    window.show()
    window.center_on_screen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
