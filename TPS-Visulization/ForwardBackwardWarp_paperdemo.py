import sys
import torch_tps_transform
import torch
import cv2
import numpy as np
import json
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog,
                             QMessageBox, QSplitter, QCheckBox, QGroupBox, QRadioButton)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QFont
from PyQt5.QtCore import Qt, QPoint

from constant import batch_size, grid_h, grid_w, device

"""
Unified TPS Warp Studio
Integrates both Backward Warping and Forward Warping into a single application.
Features an intuitive UI with mode switching and color-coded mesh states.
"""


class ControlPoint:
    def __init__(self, pos, radius=8):
        self.pos = pos
        self.initial_pos = pos
        self.radius = radius
        self.is_dragging = False

    def move(self, new_pos):
        self.pos = new_pos

    def to_dict(self):
        return {
            'x': self.pos.x(),
            'y': self.pos.y(),
            'initial_x': self.initial_pos.x(),
            'initial_y': self.initial_pos.y(),
            'radius': self.radius
        }

    @classmethod
    def from_dict(cls, data):
        pos = QPoint(data['x'], data['y'])
        initial_pos = QPoint(data['initial_x'], data['initial_y'])
        point = cls(pos, data['radius'])
        point.initial_pos = initial_pos
        return point


class ImageWidget(QWidget):
    def __init__(self, is_left_panel=False):
        super().__init__()
        self.pixmap_offset = None
        self.image = None
        self.pixmap = None
        self.control_points = []
        self.dragging_point = None
        self.is_left_panel = is_left_panel
        self.margin = 20
        self.last_warp_time = 0

        # UI/UX flags
        self.is_draggable = False  # Controlled dynamically by mode
        self.show_mesh = True
        self.show_coordinates = False
        self.show_elements = True

        # Render cache
        self.scaled_pixmap = None
        self.scale_x = 1.0
        self.scale_y = 1.0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap_cache()

    def _update_scaled_pixmap_cache(self):
        if not self.pixmap: return
        margin = self.margin
        available_width = self.width() - 2 * margin
        available_height = self.height() - 2 * margin

        if available_width <= 0 or available_height <= 0: return

        self.scaled_pixmap = self.pixmap.scaled(
            available_width, available_height,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.pixmap_offset = QPoint(
            margin + (available_width - self.scaled_pixmap.width()) // 2,
            margin + (available_height - self.scaled_pixmap.height()) // 2
        )

        if self.image and self.image.width() > 0 and self.image.height() > 0:
            self.scale_x = self.scaled_pixmap.width() / self.image.width()
            self.scale_y = self.scaled_pixmap.height() / self.image.height()

    def load_image(self, image_path):
        self.image = QImage(image_path)
        if not self.image.isNull():
            self.pixmap = QPixmap.fromImage(self.image)
            self._update_scaled_pixmap_cache()
            self.update()

    def load_image_from_array(self, image_array):
        if image_array is not None:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            self.image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
            self.pixmap = QPixmap.fromImage(self.image)
            self._update_scaled_pixmap_cache()
            self.update()

    def create_uniform_points(self):
        if not self.image: return
        self.clear_control_points()

        width = self.image.width()
        height = self.image.height()
        x_spacing = width // grid_w
        y_spacing = height // grid_h

        for i in range(grid_h + 1):
            for j in range(grid_w + 1):
                x = j * x_spacing
                y = i * y_spacing
                self.control_points.append(ControlPoint(QPoint(x, y)))
        self.update()

    def set_control_points(self, points):
        self.control_points = []
        for p in points:
            new_pt = ControlPoint(QPoint(p.pos), p.radius)
            new_pt.initial_pos = QPoint(p.initial_pos)
            self.control_points.append(new_pt)
        self.update()

    def clear_control_points(self):
        self.control_points.clear()
        self.update()

    def get_control_points(self):
        return [point.pos for point in self.control_points]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#FAFAFA"))

        if self.pixmap and self.scaled_pixmap:
            painter.drawPixmap(self.pixmap_offset, self.scaled_pixmap)
        elif not self.pixmap:
            painter.setPen(QColor("#AAAAAA"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Image Loaded")

        if not self.show_elements or not self.control_points:
            return

        if self.show_mesh:
            self.draw_mesh(painter)

        for point in self.control_points:
            self.draw_control_point(painter, point)

    def draw_mesh(self, painter):
        if len(self.control_points) != (grid_h + 1) * (grid_w + 1): return

        # Dynamic Color: Green if draggable, Soft Red if locked/rigid
        # mesh_color = QColor(0, 210, 50, 160) if self.is_draggable else QColor(255, 80, 80, 180)
        mesh_color = QColor(0, 210, 50, 160) if self.is_left_panel else QColor(255, 80, 80, 180)
        painter.setPen(QPen(mesh_color, 2, Qt.SolidLine))

        points = self.control_points
        for i in range(grid_h + 1):
            for j in range(grid_w):
                idx1 = i * (grid_w + 1) + j
                idx2 = i * (grid_w + 1) + j + 1
                if idx1 < len(points) and idx2 < len(points):
                    pos1 = self.image_to_widget(points[idx1].pos)
                    pos2 = self.image_to_widget(points[idx2].pos)
                    painter.drawLine(pos1, pos2)

        for i in range(grid_h):
            for j in range(grid_w + 1):
                idx1 = i * (grid_w + 1) + j
                idx2 = (i + 1) * (grid_w + 1) + j
                if idx1 < len(points) and idx2 < len(points):
                    pos1 = self.image_to_widget(points[idx1].pos)
                    pos2 = self.image_to_widget(points[idx2].pos)
                    painter.drawLine(pos1, pos2)

    def draw_control_point(self, painter, point):
        if not self.pixmap: return
        widget_pos = self.image_to_widget(point.pos)
        fixed_radius = 6

        painter.setPen(Qt.NoPen)
        # Dynamic Color: Green if draggable, Soft Red if locked
        # pt_color = QColor(0, 220, 50, 220) if self.is_draggable else QColor(255, 60, 60, 220)
        pt_color = QColor(0, 220, 50, 220) if self.is_left_panel else QColor(255, 60, 60, 220)
        painter.setBrush(pt_color)
        painter.drawEllipse(widget_pos, fixed_radius, fixed_radius)

        if self.show_coordinates:
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            text = f"{int(point.pos.x())}, {int(point.pos.y())}"
            metrics = painter.fontMetrics()
            rect = metrics.boundingRect(text)

            text_width = rect.width()
            text_height = rect.height()

            bg_x = widget_pos.x() - text_width // 2
            bg_y = widget_pos.y() + 8

            if bg_x < 2:
                bg_x = 2
            elif bg_x + text_width + 4 > self.width():
                bg_x = self.width() - text_width - 8
            if bg_y + text_height + 4 > self.height(): bg_y = widget_pos.y() - text_height - 10

            bg_rect = rect.translated(int(bg_x), int(bg_y))
            painter.setBrush(QColor(255, 255, 255, 230))
            painter.setPen(QColor(150, 150, 150, 200))
            painter.drawRect(bg_rect.adjusted(-3, -2, 3, 2))
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(bg_rect, Qt.AlignCenter, text)

    def image_to_widget(self, image_point):
        if not self.pixmap or not hasattr(self, 'pixmap_offset') or self.scale_x == 0:
            return image_point
        return QPoint(
            self.pixmap_offset.x() + int(image_point.x() * self.scale_x),
            self.pixmap_offset.y() + int(image_point.y() * self.scale_y)
        )

    def widget_to_image(self, widget_point):
        if not self.pixmap or not hasattr(self, 'pixmap_offset') or self.scale_x == 0:
            return widget_point
        return QPoint(
            int((widget_point.x() - self.pixmap_offset.x()) / self.scale_x),
            int((widget_point.y() - self.pixmap_offset.y()) / self.scale_y)
        )

    def mousePressEvent(self, event):
        if not self.show_elements or not self.is_draggable or not self.control_points:
            return
        if event.button() == Qt.LeftButton:
            widget_pos = event.pos()
            click_radius = 20
            for point in reversed(self.control_points):
                pt_widget_pos = self.image_to_widget(point.pos)
                if (widget_pos - pt_widget_pos).manhattanLength() < click_radius:
                    self.dragging_point = point
                    point.is_dragging = True
                    break

    def mouseMoveEvent(self, event):
        if not self.is_draggable or not self.control_points:
            return
        if self.dragging_point and event.buttons() & Qt.LeftButton:
            image_pos = self.widget_to_image(event.pos())
            self.dragging_point.move(image_pos)
            self.update()

            current_time = time.time()
            if current_time - self.last_warp_time > 0.033:
                win = getattr(self, 'window_ref', None)
                if win:
                    if win.warp_mode == "backward" or win.initial_points_set:
                        if win.realtime_checkbox.isChecked():
                            win.tps_warp_image()
                self.last_warp_time = current_time

    def mouseReleaseEvent(self, event):
        if not self.is_draggable:
            return
        if event.button() == Qt.LeftButton and self.dragging_point:
            self.dragging_point.is_dragging = False
            self.dragging_point = None

            win = getattr(self, 'window_ref', None)
            if win:
                if win.warp_mode == "backward" or win.initial_points_set:
                    if win.realtime_checkbox.isChecked():
                        win.tps_warp_image()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_path = ''
        self.warped_image = None
        self.original_cv_image = None
        self.preview_cv_image = None
        self.preview_scale_factor = 1.0

        # State Variables
        self.warp_mode = "backward"  # "backward" or "forward"
        self.initial_points_set = False

        self.setWindowTitle("TPS Warp Studio (Unified)")
        self.setGeometry(100, 100, 1500, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(15)

        self.splitter = QSplitter(Qt.Horizontal)
        self.left_image_widget = ImageWidget(is_left_panel=True)
        self.right_image_widget = ImageWidget(is_left_panel=False)
        self.left_image_widget.window_ref = self
        self.right_image_widget.window_ref = self

        self.splitter.addWidget(self.left_image_widget)
        self.splitter.addWidget(self.right_image_widget)
        self.splitter.setSizes([600, 600])

        self.main_layout.addWidget(self.splitter, stretch=1)
        self.create_sidebar_panel()
        self.update_ui_state()

    def create_sidebar_panel(self):
        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(290)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(12)

        # --- 1. Warp Mode ---
        mode_group = QGroupBox("1. Warp Mode")
        mode_layout = QVBoxLayout()
        self.radio_backward = QRadioButton("Backward Warp (Drag Left Source)")
        self.radio_forward = QRadioButton("Forward Warp (Drag Right Target)")
        self.radio_backward.setChecked(True)

        self.radio_backward.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.radio_backward)
        mode_layout.addWidget(self.radio_forward)
        mode_group.setLayout(mode_layout)
        sidebar_layout.addWidget(mode_group)

        # --- 2. Image Options ---
        file_group = QGroupBox("2. Image Operations")
        file_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_btn)

        self.save_warped_btn = QPushButton("Save Warped Image")
        self.save_warped_btn.clicked.connect(self.save_warped_image)
        file_layout.addWidget(self.save_warped_btn)
        file_group.setLayout(file_layout)
        sidebar_layout.addWidget(file_group)

        # --- 3. Mesh Configuration ---
        mesh_group = QGroupBox("3. Mesh Definition")
        mesh_layout = QVBoxLayout()
        self.create_points_btn = QPushButton("Create Uniform Points")
        self.create_points_btn.clicked.connect(self.create_uniform_points)
        mesh_layout.addWidget(self.create_points_btn)

        layout_h = QHBoxLayout()
        self.load_source_btn = QPushButton("Load")
        self.load_source_btn.clicked.connect(self.load_source_mesh)
        self.save_source_btn = QPushButton("Save")
        self.save_source_btn.clicked.connect(self.save_source_mesh)
        layout_h.addWidget(self.load_source_btn)
        layout_h.addWidget(self.save_source_btn)
        mesh_layout.addLayout(layout_h)
        mesh_group.setLayout(mesh_layout)
        sidebar_layout.addWidget(mesh_group)

        # --- 4. Forward Warp Tools ---
        fw_group = QGroupBox("4. Forward Warp Tools")
        fw_layout = QVBoxLayout()
        self.set_initial_btn = QPushButton("Lock Source (Set Initial)")
        self.set_initial_btn.clicked.connect(self.set_initial_points)
        fw_layout.addWidget(self.set_initial_btn)

        self.rigid_mesh_btn = QPushButton("Set Target to Uniform Grid and Warp")
        self.rigid_mesh_btn.clicked.connect(self.move_target_to_rigid_mesh)
        fw_layout.addWidget(self.rigid_mesh_btn)
        fw_group.setLayout(fw_layout)
        sidebar_layout.addWidget(fw_group)

        # --- 5. View & Render ---
        render_group = QGroupBox("5. View & Render")
        render_layout = QVBoxLayout()

        self.clear_btn = QPushButton("Clear All Points")
        self.clear_btn.clicked.connect(self.clear_control_points)
        render_layout.addWidget(self.clear_btn)

        self.show_coords_checkbox = QCheckBox("Show Point Coordinates")
        self.show_coords_checkbox.stateChanged.connect(self.toggle_coordinates)
        render_layout.addWidget(self.show_coords_checkbox)

        self.show_right_mesh_checkbox = QCheckBox("Show Target Mesh")
        self.show_right_mesh_checkbox.setChecked(True)
        self.show_right_mesh_checkbox.stateChanged.connect(self.toggle_right_mesh)
        render_layout.addWidget(self.show_right_mesh_checkbox)

        self.realtime_checkbox = QCheckBox("Real-time Preview")
        self.realtime_checkbox.setChecked(True)
        render_layout.addWidget(self.realtime_checkbox)

        self.manual_warp_btn = QPushButton("Warp")
        self.manual_warp_btn.clicked.connect(self.tps_warp_image)
        render_layout.addWidget(self.manual_warp_btn)

        render_group.setLayout(render_layout)
        sidebar_layout.addWidget(render_group)

        sidebar_layout.addStretch()
        self.main_layout.addWidget(sidebar_widget)

    def on_mode_changed(self):
        if self.radio_backward.isChecked():
            self.warp_mode = "backward"
            if self.left_image_widget.control_points:
                self.right_image_widget.create_uniform_points()
        else:
            self.warp_mode = "forward"
            self.initial_points_set = False
            self.right_image_widget.clear_control_points()

        self.warped_image = None
        if self.file_path:
            self.right_image_widget.load_image(self.file_path)

        self.update_ui_state()
        if self.preview_cv_image is not None and self.warp_mode == "backward":
            self.tps_warp_image()

    def update_ui_state(self):
        is_forward = (self.warp_mode == "forward")

        # Disable/Enable exclusive Forward Warp buttons
        self.set_initial_btn.setEnabled(is_forward and not self.initial_points_set)
        self.rigid_mesh_btn.setEnabled(is_forward and self.initial_points_set)

        # Control draggability & visual states
        if not is_forward:
            self.left_image_widget.is_draggable = True
            self.right_image_widget.is_draggable = False
        else:
            if not self.initial_points_set:
                self.left_image_widget.is_draggable = True
                self.right_image_widget.is_draggable = False
            else:
                self.left_image_widget.is_draggable = False
                self.right_image_widget.is_draggable = True

        self.left_image_widget.update()
        self.right_image_widget.update()

    def toggle_coordinates(self, state):
        show = (state == Qt.Checked)
        self.left_image_widget.show_coordinates = show
        self.right_image_widget.show_coordinates = show
        self.left_image_widget.update()
        self.right_image_widget.update()

    def toggle_right_mesh(self, state):
        self.right_image_widget.show_elements = (state == Qt.Checked)
        self.right_image_widget.update()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not file_path: return
        self.file_path = file_path
        self.original_cv_image = cv2.imread(file_path)

        orig_h, orig_w = self.original_cv_image.shape[:2]
        max_dim = max(orig_h, orig_w)
        if max_dim > 3000:
            self.preview_scale_factor = 0.25
        elif max_dim > 1500:
            self.preview_scale_factor = 0.5
        else:
            self.preview_scale_factor = 1.0

        if self.preview_scale_factor != 1.0:
            self.preview_cv_image = cv2.resize(self.original_cv_image, (0, 0), fx=self.preview_scale_factor,
                                               fy=self.preview_scale_factor)
        else:
            self.preview_cv_image = self.original_cv_image.copy()

        self.left_image_widget.load_image(file_path)
        self.right_image_widget.load_image(file_path)
        self.warped_image = None
        self.clear_control_points()

    def create_uniform_points(self):
        if not self.left_image_widget.image:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        self.left_image_widget.create_uniform_points()

        if self.warp_mode == "backward":
            self.right_image_widget.create_uniform_points()
        else:
            self.right_image_widget.clear_control_points()
            self.initial_points_set = False

        self.warped_image = None
        if self.file_path:
            self.right_image_widget.load_image(self.file_path)
        self.update_ui_state()

    def set_initial_points(self):
        if not self.left_image_widget.control_points:
            QMessageBox.warning(self, "Warning", "Please create and adjust source points first.")
            return

        # Lock Source -> Copy layout to Target -> Make Target draggable
        self.right_image_widget.set_control_points(self.left_image_widget.control_points)
        self.initial_points_set = True
        self.update_ui_state()
        if self.realtime_checkbox.isChecked():
            self.tps_warp_image()

    def move_target_to_rigid_mesh(self):
        if not self.initial_points_set or not self.right_image_widget.image:
            return

        width = self.right_image_widget.image.width()
        height = self.right_image_widget.image.height()
        x_spacing = width // grid_w
        y_spacing = height // grid_h

        idx = 0
        for i in range(grid_h + 1):
            for j in range(grid_w + 1):
                x = int(j * x_spacing)
                y = int(i * y_spacing)
                if idx < len(self.right_image_widget.control_points):
                    self.right_image_widget.control_points[idx].move(QPoint(x, y))
                idx += 1

        self.right_image_widget.update()
        if self.realtime_checkbox.isChecked():
            self.tps_warp_image()

    def save_source_mesh(self):
        left_points = self.left_image_widget.get_control_points()
        if not left_points:
            QMessageBox.warning(self, "Warning", "No source mesh to save.")
            return

        points_data = [{'x': p.x(), 'y': p.y()} for p in left_points]
        data = {
            'source_points': points_data,
            'grid_h': grid_h,
            'grid_w': grid_w
        }

        if self.warp_mode == "backward":
            target_points = self.right_image_widget.get_control_points()
            if target_points:
                data['target_points'] = [{'x': p.x(), 'y': p.y()} for p in target_points]

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Mesh", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                QMessageBox.information(self, "Success", "Mesh saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")

    def load_source_mesh(self):
        if not self.left_image_widget.image:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Load Mesh", "", "JSON Files (*.json)")
        if not file_path: return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if data.get('grid_h') != grid_h or data.get('grid_w') != grid_w:
                QMessageBox.warning(self, "Warning", f"Grid parameter mismatch! Current: {grid_h}x{grid_w}")
                return

            source_points = []
            for p in data.get('source_points', []):
                x = int(round(float(p['x'])))
                y = int(round(float(p['y'])))
                source_points.append(ControlPoint(QPoint(x, y)))

            if source_points:
                self.left_image_widget.set_control_points(source_points)

                if self.warp_mode == "backward":
                    if 'target_points' in data:
                        target_pts = []
                        for p in data['target_points']:
                            tx = int(round(float(p['x'])))
                            ty = int(round(float(p['y'])))
                            target_pts.append(ControlPoint(QPoint(tx, ty)))
                        self.right_image_widget.set_control_points(target_pts)
                    else:
                        self.right_image_widget.create_uniform_points()
                    self.tps_warp_image()
                else:
                    self.right_image_widget.clear_control_points()
                    self.initial_points_set = False

                self.update_ui_state()
                QMessageBox.information(self, "Success", "Mesh loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {str(e)}")

    def clear_control_points(self):
        self.left_image_widget.clear_control_points()
        self.right_image_widget.clear_control_points()
        self.initial_points_set = False
        self.warped_image = None
        if self.file_path:
            self.right_image_widget.load_image(self.file_path)
        self.update_ui_state()

    def tps_warp_image(self):
        if self.preview_cv_image is None: return
        if not self.left_image_widget.control_points: return
        if self.warp_mode == "forward" and not self.initial_points_set: return

        left_points = self.left_image_widget.get_control_points()
        right_points = self.right_image_widget.get_control_points()

        expected_len = (grid_h + 1) * (grid_w + 1)
        if len(left_points) != expected_len or len(right_points) != expected_len:
            return

        scale_factor = self.preview_scale_factor
        source_points = np.array([[p.x(), p.y()] for p in left_points], dtype=np.float32) * scale_factor
        target_points = np.array([[p.x(), p.y()] for p in right_points], dtype=np.float32) * scale_factor

        preview_image = self.preview_cv_image
        img_h, img_w = preview_image.shape[:2]

        try:
            preview_warped = self.perform_tps_warp(preview_image, source_points, target_points, img_h, img_w)
            if scale_factor != 1.0:
                orig_h, orig_w = self.original_cv_image.shape[:2]
                self.warped_image = cv2.resize(preview_warped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                self.warped_image = preview_warped

            self.right_image_widget.load_image_from_array(self.warped_image)
        except Exception as e:
            print("Warping error:", e)

    def save_warped_image(self):
        if self.warp_mode == "forward" and not self.initial_points_set:
            QMessageBox.warning(self, "Warning", "Please define target mesh first.")
            return
        if self.warp_mode == "backward" and not self.left_image_widget.control_points:
            QMessageBox.warning(self, "Warning", "Please define source mesh first.")
            return
        if self.original_cv_image is None:
            QMessageBox.critical(self, "Error", "No original image in memory.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Warped Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )

        if file_path:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                left_points = self.left_image_widget.get_control_points()
                right_points = self.right_image_widget.get_control_points()

                source_points = np.array([[p.x(), p.y()] for p in left_points], dtype=np.float32)
                target_points = np.array([[p.x(), p.y()] for p in right_points], dtype=np.float32)

                original_image = self.original_cv_image.copy()
                orig_h, orig_w = original_image.shape[:2]

                high_res_warped = self.perform_tps_warp(original_image, source_points, target_points, orig_h, orig_w)
                success = cv2.imwrite(file_path, high_res_warped)

                QApplication.restoreOverrideCursor()
                if success:
                    QMessageBox.information(self, "Success", f"Image saved to: {file_path}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save image.")

            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self, "Error", f"High-Resolution TPS transform failed:\n{str(e)}")

    def perform_tps_warp(self, input_img, source, target, img_h, img_w):
        def get_norm_mesh(mesh, height, width):
            mesh_w = mesh[..., 0] * 2. / float(width) - 1.
            mesh_h = mesh[..., 1] * 2. / float(height) - 1.
            norm_mesh = torch.stack([mesh_w, mesh_h], 3)
            return norm_mesh.reshape([batch_size, -1, 2])

        source_mesh = source.reshape([batch_size, grid_h + 1, grid_w + 1, -1])
        source_mesh = torch.tensor(source_mesh).to(device)
        target_mesh = target.reshape([batch_size, grid_h + 1, grid_w + 1, -1])
        target_mesh = torch.tensor(target_mesh).to(device)

        input_img = input_img.astype(dtype=np.float32)
        input_img = (input_img / 127.5) - 1.0
        input_img = np.transpose(input_img, [2, 0, 1])
        input_img = torch.tensor(input_img).to(device)
        input_img = input_img.unsqueeze(0)

        norm_source_mesh = get_norm_mesh(source_mesh, img_h, img_w)
        norm_target_mesh = get_norm_mesh(target_mesh, img_h, img_w)

        tps_output = torch_tps_transform.transformer(input_img, norm_target_mesh, norm_source_mesh,
                                                     (img_h, img_w))
        tps_output = (tps_output[:, 0:3, :, :] + 1) * 127.5
        tps_output = tps_output[0].cpu().detach().numpy().transpose(1, 2, 0).astype(dtype=np.uint8)
        return tps_output


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # 现代清爽的 CSS 美化方案
    app.setStyleSheet("""
        QWidget {
            color: #333333;
        }
        QMainWindow {
            background-color: #F7F7FA;
        }
        QPushButton {
            padding: 7px 12px;
            background-color: #FFFFFF;
            border: 1px solid #D0D0D5;
            border-radius: 5px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #F0F0F5;
            border: 1px solid #B0B0B5;
        }
        QPushButton:pressed {
            background-color: #E0E0E5;
        }
        QPushButton:disabled {
            background-color: #F0F0F0;
            color: #A0A0A0;
            border: 1px solid #E5E5E5;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #E0E0E5;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 15px;
            background-color: #FFFFFF;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: #444444;
        }
        QRadioButton, QCheckBox {
            spacing: 6px;
        }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())