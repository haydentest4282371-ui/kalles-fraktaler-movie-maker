import sys
import traceback
import config

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QTextEdit,
    QTabWidget,
    QFileDialog,
    QSlider,
)

from PySide6.QtCore import Qt, QTimer

from PySide6.QtGui import QImage, QPixmap


class Gui(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("KFMovieMaker")
        self.resize(1000, 800)

        self.widgets = {}

        self.preview_files = []
        self.preview_cache = {}
        self.preview_index = 0


        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(
            self.render_preview
        )


        layout = QVBoxLayout(self)


        # =========================
        # TOP BAR
        # =========================

        top = QHBoxLayout()


        self.input_path = QLineEdit()

        browse_input = QPushButton(
            "Browse"
        )

        browse_input.clicked.connect(
            self.select_input
        )


        top.addWidget(
            QLabel("Input:")
        )

        top.addWidget(
            self.input_path
        )

        top.addWidget(
            browse_input
        )


        self.output_path = QLineEdit(
            "out.mp4"
        )


        browse_output = QPushButton(
            "Browse"
        )

        browse_output.clicked.connect(
            self.select_output
        )


        top.addWidget(
            QLabel("Output:")
        )

        top.addWidget(
            self.output_path
        )

        top.addWidget(
            browse_output
        )


        self.render_button = QPushButton(
            "Start Render"
        )

        self.render_button.clicked.connect(
            self.start_render
        )

        top.addWidget(
            self.render_button
        )


        layout.addLayout(
            top
        )


        # =========================
        # TABS
        # =========================

        tabs = QTabWidget()

        layout.addWidget(
            tabs
        )


        # -------------------------
        # CONFIG TAB
        # -------------------------

        config_tab = QWidget()

        config_layout = QVBoxLayout(
            config_tab
        )


        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )


        container = QWidget()

        self.config_layout = QVBoxLayout(
            container
        )

        scroll.setWidget(
            container
        )


        config_layout.addWidget(
            scroll
        )


        tabs.addTab(
            config_tab,
            "Config"
        )


        # -------------------------
        # PREVIEW TAB
        # -------------------------

        preview_tab = QWidget()

        preview_layout = QVBoxLayout(
            preview_tab
        )


        self.preview_label = QLabel()

        self.preview_label.setAlignment(
            Qt.AlignCenter
        )


        preview_layout.addWidget(
            self.preview_label
        )


        keyframe_row = QHBoxLayout()

        self.keyframe_slider = QSlider(
            Qt.Horizontal
        )

        self.keyframe_slider.setMinimum(0)


        self.keyframe_spin = QSpinBox()

        self.keyframe_spin.setMinimum(
            0
        )


        self.keyframe_slider.valueChanged.connect(
            self.keyframe_spin.setValue
        )

        self.keyframe_spin.valueChanged.connect(
            self.keyframe_slider.setValue
        )


        self.keyframe_slider.valueChanged.connect(
            self.preview_changed
        )


        self.keyframe_spin.valueChanged.connect(
            self.preview_changed
        )


        keyframe_row.addWidget(
            QLabel("Keyframe")
        )

        keyframe_row.addWidget(
            self.keyframe_spin
        )

        keyframe_row.addWidget(
            self.keyframe_slider
        )


        preview_layout.addLayout(
            keyframe_row
        )


        self.preview_info = QLabel(
            "No preview"
        )

        preview_layout.addWidget(
            self.preview_info
        )


        tabs.addTab(
            preview_tab,
            "Preview"
        )


        # -------------------------
        # LOG
        # -------------------------

        self.log = QTextEdit()

        self.log.setReadOnly(
            True
        )

        layout.addWidget(
            self.log
        )


        self.build_config_ui()



    # =========================
    # CONFIG EDITOR
    # =========================

    def build_config_ui(self):

        for name, value in vars(config).items():

            if name.startswith("_"):
                continue

            if callable(value):
                continue


            widget = None


            if isinstance(value, bool):

                widget = QCheckBox()

                widget.setChecked(
                    value
                )

                widget.toggled.connect(
                    lambda v, n=name:
                    setattr(config, n, v)
                )


            elif isinstance(value, int):

                widget = QSpinBox()

                widget.setRange(
                    -999999999,
                    999999999
                )

                widget.setValue(
                    value
                )

                widget.valueChanged.connect(
                    lambda v, n=name:
                    setattr(config, n, v)
                )


            elif isinstance(value, float):

                widget = QDoubleSpinBox()

                widget.setRange(
                    -999999999,
                    999999999
                )

                widget.setDecimals(
                    8
                )

                widget.setValue(
                    value
                )

                widget.valueChanged.connect(
                    lambda v, n=name:
                    setattr(config, n, v)
                )


            elif isinstance(value, str):

                widget = QLineEdit(
                    value
                )

                widget.textChanged.connect(
                    lambda v, n=name:
                    setattr(config, n, v)
                )


            if widget:

                row = QHBoxLayout()

                row.addWidget(
                    QLabel(name)
                )

                row.addWidget(
                    widget
                )


                self.config_layout.addLayout(
                    row
                )

                self.widgets[name] = widget



    # =========================
    # FILE PICKERS
    # =========================

    def select_input(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select KFB/RFM Folder"
        )

        if folder:

            self.input_path.setText(
                folder
            )

            self.load_preview_files()



    def select_output(self):

        file, _ = QFileDialog.getSaveFileName(
            self,
            "Output Video",
            "",
            "MP4 Video (*.mp4)"
        )

        if file:

            self.output_path.setText(
                file
            )



    # =========================
    # PREVIEW
    # =========================

    def load_preview_files(self):

        try:
            import render

            folder = self.input_path.text()

            if not folder:
                self.log.append("No input folder selected")
                return


            self.preview_files, ext = render.discover(
                folder
            )

            if not self.preview_files:
                self.log.append(
                    "No keyframes found"
                )
                return


            maximum = len(self.preview_files) - 1


            self.keyframe_slider.blockSignals(True)
            self.keyframe_spin.blockSignals(True)


            self.keyframe_slider.setMaximum(
                maximum
            )

            self.keyframe_spin.setMaximum(
                maximum
            )


            self.keyframe_slider.setValue(0)
            self.keyframe_spin.setValue(0)


            self.keyframe_slider.blockSignals(False)
            self.keyframe_spin.blockSignals(False)


            self.preview_cache.clear()

            self.log.append(
                f"Loaded {len(self.preview_files)} {ext} files"
            )


            self.render_preview()


        except Exception:
            self.log.append(
                traceback.format_exc()
            )



    def preview_changed(self):

        # wait until the user stops dragging
        self.preview_timer.start(
            200
        )



    def render_preview(self):

        try:

            if not self.preview_files:
                return


            index = self.keyframe_spin.value()


            if index >= len(self.preview_files):
                return


            import render
            import coloring
            from numba import cuda


            if index in self.preview_cache:

                cache = self.preview_cache[index]

            else:

                kfb = render.load_kfb(
                    self.preview_files[index]
                )

                cache = render.build_render_cache(
                    kfb
                )

                self.preview_cache[index] = cache



            h, w = cache[1].shape


            pinned, d_out = coloring._get_frame_bufs(
                h,
                w
            )


            flow = 0

            if hasattr(config, "FLOW_SPEED"):
                flow = config.FLOW_SPEED


            if config.COLORING == "standard": coloring.colorize(cache, flow, d_out)
            elif config.COLORING == "contour": coloring.colorize_contour(cache, flow, d_out)
            elif config.COLORING == "audio": coloring.colorize_audio(cache, flow, d_out)
            elif config.COLORING == "image": coloring.colorize_image(cache, flow, d_out)
            elif config.COLORING == "linear": coloring.colorize_linear(cache, flow, d_out)
            elif config.COLORING == "distance": coloring.colorize_distance(cache, flow, d_out)

            cuda.synchronize()


            d_out.copy_to_host(
                pinned
            )


            self.show_image(
                pinned
            )


            self.preview_info.setText(
                f"Keyframe {index}/{len(self.preview_files)-1}"
            )


        except Exception:

            self.log.append(
                traceback.format_exc()
            )



    def show_image(self, img):

        h, w, _ = img.shape


        qimg = QImage(
            img.data,
            w,
            h,
            w * 3,
            QImage.Format_RGB888
        )


        pixmap = QPixmap.fromImage(
            qimg
        )


        self.preview_label.setPixmap(
            pixmap.scaled(
                700,
                700,
                Qt.KeepAspectRatio
            )
        )



    # =========================
    # RENDER
    # =========================

    def start_render(self):

        folder = self.input_path.text()

        output = self.output_path.text()


        if not folder:

            self.log.append(
                "No input folder selected"
            )

            return


        try:

            import render


            self.log.append(
                "Starting render..."
            )


            render.render_sequence(
                folder,
                output
            )


            self.log.append(
                "Finished"
            )


        except Exception:

            self.log.append(
                traceback.format_exc()
            )



if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = Gui()

    window.show()

    sys.exit(
        app.exec()
    )
