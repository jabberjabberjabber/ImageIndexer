import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QPushButton, QFileDialog, QTextEdit, QGroupBox
from PyQt6.QtCore import QThread, pyqtSignal, QObject

import llmii

class IndexerThread(QThread):
    output_received = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.paused = False

    def run(self):
        llmii.main(self.config, self.output_received.emit, self.check_paused)

    def check_paused(self):
        while self.paused:
            self.msleep(100)

class PauseHandler(QObject):
    pause_signal = pyqtSignal(bool)

class ImageIndexerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Indexer GUI")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        dir_button = QPushButton("Select Directory")
        dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(QLabel("Directory:"))
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(dir_button)
        layout.addLayout(dir_layout)

        # API URL and Password
        api_layout = QHBoxLayout()
        self.api_url_input = QLineEdit("http://localhost:5001")
        self.api_password_input = QLineEdit()
        api_layout.addWidget(QLabel("API URL:"))
        api_layout.addWidget(self.api_url_input)
        api_layout.addWidget(QLabel("API Password:"))
        api_layout.addWidget(self.api_password_input)
        layout.addLayout(api_layout)

        # Image Instruction
        image_instruction_layout = QHBoxLayout()
        self.image_instruction_input = QLineEdit("What do you see in the image? Be specific and descriptive")
        image_instruction_layout.addWidget(QLabel("Image Instruction:"))
        image_instruction_layout.addWidget(self.image_instruction_input)
        layout.addLayout(image_instruction_layout)

        # Checkboxes for options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        self.no_crawl_checkbox = QCheckBox("No Crawl (Disable recursive indexing)")
        self.force_rehash_checkbox = QCheckBox("Force Rehash")
        self.overwrite_checkbox = QCheckBox("Overwrite")
        self.dry_run_checkbox = QCheckBox("Dry Run")
        options_layout.addWidget(self.no_crawl_checkbox)
        options_layout.addWidget(self.force_rehash_checkbox)
        options_layout.addWidget(self.overwrite_checkbox)
        options_layout.addWidget(self.dry_run_checkbox)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # XMP/Metadata tag checkboxes
        xmp_group = QGroupBox("XMP/Metadata Tags to Write")
        xmp_layout = QVBoxLayout()
        self.keywords_checkbox = QCheckBox("Keywords")
        self.title_checkbox = QCheckBox("Title")
        self.subject_checkbox = QCheckBox("Subject")
        self.description_checkbox = QCheckBox("Description")
        xmp_layout.addWidget(self.keywords_checkbox)
        xmp_layout.addWidget(self.title_checkbox)
        xmp_layout.addWidget(self.subject_checkbox)
        xmp_layout.addWidget(self.description_checkbox)
        xmp_group.setLayout(xmp_layout)
        layout.addWidget(xmp_group)

        # Run and Pause buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Image Indexer")
        self.run_button.clicked.connect(self.run_indexer)
        #self.pause_button = QPushButton("Pause")
        #self.pause_button.clicked.connect(self.toggle_pause)
        #self.pause_button.setEnabled(False)
        button_layout.addWidget(self.run_button)
        #button_layout.addWidget(self.pause_button)
        layout.addLayout(button_layout)

        # Output area
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_area)

        self.pause_handler = PauseHandler()

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)

    def run_indexer(self):
        # Create a Config object based on GUI inputs
        config = llmii.Config()
        config.directory = self.dir_input.text()
        config.api_url = self.api_url_input.text()
        config.api_password = self.api_password_input.text()
        config.no_crawl = self.no_crawl_checkbox.isChecked()
        config.force_rehash = self.force_rehash_checkbox.isChecked()
        config.overwrite = self.overwrite_checkbox.isChecked()
        config.dry_run = self.dry_run_checkbox.isChecked()
        config.write_keywords = self.keywords_checkbox.isChecked()
        config.write_title = self.title_checkbox.isChecked()
        config.write_subject = self.subject_checkbox.isChecked()
        config.write_description = self.description_checkbox.isChecked()
        config.image_instruction = self.image_instruction_input.text()

        # Run the indexer in a separate thread
        self.indexer_thread = IndexerThread(config)
        self.indexer_thread.output_received.connect(self.update_output)
        self.indexer_thread.finished.connect(self.indexer_finished)
        #self.pause_handler.pause_signal.connect(self.indexer_thread.setPaused)
        self.indexer_thread.start()

        self.output_area.clear()
        self.output_area.append("Running Image Indexer...\n")
        self.run_button.setEnabled(False)
        #self.pause_button.setEnabled(True)
        
    def update_output(self, text):
        self.output_area.append(text)
        self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())
        QApplication.processEvents() 

    def indexer_finished(self):
        self.update_output("Image Indexer finished.")
        self.run_button.setEnabled(True)
        #self.pause_button.setEnabled(False)
        #self.pause_button.setText("Pause")

    def toggle_pause(self):
        if self.pause_button.text() == "Pause":
            self.pause_handler.pause_signal.emit(True)
            self.pause_button.setText("Resume")
            self.update_output("Indexer paused.")
        else:
            self.pause_handler.pause_signal.emit(False)
            self.pause_button.setText("Pause")
            self.update_output("Indexer resumed.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageIndexerGUI()
    window.show()
    sys.exit(app.exec())
