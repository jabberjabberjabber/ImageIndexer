import sys
import os
import json
import shutil
import base64
import requests

from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, QSize
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QLineEdit, QCheckBox, QPushButton, QFileDialog,
                           QTextEdit, QGroupBox, QSpinBox, QDoubleSpinBox, QRadioButton, QButtonGroup,
                           QProgressBar, QTableWidget, QTableWidgetItem, QComboBox,
                           QPlainTextEdit, QScrollArea, QMessageBox, QDialog, QMenuBar,
                           QMenu, QSizePolicy, QSplitter, QFrame, QPushButton, QFrame,
                           QSizePolicy, QSpacerItem)


from PyQt6.QtGui import QPixmap, QImage, QPalette, QColor, QFont, QIcon


from . import llmii
from . import help_text

class GuiConfig:
    """ Configuration class for GUI dimensions and properties
    """
    WINDOW_WIDTH = 704
    WINDOW_HEIGHT = 720
    WINDOW_FIXED = False
    
    IMAGE_PREVIEW_WIDTH = 340
    IMAGE_PREVIEW_HEIGHT = 450
    
    METADATA_WIDTH = 360
    METADATA_HEIGHT = 450
    
    LOG_WIDTH = 700
    LOG_HEIGHT = 140
    
    CONTROL_PANEL_HEIGHT = 75
    SPLITTER_HANDLE_WIDTH = 4
    
    SETTINGS_HEIGHT = 660
    SETTINGS_WIDTH = 460
    
    FONT_SIZE_NORMAL = 9
    FONT_SIZE_HEADER = 10
    
    COLOR_KEYWORD_BG = "#e1f0ff"
    COLOR_KEYWORD_TEXT = "#0066cc"
    COLOR_KEYWORD_BORDER = "#99ccff"
    COLOR_CAPTION_BG = "#f9f9f9"
    COLOR_BORDER = "#cccccc"
    
    CONTENT_MARGINS = 1
    SPACING = 1
    KEYWORDS_PER_ROW = 5
    FILENAME_LABEL_HEIGHT = 20
    CAPTION_BOX_HEIGHT = 200
    KEYWORDS_BOX_HEIGHT = abs(METADATA_HEIGHT - (FILENAME_LABEL_HEIGHT + CAPTION_BOX_HEIGHT))
    DEFAULT_INSTRUCTION = """Return a JSON object containing a Description for the image and a list of Keywords.

Write the Description using the active voice.

Generate 5 to 10 Keywords. Each Keyword is an item in a list and will be composed of a maximum of two words.

For both Description and Keywords, make sure to include:

 - Themes, concepts
 - Items, animals, objects
 - Structures, landmarks, setting
 - Foreground and background elements
 - Notable colors, textures, styles
 - Actions, activities

If humans are present, include:
 - Physical appearance
 - Gender
 - Clothing
 - Age range
 - Visibly apparent ancestry
 - Occupation/role
 - Relationships between individuals
 - Emotions, expressions, body language

Use ENGLISH only. Generate ONLY a JSON object with the keys Description and Keywords as follows {"Description": str, "Keywords": []}"""
                
class InstructionDialog(QDialog):
    def __init__(self, instruction_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Instruction")
        self.setModal(True)
        self.resize(700, 500)

        layout = QVBoxLayout(self)

        self.instruction_input = QPlainTextEdit()
        self.instruction_input.setPlainText(instruction_text)
        layout.addWidget(QLabel("Edit Instruction:"))
        layout.addWidget(self.instruction_input)

        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def get_instruction(self):
        return self.instruction_input.toPlainText()

class SkipFoldersDialog(QDialog):
    def __init__(self, skip_folders_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Skip Folders")
        self.setModal(True)
        self.resize(700, 400)

        layout = QVBoxLayout(self)

        help_label = QLabel("Enter folder names or paths to skip (one per line or separated by semicolons).\nYou can use full paths or paths relative to the working directory.")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)

        self.skip_folders_input = QPlainTextEdit()
        self.skip_folders_input.setPlainText(skip_folders_text)
        layout.addWidget(QLabel("Skip Folders:"))
        layout.addWidget(self.skip_folders_input)

        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def get_skip_folders(self):
        return self.skip_folders_input.toPlainText()

class SettingsHelpDialog(QDialog):
    """ Dialog that shows help information for all settings """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings Help")
        self.resize(600, 500)
        
        layout = QVBoxLayout(self)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        from src.help_text import get_settings_help
        help_label = QLabel(get_settings_help())
        help_label.setWordWrap(True)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setOpenExternalLinks(True)
        
        scroll_area.setWidget(help_label)
        layout.addWidget(scroll_area)
        
        button_layout = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addStretch(1)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(GuiConfig.SETTINGS_WIDTH, GuiConfig.SETTINGS_HEIGHT)
        
        layout = QVBoxLayout(self)
        
        # Scroll area in case it gets too long
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        api_layout = QHBoxLayout()
        self.api_url_input = QLineEdit("http://localhost:5001")
        self.api_password_input = QLineEdit()
        api_layout.addWidget(QLabel("API URL:"))
        api_layout.addWidget(self.api_url_input)
        api_layout.addWidget(QLabel("API Password:"))
        api_layout.addWidget(self.api_password_input)
        scroll_layout.addLayout(api_layout)

        system_instruction_layout = QHBoxLayout()
        self.system_instruction_input = QLineEdit("You are a helpful assistant.")
        system_instruction_layout.addWidget(QLabel("System Instruction:"))
        system_instruction_layout.addWidget(self.system_instruction_input)
        scroll_layout.addLayout(system_instruction_layout)

        instruction_button_layout = QHBoxLayout()
        self.edit_instruction_button = QPushButton("Edit Instruction")
        self.edit_instruction_button.clicked.connect(self.edit_instruction)
        instruction_button_layout.addWidget(self.edit_instruction_button)
        scroll_layout.addLayout(instruction_button_layout)

        skip_folders_button_layout = QHBoxLayout()
        self.edit_skip_folders_button = QPushButton("Edit Skip Folders")
        self.edit_skip_folders_button.clicked.connect(self.edit_skip_folders)
        skip_folders_button_layout.addWidget(self.edit_skip_folders_button)
        scroll_layout.addLayout(skip_folders_button_layout)

        caption_group = QGroupBox("Caption Options")
        caption_layout = QVBoxLayout()

        caption_instruction_layout = QHBoxLayout()
        self.caption_instruction_input = QLineEdit("Describe the image in detail. Be specific.")
        caption_instruction_layout.addWidget(QLabel("Caption Instruction:"))
        caption_instruction_layout.addWidget(self.caption_instruction_input)
        caption_layout.addLayout(caption_instruction_layout)

        tag_instruction_layout = QHBoxLayout()
        self.tag_instruction_input = QLineEdit('Return a JSON object with key Keywords with the value as array of Keywords and tags that describe the image as follows: {"Keywords": []}')
        tag_instruction_layout.addWidget(QLabel("Tag Instruction:"))
        tag_instruction_layout.addWidget(self.tag_instruction_input)
        caption_layout.addLayout(tag_instruction_layout)

        self.caption_radio_group = QButtonGroup(self)
        self.detailed_caption_radio = QRadioButton("Separate caption query")
        self.short_caption_radio = QRadioButton("Combined caption query")
        self.no_caption_radio = QRadioButton("No caption query")

        self.caption_radio_group.addButton(self.detailed_caption_radio)
        self.caption_radio_group.addButton(self.short_caption_radio)
        self.caption_radio_group.addButton(self.no_caption_radio)

        self.short_caption_radio.setChecked(True)

        caption_layout.addWidget(self.detailed_caption_radio)
        caption_layout.addWidget(self.short_caption_radio)
        caption_layout.addWidget(self.no_caption_radio)

        caption_group.setLayout(caption_layout)
        scroll_layout.addWidget(caption_group)

        gen_count_layout = QHBoxLayout()
        self.gen_count = QSpinBox()
        self.gen_count.setMinimum(50)
        self.gen_count.setMaximum(1000)
        self.gen_count.setValue(250)
        gen_count_layout.addWidget(QLabel("GenTokens: "))
        gen_count_layout.addWidget(self.gen_count)
        scroll_layout.addLayout(gen_count_layout)
        
        res_limit_layout = QHBoxLayout()
        self.res_limit = QSpinBox()
        self.res_limit.setMinimum(112)
        self.res_limit.setMaximum(896)
        self.res_limit.setValue(448)
        self.res_limit.setSingleStep(14)
        res_limit_layout.addWidget(QLabel("Dimension length: "))
        res_limit_layout.addWidget(self.res_limit)
        scroll_layout.addLayout(res_limit_layout)

        # Sampler Settings Group
        sampler_group = QGroupBox("Sampler Settings")
        sampler_layout = QVBoxLayout()

        # Temperature
        temp_layout = QHBoxLayout()
        self.temperature_spinbox = QDoubleSpinBox()
        self.temperature_spinbox.setMinimum(0.0)
        self.temperature_spinbox.setMaximum(2.0)
        self.temperature_spinbox.setValue(0.2)
        self.temperature_spinbox.setSingleStep(0.05)
        self.temperature_spinbox.setDecimals(2)
        temp_layout.addWidget(QLabel("Temperature:"))
        temp_layout.addWidget(self.temperature_spinbox)
        temp_layout.addStretch()
        sampler_layout.addLayout(temp_layout)

        # Top P
        top_p_layout = QHBoxLayout()
        self.top_p_spinbox = QDoubleSpinBox()
        self.top_p_spinbox.setMinimum(0.0)
        self.top_p_spinbox.setMaximum(1.0)
        self.top_p_spinbox.setValue(1.0)
        self.top_p_spinbox.setSingleStep(0.01)
        self.top_p_spinbox.setDecimals(2)
        top_p_layout.addWidget(QLabel("Top P:"))
        top_p_layout.addWidget(self.top_p_spinbox)
        top_p_layout.addStretch()
        sampler_layout.addLayout(top_p_layout)

        # Top K
        top_k_layout = QHBoxLayout()
        self.top_k_spinbox = QSpinBox()
        self.top_k_spinbox.setMinimum(0)
        self.top_k_spinbox.setMaximum(100)
        self.top_k_spinbox.setValue(100)
        top_k_layout.addWidget(QLabel("Top K:"))
        top_k_layout.addWidget(self.top_k_spinbox)
        top_k_layout.addStretch()
        sampler_layout.addLayout(top_k_layout)

        # Min P
        min_p_layout = QHBoxLayout()
        self.min_p_spinbox = QDoubleSpinBox()
        self.min_p_spinbox.setMinimum(0.0)
        self.min_p_spinbox.setMaximum(2.0)
        self.min_p_spinbox.setValue(0.05)
        self.min_p_spinbox.setSingleStep(0.01)
        self.min_p_spinbox.setDecimals(2)
        min_p_layout.addWidget(QLabel("Min P:"))
        min_p_layout.addWidget(self.min_p_spinbox)
        min_p_layout.addStretch()
        sampler_layout.addLayout(min_p_layout)

        # Repetition Penalty
        rep_pen_layout = QHBoxLayout()
        self.rep_pen_spinbox = QDoubleSpinBox()
        self.rep_pen_spinbox.setMinimum(1.0)
        self.rep_pen_spinbox.setMaximum(2.0)
        self.rep_pen_spinbox.setValue(1.01)
        self.rep_pen_spinbox.setSingleStep(0.01)
        self.rep_pen_spinbox.setDecimals(2)
        rep_pen_layout.addWidget(QLabel("Repetition Penalty:"))
        rep_pen_layout.addWidget(self.rep_pen_spinbox)
        rep_pen_layout.addStretch()
        sampler_layout.addLayout(rep_pen_layout)

        sampler_group.setLayout(sampler_layout)
        scroll_layout.addWidget(sampler_group)

        # JSON Grammar Group
        json_grammar_group = QGroupBox("Structured Output")
        json_grammar_layout = QVBoxLayout()

        self.use_json_grammar_checkbox = QCheckBox("Use JSON grammar to force structured output")
        self.use_json_grammar_checkbox.setChecked(False)
        json_grammar_layout.addWidget(self.use_json_grammar_checkbox)

        json_grammar_group.setLayout(json_grammar_layout)
        scroll_layout.addWidget(json_grammar_group)

        options_group = QGroupBox("File Options")
        options_layout = QVBoxLayout()
        
        self.no_crawl_checkbox = QCheckBox("Don't go in subdirectories")
        self.reprocess_all_checkbox = QCheckBox("Reprocess everything")
        self.reprocess_failed_checkbox = QCheckBox("Reprocess failures")
        self.reprocess_orphans_checkbox = QCheckBox("Fix any orphans")
        self.no_backup_checkbox = QCheckBox("No backups")
        self.dry_run_checkbox = QCheckBox("Pretend mode")
        self.skip_verify_checkbox = QCheckBox("No file validation")
        self.quick_fail_checkbox = QCheckBox("No retries")
        self.use_sidecar_checkbox = QCheckBox("Use metadata sidecar file instead of writing to image")
        self.write_unsafe_checkbox = QCheckBox("Use unsafe flag when writing metadata")        
        options_layout.addWidget(self.no_crawl_checkbox)
        options_layout.addWidget(self.reprocess_all_checkbox)
        options_layout.addWidget(self.reprocess_failed_checkbox)
        options_layout.addWidget(self.reprocess_orphans_checkbox)
        options_layout.addWidget(self.no_backup_checkbox)
        options_layout.addWidget(self.dry_run_checkbox)
        options_layout.addWidget(self.skip_verify_checkbox)
        options_layout.addWidget(self.quick_fail_checkbox)
        options_layout.addWidget(self.use_sidecar_checkbox)
        options_layout.addWidget(self.write_unsafe_checkbox)
        
        options_group.setLayout(options_layout)
        scroll_layout.addWidget(options_group)
        
        xmp_group = QGroupBox("Existing Metadata")
        xmp_layout = QVBoxLayout()
        
        self.update_keywords_checkbox = QCheckBox("Don't clear existing keywords (new will be added)")
        self.update_keywords_checkbox.setChecked(True)
        self.update_caption_checkbox = QCheckBox("Don't clear existing caption (new will be added surrounded by tags)")
        self.update_caption_checkbox.setChecked(False)
        
        xmp_layout.addWidget(self.update_keywords_checkbox)
        xmp_layout.addWidget(self.update_caption_checkbox)
        
        xmp_group.setLayout(xmp_layout)
        scroll_layout.addWidget(xmp_group)

        keyword_corrections_group = QGroupBox("Keyword Corrections")
        corrections_layout = QVBoxLayout()
        
        self.depluralize_checkbox = QCheckBox("Depluralize keywords")
        self.depluralize_checkbox.setChecked(False)
        self.word_limit_layout = QHBoxLayout()
        self.word_limit_checkbox = QCheckBox("Limit to")
        self.word_limit_spinbox = QSpinBox()
        self.word_limit_spinbox.setMinimum(1)
        self.word_limit_spinbox.setMaximum(5)
        self.word_limit_spinbox.setValue(2)
        self.word_limit_layout.addWidget(self.word_limit_checkbox)
        self.word_limit_layout.addWidget(self.word_limit_spinbox)
        self.word_limit_layout.addWidget(QLabel("words in keyword entry"))
        self.word_limit_layout.addStretch(1)
        self.word_limit_checkbox.setChecked(True)
        self.split_and_checkbox = QCheckBox("Split 'and'/'or' entries")
        self.split_and_checkbox.setChecked(True)
        self.ban_prompt_words_checkbox = QCheckBox("Ban prompt word repetitions")
        self.ban_prompt_words_checkbox.setChecked(True)
        self.no_digits_start_checkbox = QCheckBox("Cannot start with 3+ digits")
        self.no_digits_start_checkbox.setChecked(True)
        self.min_word_length_checkbox = QCheckBox("Words must be 2+ characters")
        self.min_word_length_checkbox.setChecked(True)
        self.latin_only_checkbox = QCheckBox("Only Latin characters")
        self.latin_only_checkbox.setChecked(True)
        
        corrections_layout.addWidget(self.depluralize_checkbox)
        corrections_layout.addLayout(self.word_limit_layout)
        corrections_layout.addWidget(self.split_and_checkbox)
        corrections_layout.addWidget(self.ban_prompt_words_checkbox)
        corrections_layout.addWidget(self.no_digits_start_checkbox)
        corrections_layout.addWidget(self.min_word_length_checkbox)
        corrections_layout.addWidget(self.latin_only_checkbox)
        
        keyword_corrections_group.setLayout(corrections_layout)
        scroll_layout.addWidget(keyword_corrections_group)
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)
        
        button_layout = QHBoxLayout()
        help_button = QPushButton("Help")
        help_button.clicked.connect(self.show_help)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(help_button)
        button_layout.addStretch(1)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.instruction_text = GuiConfig.DEFAULT_INSTRUCTION
        self.skip_folders_text = ""

        self.load_settings()
    
    def show_help(self):
        """Show the settings help dialog"""
        dialog = SettingsHelpDialog(self)
        dialog.exec()
        
    def edit_instruction(self):
        dialog = InstructionDialog(self.instruction_text, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.instruction_text = dialog.get_instruction()

    def edit_skip_folders(self):
        dialog = SkipFoldersDialog(self.skip_folders_text, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.skip_folders_text = dialog.get_skip_folders()
        
    def load_settings(self):
        try:
            if os.path.exists('settings.json'):
                with open('settings.json', 'r') as f:
                    settings = json.load(f)
                    
                self.api_url_input.setText(settings.get('api_url', 'http://localhost:5001'))
                self.api_password_input.setText(settings.get('api_password', ''))
                self.system_instruction_input.setText(settings.get('system_instruction', 'You are a helpful assistant.'))
                self.gen_count.setValue(settings.get('gen_count', 250))
                self.res_limit.setValue(settings.get('res_limit', 448))
                self.instruction_text = settings.get('instruction', GuiConfig.DEFAULT_INSTRUCTION)
                self.skip_folders_text = settings.get('skip_folders', '')
                
                self.no_crawl_checkbox.setChecked(settings.get('no_crawl', False))
                self.reprocess_failed_checkbox.setChecked(settings.get('reprocess_failed', False))
                self.reprocess_all_checkbox.setChecked(settings.get('reprocess_all', False))
                self.reprocess_orphans_checkbox.setChecked(settings.get('reprocess_orphans', True))
                self.no_backup_checkbox.setChecked(settings.get('no_backup', False))
                self.dry_run_checkbox.setChecked(settings.get('dry_run', False))
                self.skip_verify_checkbox.setChecked(settings.get('skip_verify', False))
                self.quick_fail_checkbox.setChecked(settings.get('quick_fail', False))
                self.use_sidecar_checkbox.setChecked(settings.get('use_sidecar', False))
                self.write_unsafe_checkbox.setChecked(settings.get('write_unsafe', False))
                self.caption_instruction_input.setText(settings.get('caption_instruction', 'Describe the image in detail. Be specific.'))
                self.tag_instruction_input.setText(settings.get('tag_instruction', 'Return a JSON object with key Keywords with the value as array of Keywords and tags that describe the image as follows: {"Keywords": []}'))
                
                # Set radio button based on settings
                if settings.get('detailed_caption', False):
                    self.detailed_caption_radio.setChecked(True)
                elif settings.get('no_caption', False):
                    self.no_caption_radio.setChecked(True)
                else:
                    # Default to short caption
                    self.short_caption_radio.setChecked(True)
                    
                self.update_keywords_checkbox.setChecked(settings.get('update_keywords', True))
                self.update_caption_checkbox.setChecked(settings.get('update_caption', False))
                
                # Load keyword correction settings
                self.depluralize_checkbox.setChecked(settings.get('depluralize_keywords', False))
                self.word_limit_checkbox.setChecked(settings.get('limit_word_count', True))
                self.word_limit_spinbox.setValue(settings.get('max_words_per_keyword', 2))
                self.split_and_checkbox.setChecked(settings.get('split_and_entries', True))
                self.ban_prompt_words_checkbox.setChecked(settings.get('ban_prompt_words', True))
                self.no_digits_start_checkbox.setChecked(settings.get('no_digits_start', True))
                self.min_word_length_checkbox.setChecked(settings.get('min_word_length', True))
                self.latin_only_checkbox.setChecked(settings.get('latin_only', True))

                # Load sampler settings
                self.temperature_spinbox.setValue(settings.get('temperature', 0.2))
                self.top_p_spinbox.setValue(settings.get('top_p', 1.0))
                self.top_k_spinbox.setValue(settings.get('top_k', 100))
                self.min_p_spinbox.setValue(settings.get('min_p', 0.05))
                self.rep_pen_spinbox.setValue(settings.get('rep_pen', 1.01))

                # Load JSON grammar setting
                self.use_json_grammar_checkbox.setChecked(settings.get('use_json_grammar', False))

        except Exception as e:
            print(f"Error loading settings: {e}")
            
    def save_settings(self):
        settings = {
            'api_url': self.api_url_input.text(),
            'api_password': self.api_password_input.text(),
            'system_instruction': self.system_instruction_input.text(),
            'instruction': self.instruction_text,
            'skip_folders': self.skip_folders_text,
            'gen_count': self.gen_count.value(),
            'res_limit': self.res_limit.value(),
            'no_crawl': self.no_crawl_checkbox.isChecked(),
            'reprocess_failed': self.reprocess_failed_checkbox.isChecked(),
            'reprocess_all': self.reprocess_all_checkbox.isChecked(),
            'reprocess_orphans': self.reprocess_orphans_checkbox.isChecked(),
            'no_backup': self.no_backup_checkbox.isChecked(),
            'dry_run': self.dry_run_checkbox.isChecked(),
            'skip_verify': self.skip_verify_checkbox.isChecked(),
            'quick_fail': self.quick_fail_checkbox.isChecked(),
            'update_keywords': self.update_keywords_checkbox.isChecked(),
            'caption_instruction': self.caption_instruction_input.text(),
            'tag_instruction': self.tag_instruction_input.text(),
            'detailed_caption': self.detailed_caption_radio.isChecked(),
            'short_caption': self.short_caption_radio.isChecked(),
            'no_caption': self.no_caption_radio.isChecked(),
            'update_caption': self.update_caption_checkbox.isChecked(),
            'use_sidecar': self.use_sidecar_checkbox.isChecked(),
            'write_unsafe': self.write_unsafe_checkbox.isChecked(),
            'depluralize_keywords': self.depluralize_checkbox.isChecked(),
            'limit_word_count': self.word_limit_checkbox.isChecked(),
            'max_words_per_keyword': self.word_limit_spinbox.value(),
            'split_and_entries': self.split_and_checkbox.isChecked(),
            'ban_prompt_words': self.ban_prompt_words_checkbox.isChecked(),
            'no_digits_start': self.no_digits_start_checkbox.isChecked(),
            'min_word_length': self.min_word_length_checkbox.isChecked(),
            'latin_only': self.latin_only_checkbox.isChecked(),
            'temperature': self.temperature_spinbox.value(),
            'top_p': self.top_p_spinbox.value(),
            'top_k': self.top_k_spinbox.value(),
            'min_p': self.min_p_spinbox.value(),
            'rep_pen': self.rep_pen_spinbox.value(),
            'use_json_grammar': self.use_json_grammar_checkbox.isChecked(),
        }

        try:
            with open('settings.json', 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save settings: {e}")

class APICheckThread(QThread):
    api_status = pyqtSignal(bool)
    
    def __init__(self, api_url):
        super().__init__()
        self.api_url = api_url
        self.running = True
        
    def run(self):

        while self.running:
            try:
                # Direct HTTP request to the version endpoint
                response = requests.get(f"{self.api_url}/api/extra/version", timeout=5)
                if response.status_code == 200:
                    self.api_status.emit(True)
                    break
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    self.api_status.emit(True)
                    break
            except Exception:
                self.api_status.emit(False)
            self.msleep(1000)
            
    def stop(self):
        self.running = False

class IndexerThread(QThread):
    output_received = pyqtSignal(str)
    image_processed = pyqtSignal(str, str, list, str)  # base64_image, caption, keywords, filename

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.paused = False
        self.stopped = False

    def process_callback(self, message):
        """Callback for llmii's process_file function"""
        # Check if message is a dictionary with image data
        if isinstance(message, dict) and 'type' in message and message['type'] == 'image_data':
            # Extract the image data and emit signal
            base64_image = message.get('base64_image', '')
            caption = message.get('caption', '')
            keywords = message.get('keywords', [])
            file_path = message.get('file_path', '')
            self.image_processed.emit(base64_image, caption, keywords, file_path)
        else:
            # Regular text message for the log
            self.output_received.emit(str(message))

    def run(self):
        try:
            # Pass our callback function to llmii
            llmii.main(self.config, self.process_callback, self.check_paused_or_stopped)
        except Exception as e:
            self.output_received.emit(f"Error: {str(e)}")

    def check_paused_or_stopped(self):
        if self.stopped:
            raise Exception("Indexer stopped by user")
        if self.paused:
            while self.paused and not self.stopped:
                self.msleep(100)
            if self.stopped:
                raise Exception("Indexer stopped by user")
        return self.paused

class KeywordWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.keywords = []
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS)
        
        # Create the container for keyword rows
        self.keywords_container = QWidget()
        self.keywords_layout = QVBoxLayout(self.keywords_container)
        self.keywords_container.setStyleSheet("border: none; padding: 0px")
        self.keywords_layout.setContentsMargins(GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS)
        self.keywords_layout.setSpacing(0)
        self.keywords_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
        # Set fixed size policy for container
        self.keywords_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        # Add header and container to layout
        #self.layout.addWidget(QLabel("Keywords:"))
        self.layout.addWidget(self.keywords_container)
        
        # Ensure widget doesn't expand beyond its allocated space
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    
    def clear(self):
        # Clear keywords layout
        for i in reversed(range(self.keywords_layout.count())): 
            widget = self.keywords_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.keywords = []
    
    def set_keywords(self, keywords):
        self.clear()
        self.keywords = keywords
        
        # Display keywords in rows
        max_per_row = GuiConfig.KEYWORDS_PER_ROW
        current_row = 0
        row_layouts = []
        
        # Create the first row
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)
        row_layouts.append(row_layout)
        self.keywords_layout.addWidget(row_widget)
        
        # Add keyword labels
        for i, keyword in enumerate(keywords):
            # Check if we need to start a new row
            if i > 0 and i % max_per_row == 0:
                # Create a new row
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(2)
                row_layouts.append(row_layout)
                self.keywords_layout.addWidget(row_widget)
                current_row += 1
                
            # Create the keyword label
            keyword_label = QLabel(keyword)
            keyword_label.setStyleSheet(f"""
                background-color: {GuiConfig.COLOR_KEYWORD_BG}; 
                color: {GuiConfig.COLOR_KEYWORD_TEXT};
                padding: 1px 4px;
                border-radius: 5px;
                border: 1px solid {GuiConfig.COLOR_KEYWORD_BORDER};
                margin: 2px;
                font-size: {GuiConfig.FONT_SIZE_NORMAL}px;
            """)
            
            # Add the keyword to the current row
            row_layouts[current_row].addWidget(keyword_label)
        
        # Add stretch to each row to push keywords to the left
        for row_layout in row_layouts:
            row_layout.addStretch(1)

class PauseHandler(QObject):
    pause_signal = pyqtSignal(bool)
    stop_signal = pyqtSignal()

class ImageIndexerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Apply fixed window size
        self.setWindowTitle("Image Indexer GUI")
        self.setFixedSize(GuiConfig.WINDOW_WIDTH, GuiConfig.WINDOW_HEIGHT)
        # Disable maximize button and resizing
        if GuiConfig.WINDOW_FIXED:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowMaximizeButtonHint)
            self.setFixedSize(GuiConfig.WINDOW_WIDTH, GuiConfig.WINDOW_HEIGHT)
            
        self.settings_dialog = SettingsDialog(self)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS)
        
        # Upper section with controls - fixed height
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(GuiConfig.SPACING)
        
        # Directory and Settings section
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        dir_button = QPushButton("Select Directory")
        dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(QLabel("Directory:"))
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(dir_button)
        controls_layout.addLayout(dir_layout)

        # Settings button and API status in one row
        settings_api_layout = QHBoxLayout()
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.show_settings)
        self.api_status_label = QLabel("API Status: Checking...")
        settings_api_layout.addWidget(settings_button)
        settings_api_layout.addStretch(1)
        settings_api_layout.addWidget(self.api_status_label)
        controls_layout.addLayout(settings_api_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Image Indexer")
        self.run_button.clicked.connect(self.run_indexer)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_indexer)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        controls_layout.addLayout(button_layout)
        
        # Set fixed height for controls widget
        controls_widget.setFixedHeight(GuiConfig.CONTROL_PANEL_HEIGHT)
        main_layout.addWidget(controls_widget)
        nav_widget = QWidget()
        
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS)
        nav_layout.setSpacing(GuiConfig.SPACING)

        # Create navigation buttons
        self.first_button = QPushButton("|<")  # Go to first image
        self.prev_button = QPushButton("<")    # Go to previous image
        self.position_label = QLabel("No images processed")
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_button = QPushButton(">")    # Go to next image
        self.last_button = QPushButton(">|")   # Go to most recent image

        # Add widgets to layout
        nav_layout.addWidget(self.first_button)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.position_label)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.last_button)

        # Connect button signals to slots
        self.first_button.clicked.connect(self.navigate_first)
        self.prev_button.clicked.connect(self.navigate_prev)
        self.next_button.clicked.connect(self.navigate_next)
        self.last_button.clicked.connect(self.navigate_last)

        # Set initial button states (disabled until we have images)
        self.first_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.last_button.setEnabled(False)

        # Add to the main layout
        main_layout.addWidget(nav_widget)
        
        # Middle section with image and metadata side by side
        middle_section = QWidget()
        middle_layout = QHBoxLayout(middle_section)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(GuiConfig.SPACING)
        
        # Image preview panel - fixed size
        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.Shape.Box)
        image_frame.setStyleSheet(f"border: 1px solid {GuiConfig.COLOR_BORDER}; padding: 0px;")
        
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(1, 1, 1, 1)
        
        # Image preview label with fixed size
        self.image_preview = QLabel("No image processed yet")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setFixedSize(GuiConfig.IMAGE_PREVIEW_WIDTH -12, GuiConfig.IMAGE_PREVIEW_HEIGHT -12)
        self.image_preview.setFrameShape(QFrame.Shape.NoFrame)
        self.image_preview.setStyleSheet("border: none; background-color: transparent;")
               
        image_layout.addWidget(self.image_preview, 0, Qt.AlignmentFlag.AlignCenter)
        
        image_frame.setFixedSize(GuiConfig.IMAGE_PREVIEW_WIDTH, GuiConfig.IMAGE_PREVIEW_HEIGHT)
        middle_layout.addWidget(image_frame)
        
        # Metadata panel - fixed size
        metadata_frame = QFrame()
        
        metadata_frame.setFrameShape(QFrame.Shape.Box)
        metadata_frame.setStyleSheet(f"border: 1px solid {GuiConfig.COLOR_BORDER};")
        #metadata_frame.setStyleSheet("")
        metadata_layout = QVBoxLayout(metadata_frame)
        metadata_layout.setContentsMargins(GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS)
        metadata_layout.setSpacing(GuiConfig.SPACING)
        
        # Image filename
        self.filename_label = QLabel("Filename: ")
        self.filename_label.setStyleSheet("font-weight: bold; border:none;")
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.filename_label.setFixedHeight(GuiConfig.FILENAME_LABEL_HEIGHT)
        
        metadata_layout.addWidget(self.filename_label)
        
        # Caption
        caption_group = QGroupBox("Caption")
        caption_group.setStyleSheet("QGroupBox { border: none; }")
        caption_layout = QVBoxLayout(caption_group)
        
        caption_layout.setContentsMargins(GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS)
        caption_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.caption_label = QLabel("No caption generated yet")
        self.caption_label.setWordWrap(True)
        #self.caption_label.setFrameStyle(QFrame.Shape.NoFrame)
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.caption_label.setStyleSheet(f"padding: 4px; font-weight: normal; border:none;")
        
        # Create a scroll area for caption to ensure it fits in fixed space
        caption_scroll = QScrollArea()
        caption_scroll.setWidgetResizable(True)
        caption_scroll.setWidget(self.caption_label)
        caption_scroll.setFixedHeight(200)  # Ensure fixed height for caption area
        caption_scroll.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        caption_layout.addWidget(caption_scroll, 0, Qt.AlignmentFlag.AlignTop)
        metadata_layout.addWidget(caption_group, 0, Qt.AlignmentFlag.AlignTop)
        
        # Keywords
        self.keywords_widget = KeywordWidget()
        metadata_layout.addWidget(self.keywords_widget)
        
        # Set fixed size for metadata frame
        metadata_frame.setFixedSize(GuiConfig.METADATA_WIDTH, GuiConfig.METADATA_HEIGHT)
        middle_layout.addWidget(metadata_frame)
        
        # Add middle section to main layout
        main_layout.addWidget(middle_section)
        
        # Bottom section - log output with fixed size
        log_frame = QFrame()
        log_frame.setFrameShape(QFrame.Shape.Box)
        log_frame.setFrameStyle(QFrame.Shape.NoFrame)
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS, GuiConfig.CONTENT_MARGINS)
        log_label = QLabel("Processing Log:")
        log_layout.addWidget(log_label)
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        log_layout.addWidget(self.output_area)
        
        # Set fixed size for log frame
        log_frame.setFixedSize(GuiConfig.LOG_WIDTH, GuiConfig.LOG_HEIGHT)
        main_layout.addWidget(log_frame)
        
        # Store the previous image data to keep showing something
        self.previous_image_data = None
        self.previous_caption = None
        self.previous_keywords = None
        self.previous_filename = None
        self.pause_handler = PauseHandler()
        self.api_check_thread = None
        self.api_is_ready = False
        self.run_button.setEnabled(False)
        self.image_history = []  # [(base64_image, caption, keywords, filename)]
        self.current_position = -1
        
        if os.path.exists('settings.json'):
            try:
                with open('settings.json', 'r') as f:
                    settings = json.load(f)
                    self.dir_input.setText(settings.get('directory', ''))
                    self.start_api_check(settings.get('api_url', 'http://localhost:5001'))
                    
            except Exception as e:
                print(f"Error loading settings: {e}")
                self.start_api_check('http://localhost:5001')
        else:
            self.start_api_check('http://localhost:5001')

    def show_settings(self):
        if self.settings_dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings_dialog.save_settings()
            
            self.start_api_check(self.settings_dialog.api_url_input.text())
            
            try:
                with open('settings.json', 'r') as f:
                    settings = json.load(f)
                settings['directory'] = self.dir_input.text()
                with open('settings.json', 'w') as f:
                    json.dump(settings, f, indent=4)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save directory setting: {e}")

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)

    def start_api_check(self, api_url):
        self.api_url = api_url
        if self.api_check_thread and self.api_check_thread.isRunning():
            self.api_check_thread.stop()
            self.api_check_thread.wait()
            
        self.api_is_ready = False
        self.run_button.setEnabled(False)
        self.api_status_label.setText("API Status: Checking...")
        self.api_status_label.setStyleSheet("color: orange; padding: 4px")

        self.api_check_thread = APICheckThread(api_url if api_url else self.settings_dialog.api_url_input.text())
        self.api_check_thread.api_status.connect(self.update_api_status)
        self.api_check_thread.start()

    def update_api_status(self, is_available):
        if is_available:
            self.api_is_ready = True
            self.api_status_label.setText("API Status: Connected")
            self.api_status_label.setStyleSheet("color: green; padding: 4px")
            self.run_button.setEnabled(True)
            
            # Stop the check thread once we're connected
            if self.api_check_thread:
                self.api_check_thread.stop()
        else:
            self.api_is_ready = False
            self.api_status_label.setText("API Status: Waiting for connection...")
            self.api_status_label.setStyleSheet("color: red; padding: 4px")
            self.run_button.setEnabled(False)
    
    def update_image_preview(self, base64_image, caption, keywords, filename):
        self.previous_image_data = base64_image
        self.previous_caption = caption
        self.previous_keywords = keywords
        self.previous_filename = filename
        
        # Add to history
        self.image_history.append((base64_image, caption, keywords, filename))
        
        # If user was viewing the most recent image (or this is the first image),
        # update current_position to point to the new image
        if self.current_position == -1 or len(self.image_history) <= 1:
            self.current_position = -1  # Keep at most recent
            self.display_image(base64_image, caption, keywords, filename)
        else:
            # Just update navigation buttons without changing the view
            self.update_navigation_buttons()
            
    def display_image(self, base64_image, caption, keywords, filename):
        # Update the UI with the image data
        try:
            # Convert base64 to QImage
            image_data = base64.b64decode(base64_image)
            image = QImage.fromData(image_data)
            if not image.isNull():
                pixmap = QPixmap.fromImage(image)
                
                # Scale the pixmap to fit the fixed container while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    GuiConfig.IMAGE_PREVIEW_WIDTH,
                    GuiConfig.IMAGE_PREVIEW_HEIGHT, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                
                self.image_preview.setPixmap(scaled_pixmap)
                self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                self.image_preview.setText("Error loading image")
        except Exception as e:
            self.image_preview.setText(f"Error: {str(e)}")
        
        file_basename = os.path.basename(filename)
        self.filename_label.setText(f"Filename: {file_basename}")
        self.caption_label.setText(caption or "No caption generated")
        self.keywords_widget.set_keywords(keywords or [])
        self.update_navigation_buttons()

    def navigate_first(self):
        if self.image_history:
            self.current_position = 0
            base64_image, caption, keywords, filename = self.image_history[0]
            self.display_image(base64_image, caption, keywords, filename)

    def navigate_prev(self):
        if not self.image_history:
            return
            
        if self.current_position == -1:
            # If at the most recent, go to the second most recent
            if len(self.image_history) > 1:
                self.current_position = len(self.image_history) - 2
                base64_image, caption, keywords, filename = self.image_history[self.current_position]
                self.display_image(base64_image, caption, keywords, filename)
        elif self.current_position > 0:
            self.current_position -= 1
            base64_image, caption, keywords, filename = self.image_history[self.current_position]
            self.display_image(base64_image, caption, keywords, filename)

    def navigate_next(self):
        if not self.image_history:
            return
            
        if self.current_position != -1 and self.current_position < len(self.image_history) - 1:
            self.current_position += 1
            
            # If we've reached the end, set to -1 to indicate "most recent"
            if self.current_position == len(self.image_history) - 1:
                self.current_position = -1
                
            base64_image, caption, keywords, filename = self.image_history[
                len(self.image_history) - 1 if self.current_position == -1 else self.current_position
            ]
            self.display_image(base64_image, caption, keywords, filename)

    def navigate_last(self):
        if self.image_history:
            self.current_position = -1
            base64_image, caption, keywords, filename = self.image_history[-1]
            self.display_image(base64_image, caption, keywords, filename)

    def update_navigation_buttons(self):
        history_size = len(self.image_history)
        
        if history_size == 0:
            # No images yet
            self.first_button.setEnabled(False)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.last_button.setEnabled(False)
            self.position_label.setText("No images processed")
            return
        
        # Determine position for display
        if self.current_position == -1:
            # At the most recent image
            position = history_size
            self.next_button.setEnabled(False)
            self.last_button.setEnabled(False)
        else:
            position = self.current_position + 1  # 1-based for display
            self.next_button.setEnabled(self.current_position < history_size - 1)
            self.last_button.setEnabled(self.current_position < history_size - 1)
        
        # Update position text
        self.position_label.setText(f"Image {position} of {history_size}")
        
        # Enable/disable first/prev buttons
        self.first_button.setEnabled(history_size > 1 and (self.current_position > 0 or self.current_position == -1))
        self.prev_button.setEnabled(history_size > 1 and (self.current_position > 0 or self.current_position == -1))
          
    def run_indexer(self):
        
        if not self.api_is_ready:
            QMessageBox.warning(self, "API Not Ready", 
                              "Please wait for the API to be available before running the indexer.")
            return
        
        self.image_history = []
        self.current_position = -1
        self.update_navigation_buttons()
        
        config = llmii.Config()
        
        self.image_preview.setText("No image processed yet")
        self.filename_label.setText("Filename: ")
        self.caption_label.setText("No caption generated yet")
        self.keywords_widget.clear()
        
        # Get directory from main window
        config.directory = self.dir_input.text()
        
        # Load settings from settings dialog
        config.api_url = self.settings_dialog.api_url_input.text()
        config.api_password = self.settings_dialog.api_password_input.text()
        config.system_instruction = self.settings_dialog.system_instruction_input.text()
        config.no_crawl = self.settings_dialog.no_crawl_checkbox.isChecked()
        config.reprocess_failed = self.settings_dialog.reprocess_failed_checkbox.isChecked()
        config.reprocess_all = self.settings_dialog.reprocess_all_checkbox.isChecked()
        config.reprocess_orphans = self.settings_dialog.reprocess_orphans_checkbox.isChecked()
        config.no_backup = self.settings_dialog.no_backup_checkbox.isChecked()
        config.dry_run = self.settings_dialog.dry_run_checkbox.isChecked()
        config.skip_verify = self.settings_dialog.skip_verify_checkbox.isChecked()
        config.quick_fail = self.settings_dialog.quick_fail_checkbox.isChecked()
        config.use_sidecar = self.settings_dialog.use_sidecar_checkbox.isChecked()
        config.write_unsafe = self.settings_dialog.write_unsafe_checkbox.isChecked()
        config.normalize_keywords = True
        config.depluralize_keywords = self.settings_dialog.depluralize_checkbox.isChecked()
        config.limit_word_count = self.settings_dialog.word_limit_checkbox.isChecked()
        config.max_words_per_keyword = self.settings_dialog.word_limit_spinbox.value()
        config.split_and_entries = self.settings_dialog.split_and_checkbox.isChecked()
        config.ban_prompt_words = self.settings_dialog.ban_prompt_words_checkbox.isChecked()
        config.no_digits_start = self.settings_dialog.no_digits_start_checkbox.isChecked()
        config.min_word_length = self.settings_dialog.min_word_length_checkbox.isChecked()
        config.latin_only = self.settings_dialog.latin_only_checkbox.isChecked()
        
        # Load caption settings
        config.detailed_caption = self.settings_dialog.detailed_caption_radio.isChecked()
        config.short_caption = self.settings_dialog.short_caption_radio.isChecked()
        config.no_caption = self.settings_dialog.no_caption_radio.isChecked()
        config.caption_instruction = self.settings_dialog.caption_instruction_input.text()
        config.tag_instruction = self.settings_dialog.tag_instruction_input.text()

        # Load instruction from settings
        config.instruction = self.settings_dialog.instruction_text
        
        config.update_keywords = self.settings_dialog.update_keywords_checkbox.isChecked()
        config.update_caption = self.settings_dialog.update_caption_checkbox.isChecked()
        config.gen_count = self.settings_dialog.gen_count.value()
        config.res_limit = self.settings_dialog.res_limit.value()

        # Load sampler settings
        config.temperature = self.settings_dialog.temperature_spinbox.value()
        config.top_p = self.settings_dialog.top_p_spinbox.value()
        config.top_k = self.settings_dialog.top_k_spinbox.value()
        config.min_p = self.settings_dialog.min_p_spinbox.value()
        config.rep_pen = self.settings_dialog.rep_pen_spinbox.value()

        # Load JSON grammar setting
        config.use_json_grammar = self.settings_dialog.use_json_grammar_checkbox.isChecked()

        # Parse skip folders from text (semicolon or newline separated)
        skip_folders_text = self.settings_dialog.skip_folders_text
        if skip_folders_text:
            # Split by newlines first, then by semicolons
            folders = []
            for line in skip_folders_text.split('\n'):
                for folder in line.split(';'):
                    folder = folder.strip()
                    if folder:
                        folders.append(folder)
            config.skip_folders = folders

        self.indexer_thread = IndexerThread(config)
        self.indexer_thread.output_received.connect(self.update_output)
        self.indexer_thread.image_processed.connect(self.update_image_preview)
        self.indexer_thread.finished.connect(self.indexer_finished)
        self.pause_handler.pause_signal.connect(self.set_paused)
        self.pause_handler.stop_signal.connect(self.set_stopped)
        self.indexer_thread.start()

        self.output_area.clear()
        self.output_area.append("Running Image Indexer...")
        self.run_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)

    def set_paused(self, paused):
        if self.indexer_thread:
            self.indexer_thread.paused = paused

    def set_stopped(self):
        if self.indexer_thread:
            self.indexer_thread.stopped = True

    def toggle_pause(self):
        if self.pause_button.text() == "Pause":
            self.pause_handler.pause_signal.emit(True)
            self.pause_button.setText("Resume")
            self.update_output("Indexer paused.")
        else:
            self.pause_handler.pause_signal.emit(False)
            self.pause_button.setText("Pause")
            self.update_output("Indexer resumed.")

    def stop_indexer(self):
        self.pause_handler.stop_signal.emit()
        self.update_output("Stopping indexer...")
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)

    def indexer_finished(self):
        self.update_output("\nImage Indexer finished.")
        self.run_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setText("Pause")

    def update_output(self, text):
        self.output_area.append(text)
        self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())
        QApplication.processEvents()

    # Override resizeEvent to disable it since we're using fixed sizes
    def resizeEvent(self, event):
        """We override this but it shouldn't be called since window is fixed"""
        super().resizeEvent(event)
        
    def closeEvent(self, event):
        # Clean up API check thread when closing the window
        if self.api_check_thread and self.api_check_thread.isRunning():
            self.api_check_thread.stop()
            self.api_check_thread.wait()
        event.accept()

def run_gui():
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    app.setStyle("Fusion")  # Modern cross-platform style

    
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    
    app.setPalette(palette)
    window = ImageIndexerGUI()
    window.show()
    sys.exit(app.exec())    
    
def main():
    run_gui()
    
if __name__ == "__main__":
    main()
