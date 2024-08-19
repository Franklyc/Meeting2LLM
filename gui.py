import sys
import pyttsx3
import logging
import threading

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QComboBox, QLabel, QLineEdit
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal, QObject

from audio_recording import record_audio  # Import the record_audio function from audio_recording.py
from transcription import transcribe_audio  # Import the transcribe_audio function from transcription.py
from llm_interaction import get_groq_response, get_gemini_response  # Import LLM interaction functions
from tts import TTSThread  # Import the TTSThread class from tts.py

class Worker(QObject):  # Define a worker class to handle the transcription and response generation
    finished = pyqtSignal()
    transcription_ready = pyqtSignal(str)
    response_ready = pyqtSignal(str)

    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def run(self):
        try:
            transcription = transcribe_audio(self.filename)  # Call the transcribe_audio function from transcription.py
            logging.info("Transcription completed.")  # Log transcription completion
            self.transcription_ready.emit(transcription)  # Emit signal with transcription

            response = self.get_llm_response(transcription)  # Generate LLM response
            logging.info("LLM response generated.")  # Log response generation
            self.response_ready.emit(response)  # Emit signal with response
            self.finished.emit()  # Emit signal indicating the worker is finished
        except Exception as e:
            logging.error(f"Error during transcription or response generation: {e}")  # Log error

    def get_llm_response(self, transcription):
        selected_model = self.parent().model_selector.currentText()  # Get the selected LLM model from the main window
        system_prompt = self.parent().system_prompt_input.text()  # Retrieve system prompt from the main window

        if selected_model.startswith("gemini"):  # Check if the selected model is a Gemini model
            response = get_gemini_response(transcription, selected_model, system_prompt)  # Get Gemini response from llm_interaction.py
        else:
            response = get_groq_response(transcription, selected_model, system_prompt)  # Get Groq response from llm_interaction.py

        return response  # Return the LLM response

class AudioRecorder(QMainWindow):
    """
    Main window for audio recording, transcription, and LLM response.
    """
    def __init__(self):
        """
        Initialize the main window.
        """
        super().__init__()
        self.title = 'Meeting2LLM'  # Window title
        self.left = 100  # Window position
        self.top = 100
        self.width = 400  # Window size
        self.height = 600
        self.initUI()  # Initialize the user interface
        self.tts_engine = pyttsx3.init()  # Initialize the TTS engine
        self.tts_thread = None  # Thread for TTS playback
        self.is_playing = False  # Flag to indicate if audio is being played

    def initUI(self):
        """
        Initialize the user interface.
        """
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Set window icon
        self.setWindowIcon(QIcon('M2L_icon.png'))

        # Main layout and widget
        layout = QVBoxLayout()  # Vertical layout for widgets
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)  # Set the main widget

        # --- Styling ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50; /* Green */
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3e8e41;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #ccc;
                padding: 10px;
                font-size: 16px;
                font-family: Arial, Helvetica, sans-serif;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #ccc;
                padding: 5px;
                font-size: 14px;
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #ccc;
                padding: 5px;
                font-size: 14px;
            }
        """)

        # --- System Prompt Input ---
        self.system_prompt_label = QLabel("System Prompt:", self)  # Label for system prompt input
        layout.addWidget(self.system_prompt_label)  # Add label to layout

        self.system_prompt_input = QLineEdit(self)  # Text input for system prompt
        default_prompt = "You are an experienced professional. Please respond to the following interview question in a first-person, detailed, and professional manner."
        self.system_prompt_input.setText(default_prompt)  # Set default prompt text
        layout.addWidget(self.system_prompt_input)  # Add text input to layout

        # Dropdown for model selection
        self.model_selector = QComboBox(self)  # Dropdown menu for model selection
        self.model_selector.addItem("llama3-70b-8192")  # Add model options
        self.model_selector.addItem("llama-3.1-70b-versatile")
        self.model_selector.addItem("mixtral-8x7b-32768")
        self.model_selector.addItem("gemma2-9b-it")
        self.model_selector.addItem("gemini-1.5-flash")
        self.model_selector.addItem("gemini-1.5-pro")
        layout.addWidget(self.model_selector)  # Add dropdown menu to layout

        # Buttons
        self.record_button = QPushButton('Start Recording', self)  # Button to start recording
        self.record_button.clicked.connect(self.on_click_record)  # Connect button to recording function
        layout.addWidget(self.record_button)  # Add button to layout

        self.stop_button = QPushButton('Stop Recording', self)  # Button to stop recording
        self.stop_button.clicked.connect(self.on_click_stop)  # Connect button to stop function
        self.stop_button.setEnabled(False)  # Disable button initially
        layout.addWidget(self.stop_button)  # Add button to layout

        self.play_button = QPushButton('Play Response Audio', self)  # Button to play audio response
        self.play_button.clicked.connect(self.play_audio_response)  # Connect button to playback function
        self.play_button.setEnabled(False)  # Disable button initially
        layout.addWidget(self.play_button)  # Add button to layout

        self.stop_play_button = QPushButton('Stop Audio', self)  # Button to stop audio playback
        self.stop_play_button.clicked.connect(self.stop_audio_response)  # Connect button to stop function
        self.stop_play_button.setEnabled(False)  # Disable button initially
        layout.addWidget(self.stop_play_button)  # Add button to layout

        # Text edit for transcription and LLM response
        self.text_edit = QTextEdit(self)  # Text edit area for displaying text
        self.text_edit.setPlaceholderText("Transcription and LLM responses will appear here.")  # Set placeholder text

        # Set font size for QTextEdit
        font = QFont()
        font.setPointSize(12)
        self.text_edit.setFont(font)

        layout.addWidget(self.text_edit)  # Add text edit area to layout

        self.show()  # Show the window

    @pyqtSlot()
    def on_click_record(self):
        """
        Handle the 'Start Recording' button click.
        """
        self.text_edit.clear()  # Clear the text edit area
        self.record_button.setEnabled(False)  # Disable record button
        self.stop_button.setEnabled(True)  # Enable stop button
        self.play_button.setEnabled(False)  # Disable play button
        self.stop_play_button.setEnabled(False)  # Disable stop playback button
        self.record_thread = threading.Thread(target=self.record_audio,
                                               args=("meeting_audio.wav",))  # Create recording thread
        self.record_thread.start()  # Start recording thread

    @pyqtSlot()
    def on_click_stop(self):
        """
        Handle the 'Stop Recording' button click.
        """
        self.stop_recording = True  # Set stop recording flag
        self.record_button.setEnabled(True)  # Enable record button
        self.stop_button.setEnabled(False)  # Disable stop button
        self.record_thread.join()  # Wait for recording thread to finish
        self.transcribe_and_respond("meeting_audio.wav")  # Transcribe and generate response

    def record_audio(self, filename):
        """
        Record audio from the microphone and save it to a WAV file.

        Args:
            filename: The name of the WAV file to save the recording to.
        """
        record_audio(filename)  # Call the record_audio function from audio_recording.py

    def transcribe_and_respond(self, filename):
        """
        Transcribe the audio file and generate an LLM response.

        Args:
            filename: The name of the audio file to transcribe.
        """
        # Create a separate thread for transcription and response generation
        self.worker = Worker(filename)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        # Connect worker signals to slots in the main thread
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.transcription_ready.connect(self.display_transcription)
        self.worker.response_ready.connect(self.display_response)
        # Start the thread
        self.thread.start()

    def display_transcription(self, transcription):
        """
        Display the transcription in the text edit area.
        """
        logging.info("Transcription completed.")  # Log transcription completion
        self.text_edit.append("Transcription:\n" + transcription + "\n\n")  # Append transcription to text edit area

    def display_response(self, response):
        """
        Display the LLM response in the text edit area and enable the play button.
        """
        logging.info("LLM response generated.")  # Log response generation
        self.text_edit.append("LLM Response:\n" + response + "\n")  # Append response to text edit area
        self.play_button.setEnabled(True)  # Enable play button

    def play_audio_response(self):
        """
        Play the LLM response audio.
        """
        if self.tts_thread and self.tts_thread.isRunning():  # Check if TTS thread is already running
            return  # Do nothing if TTS thread is already running

        self.tts_thread = TTSThread(self.tts_engine, self.response)  # Create a new TTS thread
        self.tts_thread.finished.connect(self.on_tts_finished)  # Connect finished signal to handler
        self.tts_thread.start()  # Start the TTS thread

        self.is_playing = True  # Set audio playing flag
        self.play_button.setEnabled(False)  # Disable play button
        self.stop_play_button.setEnabled(True)  # Enable stop playback button

    def stop_audio_response(self):
        """
        Stop the LLM response audio playback.
        """
        if self.tts_thread and self.tts_thread.isRunning():  # Check if TTS thread is running
            self.tts_thread.stop()  # Stop the TTS thread
            self.tts_thread.wait()  # Wait for the thread to finish
            self.on_tts_finished()  # Call finished signal handler

    def on_tts_finished(self):
        """
        Handle the TTS playback finished signal.
        """
        self.is_playing = False  # Clear audio playing flag
        self.play_button.setEnabled(True)  # Enable play button
        self.stop_play_button.setEnabled(False)  # Disable stop playback button