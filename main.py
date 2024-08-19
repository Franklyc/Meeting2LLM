import sys
import wave
import pyaudio
import threading
import pyttsx3
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QComboBox, QLabel, QLineEdit
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal
from groq import Groq
import google.generativeai as genai
import os

# Initialize Groq client
client = Groq()

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TTSThread(QThread):
    """
    Thread for text-to-speech playback.
    """
    finished = pyqtSignal()  # Signal emitted when playback is finished

    def __init__(self, tts_engine, text):
        """
        Initialize the TTS thread.

        Args:
            tts_engine: The pyttsx3 TTS engine.
            text: The text to be spoken.
        """
        super().__init__()
        self.tts_engine = tts_engine
        self.text = text
        self.is_stopped = False  # Flag to indicate if playback is stopped

    def run(self):
        """
        Run the TTS playback.
        """
        self.tts_engine.connect('started-word', self.check_stop)  # Connect to check for stop signal
        self.tts_engine.say(self.text)  # Speak the text
        self.tts_engine.runAndWait()  # Wait for playback to finish
        if not self.is_stopped:
            self.finished.emit()  # Emit finished signal if not stopped

    def stop(self):
        """
        Stop the TTS playback.
        """
        self.is_stopped = True
        self.tts_engine.stop()

    def check_stop(self, name, location, length):
        """
        Check if playback should be stopped.

        Args:
            name: Name of the word being spoken.
            location: Location of the word in the text.
            length: Length of the word.

        Returns:
            True if playback should continue, False otherwise.
        """
        if self.is_stopped:
            return False
        return True

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
        self.response = None # Initialize response attribute

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
        self.record_thread = threading.Thread(target=self.record_audio, args=("meeting_audio.wav",))  # Create recording thread
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
        self.stop_recording = False  # Clear stop recording flag
        CHUNK = 1024  # Audio chunk size
        FORMAT = pyaudio.paInt16  # Audio format
        CHANNELS = 2  # Number of audio channels
        RATE = 44100  # Audio sample rate

        p = pyaudio.PyAudio()  # Initialize PyAudio
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)  # Open audio stream
        frames = []  # List to store recorded audio frames

        try:
            while not self.stop_recording:
                data = stream.read(CHUNK)  # Read audio data from stream
                frames.append(data)  # Append data to frames list
        except Exception as e:
            print(f"Error during recording: {e}")  # Print error message
        finally:
            stream.stop_stream()  # Stop audio stream
            stream.close()  # Close audio stream
            p.terminate()  # Terminate PyAudio

            # Check if any data was recorded
            if frames:
                wf = wave.open(filename, 'wb')  # Open WAV file for writing
                wf.setnchannels(CHANNELS)  # Set number of channels
                wf.setsampwidth(p.get_sample_size(FORMAT))  # Set sample width
                wf.setframerate(RATE)  # Set frame rate
                wf.writeframes(b''.join(frames))  # Write audio frames to file
                wf.close()  # Close WAV file
            else:
                print("No audio data recorded, file not created.")  # Print message if no data was recorded
    

    def transcribe_and_respond(self, filename):
        """
        Transcribe the audio file and generate an LLM response.

        Args:
            filename: The name of the audio file to transcribe.
        """
        # Create a separate thread for transcription and response generation
        self.processing_thread = threading.Thread(target=self._transcribe_and_respond, args=(filename,))
        self.processing_thread.start()

    def _transcribe_and_respond(self, filename):
        """
        Transcribe the audio file and generate an LLM response in a separate thread.

        Args:
            filename: The name of the audio file to transcribe.
        """
        try:
            transcription = self.transcribe_audio(filename)  # Transcribe the audio
            logging.info("Transcription completed.")  # Log transcription completion
            self.text_edit.append("Transcription:\n" + transcription + "\n\n")  # Append transcription to text edit area

            self.response = self.get_llm_response(transcription)  # Generate LLM response
            logging.info("LLM response generated.")  # Log response generation
            self.text_edit.append("LLM Response:\n" + self.response + "\n")  # Append response to text edit area
            self.play_button.setEnabled(True)  # Enable play button
        except Exception as e:
            logging.error(f"Error during transcription or response generation: {e}")  # Log error

    def transcribe_audio(self, filename):
        """
        Transcribe the audio file using the Groq API.

        Args:
            filename: The name of the audio file to transcribe.

        Returns:
            The transcribed text.
        """
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),  # Pass the audio file to the API
                model="whisper-large-v3",  # Specify the transcription model
                response_format="verbose_json",  # Specify the response format
            )
        return transcription.text  # Return the transcribed text

    def get_llm_response(self, transcription):
        """
        Get an LLM response to the transcribed text.

        Args:
            transcription: The transcribed text.

        Returns:
            The LLM response.
        """
        selected_model = self.model_selector.currentText()  # Get the selected LLM model
        system_prompt = self.system_prompt_input.text()  # Retrieve system prompt

        if selected_model.startswith("gemini"):  # Check if the selected model is a Gemini model
            response = self.get_gemini_response(transcription, selected_model, system_prompt)  # Get Gemini response
        else:
            response = self.get_groq_response(transcription, selected_model, system_prompt)  # Get Groq response
        
        return response  # Return the LLM response

    def get_groq_response(self, transcription, model_name, system_prompt):
        """
        Get a response from a Groq LLM.

        Args:
            transcription: The transcribed text.
            model_name: The name of the Groq model to use.
            system_prompt: The system prompt to use.

        Returns:
            The Groq LLM response.
        """
        # Dictionary for model max tokens
        MODEL_MAX_TOKENS = {
            "llama3-70b-8192": 8192,
            "llama-3.1-70b-versatile": 8000,
            "mixtral-8x7b-32768": 32768,
            "gemma2-9b-it": 8192,
        }

        # Determine the correct max_tokens based on the selected model
        max_tokens = MODEL_MAX_TOKENS.get(model_name, 8000)  # Default to 8000 if not found

        completion = client.chat.completions.create(
            model=model_name,  # Specify the Groq model
            messages=[
                {
                    "role": "system",
                    "content": system_prompt  # Use the provided system prompt
                },
                {
                    "role": "user",
                    "content": transcription  # Pass the transcribed text as user input
                }
            ],
            temperature=0.7,  # Temperature parameter for response generation
            max_tokens=max_tokens,  # Maximum number of tokens for the response
            top_p=1,  # Top_p parameter for response generation
            stream=True,  # Stream the response
            stop=None,  # Stop sequence for response generation
        )
        
        response = ""  # Initialize an empty string to store the response
        for chunk in completion:  # Iterate through the streamed response chunks
            content = chunk.choices[0].delta.content or ""  # Extract the content from the chunk
            response += content  # Append the content to the response string
        return response  # Return the complete response

    def get_gemini_response(self, transcription, model_name, system_prompt):
        """
        Get a response from a Google Gemini LLM.

        Args:
            transcription: The transcribed text.
            model_name: The name of the Gemini model to use.
            system_prompt: The system prompt to use.

        Returns:
            The Gemini LLM response.
        """
        generation_config = {
            "temperature": 1,  # Temperature parameter for response generation
            "top_p": 0.95,  # Top_p parameter for response generation
            "top_k": 64,  # Top_k parameter for response generation
            "max_output_tokens": 8192,  # Maximum number of tokens for the response
            "response_mime_type": "text/plain",  # Specify the response format
        }

        model = genai.GenerativeModel(
            model_name=model_name,  # Specify the Gemini model
            generation_config=generation_config,  # Pass the generation configuration
            system_instruction=system_prompt,  # Use the provided system prompt
        )

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        transcription,  # Pass the transcribed text as user input
                    ],
                }
            ]
        )

        response = chat_session.send_message(transcription)  # Send the transcribed text to the model

        return response.text  # Return the text of the response

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

if __name__ == '__main__':
    app = QApplication(sys.argv)  # Create QApplication instance
    ex = AudioRecorder()  # Create AudioRecorder instance
    sys.exit(app.exec_())  # Run the application event loop