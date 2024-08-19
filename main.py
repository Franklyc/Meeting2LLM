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
    finished = pyqtSignal()

    def __init__(self, tts_engine, text):
        super().__init__()
        self.tts_engine = tts_engine
        self.text = text
        self.is_stopped = False

    def run(self):
        self.tts_engine.connect('started-word', self.check_stop)
        self.tts_engine.say(self.text)
        self.tts_engine.runAndWait()
        if not self.is_stopped:
            self.finished.emit()

    def stop(self):
        self.is_stopped = True
        self.tts_engine.stop()

    def check_stop(self, name, location, length):
        if self.is_stopped:
            return False
        return True

class AudioRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Meeting2LLM'
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 600
        self.initUI()
        self.tts_engine = pyttsx3.init()
        self.tts_thread = None
        self.is_playing = False

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Set window icon
        self.setWindowIcon(QIcon('M2L_icon.png'))
        
        # Main layout and widget
        layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

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
        self.system_prompt_label = QLabel("System Prompt:", self)
        layout.addWidget(self.system_prompt_label)

        self.system_prompt_input = QLineEdit(self)
        default_prompt = "You are an experienced professional. Please respond to the following interview question in a first-person, detailed, and professional manner."
        self.system_prompt_input.setText(default_prompt)
        layout.addWidget(self.system_prompt_input)

        # Dropdown for model selection
        self.model_selector = QComboBox(self)
        self.model_selector.addItem("llama3-70b-8192")
        self.model_selector.addItem("llama-3.1-70b-versatile")
        self.model_selector.addItem("mixtral-8x7b-32768")
        self.model_selector.addItem("gemma2-9b-it")
        self.model_selector.addItem("gemini-1.5-flash")
        self.model_selector.addItem("gemini-1.5-pro")
        layout.addWidget(self.model_selector)

        # Buttons
        self.record_button = QPushButton('Start Recording', self)
        self.record_button.clicked.connect(self.on_click_record)
        layout.addWidget(self.record_button)

        self.stop_button = QPushButton('Stop Recording', self)
        self.stop_button.clicked.connect(self.on_click_stop)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        self.play_button = QPushButton('Play Response Audio', self)
        self.play_button.clicked.connect(self.play_audio_response)
        self.play_button.setEnabled(False)
        layout.addWidget(self.play_button)

        self.stop_play_button = QPushButton('Stop Audio', self)
        self.stop_play_button.clicked.connect(self.stop_audio_response)
        self.stop_play_button.setEnabled(False)
        layout.addWidget(self.stop_play_button)

        # Text edit for transcription and LLM response
        self.text_edit = QTextEdit(self)
        self.text_edit.setPlaceholderText("Transcription and LLM responses will appear here.")
        
        # Set font size for QTextEdit
        font = QFont()
        font.setPointSize(12)
        self.text_edit.setFont(font)
        
        layout.addWidget(self.text_edit)

        self.show()

    @pyqtSlot()
    def on_click_record(self):
        self.text_edit.clear()
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.play_button.setEnabled(False)
        self.stop_play_button.setEnabled(False)
        self.record_thread = threading.Thread(target=self.record_audio, args=("meeting_audio.wav",))
        self.record_thread.start()

    @pyqtSlot()
    def on_click_stop(self):
        self.stop_recording = True
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_thread.join()
        self.transcribe_and_respond("meeting_audio.wav")

    def record_audio(self, filename):
        self.stop_recording = False
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = []

        try:
            while not self.stop_recording:
                data = stream.read(CHUNK)
                frames.append(data)
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Check if any data was recorded
            if frames:
                wf = wave.open(filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
            else:
                print("No audio data recorded, file not created.")
    

    def transcribe_and_respond(self, filename):
        # Create a separate thread for transcription and response generation
        self.processing_thread = threading.Thread(target=self._transcribe_and_respond, args=(filename,))
        self.processing_thread.start()

    def _transcribe_and_respond(self, filename):
        try:
            transcription = self.transcribe_audio(filename)
            logging.info("Transcription completed.")
            self.text_edit.append("Transcription:\n" + transcription + "\n\n")

            response = self.get_llm_response(transcription)
            logging.info("LLM response generated.")
            self.text_edit.append("LLM Response:\n" + response + "\n")
            self.play_button.setEnabled(True)
        except Exception as e:
            logging.error(f"Error during transcription or response generation: {e}")

    def transcribe_audio(self, filename):
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        return transcription.text

    def get_llm_response(self, transcription):
        selected_model = self.model_selector.currentText()
        system_prompt = self.system_prompt_input.text()  # Retrieve system prompt

        if selected_model.startswith("gemini"):
            response = self.get_gemini_response(transcription, selected_model, system_prompt)
        else:
            response = self.get_groq_response(transcription, selected_model, system_prompt)
        
        return response

    def get_groq_response(self, transcription, model_name, system_prompt):
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
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt  # Use the provided system prompt
                },
                {
                    "role": "user",
                    "content": transcription
                }
            ],
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        response = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            response += content
        return response

    def get_gemini_response(self, transcription, model_name, system_prompt):
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=system_prompt,  # Use the provided system prompt
        )

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        transcription,
                    ],
                }
            ]
        )

        response = chat_session.send_message(transcription)

        return response.text

    def play_audio_response(self):
        if self.tts_thread and self.tts_thread.isRunning():
            return
        
        self.tts_thread = TTSThread(self.tts_engine, self.response)
        self.tts_thread.finished.connect(self.on_tts_finished)
        self.tts_thread.start()
        
        self.is_playing = True
        self.play_button.setEnabled(False)
        self.stop_play_button.setEnabled(True)

    def stop_audio_response(self):
        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop()
            self.tts_thread.wait()
            self.on_tts_finished()

    def on_tts_finished(self):
        self.is_playing = False
        self.play_button.setEnabled(True)
        self.stop_play_button.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioRecorder()
    sys.exit(app.exec_())