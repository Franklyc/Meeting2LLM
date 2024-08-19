from PyQt5.QtCore import QThread, pyqtSignal

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