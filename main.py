import sys
from PyQt5.QtWidgets import QApplication

from gui import AudioRecorder  # Import the AudioRecorder class from gui.py

if __name__ == '__main__':
    app = QApplication(sys.argv)  # Create QApplication instance
    ex = AudioRecorder()  # Create AudioRecorder instance
    sys.exit(app.exec_())  # Run the application event loop