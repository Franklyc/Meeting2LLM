import pyaudio
import wave
import logging

def record_audio(filename, stop_recording):
    """
    Record audio from the microphone and save it to a WAV file.

    Args:
        filename: The name of the WAV file to save the recording to.
        stop_recording: A threading.Event object to signal when to stop recording.
    """
    logging.info("record_audio function started.")
    CHUNK = 1024  # Audio chunk size
    FORMAT = pyaudio.paInt16  # Audio format
    CHANNELS = 2  # Number of audio channels
    RATE = 44100  # Audio sample rate

    p = pyaudio.PyAudio()  # Initialize PyAudio
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)  # Open audio stream
    frames = []  # List to store recorded audio frames

    logging.info("Recording started.")
    try:
        while not stop_recording:
            data = stream.read(CHUNK)  # Read audio data from stream
            frames.append(data)  # Append data to frames list
            logging.debug(f"Recorded {len(frames)} frames.")
    except Exception as e:
        logging.error(f"Error during recording: {e}")  # Print error message
    finally:
        logging.info("Recording stopped.")
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
            logging.info(f"Audio saved to {filename}.")
        else:
            logging.warning("No audio data recorded, file not created.")  # Print message if no data was recorded