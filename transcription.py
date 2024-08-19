from groq import Groq

# Initialize Groq client
client = Groq()

def transcribe_audio(filename):
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