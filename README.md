# Meeting2LLM

This application allows you to record audio, transcribe it, and get a response from a large language model (LLM). It utilizes Groq's Whisper API for transcription and offers a selection of LLMs from both Groq and Google Gemini.

## Features

* **Record audio:** Capture audio input using your microphone.
* **Transcribe audio:** Transcribe the recorded audio using Groq's Whisper API.
* **LLM Response:** Generate a response to the transcribed text using a selected LLM.
* **Model Selection:** Choose from a variety of LLMs, including:
    * Groq Models:
        * llama3-70b-8192
        * llama-3.1-70b-versatile
        * mixtral-8x7b-32768
        * gemma2-9b-it
    * Google Gemini Models:
        * gemini-1.5-flash
        * gemini-1.5-pro
* **Text-to-Speech:** Listen to the LLM's response using text-to-speech functionality.

## Prerequisites

* **Python 3.7 or higher**
* **Required Python packages:** Install using `pip install -r requirements.txt`
    * pyaudio
    * wave
    * PyQt5
    * pyttsx3
    * groq
    * google-generativeai
* **Groq API Key:** Obtain an API key from [https://groq.com/](https://groq.com/) and set the `GROQ_API_KEY` environment variable.
* **Google Gemini API Key:** Obtain an API key from [https://developers.generativeai.google/](https://developers.generativeai.google/) and set the `GEMINI_API_KEY` environment variable.

## Installation

1. Clone the repository: `git clone https://github.com/your-username/meeting2LLM.git`
2. Install the required packages: `pip install -r requirements.txt`
3. Set the environment variables for your Groq and Google Gemini API keys.

## Usage

1. Run the application: `python main.py`
2. Click "Start Recording" to begin recording audio.
3. Click "Stop Recording" to stop recording and initiate transcription and LLM response generation.
4. The transcription and LLM response will be displayed in the text area.
5. Click "Play Response Audio" to listen to the LLM's response.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Groq: [https://groq.com/](https://groq.com/)
* Google Gemini: [https://developers.generativeai.google/](https://developers.generativeai.google/)
