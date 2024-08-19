# Meeting2LLM

This application allows you to record audio, transcribe it, and get a response from a large language model (LLM). It utilizes Groq's Whisper API for transcription and offers a selection of LLMs from both [Groq](https://groq.com/) and [Google Gemini](https://developers.generativeai.google/).

[![GitHub stars](https://img.shields.io/github/stars/your-username/meeting2LLM?style=social)](https://github.com/your-username/meeting2LLM)
[![GitHub forks](https://img.shields.io/github/forks/your-username/meeting2LLM?style=social)](https://github.com/your-username/meeting2LLM/fork)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

* **Record audio:** Capture audio input using your microphone.
* **Transcribe audio:** Transcribe the recorded audio using Groq's Whisper API.
* **LLM Response:** Generate a response to the transcribed text using a selected LLM.
* **Model Selection:** Choose from a variety of LLMs:

| Model Provider | Model Name                | Max Tokens |
|----------------|---------------------------|------------|
| Groq           | llama3-70b-8192          | 8192       |
| Groq           | llama-3.1-70b-versatile   | 8000       |
| Groq           | mixtral-8x7b-32768       | 32768      |
| Groq           | gemma2-9b-it             | 8192       |
| Google Gemini | gemini-1.5-flash         | 8192 (output)      |
| Google Gemini | gemini-1.5-pro           | 8192 (output)      |

* **Customizable System Prompt:** Define your own system prompt to guide the LLM's responses.
* **Text-to-Speech:** Listen to the LLM's response using text-to-speech functionality.

## Roadmap

* **Improved TTS Engine:** Evaluate and integrate a more robust and feature-rich TTS engine for better voice quality and performance.
* **Batch Processing:** Add support for transcribing and generating responses for multiple audio files in batch mode.
* **GUI Enhancements:** Improve the user interface with features like progress bars, audio visualization, and more customization options.
* **Cloud Integration:** Explore integration with cloud storage services to allow users to upload and manage audio files.
* **Offline Mode:** Investigate the feasibility of enabling offline transcription and response generation for specific models.

[![GitHub stats](https://github-readme-stats.vercel.app/api?username=your-username&show_icons=true&theme=radical)](https://github.com/anuraghazra/github-readme-stats)
[![GitHub Streak](https://github-readme-streak-stats.herokuapp.com/?user=your-username)](https://git.io/streak-stats)

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

* [Groq](https://groq.com/)
* [Google Gemini](https://developers.generativeai.google/)
* [pyttsx3](https://pyttsx3.readthedocs.io/en/latest/) (Text-to-Speech library)
* [PyQt5](https://pypi.org/project/PyQt5/) (GUI framework)
* [PyAudio](https://pypi.org/project/PyAudio/) (Audio recording library)
