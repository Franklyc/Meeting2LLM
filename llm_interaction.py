from groq import Groq
import google.generativeai as genai
import os

# Initialize Groq client
client = Groq()

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def get_groq_response(transcription, model_name, system_prompt):
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

def get_gemini_response(transcription, model_name, system_prompt):
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