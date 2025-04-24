# api/index.py
import os
from flask import Flask, request, render_template
from openai import OpenAI, APIError # Import APIError for better error handling
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

app = Flask(__name__, template_folder='../templates') # Point to the correct template folder

# --- IMPORTANT: Get API Key from Environment Variable ---
# For local testing, it reads from .env
# For Vercell, set NVIDIA_API_KEY in the Vercell dashboard environment variables
NVIDIA_API_KEY = "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"

# Initialize OpenAI client ONLY if the key exists
client = None
if NVIDIA_API_KEY:
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        # Handle initialization error if necessary
else:
    print("Warning: NVIDIA_API_KEY environment variable not set. The chatbot will not function.")


@app.route('/')
def home():
    """Renders the initial chat page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the user's message and returns the chatbot's response."""
    user_message = request.form.get('message')
    bot_response = None
    error_message = None

    if not user_message:
        error_message = "Please enter a message."
        return render_template('index.html', error=error_message)

    # Check if the client was initialized (API key was present)
    if not client:
        error_message = "Chatbot is not configured correctly (API key missing on server)."
        # Keep the user's message in the text area
        return render_template('index.html', error=error_message, user_message=user_message)

    try:
        completion = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=[{"role": "user", "content": "Give accurate information about this Skills with aproper road map to learn this skill .Follow a format 1)Steps: 2)Key prerequisites 3)Future Guidance  . Skill to be learnt : "+user_message}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True  # The API call uses streaming
        )

        # Collect the streamed response chunks into a single string
        collected_chunks = []
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                collected_chunks.append(chunk.choices[0].delta.content)

        bot_response = "".join(collected_chunks)

    except APIError as e:
        # Handle specific API errors (e.g., rate limits, invalid key)
        print(f"NVIDIA API Error: {e}")
        error_message = f"API Error: {e.status_code} - {e.body.get('message', 'Unknown error') if e.body else 'Details unavailable'}"
    except Exception as e:
        # Handle other potential errors (network issues, etc.)
        print(f"An unexpected error occurred: {e}")
        error_message = f"An unexpected error occurred: {e}"

    # Render the page again, passing the original message, the response, or an error
    return render_template('index.html', user_message=user_message, response=bot_response, error=error_message)

# Optional: for local testing convenience `python api/index.py`
# Vercell doesn't use this __main__ block
# if __name__ == '__main__':
#     app.run(debug=True, port=5001)
