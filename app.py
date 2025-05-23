import os
import traceback
import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, session, jsonify, make_response
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

# --- Flask Application Initialization ---
app = Flask(__name__)
# Set a secret key for session management. This is CRUCIAL for Flask sessions to work securely.
# os.urandom(24) generates a random 24-byte string.
app.secret_key = os.urandom(24)
# For production, it's highly recommended to load this from an environment variable:
# app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))


# --- Gemini API Configuration ---
# IMPORTANT SECURITY WARNING: Hardcoding API keys directly in your code is NOT recommended for production.
# For better security, always store API keys in environment variables (e.g., FLASK_GEMINI_API_KEY).
GEMINI_API_KEY = "AIzaSyC1S2wNk31e4gEUcAqRWhGOuE012hUbdkQ"

gemini_chatbot_model = None  # This will hold the GenerativeModel instance

# Attempt to configure and initialize the Gemini model when the app starts.
# This block runs only once when the Flask application is first loaded.
try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please provide a valid API key.")

    genai.configure(api_key=GEMINI_API_KEY)

    print("\n--- Listing Available Gemini Models ---")
    model_to_use = None
    all_available_models = []

    # Iterate through available models to find one suitable for text generation.
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            all_available_models.append(m.name)
            print(f"  Model Name: {m.name}, Supported Methods: {m.supported_generation_methods}")

    # Prioritize models based on Google's recommendations and common usage for chat.
    # 'gemini-1.5-flash-latest' is generally a good, fast choice for conversational AI.
    priority_models = [
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-flash-002",
        "models/gemini-1.5-flash-001",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro-latest",
        "models/gemini-1.5-pro-002",
        "models/gemini-1.5-pro-001",
        "models/gemini-1.5-pro",
        "models/gemini-2.5-flash-preview-05-20",  # Newer flash preview models
        "models/gemini-2.0-flash",  # Stable older flash model
        "models/gemini-pro",  # Original Pro model
        "models/gemini-1.0-pro"  # Original 1.0 Pro model
    ]

    # Select the first available model from the priority list.
    for p_model in priority_models:
        if p_model in all_available_models:
            model_to_use = p_model
            break

    print("--------------------------------------\n")

    if model_to_use is None:
        raise ValueError(
            "No suitable Gemini model found that supports 'generateContent'. Available models: " + ", ".join(
                all_available_models))

    print(f"Attempting to use model: {model_to_use}")
    # Initialize the GenerativeModel with the chosen model name.
    test_model = genai.GenerativeModel(model_to_use)

    # Perform a small test call to ensure the API key is valid and connection works.
    test_model.generate_content("Hello", safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'})

    # Assign the successfully tested model to the global variable.
    gemini_chatbot_model = test_model
    print("Gemini chatbot model initialized successfully.")

except Exception as e:
    # Log critical errors during model initialization.
    print(f"CRITICAL ERROR: Failed to configure or initialize Gemini chatbot model: {e}")
    traceback.print_exc()
    # If initialization fails, gemini_chatbot_model remains None, and the /chat route will handle this state.


# --- Helper Function for Firebase History ---
def get_user_history(user_email):
    """Fetches a user's diabetes prediction history from Firebase Realtime Database."""
    # Ensure Firebase Admin SDK is initialized before attempting to access it.
    if not firebase_admin._apps:
        print("WARNING: Firebase not initialized. Cannot fetch history.")
        return []

    ref = db.reference('results')
    # Query Firebase to get results specific to the logged-in user's email.
    all_results = ref.order_by_child('email').equal_to(user_email).get()

    history_list = []
    if all_results:
        # Convert Firebase dictionary results into a list for easier processing.
        for key, val in all_results.items():
            history_list.append(val)
    return history_list


# --- Chatbot Route ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip()
        print(f"Received message: '{user_message}'")

        if not user_message:
            return jsonify({'response': 'Message is empty.'}), 400

        # Check if the Gemini chatbot model was successfully initialized.
        if gemini_chatbot_model is None:
            print("Error: Gemini chatbot model is not initialized. Cannot process chat.")
            return jsonify(
                {'response': 'Chatbot is currently unavailable. Model not initialized.'}), 503

        # Retrieve and filter chat history from the session.
        # Ensure only 'user' and 'model' roles are kept for Gemini API compatibility.
        raw_chat_history = session.get('chat_history', [])
        chat_history_filtered = [
            turn for turn in raw_chat_history if turn['role'] in ['user', 'model']
        ]

        # Limit the chat history length to manage token usage and context window.
        MAX_HISTORY_TURNS = 5  # Keep last 5 user-model pairs
        if len(chat_history_filtered) > MAX_HISTORY_TURNS * 2:
            chat_history_filtered = chat_history_filtered[-(MAX_HISTORY_TURNS * 2):]

        # Define keywords to trigger history-aware responses.
        history_keywords = [
            "history", "past reports", "my results", "summarize my diabetes",
            "comment on my reports", "about my reports", "my previous results",
            "tell me about my history", "analysis of my reports", "review my data",
            "precautions", "what should i do", "advice", "guidance", "recommendations",
            "tips", "prevent", "manage", "lifestyle", "report", "reports", "check my diabetes",
            "check again", "gaia", "update"  # Added more general health/advice terms
        ]

        # Determine if the current query is a history-related query.
        is_history_query_current_message = any(keyword in user_message.lower() for keyword in history_keywords)

        # --- New: Manage conversation topic state in session ---
        current_topic = session.get('current_topic')

        if is_history_query_current_message:
            session['current_topic'] = 'diabetes_history'
        elif current_topic == 'diabetes_history' and not is_history_query_current_message:
            # If the topic was history, but the new message doesn't contain history keywords,
            # check if it's a general follow-up that still makes sense in context.
            # For simplicity, we'll keep the topic as history for a few turns unless a clear new topic emerges.
            # A more advanced solution might involve NLU to detect topic shifts.
            pass  # Keep current_topic as 'diabetes_history'
        else:
            session['current_topic'] = None  # Reset topic if not clearly history-related

        # If the current topic is diabetes_history:
        if session.get('current_topic') == 'diabetes_history':

            # Ensure user is logged in to access history.
            if 'user' not in session:
                bot_reply = 'Please log in to view your past reports.'
                session['chat_history'] = chat_history_filtered + [{'role': 'user', 'parts': [{'text': user_message}]},
                                                                   {'role': 'model', 'parts': [{'text': bot_reply}]}]
                return jsonify({'response': bot_reply}), 401

            user_email = session['user']
            history_data = get_user_history(user_email)

            if not history_data:
                bot_reply = "You don't have any past diabetes prediction reports yet."
                session['chat_history'] = chat_history_filtered + [{'role': 'user', 'parts': [{'text': user_message}]},
                                                                   {'role': 'model', 'parts': [{'text': bot_reply}]}]
                return jsonify({'response': bot_reply})

            # Format history data into a string for the LLM's context.
            formatted_history = "Here are the user's past diabetes prediction results:\n"
            for i, item in enumerate(history_data):
                formatted_history += (
                    f"Report {i + 1} (Date: {item.get('timestamp', 'N/A')}): "
                    f"Result: {item.get('result', 'N/A')}. "
                    f"Details: Pregnancies={item['input'].get('Pregnancies', 'N/A')}, "
                    f"Glucose={item['input'].get('Glucose', 'N/A')}, "
                    f"BMI={item['input'].get('BMI', 'N/A')}, "
                    f"Age={item['input'].get('Age', 'N/A')}.\n"
                )
                if i >= 4:  # Send up to 5 most recent reports
                    formatted_history += "...\n(More reports not shown for brevity)\n"
                    break

            # --- System prompt integrated into the user message for Gemini ---
            # Explicitly state the core topic and chatbot's role.
            history_instruction_prefix = (
                "You are a helpful assistant specialized in diabetes prediction and general diabetes health management. "
                "Your primary goal is to provide concise, helpful comments, summaries, and general advice based on the user's diabetes prediction reports. "
                "Maintain context from previous turns. Vary your response style and use unique words for each interaction. "
                "Do not ask for more information or assume future results. Keep responses brief and encouraging."
                "If the user asks for 'precautions', 'advice', 'guidance', or similar, provide general, actionable health advice related to diabetes management based on the provided history, if relevant, or general best practices."
                "\n\n"  # Separate instructions from the actual content
            )

            # Combine instruction, formatted history, and current user message.
            full_history_prompt_text = f"{history_instruction_prefix}{formatted_history}\nUser's current query: {user_message}"

            # Construct messages for Gemini API call: previous turns + current user message with instruction.
            messages_for_gemini = chat_history_filtered + [
                {'role': 'user', 'parts': [{'text': full_history_prompt_text}]}]

            print(f"Sending history-aware prompt to Gemini: {full_history_prompt_text[:200]}...")

            history_response = gemini_chatbot_model.generate_content(
                messages_for_gemini,
                safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
            )
            bot_reply = history_response.text.strip()
            print(f"Replying with history comment: '{bot_reply}'")

            # Update session history with user's original message and bot's response.
            session['chat_history'] = chat_history_filtered + [{'role': 'user', 'parts': [{'text': user_message}]},
                                                               {'role': 'model', 'parts': [{'text': bot_reply}]}]
            return jsonify({'response': bot_reply})

        # --- Standard Chat Logic (if not a history query, or history topic was reset) ---
        # --- System prompt integrated into the user message for Gemini ---
        # Explicitly state the core topic and chatbot's role for all general queries.
        general_instruction_prefix = (
            "You are a helpful assistant specialized in diabetes prediction and general diabetes health management. "
            "Respond to general queries about diabetes, health, or the app's features. "
            "Maintain context from previous turns. Vary your response style and use unique words for each interaction."
            "\n\n"  # Separate instructions from the actual content
        )

        # Combine instruction and current user message.
        full_general_prompt_text = f"{general_instruction_prefix}User's current query: {user_message}"

        # Construct messages for Gemini API call: previous turns + current user message with instruction.
        messages_for_gemini = chat_history_filtered + [{'role': 'user', 'parts': [{'text': full_general_prompt_text}]}]

        response = gemini_chatbot_model.generate_content(
            messages_for_gemini,
            safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
        )
        bot_reply = response.text.strip()
        print(f"Replying with: '{bot_reply}'")

        # Update session history with user's original message and bot's response.
        session['chat_history'] = chat_history_filtered + [{'role': 'user', 'parts': [{'text': user_message}]},
                                                           {'role': 'model', 'parts': [{'text': bot_reply}]}]
        return jsonify({'response': bot_reply})

    except Exception as e:
        print(f"Error during chat processing: {e}")
        traceback.print_exc()
        bot_reply = 'Sorry, something went wrong with the chatbot. Please check server logs.'
        # Ensure history is updated even on error for debugging.
        session['chat_history'] = chat_history_filtered + [{'role': 'user', 'parts': [{'text': user_message}]},
                                                           {'role': 'model', 'parts': [{'text': bot_reply}]}]
        return jsonify({'response': bot_reply}), 500


# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the Gemini chatbot model is initialized."""
    if gemini_chatbot_model is not None and isinstance(gemini_chatbot_model, genai.GenerativeModel):
        return jsonify({'status': 'ok', 'message': 'Gemini chatbot model is initialized.'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Gemini chatbot model not initialized.'}), 503


# --- ML Model and Firebase Initialization ---
# Load the pre-trained machine learning model and scaler for diabetes prediction.
try:
    # Ensure 'model/model.pkl' and 'model/scaler.pkl' paths are correct relative to app.py.
    model = pickle.load(open("model/model.pkl", 'rb'))
    scaler = pickle.load(open("model/scaler.pkl", 'rb'))
    print("Diabetes prediction model and scaler loaded successfully.")
except FileNotFoundError:
    print("WARNING: ML model or scaler files not found. Prediction functionality might be affected.")
    model = None
    scaler = None
except Exception as e:
    print(f"ERROR: Failed to load ML model or scaler: {e}")
    traceback.print_exc()
    model = None
    scaler = None

# Initialize Firebase Admin SDK for Realtime Database access.
# This block is guarded to prevent multiple initializations, which causes errors with Flask's reloader.
if not firebase_admin._apps:  # Check if the default Firebase app is NOT already initialized.
    try:
        cred = credentials.Certificate("firebase_admin.json")  # Path to your Firebase Admin SDK JSON key.
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://diabetisprediction-default-rtdb.firebaseio.com'
            # Replace with your Realtime DB URL.
        })
        print("Firebase Admin SDK initialized successfully.")
    except FileNotFoundError:
        print("WARNING: 'firebase_admin.json' not found. Firebase functionality might be affected.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Firebase Admin SDK: {e}")
        traceback.print_exc()
else:
    print("Firebase Admin SDK already initialized.")


# --- Flask Routes ---
@app.route('/')
def login():
    """Renders the login page."""
    return render_template('login.html')


@app.route('/set_user', methods=['POST'])
def set_user():
    """Sets the logged-in user's email in the session."""
    data = request.json
    # Defensive check for secret key, though it should be set globally.
    if not app.secret_key:
        raise RuntimeError("Flask secret key is not set. Session is unavailable.")
    session['user'] = data.get('email')
    return jsonify({'message': 'User session set'})


@app.route('/home')
def home():
    """Renders the home page after successful login."""
    if 'user' not in session:
        return redirect('/')  # Redirect to login if not authenticated.
    response = make_response(render_template('home.html', user=session['user']))
    # Headers to prevent caching of the home page.
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/predict-form')
def predict_form():
    """Renders the diabetes prediction form page."""
    if 'user' not in session:
        return redirect('/')
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the diabetes prediction request."""
    if 'user' not in session:
        return redirect('/')

    if model is None or scaler is None:
        return render_template('index.html',
                               prediction_text="Error: Prediction model not loaded. Please check server logs."), 500

    input_features = []
    field_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    # Extract and validate form inputs.
    for field in field_names:
        value = request.form.get(field)
        if value is None:
            return render_template('index.html', prediction_text=f"Error: Missing input for {field}."), 400
        try:
            input_features.append(float(value))
        except ValueError:
            return render_template('index.html', prediction_text=f"Error: Invalid number format for {field}."), 400

    # Scale inputs and make a prediction using the loaded ML model.
    input_np = np.array(input_features).reshape(1, -1)
    std_data = scaler.transform(input_np)
    prediction = model.predict(std_data)[0]

    prediction_text = "The person is diabetic" if prediction else "The person is not diabetic"

    # Save the prediction result to Firebase Realtime Database.
    if firebase_admin._apps:  # Ensure Firebase is initialized before attempting to save.
        ref = db.reference('results')
        ref.push({
            'email': session['user'],
            'input': dict(zip(field_names, input_features)),  # Store inputs with their names.
            'result': prediction_text,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    else:
        print("WARNING: Firebase not initialized. Prediction result not saved to DB.")

    # Render the form page again, displaying the prediction result.
    return render_template('index.html', prediction_text=prediction_text)


@app.route('/history')
def history():
    """Renders the user's prediction history page."""
    if 'user' not in session:
        return redirect('/')

    if not firebase_admin._apps:
        return render_template('history.html', results=[], error="Firebase not initialized. Cannot fetch history.")

    results_list = get_user_history(session['user'])  # Use the helper function to fetch history.

    return render_template('history.html', results=results_list)


@app.route('/logout')
def logout():
    """Logs out the user by clearing the session and redirects to login."""
    session.pop('user', None)
    return redirect('/')


# --- Main entry point for running the Flask app ---
if __name__ == "__main__":
    app.run(debug=True)  # Run in debug mode for development. Set debug=False for production.
