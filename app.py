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

app = Flask(__name__)
# Consider using a more secure method for secret key in production
GEMINI_API_KEY = "AIzaSyC1S2wNk31e4gEUcAqRWhGOuE012hUbdkQ"
gemini_chatbot_model = None  # This will hold the GenerativeModel instance
try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please provide a valid API key.")
    genai.configure(api_key=GEMINI_API_KEY)
    print("\n--- Listing Available Gemini Models ---")
    model_to_use = None
    all_available_models = []
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            all_available_models.append(m.name)
            print(f"  Model Name: {m.name}, Supported Methods: {m.supported_generation_methods}")
    priority_models = [
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-flash-002",
        "models/gemini-1.5-flash-001",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro-latest",
        "models/gemini-1.5-pro-002",
        "models/gemini-1.5-pro-001",
        "models/gemini-1.5-pro",
        "models/gemini-2.5-flash-preview-05-20",
        "models/gemini-2.0-flash",
        "models/gemini-pro",
        "models/gemini-1.0-pro"
    ]
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
    test_model = genai.GenerativeModel(model_to_use)
    test_model.generate_content("Hello", safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'})
    gemini_chatbot_model = test_model  # Assign the successfully tested model
    print("Gemini chatbot model initialized successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to configure or initialize Gemini chatbot model: {e}")
    traceback.print_exc()
def get_user_history(user_email):
    """Fetches a user's prediction history from Firebase."""
    if not firebase_admin._apps:
        print("WARNING: Firebase not initialized. Cannot fetch history.")
        return []
    ref = db.reference('results')
    # Order by child and equal to user_email to get only relevant results
    all_results = ref.order_by_child('email').equal_to(user_email).get()
    history_list = []
    if all_results:
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
        if gemini_chatbot_model is None:
            print("Error: Gemini chatbot model is not initialized. Cannot process chat.")
            return jsonify(
                {'response': 'Chatbot is currently unavailable. Model not initialized.'}), 503
        chat_history = session.get('chat_history', [])
        MAX_HISTORY_TURNS = 5
        if len(chat_history) > MAX_HISTORY_TURNS * 2:
            chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]
        history_keywords = [
            "history", "past reports", "my results", "summarize my diabetes",
            "comment on my reports", "about my reports", "my previous results",
            "tell me about my history", "analysis of my reports", "review my data",
            "precautions", "what should i do", "advice", "guidance"
        ]
        is_history_query = any(keyword in user_message.lower() for keyword in history_keywords)
        if is_history_query or \
                (len(chat_history) >= 1 and "diabetes prediction reports" in chat_history[-1]['parts'][0][
                    'text'].lower()):  # Check previous bot response

            if 'user' not in session:
                bot_reply = 'Please log in to view your past reports.'
                # Only append user message and bot reply to history for session
                session['chat_history'] = chat_history + [{'role': 'user', 'parts': [{'text': user_message}]},
                                                          {'role': 'model', 'parts': [{'text': bot_reply}]}]
                return jsonify({'response': bot_reply}), 401
            user_email = session['user']
            history_data = get_user_history(user_email)
            if not history_data:
                bot_reply = "You don't have any past diabetes prediction reports yet."
                # Only append user message and bot reply to history for session
                session['chat_history'] = chat_history + [{'role': 'user', 'parts': [{'text': user_message}]},
                                                          {'role': 'model', 'parts': [{'text': bot_reply}]}]
                return jsonify({'response': bot_reply})
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
                if i >= 4:
                    formatted_history += "...\n(More reports not shown for brevity)\n"
                    break
            history_instruction = (
                "You are a helpful assistant for users of a diabetes prediction app. "
                "Your primary goal is to provide concise, helpful comments, summaries, and general advice based on the user's diabetes prediction reports. "
                "Vary your response style and use unique words for each interaction. "
                "Do not ask for more information or assume future results. Keep responses brief and encouraging."
            )
            full_history_prompt_text = f"{history_instruction}\n\n{formatted_history}\n\nUser: {user_message}"
            messages_for_gemini = chat_history[:-1] + [{'role': 'user', 'parts': [{'text': full_history_prompt_text}]}]
            print(f"Sending history-aware prompt to Gemini: {full_history_prompt_text[:200]}...")
            history_response = gemini_chatbot_model.generate_content(
                messages_for_gemini,
                safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
            )
            bot_reply = history_response.text.strip()
            print(f"Replying with history comment: '{bot_reply}'")
            session['chat_history'] = chat_history + [{'role': 'user', 'parts': [{'text': user_message}]},
                                                      {'role': 'model', 'parts': [{'text': bot_reply}]}]
            return jsonify({'response': bot_reply})
        general_instruction = (
            "You are a helpful assistant for users of a diabetes prediction app. "
            "Respond to general queries about diabetes, health, or the app's features. "
            "Vary your response style and use unique words for each interaction."
        )
        full_general_prompt_text = f"{general_instruction}\n\nUser: {user_message}"
        messages_for_gemini = chat_history[:-1] + [{'role': 'user', 'parts': [{'text': full_general_prompt_text}]}]
        response = gemini_chatbot_model.generate_content(
            messages_for_gemini,
            safety_settings={'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'}
        )
        bot_reply = response.text.strip()
        print(f"Replying with: '{bot_reply}'")
        session['chat_history'] = chat_history + [{'role': 'user', 'parts': [{'text': user_message}]},
                                                  {'role': 'model', 'parts': [{'text': bot_reply}]}]
        return jsonify({'response': bot_reply})
    except Exception as e:
        print(f"Error during chat processing: {e}")
        traceback.print_exc()
        bot_reply = 'Sorry, something went wrong with the chatbot. Please check server logs.'
        # Ensure history is updated even on error for debugging
        session['chat_history'] = chat_history + [{'role': 'user', 'parts': [{'text': user_message}]},
                                                  {'role': 'model', 'parts': [{'text': bot_reply}]}]
        return jsonify({'response': bot_reply}), 500


# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    if gemini_chatbot_model is not None and isinstance(gemini_chatbot_model, genai.GenerativeModel):
        return jsonify({'status': 'ok', 'message': 'Gemini chatbot model is initialized.'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'Gemini chatbot model not initialized.'}), 503

# --- ML Model and Firebase Initialization ---
# Load ML model and scaler
try:
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

# Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate("firebase_admin.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://diabetisprediction-default-rtdb.firebaseio.com'
    })
    print("Firebase Admin SDK initialized successfully.")
except FileNotFoundError:
    print("WARNING: 'firebase_admin.json' not found. Firebase functionality might be affected.")
except Exception as e:
    print(f"ERROR: Failed to initialize Firebase Admin SDK: {e}")
    traceback.print_exc()

# --- Flask Routes ---
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/set_user', methods=['POST'])
def set_user():
    data = request.json
    session['user'] = data.get('email')
    return jsonify({'message': 'User session set'})

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/')
    response = make_response(render_template('home.html', user=session['user']))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/predict-form')
def predict_form():
    if 'user' not in session:
        return redirect('/')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
    for field in field_names:
        value = request.form.get(field)
        if value is None:
            return render_template('index.html', prediction_text=f"Error: Missing input for {field}."), 400
        try:
            input_features.append(float(value))
        except ValueError:
            return render_template('index.html', prediction_text=f"Error: Invalid number format for {field}."), 400
    input_np = np.array(input_features).reshape(1, -1)
    std_data = scaler.transform(input_np)
    prediction = model.predict(std_data)[0]
    prediction_text = "The person is diabetic" if prediction else "The person is not diabetic"
    if firebase_admin._apps:
        ref = db.reference('results')
        ref.push({
            'email': session['user'],
            'input': dict(zip(field_names, input_features)),
            'result': prediction_text,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    else:
        print("WARNING: Firebase not initialized. Prediction result not saved to DB.")
    return render_template('index.html', prediction_text=prediction_text)
@app.route('/history')
def history():
    if 'user' not in session:
        return redirect('/')
    if not firebase_admin._apps:
        return render_template('history.html', results=[], error="Firebase not initialized. Cannot fetch history.")
    results_list = get_user_history(session['user'])  # Use the helper function
    return render_template('history.html', results=results_list)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)