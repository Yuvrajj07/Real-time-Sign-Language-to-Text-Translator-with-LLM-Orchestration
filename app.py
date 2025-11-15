# app.py (Modified)

from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import pyttsx3              # ✅ Add this import
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from flask import Flask, request, jsonify
from gtts import gTTS

import re
import json
import os
import base64 
import time
import uuid  


# from langchain.schema.output_parser import StrOutputParser
# ------------------------------------------------


app = Flask(__name__)


os.environ["GROQ_API_KEY"] = "GROQ_API_KEY_HERE"

# Create the prompt
deaf_prompt = ChatPromptTemplate.from_template("""
You are an intelligent interpreter helping a deaf person communicate.
The person expresses themselves by giving a list of simple words or short phrases.

Your task:
- Understand what the person most likely means.
- Convert the given words into a natural, fluent English sentence.
- Maintain correct grammar and tone.
- If uncertain, give your best possible interpretation politely.

Examples:
Input: ["go", "school", "tomorrow"]
Output: "I will go to school tomorrow."

Input: ["friend", "come", "home", "today"]
Output: "My friend is coming home today."

Input: ["you", "help", "me", "homework"]
Output: "Can you help me with my homework?"

please don't give any explanations, only provide the final sentence. 
                                               
Input words: {words_list}
""")

# Initialize model
llm = ChatGroq(
    model="qwen/qwen3-32b",
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0.7
)

# Combine components
chain = deaf_prompt | llm | StrOutputParser()
# ------------------------------------------------

# --- Configuration ---
DATA_DIR = 'data'
MODEL_DIR = 'models'
GESTURES_DATA_FILE = os.path.join(DATA_DIR, 'gestures_data.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_model.h5')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global variables for model and label encoder
trained_model = None
label_encoder = None
scaler = None # For feature scaling

# --- NEW GLOBAL VARIABLES FOR WORD SAVING ---
saved_words = []
last_prediction = "No Hand/Gesture Detected"
last_saved_word = "" # Tracks the last word successfully added to the sentence
# -------------------------------------------


# Load model and label encoder if they exist (function remains the same)
def load_trained_model():
    global trained_model, label_encoder, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        try:
            trained_model = tf.keras.models.load_model(MODEL_PATH)
            with open(LABELS_PATH, 'r') as f:
                labels_dict = json.load(f)
                label_encoder = LabelEncoder()
                label_encoder.classes_ = np.array(labels_dict['classes'])
                if 'scaler_mean' in labels_dict and 'scaler_scale' in labels_dict:
                    scaler = StandardScaler()
                    scaler.mean_ = np.array(labels_dict['scaler_mean'])
                    scaler.scale_ = np.array(labels_dict['scaler_scale'])
            print("Model and label encoder loaded successfully.")
        except Exception as e:
            print(f"Error loading model or label encoder: {e}")
            trained_model = None
            label_encoder = None
            scaler = None
    else:
        print("No trained model or labels found. Please train a model first.")

load_trained_model()

# --- Utility Functions (extract_landmarks, draw_landmarks, collect_gesture_data remain the same) ---

def extract_landmarks(image):
    with mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                return np.array(landmarks)
        return None

def draw_landmarks(image, hand_landmarks):
    # This function isn't used directly but kept for completeness
    if hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return image

def collect_gesture_data(frame, gesture_label):
    landmarks = extract_landmarks(frame)
    if landmarks is not None:
        # Create a DataFrame row from landmarks and label
        data_row = pd.DataFrame([landmarks], columns=[f'landmark_{i}' for i in range(len(landmarks))])
        data_row['label'] = gesture_label

        # Append to CSV
        if not os.path.exists(GESTURES_DATA_FILE):
            data_row.to_csv(GESTURES_DATA_FILE, mode='w', header=True, index=False)
        else:
            data_row.to_csv(GESTURES_DATA_FILE, mode='a', header=False, index=False)
        return True
    return False

# --- Flask Routes (index, capture, train_page, video_feed_capture, save_gesture_data, start_training remain the same) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

# ---------- ROUTE: GENERATE SENTENCE ----------

@app.route('/generate_sentence', methods=['POST'])
def generate_sentence():
    try:
        data = request.get_json()
        words_list = data.get("words_list", [])

        # Your model logic
        response = chain.invoke({"words_list": words_list})

        cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        # === TTS ===
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)

        # Generate unique filename each time
        filename = f"output_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join("static", filename)

        engine.save_to_file(cleaned_response, audio_path)
        engine.runAndWait()
        engine.stop()

        return jsonify({
            "sentence": cleaned_response,
            "audio_url": f"/static/{filename}"
        })

    except Exception as e:
        print("Error generating sentence:", e)
        return jsonify({"error": str(e)}), 500


# @app.route('/generate_sentence', methods=['POST'])
# def generate_sentence():
#     try:
#         data = request.get_json()
#         words_list = data.get("words_list", [])

#         # Run LangChain model
#         response = chain.invoke({"words_list": words_list})

#         # Clean text
#         cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

#         # === TEXT TO SPEECH ===
#         tts = gTTS(text=cleaned_response, lang='en', slow=False)
#         audio_path = os.path.join("static", "output_audio.mp3")
#         tts.save(audio_path)

#         return jsonify({
#             "sentence": cleaned_response,
#             "audio_url": f"/{audio_path}"
#         })

#     except Exception as e:
#         print("Error generating sentence:", e)
#         return jsonify({"error": str(e)}), 500


# ------------------------------------------------

def generate_capture_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed_capture')
def video_feed_capture():
    return Response(generate_capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_gesture_data', methods=['POST'])
def save_gesture_data():
    data = request.json
    image_data_url = data['imageDataUrl']
    gesture_label = data['gestureLabel']

    nparr = np.frombuffer(base64.b64decode(image_data_url.split(',')[1]), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if collect_gesture_data(frame, gesture_label):
        return jsonify({'status': 'success', 'message': f'Gesture "{gesture_label}" data saved.'})
    else:
        return jsonify({'status': 'error', 'message': 'No hand detected or failed to save data.'}), 400

@app.route('/start_training', methods=['POST'])
def start_training():
    global trained_model, label_encoder, scaler
    try:
        if not os.path.exists(GESTURES_DATA_FILE):
            return jsonify({'status': 'error', 'message': 'No gesture data found. Please capture data first.'}), 400

        df = pd.read_csv(GESTURES_DATA_FILE)
        
        if df.empty or 'label' not in df.columns:
            return jsonify({'status': 'error', 'message': 'No valid gesture data found.'}), 400

        X = df.drop('label', axis=1).values
        y = df['label'].values

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Reshape for LSTM (samples, timesteps, features) - 1 timestep for static gestures
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

        X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

        # Build a simple LSTM model
        model = Sequential([
            LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu'),
            Dropout(0.2),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Training model...")
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        print("Model training complete.")

        # Save the model
        model.save(MODEL_PATH)
        trained_model = model

        # Save label encoder classes and scaler parameters
        labels_dict = {
            'classes': label_encoder.classes_.tolist(),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist()
        }
        with open(LABELS_PATH, 'w') as f:
            json.dump(labels_dict, f)

        return jsonify({'status': 'success', 'message': 'Model trained and saved successfully!', 'accuracy': model.evaluate(X_test, y_test)[1]})
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({'status': 'error', 'message': f'Error during training: {str(e)}'}), 500

# --- NEW ROUTE FOR SAVING WORD (MODIFIED FOR DEBOUNCING) ---
@app.route('/save_word', methods=['POST'])
def save_word():
    global saved_words, last_prediction, last_saved_word
    
    if last_prediction and not last_prediction.startswith("No Hand"):
        try:
            # Extract the pure word (e.g., from "Gesture: WORD (Confidence)")
            word_to_save = last_prediction.split(':')[1].split('(')[0].strip()
        except IndexError:
            word_to_save = last_prediction.strip()
            
        # Debouncing Logic: Only save if the detected word is different from the last word saved
        if word_to_save and word_to_save != last_saved_word:
            saved_words.append(word_to_save)
            last_saved_word = word_to_save # Update the tracking variable
            
            return jsonify({
                'status': 'success', 
                'message': 'Word saved.', 
                'last_word_saved': word_to_save,
                'current_sentence': ' '.join(saved_words) # This contains all recognized words
            })
        else:
            return jsonify({
                'status': 'error', 
                'message': f'The word "{last_saved_word}" was just saved. Please perform a different gesture before saving again.'
            }), 400
    else:
        return jsonify({
            'status': 'error', 
            'message': 'No valid gesture recently detected to save.'
        }), 400
# ----------------------------------

# --- NEW ROUTE FOR DELETING LAST WORD ---
@app.route('/delete_last_word', methods=['POST'])
def delete_last_word():
    global saved_words, last_saved_word
    
    if saved_words:
        deleted_word = saved_words.pop()
        
        # Reset last_saved_word tracker if the list is now empty
        if not saved_words:
            last_saved_word = ""
        else:
            # Update last_saved_word to be the new last word in the list
            last_saved_word = saved_words[-1]

        return jsonify({
            'status': 'success', 
            'message': 'Last word deleted.', 
            'deleted_word': deleted_word,
            'current_sentence': ' '.join(saved_words)
        })
    else:
        return jsonify({
            'status': 'error', 
            'message': 'The sentence is already empty. Nothing to delete.'
        }), 400
# ----------------------------------


# Video feed for real-time detection (MODIFIED TO UPDATE GLOBAL last_prediction)
def generate_detection_frames():
    global last_prediction
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            predicted_text = "No Hand/Gesture Detected"

            if trained_model and label_encoder and scaler:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        )

                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])

                        # Prepare for prediction: scale and reshape
                        landmarks_scaled = scaler.transform(np.array(landmarks).reshape(1, -1))
                        landmarks_reshaped = landmarks_scaled.reshape(1, 1, landmarks_scaled.shape[1])

                        prediction = trained_model.predict(landmarks_reshaped, verbose=0)
                        predicted_class_index = np.argmax(prediction)
                        confidence = prediction[0][predicted_class_index] * 100

                        if confidence > 70: # Confidence threshold
                            recognized_word = label_encoder.inverse_transform([predicted_class_index])[0]
                            predicted_text = f"Gesture: {recognized_word} ({confidence:.2f}%)"
                        else:
                            predicted_text = "Gesture: Unknown (Low Confidence)"
                        break # Only process the first detected hand for simplicity

            # Update the global variable with the latest prediction before drawing
            last_prediction = predicted_text
            
            # Display predicted text on frame
            cv2.putText(frame, predicted_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed_detect')
def video_feed_detect():
    return Response(generate_detection_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)