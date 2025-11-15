
<h1 align="center">🤟 Sign Language Detection Web App</h1>
<h3 align="center"> | Deep Learning × Computer Vision × Real-Time Browser App × AI Sentence Generation |</h3>

<p align="center">
  <img src="https://komarev.com/ghpvc/?username=sign-lang-detection&label=PROJECT+VIEWS" alt="views" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Flask-blue?logo=flask" />
  <img src="https://img.shields.io/badge/Frontend-JavaScript-yellow?logo=javascript" />
  <img src="https://img.shields.io/badge/DeepLearning-TensorFlow-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/AI-LangChain×Groq-green?logo=OpenAI" />
</p>

---

## 📌 Description

A web-based application that detects hand signs using a deep learning model **and converts recognized words into meaningful English sentences using LangChain + Groq AI.**

This project combines:
- 🧠 A trained CNN model for gesture recognition
- 💬 LangChain + Groq LLM for intelligent sentence generation
- 🌐 A real-time browser-based interface built with Flask and JavaScript

---

## 🚀 Features

- 📷 Real-time webcam gesture detection in browser  
- 🤖 Pre-trained Keras model (`gesture_model.h5`) for ASL alphabets (A–Z)  
- 🧠 AI-powered **sentence generation** using LangChain and Groq LLM (`qwen/qwen3-32b`)  
- 🔠 Converts recognized signs into grammatically correct English sentences  
- 🖥️ Clean frontend using HTML, CSS, and Vanilla JS  
- 🔁 Flask-powered Python backend  

---

## 📁 Project Structure

- 📄 **app.py** — Flask backend
- 📂 **data/**
  - 📄 gestures_data.csv — Dataset used for training
- 📂 **models/**
  - 🧠 gesture_model.h5 — Trained Keras model
  - 📝 labels.json — Gesture-label mappings
- 📂 **static/**
  - 📂 css/
    - 🎨 style.css — Custom styles
  - 📂 js/
    - ✍️ capture.js — Capture gesture data
    - 👁️ detect.js — Real-time gesture detection
    - 🛠️ train.js — Trigger training routines
- 📂 **templates/**
  - 🏠 index.html — Homepage
  - ✋ capture.html — Gesture capture interface
  - 🧪 train.html — Model training interface
- 📁 **venv/** — Python virtual environment (optional)



---

## 🧠 LangChain + Groq Integration

When the **“Generate Sentence”** button is clicked:
1. The recognized words are sent to the Flask backend.
2. Flask runs a **LangChain pipeline** with **Groq’s Qwen model**.
3. The model interprets and converts the recognized words into a fluent English sentence.
4. The result is displayed in the “Possible Sentence” section on the UI.

Example:
Input: ["go", "school", tomorrow"]
Output: "I will go to school tomorrow."


---

## 🚀 Getting Started

### 🔧 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


▶️ Run the App
python app.py

📦 Requirements
flask
tensorflow
numpy
mediapipe
opencv-python
pandas
scikit-learn
langchain-core
langchain-community
langchain-groq

🙋‍♂️ Author

Yuvraj Singh
📧 vyuvrajsingh98@gmail.com

🌐 https://github.com/Yuvrajj07






