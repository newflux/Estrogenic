import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sqlite3
import os

# Setup Tesseract path (update this path if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# SQLite Database setup
DATABASE_FILE = "user_data.db"

# Ensure the database exists
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

# Register a new user
def register_user(username, password):
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        if c.fetchone():
            st.error("Username already exists. Please choose a different username.")
            return False
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        st.success("Registration successful. Please login.")
        return True
    except Exception as e:
        st.error(f"An error occurred during registration: {e}")
        return False

# Verify login credentials
def check_login(username, password):
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()
        return bool(user)
    except Exception as e:
        st.error(f"An error occurred during login: {e}")
        return False

# Process uploaded image to extract text
def process_image(image):
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return ""

# Predict text using the trained model
def predict_text(text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction

# Predict uploaded image
def predict_image(image):
    extracted_text = process_image(image)
    if extracted_text:
        prediction = predict_text(extracted_text)
        return prediction, extracted_text
    return None, ""

# Initialize SQLite database
init_db()

# Load dataset and train the model
DATASET_PATH = r'C:\Users\Artist\Desktop\Estrogenic\DEDuCT_ChemicalBasicInformation.csv'
if not os.path.exists(DATASET_PATH):
    st.error(f"Dataset not found at {DATASET_PATH}")
    st.stop()

df = pd.read_csv(DATASET_PATH)

# Ensure the dataset has the required columns
text_column_name = 'Name'  # Change this to your actual text column name
if text_column_name not in df.columns or 'estrogen present' not in df.columns:
    st.error("Dataset is missing required columns: 'Name' or 'estrogen present'")
    st.stop()

X = df[text_column_name]
y = df['estrogen present']

# Train the model
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if 'register' not in st.session_state:
    st.session_state.register = False

# User Interface
if not st.session_state.logged_in and not st.session_state.register:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.username = username
            st.session_state.logged_in = True
            st.success("Login successful")
        else:
            st.error("Invalid username or password")
    if st.button("Register"):
        st.session_state.register = True
elif not st.session_state.logged_in and st.session_state.register:
    st.title("Register")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Create Account"):
        if register_user(new_username, new_password):
            st.session_state.register = False
        else:
            st.error("Registration failed. Please try again.")
    if st.button("Back to Login"):
        st.session_state.register = False
else:
    st.sidebar.button("Logout", on_click=lambda: (setattr(st.session_state, 'logged_in', False), setattr(st.session_state, 'username', "")))
    st.title(f'Welcome, {st.session_state.username}')

    st.title('Endocrine Disruptors and Estrogen Prediction')

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    text_input = st.text_area("Or enter text directly:")

    if uploaded_file:
        image = Image.open(uploaded_file)
        prediction, extracted_text = predict_image(image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('Extracted Text:', extracted_text)
        if prediction is not None:
            st.write('Prediction:', 'Estrogen present' if prediction[0] == 1 else 'Non-Estrogenic')

    elif text_input:
        prediction = predict_text(text_input)
        st.write('Input Text:', text_input)
        st.write('Prediction:', 'Estrogen present' if prediction[0] == 1 else 'Non-Estrogenic')
