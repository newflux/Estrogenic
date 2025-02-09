import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import sqlite3
import os

# Setup Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

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

# Verify login credentials
def check_login(username, password):
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return bool(user)

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

# Predict text (always estrogenic)
def predict_text(text):
    return check_estrogenicity(text)

# Predict uploaded image (always estrogenic)
def predict_image(image):
    extracted_text = process_image(image)
    st.write("Extracted Text:", extracted_text)
    prediction = check_estrogenicity(extracted_text)
    return prediction, extracted_text

# Load the chemical database (CSV)
DATABASE_PATH = r"C:/Users/Artist/Desktop/Estrogenic\DEDuCT_ChemicalBasicInformation.csv"
try:
    chemical_data = pd.read_csv(DATABASE_PATH)
except Exception as e:
    st.error(f"Error reading the CSV file: {e}")
    chemical_data = pd.DataFrame()  # Empty DataFrame if file read fails

# Check if the 'ChemicalName' and 'Estrogenic' columns exist
if 'Name' not in chemical_data.columns or 'estrogen present' not in chemical_data.columns:
    st.error("The CSV file does not contain the required columns: 'ChemicalName' and 'Estrogenic'. Please check the file.")

# Check if the chemical name exists in the database and return estrogenic status
def check_estrogenicity(chemical_name):
    # Ensure that chemical_name is a string and strip any extra spaces
    if not isinstance(chemical_name, str) or not chemical_name.strip():
        st.error("Invalid chemical name.")
        return None
    
    chemical_name = chemical_name.strip()

    # Check for matches in the DataFrame, case insensitive
    matches = chemical_data[chemical_data['Name'].str.contains(chemical_name, case=False, na=False)]
    
    # Debugging: Check if matches were found
    if not matches.empty:
        st.write("Matching Chemicals Found:", matches)
        estrogenic_status = matches['estrogen present'].iloc[0]  # Get the estrogenic status of the first match
        if estrogenic_status == 1:
            return "Estrogenic"
        else:
            return "Non Estrogenic"
    else:
        st.error(f"No match found for '{chemical_name}' in the database.")
        return "No match found"

# Initialize SQLite database
init_db()

# Streamlit session state
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

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    text_input = st.text_area("Or enter text directly:")

    if uploaded_file:
        image = Image.open(uploaded_file)
        prediction, extracted_text = predict_image(image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('Extracted Text:', extracted_text)
        st.write('Prediction:', prediction)

    elif text_input:
        prediction = predict_text(text_input)
        st.write('Input Text:', text_input)
        st.write('Prediction:', prediction)
       
