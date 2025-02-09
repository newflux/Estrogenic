# Chemical Estrogenicity Classification using OCR and Machine Learning

This project classifies chemicals as Estrogenic or Non-Estrogenic using Optical Character Recognition (OCR) and Machine Learning (ML). The system takes text or image input, extracts chemical information via OCR, and processes it through an ML model. The deployment is handled using Streamlit for an interactive user interface.


https://github.com/user-attachments/assets/3138eff8-8731-4da4-ad2b-88ece27c9a36



------------

## Project Overview

- **Input**: Chemical name (via text or image)

- **Processing:**
  + Extract text from images using OCR (Tesseract/Pytesseract)
 +  Preprocess the extracted text for ML classification
  + Classify chemicals using a trained ML model

- **Output**: Classification as Estrogenic or Non-Estrogenic

- **Deploymen**t: Interactive Streamlit web application

------------

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/Estrogenic.git
cd Estrogenic
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

`pip install -r requirements.txt`

------------

## Usage

### 1. Run the Streamlit Application

`streamlit run app.py`

### 2. Input Options

- Upload an Image: The system extracts chemical names using OCR and classifies them.
- Enter Chemical Name: Direct text input for classification.

### 3. View Results

- The application displays whether the chemical is Estrogenic or Non-Estrogenic.
- Confidence scores and additional details are provided.

------------





