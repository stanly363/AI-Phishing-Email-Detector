# Phishing Email Classifier (TensorFlow)

A simple TensorFlow/Keras pipeline to train a binary classifier on a CEAS-style phishing email dataset.  
The model uses an embedding + Bi-LSTM architecture and includes early-stopping and checkpointing.

## 📋 Contents

- `phishing.csv`  
  - CSV with columns: `sender, receiver, date, subject, body, label, urls`
  - `label`: `1` for spam/phishing, `0` for ham/legitimate

- `tensor.py`  
  - Loads and preprocesses data  
  - Builds `tf.data.Dataset` pipelines  
  - Vectorizes text with `TextVectorization`  
  - Defines & compiles an Embedding + Bi-LSTM model  
  - Trains with early stopping & checkpoint callbacks  
  - Prints GPU availability  

## 🛠️ Requirements

- Python 3.8–3.12  
- TensorFlow 2.x  
- scikit-learn  
- pandas  

## 🚀 Installation

1. **Activate your virtual environment**  
   - **macOS / Linux**  
     ```bash
     source venv/bin/activate
     ```  
   - **Windows (PowerShell)**  
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```  
   - **Windows (cmd.exe)**  
     ```bat
     .\venv\Scripts\activate.bat
     ```

2. **Install dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

## Run the model
By running
  ```bash
  python tensor.py
