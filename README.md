# 🍄 Mushroom Classification Web App

This project is a web-based application that classifies mushroom images into different categories using a trained machine learning model.
It has been developed as part of an academic assignment.

## 🚀 Features

* Upload an image of a mushroom
* View top predictions with confidence scores
* Displays final predicted genus
* Shows edible vs poisonous probability

## ▶️ Setup & Run

### Important Note on Python Versions
TensorFlow (required by this project) currently supports **Python 3.12 or older**, which causes installation issues (`ModuleNotFoundError: No module named 'tensorflow'`) if you try running this project directly using newer Python versions like Python 3.13 or 3.14. 

To easily manage this without touching your system's global Python installation, we rely on the `uv` tool to automatically download a compatible Python environment.

### Installation Instructions

1. **Install `uv`** (if not already installed):
```bash
pip install uv
```

2. **Create a compatible virtual environment** (forces Python 3.12):
```bash
python -m uv venv --python 3.12 venv
```

3. **Install the project dependencies** into the virtual environment:
```bash
python -m uv pip install --python .\venv\Scripts\python.exe -r requirements.txt
```

4. **Run the web application**:
```bash
.\venv\Scripts\python.exe app.py
```

Then hold `CTRL` and click the `http://127.0.0.1:5000/` link in your terminal to open the application in your browser.

## 📁 Files

* `app.py` – Flask app
* `mushroom_model.h5` – trained model
* `mushroom_classification.ipynb` – training notebook

## ⚠️ Disclaimer

* This is an academic/practice project and the model is not fully accurate.
* The predictions should **not be used for real-life consumption or safety decisions**.

## 👩‍💻 Author

Aarohi Nagdeote
