# Forest Fire Prediction System

An AI-powered web application for early detection and prediction of forest fires using a dual-mode verification pipeline:
1. **Weather Sensor Analysis**: Tabular model based on meteorological inputs (utilizing the Algerian Forest Fires dataset) to classify fire risk and predict the Fire Weather Index (FWI).
2. **Visual/Image Analysis**: Combined ensemble of a **Convolutional Neural Network (CNN)** (MobileNetV3), **Gradient Boosting Classifier** (handcrafted color/HSV features), and a **Smart Fire Heuristic** veto system to detect fires from uploaded images.

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Make sure you have Python 3.10+ installed.

Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

*(Optional)* To train or run the CNN (MobileNetV3), install PyTorch and torchvision:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Dataset & Training

To train the machine learning models:

1. **Download the Datasets** (if not already downloaded):
   ```bash
   python download_data.py
   ```
2. **Train the Tabular and Image-based Gradient Boosting Models**:
   ```bash
   python train_models.py
   ```
   This will train the models and save the pickle bundles (`image_model.pkl`, `scaler.pkl`, `classifier.pkl`, `regressor.pkl`) inside the `models/` directory.

3. **Train the CNN Model** (Optional):
   ```bash
   python train_cnn.py
   ```

---

## 💻 Running the Application

Start the Flask web server:
```bash
python app.py
```

The application will start locally on:
👉 **[http://127.0.0.1:8080](http://127.0.0.1:8080)**

---

## 🛠 Project Structure

- `app.py`: Main Flask application server hosting the API endpoints and predictions logic.
- `train_models.py`: Model training script for tabular data (Algerian dataset) and image feature extraction (Gradient Boosting).
- `train_cnn.py`: PyTorch training routine for the CNN classifier.
- `templates/`: HTML views for the front-end interface.
  - `index.html`: Main user dashboard with form inputs and image drop zone.
- `static/`: Static resources and images (e.g. background image).
- `models/`: Destination folder for trained model checkpoint weights and scikit-learn artifacts.
- `requirements.txt`: Python package dependencies.
