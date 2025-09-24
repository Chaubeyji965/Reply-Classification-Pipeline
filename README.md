# Reply Classification with Neural Network (MLP)

This project implements a Feedforward Neural Network (MLP) for classifying email replies into three categories: **positive**, **negative**, and **neutral**.  
The model uses **TF-IDF features** and a multi-layer perceptron architecture built with **TensorFlow/Keras**.

---

## Project Structure

├── app.py # Flask API application
├── SvaraAi.ipynb # Neural network training notebook
├── answers.md # Reasoning answers
├── best_nn_model.h5 # Trained neural network model
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── cleaned_dataset.csv # Processed dataset
├── reply_classification_dataset.csv # Original dataset
├── requirements.txt # Python dependencies
├── Dockerfile # Docker configuration
└── README.md # This file


## Model Performance

- **Best Model**: Multi-Layer Perceptron (MLP)
- **Accuracy**: ~93%
- **F1 Score**: ~93%

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
** 2. Training**
To train the model from scratch:

bash
Copy code
python notebook.ipynb
This will:

Preprocess and clean the dataset

Generate TF-IDF embeddings

Train a 3-layer neural network with dropout

Save the trained model (best_nn_model.h5) and vectorizer (vectorizer.pkl)

API Usage
Starting the Server
bash
Copy code
python app.py
Server runs on http://localhost:5000

API Endpoints
POST /predict
Classify a reply text.

Request:


{
    "text": "Looking forward to the demo!"
}
Response:


{
    "label": "positive",
    "confidence": 0.99
}

**Data Preprocessing**
The preprocessing pipeline includes:

Text cleaning (lowercasing, removing special characters)

Abbreviation expansion (u → you, plz → please)

Duplicate removal

Label standardization

TF-IDF vectorization (max features = 2000, bigram support)

**Model Architecture**
Input Layer: TF-IDF features

Hidden Layers:

Dense (256 units, ReLU) + Dropout(0.3)

Dense (128 units, ReLU) + Dropout(0.3)

Output Layer: Dense (3 units, softmax)

Optimizer: Adam

Loss: Categorical Cross-Entropy

**Docker Deployment**
Build and run with Docker:


docker build -t reply-classifier .
docker run -p 5000:5000 reply-classifier
Production Considerations
Save/load both model and TF-IDF vectorizer

Handle unknown inputs and edge cases

Include a confidence threshold for predictions

Monitor performance over time

Scalable deployment with Docker

**Future Improvements**
Replace TF-IDF with word embeddings (Word2Vec, GloVe, FastText)

Experiment with CNNs or RNNs for sequence modeling

Hyperparameter tuning for deeper/broader networks

Active learning loop for continuous improvement
