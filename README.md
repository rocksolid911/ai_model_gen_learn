# Complete End-to-End ML Project: Text & Image Classification

A comprehensive machine learning project demonstrating practical AI using **NumPy**, **pandas**, **scikit-learn**, and **TensorFlow**. This project includes two complete pipelines: news text classification and image recognition from scratch.

## 🎯 Project Overview

### Task 1: News Text Classification
Classify news articles into 4 categories: `neutral`, `left`, `right`, or `propaganda`.
- **Baseline Model**: Scikit-learn with TF-IDF features
- **Deep Learning Model**: TensorFlow/Keras with embeddings and neural networks

### Task 2: Image Recognition
Build a CNN from scratch for image classification.
- **Custom CNN**: Built from scratch using TensorFlow/Keras
- **GPU Support**: Optimized for laptop GPU training

## 📁 Project Structure

```
.
├── data/                          # Data directory
│   ├── news.csv                   # News dataset (generated or your own)
│   └── images/                    # Image dataset folder
│       └── <class_name>/          # Each subfolder = one class
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_eda_news.ipynb         # Exploratory data analysis for news
│   ├── 02_train_news_models.ipynb # Train news classifiers
│   └── 03_train_image_model.ipynb # Train image classifier
├── src/                           # Source code
│   ├── data_loading/              # Data loading modules
│   │   ├── news_dataset.py       # News data loader
│   │   └── image_dataset.py      # Image data loader
│   ├── preprocessing/             # Preprocessing modules
│   │   ├── text_preprocessing.py # Text cleaning and feature extraction
│   │   └── image_preprocessing.py # Image preprocessing
│   ├── models/                    # Model architectures
│   │   ├── news_baseline_sklearn.py  # Sklearn baseline
│   │   ├── news_deep_tf.py           # TensorFlow news model
│   │   └── image_cnn_tf.py           # TensorFlow CNN
│   ├── training/                  # Training scripts
│   │   ├── train_news_baseline.py
│   │   ├── train_news_tf.py
│   │   └── train_image_cnn.py
│   ├── evaluation/                # Evaluation scripts
│   │   ├── evaluate_news_models.py
│   │   └── evaluate_image_model.py
│   └── inference/                 # Inference scripts
│       ├── predict_news.py       # Predict news article class
│       └── predict_image.py      # Predict image class
├── models/                        # Saved trained models
├── scripts/                       # Utility scripts
│   └── generate_sample_data.py   # Generate sample data
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Setup Instructions

### 1. Create Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Download NLTK data (for text preprocessing)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 3. Verify GPU Support (Optional but Recommended)

```bash
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

If GPUs are detected, TensorFlow will automatically use them for training.

## 📊 Data Preparation

### Option 1: Generate Sample Data

Use the provided script to generate sample datasets for testing:

```bash
python scripts/generate_sample_data.py
```

This creates:
- `data/news.csv` with 1000 sample news articles
- `data/images/` with sample images in multiple class folders

### Option 2: Use Your Own Data

#### For News Classification:
Place your CSV file at `data/news.csv` with columns:
- `id`: Unique identifier
- `title`: News article title
- `content`: Article body text
- `label`: One of [`neutral`, `left`, `right`, `propaganda`]

#### For Image Classification:
Organize images in folders:
```
data/images/
├── class_1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── class_2/
│   └── ...
└── class_n/
    └── ...
```

## 🏃 Running the Project

### News Text Classification

#### Step 1: Train the Baseline Model (Sklearn + TF-IDF)

```bash
python src/training/train_news_baseline.py
```

**What this does:**
- Loads and preprocesses news data
- Extracts TF-IDF features
- Trains a Logistic Regression classifier
- Saves model to `models/news_baseline_model.pkl`

#### Step 2: Train the Deep Learning Model (TensorFlow)

```bash
python src/training/train_news_tf.py
```

**What this does:**
- Tokenizes and sequences text data
- Builds a neural network (Embedding + Conv1D/LSTM + Dense)
- Trains on GPU if available
- Saves model to `models/news_deep_model.h5`

#### Step 3: Evaluate Models

```bash
python src/evaluation/evaluate_news_models.py
```

**Outputs:**
- Accuracy, Precision, Recall, F1-score for each class
- Confusion matrices
- Comparison between baseline and deep learning models

#### Step 4: Make Predictions

```bash
# Predict from text
python src/inference/predict_news.py --text "Your news article text here..."

# Predict from file
python src/inference/predict_news.py --file path/to/article.txt

# Choose model type
python src/inference/predict_news.py --text "..." --model baseline  # or 'deep'
```

### Image Classification

#### Step 1: Train the CNN Model

```bash
python src/training/train_image_cnn.py
```

**What this does:**
- Loads images from `data/images/`
- Applies preprocessing and augmentation
- Builds a custom CNN from scratch
- Trains on GPU if available
- Saves model to `models/image_cnn_model.h5`

#### Step 2: Evaluate the Model

```bash
python src/evaluation/evaluate_image_model.py
```

**Outputs:**
- Test accuracy and loss
- Per-class metrics
- Confusion matrix
- Sample predictions with visualizations

#### Step 3: Make Predictions

```bash
python src/inference/predict_image.py --image path/to/image.jpg

# Show top-k predictions
python src/inference/predict_image.py --image path/to/image.jpg --top-k 3
```

## 📓 Jupyter Notebooks

For interactive exploration and visualization:

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_eda_news.ipynb - Explore news data
# 2. notebooks/02_train_news_models.ipynb - Train and compare news models
# 3. notebooks/03_train_image_model.ipynb - Train and visualize image model
```

## 🎓 Learning Notes

### Key ML Concepts Covered

1. **Data Preprocessing**
   - Why: Raw data is messy; models need clean, normalized inputs
   - Text: Tokenization, stopword removal, TF-IDF vectorization
   - Images: Resizing, normalization, augmentation

2. **Train/Validation/Test Split**
   - Why: Prevent overfitting and get honest performance estimates
   - Training set: Learn patterns
   - Validation set: Tune hyperparameters
   - Test set: Final evaluation (never seen during training)

3. **Feature Engineering**
   - Traditional ML: TF-IDF captures word importance
   - Deep Learning: Neural networks learn features automatically

4. **Model Selection**
   - Baseline (Sklearn): Fast, interpretable, good starting point
   - Deep Learning: Can learn complex patterns but needs more data/compute

5. **Evaluation Metrics**
   - Accuracy: Overall correctness
   - Precision: Of predicted positives, how many are correct?
   - Recall: Of actual positives, how many did we find?
   - F1-Score: Harmonic mean of precision and recall

6. **Neural Network Components**
   - Embeddings: Convert words to dense vectors
   - Convolutional layers: Extract local patterns
   - LSTM/GRU: Capture sequential dependencies
   - Dense layers: Learn high-level representations
   - Softmax: Multi-class probability distribution

## 🔧 Troubleshooting

### Out of Memory Errors
- Reduce batch size in training scripts
- Use smaller image sizes (e.g., 128x128 instead of 224x224)
- Train on smaller data subsets

### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi

# Reinstall TensorFlow GPU
pip install tensorflow==2.15.0
```

### Slow Training
- Ensure GPU is being used (check script output)
- Reduce model complexity
- Use smaller datasets for testing

## 📚 Further Learning

- **Scikit-learn docs**: https://scikit-learn.org/
- **TensorFlow tutorials**: https://www.tensorflow.org/tutorials
- **Deep Learning book**: http://www.deeplearningbook.org/

## 🤝 Contributing

This is a learning project. Feel free to:
- Experiment with different architectures
- Try different hyperparameters
- Add new models or features
- Improve documentation

## 📝 License

This project is for educational purposes. Use freely for learning and experimentation.

---

**Happy Learning! 🚀🤖**
