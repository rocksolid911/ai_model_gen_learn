# Quick Start Guide

Get up and running with this ML project in 5 minutes!

## Step 1: Setup Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Step 2: Generate Sample Data

```bash
# Generate synthetic news articles and images
python scripts/generate_sample_data.py
```

This creates:
- `data/news.csv` - 1000 news articles
- `data/images/` - 500 images across 5 classes

## Step 3: Train Models

### News Classification (Baseline)
```bash
# Train TF-IDF + Logistic Regression (fast, ~1 minute)
python src/training/train_news_baseline.py
```

### News Classification (Deep Learning)
```bash
# Train CNN/LSTM model (slower, ~5-10 minutes)
python src/training/train_news_tf.py
```

### Image Classification
```bash
# Train CNN from scratch (~10-15 minutes on CPU, ~3-5 minutes on GPU)
python src/training/train_image_cnn.py
```

## Step 4: Evaluate Models

```bash
# Compare news models
python src/evaluation/evaluate_news_models.py

# Evaluate image model
python src/evaluation/evaluate_image_model.py
```

## Step 5: Make Predictions

### Predict News Article

```bash
# Using baseline model
python src/inference/predict_news.py --model baseline --text "Your news article here..."

# Using deep learning model
python src/inference/predict_news.py --model deep --text "Your news article here..."

# From file
python src/inference/predict_news.py --model deep --file path/to/article.txt
```

### Predict Image Class

```bash
python src/inference/predict_image.py --image path/to/image.jpg

# Show top 3 predictions
python src/inference/predict_image.py --image path/to/image.jpg --top-k 3
```

## Step 6: Explore with Jupyter

```bash
# Start Jupyter notebook
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_eda_news.ipynb - Explore data
# 2. notebooks/02_train_news_models.ipynb - Train and compare models
# 3. notebooks/03_train_image_model.ipynb - Train CNN
```

## Project Structure

```
.
├── data/                   # Datasets
├── models/                 # Saved trained models
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── src/
│   ├── data_loading/      # Data loaders
│   ├── preprocessing/     # Preprocessing pipelines
│   ├── models/            # Model architectures
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation scripts
│   └── inference/         # Prediction scripts
├── requirements.txt       # Dependencies
└── README.md             # Full documentation
```

## Common Issues

### GPU Not Detected
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Out of Memory
- Reduce batch size in training scripts
- Use smaller image size (e.g., 64x64 instead of 128x128)
- Train on smaller dataset subset

### Slow Training
- Ensure GPU is being used (check training output)
- Reduce model complexity
- Use smaller datasets for testing

## Next Steps

1. **Customize**: Modify architectures in `src/models/`
2. **Tune**: Adjust hyperparameters in training scripts
3. **Extend**: Add new models or features
4. **Deploy**: Use inference scripts for production

## Learn More

- Full documentation: `README.md`
- Model architectures: `src/models/`
- Training details: `src/training/`
- Evaluation metrics: `src/evaluation/`

Happy Learning! 🚀
