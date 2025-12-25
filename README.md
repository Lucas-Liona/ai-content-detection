# AI Content Detector

A machine learning project that detects AI-generated images using both traditional feature engineering and deep learning approaches.

## Overview

This project implements two different classifiers to distinguish between AI-generated and real images:

1. **Random Forest Classifier** (~71% accuracy) - Uses engineered features including:
   - Statistical moments (mean, variance, skewness, kurtosis, higher-order moments)
   - Discrete Cosine Transform (DCT) coefficients
   - Fast Fourier Transform (FFT) frequency analysis

2. **ResNet50 Deep Learning Model** (~79% accuracy) - Fine-tuned pretrained model for end-to-end image classification

Both models include explainability analysis using LIME (Local Interpretable Model-Agnostic Explanations) and Integrated Gradients to understand decision-making processes.

## Features

- **Feature Engineering**: Extracts frequency-domain and statistical features from images
- **Transfer Learning**: Fine-tunes ResNet50 for AI content detection
- **Model Explainability**: Visualizes which features and image regions influence predictions
- **Comparative Analysis**: Evaluates traditional ML vs deep learning approaches

## Dataset

Uses the [AI vs. Real](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images/) dataset from Kaggle, containing 975 images (539 AI-generated, 436 real).

## Results

| Model | Accuracy |
|-------|----------|
| Random Forest (Engineered Features) | ~71% |
| ResNet50 (Deep Learning) | ~79% |

### Key Findings

- **Random Forest**: Variance and 5th-order statistical moments are the most important features. DCT and FFT frequency ratios add meaningful discriminative power.
- **ResNet50**: Achieves higher accuracy through end-to-end learning, with Integrated Gradients showing focus on texture patterns and edge characteristics.

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd assignment-03-ai-content-detector-Lucas-Liona

# Install dependencies using poetry
poetry install

# Or using pip
pip install -r requirements.txt  # (if you have one)
```

## Usage

Open and run the notebook:

```bash
jupyter notebook src/ai_image_detector.ipynb
```

The notebook walks through:
1. Data loading and preprocessing
2. Feature engineering for Random Forest
3. Training the Random Forest classifier
4. Training the ResNet50 model
5. Explainability analysis for both models

## Technical Details

### Random Forest Features
- **Statistical**: Mean, variance, skewness, kurtosis, 5th-8th order moments
- **DCT**: DC coefficient, AC statistics, frequency band energy ratios
- **FFT**: Magnitude spectrum, low/high frequency ratios

### ResNet50 Training
- Two-phase training: freeze backbone for 3 epochs, then fine-tune full model for 7 epochs
- Dropout regularization to prevent overfitting
- Cross-entropy loss with Adam optimizer

## License

This repository is provided with an MIT license. See the `LICENSE` file.

## Contact

For questions or suggestions, please open an issue on GitHub.
