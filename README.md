# Deep_tabular_methods
This repository contains the implementation for the paper titled "Non-Contact Stress Assessment Based on Tabular Data Learning Architecture", which explores a non-invasive approach to stress detection using deep tabular learning. The model leverages remote photoplethysmography (rPPG) signals extracted from facial videos to classify stress tasks and stress levels.

Abstract
Stress is a significant factor impacting human health, potentially leading to serious conditions such as heart disease, diabetes, and anxiety. This work proposes a non-contact method for stress detection based on rPPG signals and a tabular deep learning architecture. The method has been validated on the publicly available UBFC-Phys dataset, achieving state-of-the-art results for binary and multi-level stress classification tasks.

Repository Contents
src/: Contains Python scripts for preprocessing, feature extraction, and model training.
notebooks/: Jupyter notebooks for exploratory data analysis and experimentation.
data/: Placeholder for dataset (not included for privacy reasons).
models/: Pretrained models and training checkpoints.
results/: Outputs, evaluation metrics, and visualizations.

Dependencies
Python 3.8+
Required Libraries:
numpy, pandas, scikit-learn, torch, opencv-python, mediapipe, matplotlib
Install dependencies using:
pip install -r requirements.txt

Usage
Clone the repository:
git clone https://github.com/Heeya2205/Deep_tabular_methods.git
cd Deep_tabular_methods
Download the UBFC-Phys dataset and place it in the data/ directory.
Run preprocessing:
python src/preprocess.py
Train the model:
python src/train.py



