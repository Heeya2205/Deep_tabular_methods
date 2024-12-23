# Deep_tabular_methods
This repository contains the implementation of the paper titled "Non-Contact Stress Assessment Based on Deep Tabular Method." The paper presents a non-invasive approach for classifying stress tasks (binary and ternary) and stress levels. The model leverages remote photoplethysmography (rPPG) signals extracted from facial videos to classify both stress tasks and stress levels.

Abstract
Stress is a significant factor impacting human health, potentially leading to serious conditions such as heart disease, diabetes, and anxiety. This work proposes a non-contact method for stress detection based on rPPG signals and a tabular deep learning architecture. The method has been validated on the publicly available UBFC-Phys dataset, achieving state-of-the-art results for binary and multi-level stress classification tasks.

Repository Contents
preprocess & color_traces/: Contains Python scripts for preprocessing.
pos/: Contains the algorithm to extract features.
weights/: Contains the weights for binary and ternary task classifications.
level_weights/: Contains weights for level classification.
binaryclassification/: Holds the Python code (.py files) for binary stress task classifications.
levelclassification/: Contains the Python code (.py files) for stress level classifications.

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
Download the UBFC-Phys dataset from "Rita Meziati, Yannick Benezeth, Pierre De Oliveira, Julien Chappé, Fan Yang, March 3, 2021, "UBFC-Phys", IEEE Dataport, doi: https://dx.doi.org/10.21227/5da0-7344".
Run preprocessing:
python preprocess.py
Train the model:
python binaryclassification.py
python levelclassification.py



