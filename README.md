# Heart Disease Prediction with Machine Learning

This project uses advanced preprocessing techniques and various machine learning models to predict the likelihood of heart disease. It includes data imputation, encoding, scaling, dimensionality reduction with PCA, model optimization, and evaluation using AUC-ROC and confusion matrices.

![Heart health](https://github.com/user-attachments/assets/919283d3-43ba-4ba7-a4ab-0560ef709502)

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Data Preprocessing](#data-preprocessing)  
- [Models Evaluated](#models-evaluated)  
- [Results](#results)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Acknowledgments](#acknowledgments)  
- [Contact](#contact)  
- [References](#references)  
- [License](#license)  

---

## Overview

This project focuses on predicting heart disease using machine learning models. It preprocesses data efficiently and evaluates several models to determine the best performer. The dataset used in this project is `heart_disease.csv`.

---

## Features

- Handles missing values with `SimpleImputer` (Median/Most Frequent).
- Encodes categorical variables using `LabelEncoder` and `OneHotEncoder`.
- Scales numerical features with `StandardScaler`.
- Applies PCA for dimensionality reduction.
- Optimizes models using `GridSearchCV`.
- Evaluates models with AUC-ROC and confusion matrices.
- Saves preprocessing and model artifacts using `joblib`.

---

## Data Preprocessing

The following steps are applied:  

1. **Imputation**: Missing values in numerical and categorical data are handled using median and most frequent strategies, respectively.
2. **Encoding**: Binary variables are label-encoded, and multi-class variables are one-hot encoded.
3. **Scaling**: Numerical features are standardized.
4. **Dimensionality Reduction**: PCA is applied to reduce feature dimensions while retaining essential information.

---

## Models Evaluated

The following models were tested:

1. Logistic Regression  
2. Random Forest  
3. Support Vector Machines (SVM)  
4. K-Nearest Neighbors (KNN)  
5. Gradient Boosting  

---

## Results

The best model identified was **[Best Model Name]**, achieving an AUC score of **[Best AUC Score]**. Confusion matrices and AUC-ROC curves for all models are included in the results visualization.

E:\01_Personal Gigs\00_Repository\05_Heart_Disease\output1.png

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
```

---

## Usage

1. Place the `heart_disease.csv` file in the project directory.  
2. Run the preprocessing and training script:  

   ```bash
   python train_models.py
   ```

3. View the results and saved models in the project directory.

---

## Acknowledgments

This project was inspired by advancements in healthcare analytics and machine learning. Special thanks to the contributors who supported the development of this repository.

---

## Contact

- **Name**: Rod Will  
- **Email**: [your.email@example.com](mailto:rhudwill@gmail.com)  
- **GitHub**: [Your GitHub Profile](https://github.com/rod-will)  

---

## References

1. Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)  
2. Matplotlib Documentation: [https://matplotlib.org/](https://matplotlib.org/)  
3. Kaggle Heart Disease Dataset: [https://kaggle.com/](https://kaggle.com/)  

---

## License

This project is licensed under the CC0-1.0 License. For more details, see the [LICENSE](LICENSE) file.

---
