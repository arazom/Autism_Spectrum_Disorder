# ASD Severity Classification using Machine Learning

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue)]()

> **Behavioral Analysis and Severity Classification of Children with Autism Spectrum Disorder using Unsupervised Clustering and Classification Models on Multidimensional Behavioral Questionnaire Data**

This repository contains the complete implementation of machine learning models for classifying Autism Spectrum Disorder (ASD) severity into three levels: **Light (Mild)**, **Moderate**, and **Severe**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition characterized by challenges in social interaction, communication, and repetitive behaviors. Accurate severity assessment is crucial for appropriate intervention planning and resource allocation.

This research addresses the following objectives:

1. **Data Collection:** Assembled a comprehensive dataset of 340 children from three autism rehabilitation centers in Kirkuk, Iraq
2. **Advanced ML Methods:** Developed modern machine learning approaches for rapid and effective ASD severity classification
3. **Reduced Diagnosis Time:** Minimized required features from 26 to 20 (23% reduction) while maintaining acceptable accuracy
4. **Improved Performance:** Achieved 64.71% accuracy with 82.24% specificity using SVM
5. **Differential Diagnosis:** Developed effective methods to distinguish ASD from similar disorders

---

## ✨ Key Features

- **Unsupervised Clustering:** K-Means, Gaussian Mixture Models (GMM), and Hierarchical Clustering
- **Supervised Classification:** 11 machine learning models including SVM, Random Forest, Gradient Boosting, and Neural Networks
- **Feature Selection:** Multiple methods (Chi-Square, Mutual Information, RFE, LASSO) to identify the most discriminative behavioral markers
- **Comprehensive Evaluation:** ROC curves, Precision-Recall curves, confusion matrices, and detailed performance metrics
- **Reproducible Research:** Jupyter notebooks for each analysis stage
- **Well-Documented Code:** Clear documentation and inline comments

---

## 📊 Dataset

### **Dataset Information**

- **Source:** Three autism rehabilitation centers in Kirkuk, Iraq
- **Sample Size:** 340 children (ages 2-16 years)
- **Features:** 34 attributes (8 demographic + 26 behavioral)
- **Target Variable:** 3 severity levels (Light, Moderate, Severe)
- **Missing Values:** None
- **Class Distribution:** Balanced (30.6%, 35.0%, 34.4%)

### **Behavioral Domains**

The 26 behavioral features are organized into 5 domains:

1. **Social Interaction** (6 features): Eye contact, name response, joint attention, empathy, etc.
2. **Communication** (6 features): Speech clarity, pointing, gestures, echolalia, etc.
3. **Repetitive Behaviors** (8 features): Object lining, finger movements, blank staring, etc.
4. **Sensory Sensitivity** (2 features): Sensory seeking, noise sensitivity
5. **Language Development** (3 features): Vocabulary size, first words, adaptability

### **Data Availability**

- **Full Dataset:** Available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/) under the name "ASD_LMS"
- **Sample Data:** A small sample (20 instances) is provided in `data/sample/` for testing

**⚠️ Note:** The full dataset is not included in this repository to protect participant privacy. Please download it from UCI or contact the authors.

---

## 🔬 Methodology

### **1. Data Preprocessing**

- Data cleaning and validation
- Handling categorical variables
- Feature scaling (StandardScaler)
- Train-test split (80-20)

### **2. Unsupervised Clustering**

Three clustering algorithms were applied to discover natural groupings:

| Algorithm | ARI | Silhouette Score |
|-----------|-----|------------------|
| **K-Means** | **0.133** | 0.068 |
| GMM | 0.121 | - |
| Hierarchical | 0.102 | - |

### **3. Supervised Classification**

11 machine learning models were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM (Linear)** | **64.71%** | **65.25%** | **65.25%** | **65.15%** |
| AdaBoost | 61.76% | 62.13% | 61.76% | 61.94% |
| Logistic Regression | 61.76% | 62.80% | 61.76% | 62.47% |
| Random Forest | 58.82% | 59.74% | 58.82% | 59.27% |
| Gradient Boosting | 57.35% | 58.88% | 57.35% | 58.10% |

### **4. Feature Selection**

Multiple feature selection methods were applied:

- **Chi-Square Test**
- **Mutual Information**
- **Recursive Feature Elimination (RFE)**
- **LASSO (L1 Regularization)**

**Result:** Reduced features from 26 to 20 (23% reduction) with only 7% accuracy drop (64.71% → 57.35%)

### **5. Model Evaluation**

- **Sensitivity (Recall):** 65.25% average
- **Specificity:** 82.24% average ⭐
- **ROC-AUC:** 0.883 (Mild), 0.726 (Moderate), 0.933 (Severe)
- **Cross-Validation:** 5-fold CV for robustness

---

## 🚀 Installation

### **Prerequisites**

- Python 3.8 or higher
- pip or conda package manager

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/ASD-Severity-Classification.git
cd ASD-Severity-Classification
```

### **Step 2: Create Virtual Environment (Recommended)**

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n asd-classification python=3.8
conda activate asd-classification
```

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### **Quick Start**

```python
# Load the data
import pandas as pd
data = pd.read_csv('data/sample/sample_data.csv')

# Preprocess
from src.data_preprocessing import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train SVM model
from src.classification import train_svm
model = train_svm(X_train, y_train)

# Evaluate
from src.evaluation import evaluate_model
metrics = evaluate_model(model, X_test, y_test)
print(metrics)
```

### **Using Jupyter Notebooks**

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_preprocessing.ipynb
# 3. notebooks/03_clustering.ipynb
# 4. notebooks/04_classification.ipynb
# 5. notebooks/05_feature_selection.ipynb
# 6. notebooks/06_evaluation.ipynb
```

### **Running Scripts**

```bash
# Train all models
python scripts/train_models.py

# Evaluate models
python scripts/evaluate_models.py

# Generate report
python scripts/generate_report.py

# Run complete pipeline
bash scripts/run_all.sh
```

---

## 📈 Results

### **Best Model: SVM (Linear)**

- **Overall Accuracy:** 64.71%
- **Average Specificity:** 82.24% (excellent for avoiding false positives)
- **Average Sensitivity:** 65.25%

### **Performance by Severity Level**

| Level | Sensitivity | Specificity | F1-Score |
|-------|-------------|-------------|----------|
| **Mild** | 76.19% | 85.11% | 0.70 |
| **Moderate** | 50.00% | 72.73% | 0.58 |
| **Severe** | 69.57% | 88.89% | 0.68 |

### **Feature Importance**

Top 10 most discriminative features:

1. Sensory seeking (r = 0.398)
2. Blank staring (r = 0.373)
3. Finger movements (r = 0.365)
4. Repetitive manipulation (r = 0.359)
5. Vocabulary size (r = 0.344)
6. Eye contact (r = 0.333)
7. Repetitive behaviors (r = 0.307)
8. Noise sensitivity (r = 0.228)
9. Joint attention (r = 0.289)
10. Speech clarity (r = 0.180)

### **Visualizations**

All figures are available in `results/figures/`:

- Confusion matrices for top 3 models
- ROC curves (3 models × 3 classes)
- Precision-Recall curves
- Feature importance plots
- Clustering visualizations
- Performance comparison charts

---

## 📁 Project Structure

```
ASD-Severity-Classification/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Data directory
│   ├── README.md                      # Data description
│   ├── sample/                        # Sample data for testing
│   │   └── sample_data.csv
│   └── [raw/ and processed/ - not included]
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_classification.ipynb
│   ├── 05_feature_selection.ipynb
│   └── 06_evaluation.ipynb
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── clustering.py
│   ├── classification.py
│   ├── feature_selection.py
│   ├── evaluation.py
│   └── visualization.py
│
├── models/                            # Trained models
│   ├── README.md
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   └── model_metadata.json
│
├── results/                           # Results and outputs
│   ├── figures/                       # Visualizations
│   ├── tables/                        # Result tables
│   └── logs/                          # Training logs
│
├── scripts/                           # Executable scripts
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── generate_report.py
│   └── run_all.sh
│
├── tests/                             # Unit tests
│   ├── test_preprocessing.py
│   ├── test_clustering.py
│   └── test_classification.py
│
├── docs/                              # Documentation
│   ├── installation.md
│   ├── usage.md
│   └── methodology.md
│
└── paper/                             # Research paper
    ├── paper.pdf
    └── citation.bib
```

---

## 📦 Requirements

### **Core Libraries**

- numpy >= 1.24.3
- pandas >= 2.0.3
- scikit-learn >= 1.3.0
- scipy >= 1.11.1

### **Visualization**

- matplotlib >= 3.7.2
- seaborn >= 0.12.2
- plotly >= 5.15.0

### **Optional**

- imbalanced-learn >= 0.11.0
- xgboost >= 1.7.6
- jupyter >= 1.0.0

See `requirements.txt` for complete list.

---

## 📄 Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{yourname2025asd,
  title={Behavioral Analysis and Severity Classification of Children with Autism Spectrum Disorder using Unsupervised Clustering and Classification Models},
  author={[Your Name] and [Co-authors]},
  journal={[Journal Name]},
  year={2025},
  volume={XX},
  number={XX},
  pages={XX--XX},
  doi={10.xxxx/xxxxx}
}
```

**Paper:** [Link to published paper]  
**Dataset:** [Link to UCI repository]

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [arazom]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 👥 Contact

**Principal Investigator:**
- Name: arazo muhamed
- Email: arazo.2007@yahoo.com
- Institution: kirkuk unvercity
- ORCID: https://orcid.org/0000-0001-7987-1295


**For questions about:**
- **Code:** Open an issue on GitHub
- **Dataset:** Contact via email or UCI repository
- **Collaboration:** Email the principal investigator

---

## 🙏 Acknowledgments

We would like to thank:

- The three autism rehabilitation centers in Kirkuk, Iraq, for their collaboration
- All families who participated in this study
- The trained professionals who supervised data collection
- [Funding agency, if applicable]

---

## 🔗 Related Resources

- **UCI Dataset:** [Link to UCI repository]
- **Research Paper:** [Link to paper]
- **Supplementary Materials:** [Link to supplementary]
- **Project Website:** [If available]

---

## 📊 Statistics

![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/ASD-Severity-Classification?style=social)
![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/ASD-Severity-Classification?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/YOUR_USERNAME/ASD-Severity-Classification?style=social)

---

## 🗓️ Version History

- **v1.0.0** (2025-01-01): Initial release
  - Complete implementation of all models
  - Jupyter notebooks for reproducibility
  - Comprehensive documentation

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 🐛 Known Issues

- None currently. Please report any issues on the GitHub Issues page.

---

## 🔮 Future Work

- [ ] Deep learning models (CNN, LSTM)
- [ ] Explainable AI (SHAP, LIME)
- [ ] Web application for clinical use
- [ ] Multi-language support
- [ ] Longitudinal analysis

---

## 📚 References

1. American Psychiatric Association. (2013). *Diagnostic and statistical manual of mental disorders (5th ed.)*.
2. Lord, C., et al. (2018). Autism spectrum disorder. *The Lancet*, 392(10146), 508-520.
3. [Add other relevant references]

---

**Made with ❤️ for the autism research community**

---

*Last updated: December 2025*

