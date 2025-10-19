#  Parkinson’s Disease Severity Prediction

This project explores multiple machine learning and deep learning models to predict **Parkinson’s disease severity** based on **5,875 voice recording features from 42 patients**.  
The goal was to compare **traditional ML approaches (Scikit-learn)** with **deep learning models (TensorFlow)** to identify which performs best in predicting Parkinson’s severity (classification task).

---

##  Project Objectives

- Compare **traditional ML algorithms** and **deep learning architectures** for Parkinson’s prediction.  
- Implement **a complete ML pipeline**: preprocessing, feature scaling, model training, evaluation, and comparison.  
- Use both **Scikit-learn** and **TensorFlow (Sequential & Functional API with tf.data)**.  
- Conduct **7 experiments** and analyze model performance and dataset limitations.

---

##  Models Tested

### **Traditional Machine Learning Models (Scikit-learn)**
1. Logistic Regression *(Baseline)*
2. Support Vector Machine (SVM)
3. Balanced Random Forest
4. XGBoost (with Class Weights)
5. Stacking Classifier

### **Deep Learning Models (TensorFlow)**
6. Sequential Neural Network  
7. Functional API Model (with `tf.data`)

---

##  Final Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Training Time (s) |
|-------|-----------|------------|---------|-----------|----------|-------------------|
| **Stacking Classifier** | **0.9029** | **0.9041** | **0.9029** | **0.9034** | **0.9720** | 27.63 |
| **XGBoost (Weighted)** | 0.8996 | 0.9001 | 0.8996 | 0.8998 | 0.9750 | 0.7 |
| **Balanced Random Forest** | 0.8826 | 0.8827 | 0.8826 | 0.8826 | 0.9538 | 2.16 |
| **Functional API (tf.data)** | 0.8536 | 0.8591 | 0.8536 | 0.8551 | 0.9353 | 34.51 |
| **Sequential Neural Network** | 0.8434 | 0.8537 | 0.8434 | 0.8457 | 0.9203 | 30.94 |
| **Support Vector Machine (SVM)** | 0.7311 | 0.7733 | 0.7311 | 0.7370 | 0.8282 | 11.8|
| **Logistic Regression (Baseline)** | 0.5515 | 0.5975 | 0.5515 | 0.5613 | 0.6086 | 0.08 |

---

##  Analysis & Insights

###  **Top Performer: Stacking Classifier**
- Achieved the highest **accuracy (90.3%)** and **F1-score (0.903)**.
- Combines strengths of multiple base learners, improving robustness.

###  **Traditional ML Summary**
- **XGBoost** followed closely, showcasing powerful gradient boosting capabilities.  
- **Balanced Random Forest** effectively managed class imbalance.  
- **SVM** and **Logistic Regression** provided useful baselines but struggled with dataset complexity.

###  **Deep Learning Summary**
- Both **Sequential** and **Functional API** models performed competitively.
- **Functional API with `tf.data`** slightly outperformed the Sequential model, thanks to efficient data pipelines.
- Longer training times indicate a trade-off between computational cost and performance.

---

##  Key Takeaways

- **Traditional ensemble models** (Stacking, XGBoost) outperform deep learning on this dataset.  
- **Deep learning models** are still valuable for capturing complex, non-linear relationships.
- The **Functional API** model shows potential if trained longer or with more data.
- The **Logistic Regression** baseline highlights the challenge and non-linearity of the dataset.

---

##  Experiments Conducted

| Experiment | Category | Model / Technique | Framework |
|-------------|------------|-------------------|------------|
| 1 | Traditional | Logistic Regression | Scikit-learn |
| 2 | Traditional | Support Vector Machine | Scikit-learn |
| 3 | Traditional | Balanced Random Forest | Scikit-learn |
| 4 | Traditional | XGBoost (Weighted) | XGBoost |
| 5 | Traditional | Stacking Classifier | Scikit-learn |
| 6 | Deep Learning | Sequential Neural Network | TensorFlow (Sequential API) |
| 7 | Deep Learning | Functional API Model (tf.data) | TensorFlow (Functional API) |

---

##  Conclusion

This study demonstrates that **ensemble machine learning approaches** outperform deep learning models for this particular Parkinson’s dataset.  
While neural networks performed respectably, the **Stacking Classifier and XGBoost** offered superior generalization and efficiency.  
With further tuning, data augmentation, or more diverse input features, deep learning could potentially surpass ensemble methods.

---

##  Installation & Quick Start

```bash
# Clone repository
git clone https://github.com/denismitali17/parkinsons-voice-analyzer
cd parkinsons-voice-analyzer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Open and Run Jupyter Notebook
jupyter notebook Denis_Mitali_Parkinson_Voice_Analysis.ipynb

```

---

### Author: Denis Mitali
