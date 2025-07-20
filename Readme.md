# **Machine Learning Pipeline with Streamlit Interface**

## **Overview**

This project, **CardioPredict AI**, provides a comprehensive machine learning pipeline for binary classification, specifically targeting early detection and monitoring of cardiovascular diseases (CVDs). It features an end-to-end workflow from data ingestion to model deployment, emphasizing reproducibility, modularity, and transparency. The system integrates five public datasets (1,871 rows), employs a Random Forest Classifier optimized for recall (94.75%), and includes SHAP-based interpretability. A Streamlit application enables real-time CVD risk prediction, with comparisons against Logistic Regression, SVM, XGBoost, KNN, and a Voting Classifier. Hosted at [https://cardiovascularprediction-jc78xmh2kjldht53ac2juf.streamlit.app/](https://cardiovascularprediction-jc78xmh2kjldht53ac2juf.streamlit.app/)

## **Features**

* **Automated Data Pipeline**: Loads and merges five datasets (Statlog, Heart\_Disease\_Prediction, etc.), ensuring consistency.  
* **Configurable Machine Learning Workflow**: Externalizes parameters via config.yaml for data processing, feature selection, and training.  
* **Feature Selection Module**: Uses Random Forest importance to select top 12 features (e.g., thalach, oldpeak).  
* **Model Training and Persistence**: Trains a RandomForestClassifier (n\_estimators=600, class\_weight='balanced') and saves it as random\_forest\_model.pkl.  
* **Comprehensive Model Evaluation**: Assesses accuracy, precision, recall (94.75%), F1-score, and ROC-AUC, recorded in metrics.json.  
* **Interactive Streamlit Application**: Offers a user-friendly interface with prediction input, SHAP visualization, and basic error handling.

## **Project Structure**

.  
├── README.md  
├── requirements.txt  
├── .gitignore  
├── .devcontainer/  
├── .streamlit/  
├── Cardiovascular.ipynb  
├── app.py  
├── packages.txt  
├── random\_forest\_model.pkl  
├── scaler.pkl  
├── data/  
│   └── raw/  
│       ├── Heart\_Disease\_Prediction (1).csv  
│       ├── Cardiovascular\_Disease\_Dataset.csv  
│       ├── hear\_LAPPt.csv  
│       └── heart nandal.csv  
├── reports/  
│   └── metrics.json  
└── config/  
    └── config.yaml

* README.md: This documentation.  
* requirements.txt: Python dependency list.  
* .gitignore: Specifies intentionally untracked files to ignore.  
* .devcontainer/: Configuration for development containers.  
* .streamlit/: Streamlit configuration files.  
* Cardiovascular.ipynb: Jupyter Notebook containing the main machine learning pipeline (data loading, preprocessing, model training, evaluation).  
* app.py: Streamlit application for interactive predictions.  
* packages.txt: Additional package dependencies (if any).  
* random\_forest\_model.pkl: The trained Random Forest model.  
* scaler.pkl: The trained MinMaxScaler for feature scaling.  
* data/raw/: Directory for the raw input datasets.  
* reports/: Directory for evaluation metrics (e.g., metrics.json).  
* config/: Directory for configuration files (e.g., config.yaml).

## **Installation**

### **Prerequisites**

* Python 3.8 or higher  
* pip (Python package installer)

### **Steps**

1. Clone the Repository:  
   git clone [https://github.com/CodeRishiX/Cardiovascularprediction.git](https://github.com/CodeRishiX/Cardiovascularprediction.git)  
   cd Cardiovascularprediction

2. Create a Virtual Environment:  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows: venv\\Scripts\\activate

3. Install Dependencies:  
   pip install \-r requirements.txt

   (Ensure requirements.txt contains: pandas numpy scikit-learn seaborn matplotlib joblib shap xgboost streamlit)

### **Table 1: Project Dependencies**

| Package Name | Version |
| :---- | :---- |
| pandas | 1.3.5 |
| scikit-learn | 1.0.2 |
| numpy | 1.21.6 |
| streamlit | 1.10.0 |
| pyyaml | 6.0 |
| joblib | 1.1.0 |
| shap | 0.41.0 |
| xgboost | 1.5.0 |

## **Dataset**

The project utilizes five datasets stored in data/raw/: Heart\_Disease\_Prediction (1).csv, Cardiovascular\_Disease\_Dataset.csv, hear\_LAPPt.csv, heart nandal.csv, and Statlog (fetched from UCI). These are merged into 1,871 rows after removing duplicates. Preprocessing (as performed in Cardiovascular.ipynb) includes:

* **Missing Value Handling**: Imputes numerical columns (age, trestbps, etc.) with means and categorical columns (cp, restecg) with modes, configurable in config.yaml.  
* **Feature Scaling/Normalization**: Applies MinMaxScaler to numerical features using scaler.pkl.  
* Categorical Encoding: Uses one-hot encoding for cp, restecg, slope.  
  The processed data is typically used within the notebook, and a processed version could be saved (e.g., data/processed/processed\_data.csv) if needed for direct loading.

## **Methodology**

### **Data Loading and Preprocessing**

* **Data Loading**: Handled within Cardiovascular.ipynb, which loads the five datasets and standardizes column names (e.g., BP to trestbps).  
* **Preprocessing**: Performed within Cardiovascular.ipynb, merging data, removing duplicates, imputing missing values, scaling numerical features, and one-hot encoding categorical variables, all parameterized via config.yaml.

### **Feature Engineering and Selection**

* **Feature Selection**: Within Cardiovascular.ipynb, a temporary RandomForestClassifier (n\_estimators=300, max\_depth=30) is used to select the top 12 features: \['thalach', 'oldpeak', 'ca', 'cp\_4.0', 'cp\_2.0', 'age', 'trestbps', 'chol', 'restecg\_2.0', 'slope\_1.0', 'slope\_3.0', 'slope\_2.0'\].

### **Table 3: Illustrative Feature Importance**

| Feature Name | Importance Score (Illustrative) |
| :---- | :---- |
| thalach | 0.25 |
| oldpeak | 0.20 |
| ca | 0.15 |
| cp\_4.0 | 0.10 |
| cp\_2.0 | 0.08 |
| ... | ... |

### **Model Training**

* **Model Choice**: Within Cardiovascular.ipynb, a RandomForestClassifier is trained with n\_estimators=600, max\_depth=18, min\_samples\_split=8, max\_features='log2', and class\_weight='balanced'.  
* **Hyperparameters**: Configured in config.yaml for reproducibility.  
* **Model Persistence**: The trained model is saved as random\_forest\_model.pkl and the scaler as scaler.pkl.

### **Model Evaluation**

* **Evaluation**: Performed within Cardiovascular.ipynb, computing accuracy, precision, recall, F1-score, and ROC-AUC. These metrics can be saved to reports/metrics.json.  
* **Configuration**: Threshold (0.44) for recall optimization is in config.yaml.

### **Table 2: Model Performance Metrics**

| Metric | Value (Illustrative) |
| :---- | :---- |
| Accuracy | 90.21% |
| Precision | 89.50% |
| Recall | 94.75% |
| F1-Score | 91.31% |

## **Usage**

### **Running the Machine Learning Pipeline (via Jupyter Notebook)**

1. Ensure dataset files are in data/raw/.  
2. Open and run the Cardiovascular.ipynb notebook in a Jupyter environment (e.g., Jupyter Lab, VS Code with Jupyter extension).  
3. The notebook will handle data loading, preprocessing, model training, and evaluation.  
4. Check outputs:  
   * Trained model: random\_forest\_model.pkl  
   * Trained scaler: scaler.pkl  
   * Metrics (if saved by the notebook): reports/metrics.json

### **Running the Streamlit Application**

1. Ensure random\_forest\_model.pkl and scaler.pkl exist in the project root.  
2. Launch the app from the project root:  
   streamlit run app.py

3. Interact with the app at http://localhost:8501:  
   * Input feature values (e.g., age, thalach).  
   * View predictions and SHAP values.

## **Results and Discussion**

The RandomForestClassifier achieved **90.21% accuracy**, **94.75% recall**, and **91.31% F1-score**, with thalach and oldpeak as top contributors per SHAP. The high recall meets medical standards (\<20% false negatives), though a 7.35% train-test gap suggests mild overfitting. Further analysis of the confusion matrix could refine insights.

## **Conclusions**

**CardioPredict AI** delivers a reproducible CVD prediction pipeline with a robust RandomForestClassifier and an interactive Streamlit app. Its modularity and transparency support further research, validated by multi-metric evaluation and SHAP interpretability.

## **Future Work**

* Explore XGBoost or neural networks for improved performance.  
* Augment data with CVD-specific features.  
* Deploy on a cloud platform (e.g., AWS).  
* Add model monitoring for drift detection.

## **Contributing**

* Report bugs or suggest features via GitHub Issues.  
* Submit pull requests with new branches, adhering to code style.

## **License**

MIT License (see LICENSE file).

## **Contact**

Open an issue on [https://github.com/CodeRishiX/Cardiovascularprediction](https://github.com/CodeRishiX/Cardiovascularprediction).