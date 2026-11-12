# Social Media Tourism Intelligence
### Customer Segmentation & Purchase Propensity Prediction Using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Data%20Science-2563EB?style=flat-square" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Binary%20Classification-7C3AED?style=flat-square" />
  <img src="https://img.shields.io/badge/Analytics-Customer%20Segmentation-0891B2?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Exploratory%20Project-orange?style=flat-square" />
</p>

## Project Overview

How can a tourism company identify potential customers without relying entirely on traditional telemarketing?

This project explores how social-media engagement, travel preferences, customer characteristics, and online browsing behaviour can be used to understand purchasing patterns and predict customer purchase propensity.

The objective is to help a tourism and aviation company transition from broad customer outreach towards a more targeted, data-driven digital marketing strategy.

Using a dataset of **11,760 customer records**, the project combines exploratory data analysis, customer segmentation, and supervised machine learning to investigate the factors associated with purchasing a travel-related product.

The analytical workflow includes data preprocessing, behavioural analysis, K-Means clustering, device-based customer segmentation, classification model development, hyperparameter tuning, model evaluation, and feature-importance analysis.

Two customer segments are investigated independently:

- **Laptop users:** 1,108 records.
- **Mobile users:** 10,652 records.

Multiple machine learning algorithms are compared to identify promising approaches for predicting potential purchasers within each segment.

---

## Business Problem

An aviation company offering domestic and international travel products traditionally relies on telecalling to reach prospective customers.

However, contacting customers individually can be time-consuming and may not be the most efficient way to allocate marketing resources.

The company wants to explore a more targeted digital advertising strategy by leveraging customer information available through social-media platforms.

The central business problem is:

**Can customer characteristics and digital engagement behaviour be used to identify individuals who are more likely to purchase a travel-related product?**

Addressing this problem involves understanding customer behaviour, identifying meaningful customer segments, and developing predictive models that can support more focused marketing decisions.

### Key Business Questions

- Which customer characteristics and engagement patterns are associated with product purchases?
- Does travel-related social-media activity provide useful signals for identifying potential buyers?
- How do purchasing patterns differ between laptop and mobile users?
- Can customers be grouped into meaningful behavioural segments?
- Which classification algorithms are most effective at identifying actual purchasers?
- How can predictive insights support targeted digital advertising and customer acquisition?

---

## Project Objectives

1. Perform exploratory data analysis to understand customer characteristics and purchasing behaviour.
2. Clean and preprocess the dataset by addressing missing values, inconsistent categories, and outliers.
3. Examine relationships between customer attributes and purchase outcomes.
4. Apply K-Means clustering to explore behavioural customer segments.
5. Develop separate classification models for laptop and mobile users.
6. Compare multiple machine learning algorithms using appropriate evaluation metrics.
7. Apply hyperparameter tuning to selected models.
8. Analyse XGBoost feature importance to investigate influential predictive attributes.
9. Translate analytical findings into potential customer-targeting and digital marketing strategies.

---

## Dataset Description

The project uses the **Social Media Data for DSBA** dataset.

| Dataset characteristic | Description |
|---|---|
| Total observations | 11,760 |
| Original attributes | 17 |
| Total cells | 199,920 |
| Duplicate records | 0, as reported in the EDA notebook |
| Target variable | `Taken_product` |
| Problem type | Binary classification |
| Positive class | Customers who purchased the product |

### Target Variable

The target variable is `Taken_product`.

| Value | Meaning |
|---|---|
| 1 / Yes | Customer purchased the product |
| 0 / No | Customer did not purchase the product |

### Class Distribution

The target variable is imbalanced:

| Customer category | Records | Percentage |
|---|---:|---:|
| Purchased | 1,896 | 16% |
| Did not purchase | 9,864 | 84% |
| **Total** | **11,760** | **100%** |

This class imbalance makes recall, precision, and other classification metrics particularly important. A model should not be considered effective simply because it achieves high overall accuracy.

### Feature Description

The original dataset contains the following attributes:

| Feature | Description |
|---|---|
| `UserID` | Unique customer identifier |
| `Taken_product` | Whether the customer purchased the product |
| `Yearly_avg_view_on_travel_page` | Average yearly views on travel pages |
| `preferred_device` | Customer's preferred device |
| `total_likes_on_outstation_checkin_given` | Likes given on outstation check-ins |
| `yearly_avg_Outstation_checkins` | Average yearly outstation check-ins |
| `member_in_family` | Number of family members |
| `preferred_location_type` | Preferred travel or location category |
| `Yearly_avg_comment_on_travel_page` | Average yearly comments on travel pages |
| `total_likes_on_outofstation_checkin_received` | Likes received on outstation check-ins |
| `week_since_last_outstation_checkin` | Weeks since the last outstation check-in |
| `following_company_page` | Whether the customer follows the company page |
| `montly_avg_comment_on_company_page` | Average monthly comments on the company page |
| `working_flag` | Employment-related indicator |
| `travelling_network_rating` | Travel-related network rating |
| `Adult_flag` | Adult indicator |
| `Daily_Avg_mins_spend_on_traveling_page` | Average daily time spent on travel pages |

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Data manipulation | Pandas, NumPy |
| Data visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Clustering | K-Means |
| Classification | Logistic Regression, Decision Tree, KNN, Random Forest |
| Gradient boosting | XGBoost |
| Deep learning | TensorFlow / Keras |
| Model optimisation | GridSearchCV |
| Development environment | Jupyter Notebook, Google Colab |

---

## Project Workflow

The project is organised into two primary analytical stages.

### 1. Exploratory Data Analysis & Customer Segmentation

Notebook: `social-media-tourism-eda.ipynb`

This stage focuses on understanding the dataset, identifying behavioural patterns, and exploring customer segments.

**Data preparation**

- Dataset inspection and structural analysis.
- Identification of missing values and duplicate records.
- Standardisation of selected categorical attributes.
- Handling of inconsistent and non-numeric values.
- Numerical imputation using median values.
- Categorical imputation using the most frequent values.
- IQR-based outlier treatment.
- Preparation of the dataset for further analysis.

**Exploratory analysis**

- Univariate analysis of numerical and categorical attributes.
- Distribution and spread analysis using histograms and box plots.
- Bivariate analysis of customer attributes and purchase outcomes.
- Correlation analysis.
- Comparison of customer engagement and travel-related behaviour.

**Customer segmentation**

K-Means clustering is used to explore behavioural similarities between customers.

The notebook investigates cluster counts using:

- Elbow method and within-cluster sum of squared errors.
- Silhouette analysis.
- Cluster profiling based on customer characteristics and purchasing behaviour.

The notebook proceeds with a four-cluster solution to investigate customer segments.

### 2. Machine Learning & Model Development

Notebook: `social_media_tourism_modeling_final.ipynb`

The second stage focuses on predicting customer purchase behaviour.

**Device-based segmentation**

The preprocessed dataset is divided into:

| Segment | Records |
|---|---:|
| Laptop users | 1,108 |
| Mobile users | 10,652 |
| **Total** | **11,760** |

The preferred-device feature is removed from the individual modelling datasets after segmentation.

**Train-test split**

Each segment is divided into training and testing subsets using:

- Training set: 70%
- Testing set: 30%
- Random state: 123
- Stratified sampling based on `Taken_product`

| Segment | Training records | Testing records |
|---|---:|---:|
| Laptop | 775 | 333 |
| Mobile | 7,456 | 3,196 |

**Classification algorithms**

The following algorithms are investigated:

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Linear classification baseline |
| Decision Tree (CART) | Rule-based non-linear classification |
| K-Nearest Neighbours | Distance-based classification |
| Random Forest | Tree-based ensemble learning |
| XGBoost | Gradient-boosted decision trees |
| Artificial Neural Network | Deep learning-based classification |

**Hyperparameter tuning**

GridSearchCV is applied to selected classification algorithms to investigate alternative hyperparameter configurations.

The project compares basic and tuned model performance to understand the effect of model optimisation.

XGBoost is also evaluated as a gradient-boosting approach, while ANN models provide an additional deep-learning benchmark.

**Model evaluation**

The models are evaluated using:

- Accuracy
- Recall
- ROC-AUC
- Classification reports
- Confusion matrices
- ROC curves

Particular emphasis is placed on recall because identifying actual purchasers is important for the customer acquisition objective.

The notebook also uses a probability threshold of 0.29 in several prediction workflows to classify customers as purchasers or non-purchasers.

---

## Experimental Results

The following results are taken from the saved model-comparison outputs in the modelling notebook. Values are rounded to two decimal places.

They represent the current notebook experiments and should be interpreted as preliminary results pending the validation improvements described later in this README.

### Laptop Users — Basic Models

| Model | Test Accuracy | Test Recall | Test ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 74% | 66% | 0.79 |
| CART | 93% | 86% | 0.91 |
| KNN | 91% | 72% | 0.95 |
| Random Forest | 95% | 81% | 1.00 |
| Basic ANN | 97% | 90% | 0.99 |

### Laptop Users — Tuned Models

| Model | Test Accuracy | Test Recall | Test ROC-AUC |
|---|---:|---:|---:|
| Tuned Logistic Regression | 80% | 67% | 0.85 |
| Tuned CART | 80% | 86% | 0.89 |
| Tuned KNN | 96% | 92% | 0.99 |
| Tuned Random Forest | 97% | 95% | 0.99 |
| XGBoost | 97% | 95% | 1.00 |

### Mobile Users — Basic Models

| Model | Test Accuracy | Test Recall | Test ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 84% | 18% | 0.71 |
| CART | 97% | 89% | 0.94 |
| KNN | 96% | 84% | 0.98 |
| Random Forest | 98% | 86% | 1.00 |
| Basic ANN | 98% | 94% | 0.99 |

### Mobile Users — Tuned Models

| Model | Test Accuracy | Test Recall | Test ROC-AUC |
|---|---:|---:|---:|
| Tuned Logistic Regression | 83% | 40% | 0.79 |
| Tuned CART | 98% | 91% | 0.95 |
| Tuned KNN | 99% | 96% | 1.00* |
| Tuned Random Forest | 98% | 97% | 1.00 |
| XGBoost | 99% | 97% | 1.00 |

\*The mobile KNN ROC-AUC value in the notebook's consolidated comparison table uses the training ROC-AUC variable instead of the test ROC-AUC variable. It must be corrected before the test result is treated as verified.

### Key Results

**Laptop segment**

- Tuned Random Forest and XGBoost achieve the highest reported test accuracy of 97%.
- Both models achieve approximately 95% purchaser recall.
- Tuned KNN also performs strongly, with 96% test accuracy and 92% recall.

**Mobile segment**

- XGBoost achieves the highest reported test accuracy of 99%.
- XGBoost and tuned Random Forest both achieve approximately 97% purchaser recall.
- Tuned KNN reports 99% test accuracy and 96% recall, although its consolidated test ROC-AUC requires correction.

### Model Interpretation

The experimental results indicate that ensemble-based methods, particularly Random Forest and XGBoost, are promising approaches for identifying potential purchasers in both device segments.

Logistic Regression performs comparatively poorly in terms of purchaser recall, particularly for mobile users. This suggests that a simple linear model may not capture all the relationships present in the customer attributes.

The results also demonstrate why accuracy should not be considered in isolation when dealing with an imbalanced target variable.

---

## Feature Importance Analysis

The modelling notebook includes feature-importance visualisations generated from the trained XGBoost models for both laptop and mobile customer segments.

The analysis uses the models' `feature_importances_` attribute to rank the input variables according to their contribution to the fitted models.

This provides an initial understanding of which customer attributes the models rely on when predicting purchase outcomes.

Feature importance is useful for exploratory model interpretation, but it does not establish causal relationships. Further explainability techniques, such as SHAP and permutation importance, would provide a more detailed interpretation of individual predictions and feature contributions.

---

## Key Observations

The experiments provide several useful observations:

1. **Class imbalance is an important modelling consideration.** Approximately 84% of customers did not purchase the product, making recall an important metric for evaluating the identification of actual purchasers.

2. **Predictive performance differs across algorithms.** Tree-based and ensemble models demonstrate stronger reported performance than the basic Logistic Regression models.

3. **Device-based modelling provides a useful comparison.** Separate models allow the company to investigate potential differences between laptop and mobile customer segments.

4. **Ensemble learning performs strongly in both segments.** Random Forest and XGBoost achieve high reported accuracy and purchaser recall.

5. **Hyperparameter tuning can improve predictive performance.** The extent of improvement varies by algorithm and evaluation metric.

6. **Model interpretation can support marketing analysis.** XGBoost feature importance provides an initial way to investigate behavioural variables associated with model predictions.

---

## Business Recommendations

The findings suggest several potential applications for a data-driven customer acquisition strategy.

### 1. Predictive Customer Targeting

Use validated purchase-propensity estimates to prioritise customers who are more likely to purchase the travel product.

### 2. Device-Aware Marketing

Explore separate marketing strategies for laptop and mobile audiences, including device-optimised advertisements and booking experiences.

### 3. Engagement-Based Segmentation

Use travel-page views, comments, likes, and outstation check-ins to explore customer groups with different levels of travel interest and digital engagement.

### 4. Retargeting Opportunities

Investigate customers who repeatedly explore travel content but have not purchased. Test relevant destination promotions, offers, and personalised travel recommendations.

### 5. Threshold Optimisation

Choose prediction thresholds based on the balance between missed potential purchasers and the cost of targeting non-purchasers.

### 6. Campaign Measurement

Evaluate the business impact of predictive targeting using controlled experiments and metrics such as conversion rate, customer acquisition cost, incremental purchases, and return on advertising spend.

These recommendations are proposed applications of the analytical findings; the notebooks do not report a completed advertising campaign or measured marketing uplift.

---

## Limitations & Validation Considerations

The project provides a useful foundation for customer analytics, but several methodological considerations should be addressed before the predictive results are used in a real-world setting.

### Data preprocessing

Some preprocessing operations, including imputation and outlier treatment, are performed before the train-test split. A more robust approach would fit preprocessing transformations exclusively on training data using a scikit-learn Pipeline and ColumnTransformer.

### Feature scaling

The KNN section fits a separate scaler on the test data using `fit_transform()`. The test set should instead be transformed using the scaler fitted on the corresponding training data.

### Device categorisation

The EDA notebook groups every preferred-device category other than the exact `Laptop` label into the `Mobile` category. The original categories should be verified before treating the two segments as strictly laptop and mobile users.

### Model validation

The reported model performance is based on a single stratified train-test split. Repeated stratified cross-validation and an independently held-out test dataset would provide stronger evidence of generalisation.

### Evaluation metrics

The current consolidated model comparisons focus on accuracy, recall, and ROC-AUC. Precision, F1-score, PR-AUC, specificity, and probability calibration should also be evaluated.

### Classification threshold

A threshold of 0.29 is used in several prediction workflows. Threshold selection should be justified using validation data and the intended marketing objective.

### Reported performance

Some models produce exceptionally high reported test scores. These should be independently reproduced after correcting the preprocessing and scaling issues before being considered reliable estimates of real-world performance.

---

## Future Scope

The project can be extended in several directions:

- **Explainable AI:** Apply SHAP to understand global feature contributions and individual customer predictions.
- **Improved class-imbalance handling:** Investigate class weighting and resampling approaches within cross-validation.
- **Robust model validation:** Introduce repeated stratified cross-validation and an independent holdout dataset.
- **Probability calibration:** Evaluate the reliability of predicted purchase probabilities.
- **Advanced clustering:** Compare K-Means with approaches designed for mixed numerical and categorical data.
- **Model deployment:** Develop an interactive prediction application or API.
- **Marketing intelligence dashboard:** Visualise customer segments, purchase propensity, and campaign performance.
- **Campaign experimentation:** Measure incremental conversion and return on advertising spend through controlled marketing experiments.

---

## Repository Structure

```text
Social-Media-Tourism/
│
├── social-media-tourism-eda.ipynb
│   └── Exploratory Data Analysis,
│       preprocessing and customer segmentation
│
├── social_media_tourism_modeling_final.ipynb
│   └── Classification, model tuning,
│       evaluation and feature importance
│
├── Social Media Data for DSBA.csv
│   └── Original dataset (required)
│
├── PreProcessed-SocialMediaData.csv
│   └── Generated by the EDA notebook
│
├── README.md
│
└── requirements.txt
```

The CSV files are required for execution but are not assumed to be included in this repository. Their availability and redistribution permissions should be confirmed before publishing them.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Social-Media-Tourism.git

cd Social-Media-Tourism
```

Replace `YOUR-USERNAME` with the appropriate GitHub username and update the repository name if necessary.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the datasets

Place the original dataset in the project directory:

```text
Social Media Data for DSBA.csv
```

Run the EDA notebook first:

```text
social-media-tourism-eda.ipynb
```

This notebook generates the preprocessed dataset:

```text
PreProcessed-SocialMediaData.csv
```

The modelling notebook expects both the original and preprocessed datasets. Its current CSV-loading cells use Google Colab / Google Drive paths, so update those paths to your local project directory or mount the appropriate Google Drive location before execution.

### 5. Run the modelling notebook

Open:

```text
social_media_tourism_modeling_final.ipynb
```

Execute the cells in sequence to reproduce the device-based model development, evaluation, and feature-importance analysis.

---

## Requirements

The notebooks use the following Python libraries:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
tensorflow
jupyter
```

A suitable Python environment with compatible versions of these packages is required. TensorFlow installation may depend on the operating system and Python version.

---

## Acknowledgements & Attribution

The notebook files identify **Sudhanshu Singh** as their author and include the original GitHub profile:

[Sudhanshu Singh – GitHub](https://github.com/sudhanshusinghaiml)

The notebooks should retain appropriate original attribution when shared or adapted. Dataset ownership, source, and redistribution permissions should also be verified before public distribution.

---

## Project Takeaway

This project demonstrates an end-to-end application of exploratory analytics, customer segmentation, and supervised machine learning to a tourism customer acquisition problem.

By investigating social-media engagement and travel-related behaviour, the analysis explores how customer data can support purchase propensity prediction and more focused digital marketing strategies.

The strongest reported experimental results come from Random Forest and XGBoost, particularly within the device-specific modelling workflow. Further leakage-free validation, model interpretation, and business experimentation are necessary to establish their reliability and practical value.

**The objective is not just to predict who might purchase, but to understand customer behaviour well enough to make marketing decisions more informed, measurable, and effective.**
