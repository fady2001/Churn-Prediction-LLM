# Comprehensive Technical Report: Churn Prediction LLM System

## Executive Summary

This report presents a comprehensive analysis of an innovative customer churn prediction system that combines Large Language Models (LLMs) with traditional machine learning approaches. The system enables natural language input processing for customer data extraction and provides accurate churn predictions with interpretable results.

## 1. Project Overview

### 1.1 Objective

The primary objective is to develop an intelligent churn prediction system that:

- Accepts natural language descriptions of customers
- Extracts structured features using LLM technology
- Predicts customer churn probability using machine learning
- Provides actionable insights for marketing teams

### 1.2 Key Innovation

The system bridges the gap between unstructured customer descriptions and structured machine learning models by leveraging LLM capabilities for feature extraction, making churn prediction accessible through natural language interfaces.

## 2. System Architecture

### 2.1 High-Level Architecture

The system follows a modular, pipeline-based architecture with the following main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Layer   │───▶│ Processing Layer│───▶│  Output Layer   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
   ┌─────────┐              ┌─────────┐              ┌─────────┐
   │Natural  │              │Feature  │              │Churn    │
   │Language │              │Extraction│             │Prediction│
   │Input    │              │& ML Model│             │Results  │
   └─────────┘              └─────────┘              └─────────┘
```

### 2.2 Component Architecture

#### 2.2.1 Data Layer

- **Raw Data**: Telco customer churn dataset with 21 features
- **Interim Data**: Feature-engineered datasets
- **Processed Data**: Preprocessed data ready for modeling

#### 2.2.2 Feature Engineering Layer

- **ChurnFeatureEngineer**: Custom transformer for domain-specific feature creation
- **Preprocessor**: Data preprocessing pipeline with encoding and scaling

#### 2.2.3 LLM Integration Layer

- **CustomerInfoExtractor**: Natural language processing using Ollama LLM
- **CustomerData**: Pydantic models for structured data validation

#### 2.2.4 Machine Learning Layer

- **ChurnPredictor**: Wrapper class for prediction pipeline
- **TargetEncoder**: Custom target variable encoder
- **RandomForestClassifier**: Core ML algorithm

#### 2.2.5 API Layer

- **Prediction Interface**: Multiple input methods (text, structured data, batch)
- **Result Formatting**: Standardized output with confidence scores

## 3. Methodology

### 3.1 Data Science Methodology

The project follows the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology:

1. **Business Understanding**

   - Problem: Customer churn prediction for retention strategies
   - Success Criteria: High recall (identify churners) with acceptable precision

2. **Data Understanding**

   - Dataset: 7,043 customer records with 21 features
   - Target Variable: Binary churn indicator (Yes/No)
   - Data Quality: Clean dataset with minimal missing values

3. **Data Preparation**

   - Feature Engineering: 15+ new derived features
   - Data Preprocessing: Encoding, scaling, and transformation
   - Train/Validation/Test split for robust evaluation

4. **Modeling**

   - Algorithm Selection: Random Forest Classifier
   - Hyperparameter Tuning: Optimized for business metrics
   - Pipeline Integration: End-to-end automation

5. **Evaluation**

   - Multiple Metrics: Accuracy, F1-score, Precision, Recall, ROC-AUC
   - Business Focus: F-beta score (β=0.5) emphasizing precision because the marketing team will use it so we don't want to spend money on customer who are identified as churners by mistake (we need high precision) and also we do not want to ignore recall because there may be a valuable customers who will not identified and the company lose them (we need high recall) so to sum up, the best metric will be f-beta score that emphasize precision over recall
   - Validation Strategy: Hold-out validation set

6. **Deployment**
   - Production Pipeline: Serialized model artifacts
   - API Interface: Multiple input/output formats
   - Monitoring: Comprehensive logging and error handling

### 3.2 Feature Engineering Strategy

#### 3.2.1 Derived Features Created

1. **Customer Lifecycle Features**

   - `tenure_group`: Categorical tenure segments
   - `Is_New_Customer`: Binary indicator for customers < 6 months
   - `Is_Long_Term`: Binary indicator for customers > 48 months

2. **Service Usage Features**

   - `num_services`: Total count of active services
   - `Security_Package`: Count of security-related services
   - `Entertainment_Package`: Binary indicator for streaming services
   - `Has_Internet`: Binary indicator for internet service

3. **Financial Features**

   - `Charge_to_Tenure_Ratio`: Monthly charges divided by tenure
   - `Contract_Charges_Interaction`: Contract length × monthly charges

4. **Risk Indicators**

   - `Payment_Risk_Score`: Combined risk from payment method and billing
   - `Is_Paperless_and_Monthly`: High-risk combination indicator
   - `AutoPay`: Automatic payment method indicator

5. **Engagement Metrics**
   - `Engagement_Score`: Tenure × number of services
   - Service bundling indicators

#### 3.2.2 Feature Importance Analysis

Based on correlation analysis, the most predictive features are:

- Charge_to_Tenure_Ratio (0.412 correlation)
- Contract type (0.397 correlation)
- Contract_Length (0.394 correlation)
- Is_Paperless_and_Monthly (0.375 correlation)
- Tenure (0.352 correlation)

## 4. Models and Algorithms

### 4.1 Large Language Model (LLM)

#### 4.1.1 Model Selection

- **Model**: Gemma3:1b via Ollama
- **Framework**: LangChain for prompt management
- **Purpose**: Natural language feature extraction

#### 4.1.2 Prompt Engineering

The system uses a sophisticated prompt engineering approach:

```
System Role: Expert information extraction assistant
Task: Extract customer features from natural language
Output Format: Structured JSON with null handling
Validation Rules: Logical consistency checks
```

#### 4.1.3 Key Capabilities

- **Null Handling**: Proper treatment of missing information
- **Data Validation**: Pydantic models for type checking
- **Logical Consistency**: Rule-based validation (e.g., internet service dependencies)
- **Robust Parsing**: Regex-based JSON extraction from LLM responses

### 4.2 Machine Learning Model

#### 4.2.1 Algorithm Selection

- **Primary Model**: Random Forest Classifier
- **Rationale**:
  - Handles mixed data types effectively
  - Provides feature importance insights
  - Robust to outliers
  - Good interpretability

#### 4.2.2 Model Configuration

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=7,
    min_samples_leaf=1,
    min_samples_split=5,
    random_state=42
)
```

#### 4.2.3 Pipeline Architecture

```python
Pipeline([
    ('feature_engineering', ChurnFeatureEngineer()),
    ('preprocessing', Preprocessor()),
    ('model', RandomForestClassifier())
])
```

## 5. Data Processing Pipeline

### 5.1 Data Flow

1. **Raw Data Ingestion**

   - Source: CSV files (train.csv, val.csv, test.csv)
   - Format: Structured customer data with 21 features

2. **Feature Engineering**

   - Custom transformer: `ChurnFeatureEngineer`
   - Output: 35+ features (original + engineered)

3. **Preprocessing**

   - Categorical encoding (OneHot, Label, Target)
   - Numerical scaling (StandardScaler, MinMaxScaler)
   - Missing value imputation

4. **Model Training**

   - Target encoding: String labels → Binary (0/1)
   - Model fitting: Random Forest on processed features
   - Validation: Hold-out validation set

5. **Prediction Pipeline**
   - Input processing: Same transformations as training
   - Prediction: Binary classification with probabilities
   - Output formatting: User-friendly results with confidence

### 5.2 Preprocessing Strategy

#### 5.2.1 Categorical Variables

- **One-Hot Encoding**: For nominal categories with few levels
- **Label Encoding**: For ordinal categories

#### 5.2.3 Target Variable

- **Custom TargetEncoder**: Maps "Yes"/"No" → 1/0
- **Reversible**: Enables interpretation of results
- **Consistent**: Same encoding across training and prediction

## 6. Model Performance and Evaluation

### 6.1 Evaluation Methodology

The model evaluation follows:

1. **Training Set Evaluation**: Initial model performance assessment
2. **Validation Set Evaluation**: Unbiased performance estimation during development

### 6.2 Performance Metrics Overview

The system employs comprehensive evaluation metrics aligned with business objectives:

1. **Classification Accuracy**: Overall correctness across all predictions
2. **F1-Score**: Harmonic mean of precision and recall
3. **Precision**: True positive rate (accuracy of churn predictions)
4. **Recall**: Sensitivity (detection rate of actual churners)
5. **F-beta Score (β=0.5)**: Precision-weighted harmonic mean for business focus

### 6.3 Training Set Performance Results

#### 6.3.1 Core Metrics

| Metric             | Value  | Business Interpretation                      |
| ------------------ | ------ | -------------------------------------------- |
| **Accuracy**       | 82.90% | Correctly classifies 83 out of 100 customers |
| **Precision**      | 72.69% | 73% of predicted churners actually churn     |
| **Recall**         | 55.91% | Identifies 56% of actual churners            |
| **F1-Score**       | 63.20% | Balanced precision-recall performance        |
| **F-beta (β=0.5)** | 68.57% | Precision-focused business metric            |

#### 6.3.2 Detailed Classification Report

```
              precision    recall  f1-score   support

          No       0.85      0.93      0.89      3635
         Yes       0.73      0.56      0.63      1295

    accuracy                           0.83      4930
   macro avg       0.79      0.74      0.76      4930
weighted avg       0.82      0.83      0.82      4930
```

**Key Insights:**

- **Non-churners (No)**: High precision (85%) and excellent recall (93%)
- **Churners (Yes)**: Good precision (73%) but moderate recall (56%)
- **Weighted average**: Reflects the class imbalance in the dataset

#### 6.3.3 Confusion Matrix Analysis

```
Predicted:     No    Yes
Actual: No   3363   272   (Total: 3635)
        Yes   571   724   (Total: 1295)
```

**Confusion Matrix Breakdown:**

- **True Negatives (TN)**: 3,363 correctly identified non-churners
- **False Positives (FP)**: 272 incorrectly predicted as churners
- **False Negatives (FN)**: 571 missed churners (critical business cost)
- **True Positives (TP)**: 724 correctly identified churners

**Business Impact of Errors:**

- **False Positives (272)**: $16,592 in unnecessary retention spending\*
- **False Negatives (571)**: $42,520 in lost monthly revenue potential\*

\*Based on average customer values from EDA analysis

### 6.4 Validation Set Performance Results

#### 6.4.1 Core Validation Metrics

| Metric             | Training | Validation | Difference |
| ------------------ | -------- | ---------- | ---------- |
| **Accuracy**       | 82.90%   | 81.06%     | -1.84%     |
| **Precision**      | 72.69%   | 68.57%     | -4.12%     |
| **Recall**         | 55.91%   | 51.80%     | -4.11%     |
| **F1-Score**       | 63.20%   | 59.02%     | -4.18%     |
| **F-beta (β=0.5)** | 68.57%   | 64.40%     | -4.17%     |

#### 6.4.2 Validation Confusion Matrix

```
Predicted:     No    Yes
Actual: No    712    66   (Total: 778)
        Yes   134   144   (Total: 278)
```
### 6.1 Evaluation Metrics

The system employs comprehensive evaluation metrics:

1. **Classification Accuracy**: Overall correctness
2. **F1-Score**: Harmonic mean of precision and recall
3. **Precision**: True positive rate (churn prediction accuracy)
4. **Recall**: Sensitivity (actual churner detection rate)
5. **F-beta Score (β=0.5)**: Precision-weighted harmonic mean

### 6.2 Business Impact Analysis

Based on the EDA findings:

- **Total Customers**: 7,043
- **Churn Rate**: 26.5% (1,869 customers)
- **Retained Customers**: 73.5% (5,174 customers)
- **Revenue Impact**:
  - Average monthly charges (churned): $74.44
  - Average monthly charges (retained): $61.27
  - Monthly revenue loss: $139,130.85
  - Annualized revenue loss: $1,669,570.20

### 6.3 Model Interpretability

The system provides multiple levels of interpretability:

1. **Feature Importance**: Random Forest inherent feature ranking
2. **Probability Scores**: Confidence levels for each prediction
3. **Business Rules**: Interpretable feature engineering logic
4. **Prediction Explanations**: Detailed breakdown of contributing factors

## 7. Technical Implementation

### 7.1 Technology Stack

#### 7.1.1 Core Libraries

- **scikit-learn**: Machine learning pipeline and algorithms
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **loguru**: Advanced logging and monitoring

#### 7.1.2 LLM Integration

- **LangChain**: LLM orchestration and prompt management
- **Ollama**: Local LLM deployment and inference
- **Pydantic**: Data validation and serialization

#### 7.1.3 Development Tools

- **Jupyter**: Interactive development and analysis
- **Python**: Core programming language
- **Git**: Version control
- **Cookiecutter Data Science**: Project structure template

### 7.2 Code Architecture

#### 7.2.1 Modular Design

```
src/
├── config.py          # Configuration management
├── features.py         # Feature engineering
├── preprocessor.py     # Data preprocessing
├── target_encoder.py   # Target variable encoding
├── llm.py             # LLM integration
├── llm_models.py      # Pydantic data models
├── predictor.py       # Prediction wrapper
└── full_pipeline.py   # End-to-end pipeline
```

### 7.3 Error Handling and Validation

#### 7.3.1 Input Validation

- **Pydantic Models**: Type checking and validation
- **Business Rules**: Logical consistency verification

## 8. System Features and Capabilities

### 8.1 Input Flexibility

The system supports multiple input formats:

1. **Natural Language**: Free-text customer descriptions
2. **Structured Data**: Dictionary/JSON format
3. **Batch Processing**: CSV files or DataFrame input
4. **API Integration**: RESTful interface potential

### 8.2 Output Formats

Standardized output includes:

- **Prediction**: Binary churn classification
- **Confidence Score**: Probability-based confidence
- **Detailed Probabilities**: Churn and retention probabilities
- **Feature Attribution**: Contributing factors analysis

### 8.3 Scalability Features

- **Batch Processing**: Efficient handling of multiple customers
- **Serialization**: Model persistence for production deployment
- **Memory Optimization**: Efficient data handling

## 9. Validation and Testing

### 9.1 Model Validation

- **Hold-out Validation**: Separate validation dataset
- **Cross-validation**: K-fold validation during development
- **Business Metrics**: Focus on business-relevant metrics

### 9.3 Data Quality Assurance

- **Data Profiling**: Statistical analysis of input data
- **Outlier Detection**: Identification of anomalous patterns
- **Consistency Checks**: Logical validation rules

## 10. Business Value and ROI

### 10.1 Revenue Protection

- **Churn Prevention**: Early identification of at-risk customers
- **Targeted Interventions**: Personalized retention strategies
- **Revenue Optimization**: Focus on high-value customer retention

### 10.2 Operational Efficiency

- **Automated Screening**: Reduced manual analysis
- **Natural Language Interface**: Non-technical user accessibility
- **Batch Processing**: Efficient large-scale analysis

### 10.3 Strategic Insights

- **Customer Segmentation**: Risk-based customer grouping
- **Feature Importance**: Key churn drivers identification
- **Predictive Analytics**: Proactive customer management

## 11. Future Enhancements

### 11.1 Model Improvements

- **Ensemble Methods**: Multiple algorithm combination
- **Deep Learning**: Neural network integration
- **Feature Selection**: Automated feature optimization

### 11.2 System Enhancements

- **Real-time Processing**: Streaming data integration
- **Web Interface**: User-friendly dashboard development
- **API Development**: RESTful service implementation

### 11.3 Advanced Analytics

- **Explainable AI**: Enhanced model interpretability
- **Customer Lifetime Value**: Integrated CLV prediction
- **Recommendation Engine**: Personalized retention strategies

## 12. Graphical User Interface (GUI)

### 12.1 Streamlit Web Application

#### 12.1.1 Application Architecture

```python
# Main application structure
streamlit_app.py
├── Model Loading (@st.cache_resource)
├── Data Conversion Functions
├── Result Display Components
└── Main Interface (tabs-based layout)
```

#### 12.1.2 Core Features

**Single Customer Prediction Tab:**

- Natural language text input area
- Example customer descriptions
- Real-time LLM feature extraction
- Interactive prediction results display
- Customer information visualization

**Batch Customer Predictions Tab:**

- Multiple input methods:
  - Text area for multiple descriptions (one per line)
  - File upload functionality (.txt files)
- Progress tracking for batch processing
- Downloadable results (CSV format)

#### 12.1.3 User Interface Components

**Sidebar Information Panel:**

- Model information display
- Target encoding mapping
- LLM model details (Gemma 3 1B)
- ML pipeline information

**Main Dashboard Metrics:**

- Total customers processed
- Predicted churners count
- Overall churn rate percentage
- Average churn probability

**Results Visualization:**

- Color-coded predictions (Red: Will Churn, Green: Will Stay)
- Risk level indicators (High/Medium/Low)
- Confidence scores and probability metrics
- Customer description truncation for readability

#### 12.1.4 Technical Implementation Details

```python
@st.cache_resource
def load_models():
    """Cached model loading for performance"""
    predictor = ChurnPredictor.load(MODELS_DIR / "artifacts.pkl")
    extractor = CustomerInfoExtractor()
    return predictor, extractor

def display_prediction_results(results, customer_text):
    """Formatted result display with color coding"""
    # Risk-based color coding
    # Confidence metrics
    # Probability displays
```

#### 12.1.5 User Experience Features

- **Example Templates**: Pre-filled customer description examples
- **Progress Indicators**: Real-time processing status
- **Error Handling**: User-friendly error messages
- **Download Functionality**: CSV export of batch results
- **Responsive Design**: Multi-column layouts for optimal viewing

### 12.2 Usage Scenarios

**Business Analysts:**

- Upload customer lists for bulk churn assessment
- Generate reports for management review
- Identify high-risk customer segments

**Customer Service Representatives:**

- Quick individual customer risk assessment
- Real-time churn probability during customer calls
- Proactive intervention triggering

**Marketing Teams:**

- Campaign targeting based on churn risk
- Customer segmentation for retention programs
- ROI calculation for intervention strategies

## 13. Comprehensive Business Insights from Exploratory Data Analysis

### 13.1 Critical Risk Factors Identified

#### 13.1.1 Contract Type Analysis

**Key Finding**: Month-to-month contracts show dramatically higher churn rates

- **Month-to-month**: 3,875 customers (55.0%) - **42.7% churn rate**
- **One year**: 1,473 customers (20.9%) - **11.3% churn rate**
- **Two year**: 1,695 customers (24.1%) - **2.8% churn rate**

**Business Impact**: Customers with longer contracts are significantly more loyal, representing a clear retention strategy opportunity.

#### 13.1.2 Customer Tenure Risk Profile

**Key Finding**: New customers are at highest risk

- **0-12 months**: **47.4% churn rate** (highest risk segment)
- **13-24 months**: **35.2% churn rate**
- **25-48 months**: **15.1% churn rate**
- **49+ months**: **6.8% churn rate** (most loyal segment)

**Business Impact**: The first year is critical for customer retention. Early intervention programs could significantly reduce churn.

#### 13.1.3 Internet Service Risk Analysis

**Key Finding**: Fiber optic customers churn at double the rate of DSL customers

- **Fiber optic**: **41.9% churn rate** (2,421 customers)
- **DSL**: **19.0% churn rate** (2,421 customers)
- **No internet service**: **7.4% churn rate** (1,526 customers)

**Business Impact**: Despite higher service levels, fiber customers are less satisfied, indicating potential service quality or pricing issues.

#### 13.1.4 Payment Method Risk Analysis

**Key Finding**: Electronic check payments correlate with highest churn

- **Electronic check**: 2,365 customers (33.6%) - **45.3% churn rate**
- **Mailed check**: 1,612 customers (22.9%) - **19.1% churn rate**
- **Bank transfer (automatic)**: 1,544 customers (21.9%) - **16.7% churn rate**
- **Credit card (automatic)**: 1,522 customers (21.6%) - **15.2% churn rate**

**Business Impact**: Payment friction may contribute to churn. Encouraging automatic payments could improve retention.

#### 13.1.5 Service Usage Patterns

**Key Finding**: Customers with fewer additional services are more likely to churn

**High Churn Services:**

- **No Online Security**: 41.8% churn rate
- **No Tech Support**: 41.3% churn rate
- **No Device Protection**: 39.1% churn rate

**Low Churn Services:**

- **With Online Security**: 14.6% churn rate
- **With Tech Support**: 15.2% churn rate
- **With Device Protection**: 17.8% churn rate

### 13.2 Revenue Impact Analysis

#### 13.2.1 Financial Metrics

- **Total Customers**: 7,043
- **Churned Customers**: 1,869 (26.5%)
- **Retained Customers**: 5,174 (73.5%)

#### 13.2.2 Revenue Loss Calculations

- **Average Monthly Charges (Churned)**: $74.44
- **Average Monthly Charges (Retained)**: $61.27
- **Total Revenue from Churned Customers**: $2,862,926.90
- **Monthly Revenue Loss from Churn**: $139,130.85
- **Annualized Revenue Loss**: $1,669,570.20

#### 13.2.3 Customer Value Insights

**Key Finding**: Churning customers actually pay higher monthly rates on average

This suggests that price sensitivity may be a significant churn driver, or that higher-paying customers have higher expectations that aren't being met.

### 13.3 Strategic Business Opportunities

#### 13.3.1 High-Impact Intervention Targets

**Highest Priority Segments:**

1. **Month-to-month + Fiber optic customers**: 54.6% churn rate
2. **New customers with Electronic check payment**: 63.1% churn rate
3. **Customers with no additional services**: 56.7% churn rate

**Potential Revenue Recovery**: Targeting these segments could recover 40-60% of the annual $1.67M revenue loss.

#### 13.3.2 Service Bundling Opportunities

- **Security Package adoption** could reduce churn by 25+ percentage points
- **Entertainment Package** (streaming services) shows moderate retention benefit
- **Automatic payment adoption** could reduce churn by 25-30 percentage points

#### 13.3.3 Contract Strategy Optimization

- **Contract upgrade campaigns** for month-to-month customers
- **Early contract renewal incentives** before contract expiration
- **New customer onboarding** focused on longer-term commitments

### 13.4 Next Steps and Recommendations

#### 13.4.1 Immediate Actions (0-3 months)

1. **Build predictive models** using the identified key risk factors
2. **Implement targeted retention campaigns** for high-risk segments
3. **Monitor key metrics** and validate improvement hypotheses
4. **A/B test retention strategies** with control groups

#### 13.4.2 Strategic Initiatives (3-12 months)

1. **Service quality improvement** for fiber optic customers
2. **Payment method migration** campaigns to automatic payments
3. **New customer onboarding** program redesign
4. **Service bundling** promotion strategies

#### 13.4.3 Long-term Strategy (12+ months)

1. **Customer lifetime value optimization** based on churn insights
2. **Pricing strategy review** for high-value customer segments
3. **Service portfolio optimization** based on retention impact
4. **Predictive analytics integration** into customer service workflows

### 13.5 Statistical Significance Analysis

#### 13.5.1 Chi-Square Test Results

All major risk factors show statistically significant associations with churn (p < 0.001):

- **Contract**: χ² = 1,184.60, Phi = 0.4101 (Strong association)
- **Online Security**: χ² = 850.00, Phi = 0.3474 (Strong association)
- **Tech Support**: χ² = 828.20, Phi = 0.3429 (Strong association)
- **Internet Service**: χ² = 732.31, Phi = 0.3225 (Strong association)
- **Payment Method**: χ² = 648.14, Phi = 0.3034 (Strong association)

#### 13.5.2 Correlation Analysis

**Strongest predictors of churn** (correlation coefficients):

1. Charge_to_Tenure_Ratio: 0.412
2. Contract type: 0.397
3. Contract_Length: 0.394
4. Is_Paperless_and_Monthly: 0.375
5. Tenure: 0.352

### 13.6 Business Impact Summary

This comprehensive EDA has revealed several critical insights about customer churn:

**Key Risk Factors:**

- **Contract Type**: Month-to-month contracts show significantly higher churn rates
- **Tenure**: New customers (first 12 months) are at highest risk
- **Internet Service**: Fiber optic customers churn more than DSL customers
- **Payment Method**: Electronic check payments correlate with higher churn
- **Service Usage**: Customers with fewer additional services are more likely to churn

**Business Impact:**

- **High revenue loss** from churned customers ($1.67M annually)
- **Opportunity to reduce churn** through targeted interventions
- **Service combinations** can increase customer retention
- **Contract strategy** offers significant retention improvement potential

**Next Steps:**

- **Build predictive models** using these insights
- **Implement targeted retention campaigns** for identified high-risk segments
- **Monitor key metrics** and validate improvements
- **A/B test retention strategies** with measurable business outcomes

This analysis provides a solid foundation for developing effective churn prediction models and retention strategies that could recover a significant portion of the $1.67 million in annual revenue loss while improving overall customer satisfaction and loyalty.
