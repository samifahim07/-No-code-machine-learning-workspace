# ML Workflow Helper

ML Workflow Helper is a Streamlit-based machine learning web application that guides users through a complete machine learning workflow using any CSV dataset.

The application is designed to reduce repetitive notebook work by providing a structured, step-by-step interface for data inspection, preprocessing, visualization, model training, tuning, comparison, evaluation, and model export.

It supports both classification and regression problems and keeps the user in control of the important choices throughout the workflow.

## Project Demonstration

A complete video demonstration of this project is available on Google Drive.

[Watch the Project Demo](https://drive.google.com/drive/folders/14cvLyeL-PUc4mWojmw6LXiuh-WcGWPLe?usp=drive_link)
## Overview

Instead of repeatedly writing the same preprocessing and model-training code for every new dataset, ML Workflow Helper provides a reusable interface where users can:

- Upload a CSV dataset
- Select the target column
- Choose classification or regression
- Inspect the dataset
- Handle missing values and duplicates
- Visualize important patterns
- Split the dataset
- Encode categorical features
- Scale numerical features
- Train multiple machine learning models
- Tune models with RandomizedSearchCV
- Compare models using evaluation metrics
- Detect possible overfitting
- Save the best model and preprocessing components

The application currently runs from a single `app.py` file and uses a nine-step workflow in the Streamlit sidebar.

## Application Workflow

### Step 1: Upload Dataset and Select Target

The user uploads a CSV file and selects:

- CSV separator
- Target column
- Problem type
  - Classification
  - Regression

The application can automatically guess the problem type based on the target column, but the user can override the suggestion.

### Step 2: Dataset Overview

The application provides useful dataset information including:

- Head
- Tail
- Column names
- Dataset shape
- Data types
- Dataset information
- Descriptive statistics

Users can also:

- Select the number of rows to display
- Include text columns in descriptive statistics
- Drop unnecessary columns

### Step 3: Missing Value Analysis

The application displays:

- Missing value count
- Missing value percentage

Users can choose to:

- Only inspect missing values
- Drop columns based on a missing-value threshold
- Apply a numeric missing-value strategy
- Apply a text missing-value strategy

Rows where the target value is missing are always removed instead of being filled.

### Step 4: Duplicate Handling

The application shows:

- Total duplicate rows
- Sample duplicate records

Available options include:

- Only inspect duplicates
- Drop duplicates while keeping the first occurrence
- Drop duplicates while keeping the last occurrence

### Step 5: Data Visualization

The application supports several exploratory data analysis plots:

- Target distribution
- Histograms
- Boxplots
- Correlation heatmap
- Countplots

Users can choose which visualizations to generate and how many columns to display.

### Step 6: Train-Test Split, Encoding, and Scaling

The application prepares the dataset for machine learning and displays previews and shapes of:

- `X_train`
- `X_test`
- `y_train`
- `y_test`

Users can configure:

- Test size
- Random state
- Stratified splitting
- Encoding method
- Scaling method

Supported encoding methods:

- One-Hot Encoding
- Ordinal Encoding

Supported scaling methods:

- StandardScaler
- MinMaxScaler
- RobustScaler
- No scaling

The encoder and scaler are fitted using the training data and then applied to the test data.

### Step 7: Model Training and Hyperparameter Tuning

Users can train selected machine learning algorithms using either:

- Default mode
- Tuned mode

Default mode trains models using their normal library defaults with a few fixed settings where required.

Tuned mode uses `RandomizedSearchCV` with cross-validation and a limited parameter search space to find stronger model configurations without the cost of a full grid search.

Classification tuning uses accuracy as the optimization score.

Regression tuning uses R2 as the optimization score.

Default and tuned versions of the same model are stored separately so they can be compared in the final results.

## Supported Models

### Classification

| Model |
|---|
| Logistic Regression |
| Random Forest Classifier |
| Decision Tree Classifier |
| Support Vector Machine |
| CatBoost Classifier |
| AdaBoost Classifier |
| LightGBM Classifier |

### Regression

| Model |
|---|
| Linear Regression |
| Random Forest Regressor |
| Decision Tree Regressor |
| Support Vector Regressor |
| CatBoost Regressor |
| AdaBoost Regressor |
| LightGBM Regressor |

## Model Evaluation

### Classification Metrics

The application evaluates classification models using more than accuracy alone.

Supported metrics include:

- Training accuracy
- Test accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- Sensitivity
- Specificity
- ROC AUC
- ROC curve

Sensitivity represents the proportion of actual positive cases correctly identified.

Specificity represents the proportion of actual negative cases correctly identified.

For multiclass classification, sensitivity and specificity are calculated using averaged one-vs-rest values.

### Regression Metrics

Regression models are evaluated using:

- R2
- RMSE
- MAE
- RSE
- RAE

#### R2

Measures the proportion of variance explained by the model.

#### RMSE

Represents the typical prediction error in the same unit as the target variable while penalizing larger errors more heavily.

#### MAE

Represents the average absolute prediction error.

#### RSE

In this project, RSE means Relative Squared Error.

It is calculated by comparing the model's squared error against the squared error produced by always predicting the target mean.

Values below `1` indicate performance better than the mean baseline.

#### RAE

Relative Absolute Error compares the model's absolute error against the absolute error of the mean baseline.

Values below `1` indicate performance better than the baseline.

## Model Comparison

After training, the application creates a comparison view containing:

- Model performance table
- Ranking based on a selected metric
- Bar chart comparison
- Best-performing model
- Possible overfitting warning

The user can select which metric should be used to rank the trained models.

## Saving the Best Model

The final step allows the user to export the best model.

The saved pickle bundle contains:

- Trained model
- Encoder
- Scaler
- Label encoder
- Feature column names
- Class names

This allows new data to be prepared using the same preprocessing configuration that was used during training.

Only load pickle files that you created or trust. Pickle files can execute unsafe code, and compatibility may also break when library versions change.

## Project Structure

```text
ML-Workflow-Helper/
|
|-- app.py
|-- requirements.txt
|-- README.md
```

The project currently keeps the application files in a simple flat structure.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application should open in your browser at:

```text
http://localhost:8501
```

Then:

1. Upload a CSV dataset.
2. Select the target column.
3. Confirm whether the problem is classification or regression.
4. Follow the workflow from Step 1 to Step 9.
5. Train and compare models.
6. Save the best model if required.

## Important Design Decisions

Several design choices are used to keep the workflow consistent:

- The encoder is fitted only on training data.
- The scaler is fitted only on training data.
- Test data is transformed using preprocessing learned from the training set.
- Rows with a missing target are removed.
- The application can guess the problem type, but the user can override it.
- Changing an earlier preprocessing step clears previous train-test splits and trained models.
- Scaling is applied after encoding.
- User actions are applied only after the relevant apply button is clicked.
- Workflow choices are recorded in a log.

## Current Limitations

The current version has several known limitations:

- Missing-value filling happens before the train-test split, which can introduce a small amount of data leakage.
- Very large datasets can be slow.
- Heavy hyperparameter tuning can take significant time.
- One-Hot Encoding on high-cardinality categorical columns can create a very large number of features.
- Only CSV files are currently supported.
- Cross-validated model comparison is not yet implemented for the full comparison stage.
- A dedicated prediction page is not yet available.
- Feature importance and SHAP explanations are not yet included.
- Automated unit tests are not yet included.

## Planned Improvements

Future versions can improve the project by adding:

- A complete scikit-learn preprocessing and modeling Pipeline
- Imputation fitted only on training data
- Cross-validated model comparison
- Feature importance
- SHAP explainability
- A dedicated prediction interface for new records
- Excel file support
- Built-in sample datasets
- Row sampling for large datasets
- Tuning time limits
- High-cardinality categorical feature warnings
- Target encoding
- Unit tests for evaluation functions
- Improved multiclass ROC AUC testing

## Recommended Development Roadmap

A practical extension order for the project is:

1. Validate the full workflow using one classification dataset and one regression dataset.
2. Add a prediction page.
3. Add feature importance and explainability.
4. Refactor preprocessing and modeling into a scikit-learn Pipeline.
5. Add automated tests.
6. Add screenshots and demonstrations to the repository.

## Why This Project Matters

ML Workflow Helper is intended to be more than a single-dataset machine learning notebook.

It demonstrates an end-to-end machine learning workflow that includes:

- Data inspection
- Data cleaning
- Feature preprocessing
- Classification
- Regression
- Hyperparameter tuning
- Model evaluation
- Model comparison
- Model persistence

The project is especially useful for learning how the individual stages of a machine learning pipeline connect together while also creating a reusable tool for future datasets.

## Screenshots

Add screenshots of the application here after uploading them to the repository.

Example:

```markdown
![Dataset Overview](images/dataset-overview.png)

![Model Comparison](images/model-comparison.png)
```

## License

Add the license you want to use for this repository.

For example, you can add an MIT License if you want others to freely use, modify, and distribute the project.

## Author

Developed as a machine learning workflow and portfolio project.
