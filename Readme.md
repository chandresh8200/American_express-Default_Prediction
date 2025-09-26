

-----

# American Express - Default Prediction

This repository contains the code for the American Express - Default Prediction Kaggle competition. The goal of this competition is to predict the probability that a customer will default on their credit card balance in the future based on their monthly statements.

This solution employs a multi-stage feature engineering pipeline, followed by training and ensembling LightGBM and Neural Network models to generate the final predictions.

## Solution Pipeline

The project is structured as a sequential pipeline, with each script responsible for a specific stage of the process. The scripts are numbered to be run in order.

### 1\. Data Denoising (`S1_denoise.py`)

  - **Purpose**: Initial data cleaning and preprocessing.
  - **Actions**:
      - Categorical features `D_63` and `D_64` are manually encoded into integer representations.
      - All other numerical features are multiplied by 100 and floored to convert them to integers, which can help reduce noise and memory usage.
      - The processed dataframes are saved in the efficient `.feather` format.

### 2\. Manual Feature Engineering (`S2_manual_feature.py`)

  - **Purpose**: Generate a rich set of aggregated features from the customer statement data.
  - **Process**:
      - The script creates features based on the full history of each customer, as well as features based on only the last 3 and 6 statements.
      - It computes aggregations (`mean`, `std`, `min`, `max`, `sum`, `last`, `nunique`) for numerical, categorical (after one-hot encoding), and difference features.
      - The script leverages multiprocessing (`ThreadPool`) to speed up the feature generation process significantly.

### 3\. Time Series Feature Generation (`S3_series_feature.py`)

  - **Purpose**: Create features that capture the time-series nature of the data and train a LightGBM model on them.
  - **Process**:
      - A LightGBM model is trained using categorical features like `B_30`, `B_38`, `D_63`, etc.
      - This script uses a LightGBM model with `dart` boosting.
      - The out-of-fold (OOF) predictions and test set predictions from this model are saved to be used as features in later stages.

### 4\. Feature Combination (`S4_feature_combined.py`)

  - **Purpose**: Consolidate all engineered features into a single dataset for the main models.
  - **Actions**:
      - Combines features from different categories: `cat`, `num`, `diff`, `rank_num`, and features from the last 3/6 months.
      - Appends the OOF predictions generated in `S3` as new features.
      - Creates a separate, normalized dataset specifically for the Neural Network model (`nn_all_feature.feather`).

### 5\. LightGBM Model Training (`S5_LGB_main.py`)

  - **Purpose**: Train the main LightGBM models on the complete feature set.
  - **Models**:
    1.  **Model 1**: Trained on all manual features.
    2.  **Model 2**: Trained on all manual features plus the time-series OOF predictions from `S3`.
  - **Configuration**: Uses 5-fold stratified cross-validation and a `dart` boosting LightGBM model.

### 6\. Neural Network Model Training (`S6_NN_main.py`)

  - **Purpose**: Train a Neural Network to capture sequential patterns in the data.
  - **Architecture (`model.py`)**:
      - The model uses a Gated Recurrent Unit (GRU) to process the time-series of customer statements.
      - The output from the GRU is combined with the aggregated features before being passed to a final prediction head.
  - **Models**: Two variations of the NN model are trained, likely with and without certain feature sets to enhance diversity for ensembling.

### 7\. Ensembling (`S7_ensemble.py`)

  - **Purpose**: Combine the predictions from all trained models to create the final submission file.
  - **Method**: A weighted average of the predictions from the four models (2 LGB, 2 NN) is calculated. The weights are:
      - LGB (manual features): `0.3`
      - LGB (manual + series features): `0.35`
      - NN (series features): `0.15`
      - NN (series + all features): `0.1`

## How to Run the Pipeline

To reproduce the results, follow these steps:

1.  **Prerequisites**

      - Ensure you have all the necessary libraries installed.
        ```bash
        pip install -r requirements.txt
        ```

2.  **Data Setup**

      - Place the competition's `train_data.csv`, `test_data.csv`, and `train_labels.csv` files inside an `./input/` directory.

3.  **Execute the Scripts in Order**

      - Run the scripts sequentially from your terminal.

    <!-- end list -->

    ```bash
    python S1_denoise.py
    python S2_manual_feature.py
    python S3_series_feature.py
    python S4_feature_combined.py
    python S5_LGB_main.py
    python S6_NN_main.py
    python S7_ensemble.py
    ```

4.  **Final Submission**

      - The final submission file will be generated at `./output/final_submission.csv.zip`.

## Dependencies

This project relies on the following major libraries:
  - `pandas`
  - `numpy`
  - `lightgbm`
  - `torch`
  - `tqdm`
  - `matplotlib`

