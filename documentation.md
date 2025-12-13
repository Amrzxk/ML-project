# Project Documentation

This document provides detailed information about the machine learning models implemented in this project. Two notebooks are covered:

1. **Numerical.ipynb** – Linear Regression & KNN as regressors on a numerical (tabular) dataset  
2. **Image.ipynb** – Logistic Regression & KMeans as classifiers on an image dataset

---

## Table of Contents

- [Numerical Dataset (Numerical.ipynb)](#numerical-dataset-numericalipynb)
  - [Dataset General Information](#dataset-general-information)
  - [Model 1: Linear Regression (Manual Gradient Descent)](#model-1-linear-regression-manual-gradient-descent)
  - [Model 2: KNN Regressor](#model-2-knn-regressor)
- [Image Dataset (Image.ipynb)](#image-dataset-imageipynb)
  - [Dataset General Information](#dataset-general-information-1)
  - [Model 1: Logistic Regression (SGD)](#model-1-logistic-regression-sgd)
  - [Model 2: KMeans Classifier](#model-2-kmeans-classifier)

---

# Numerical Dataset (Numerical.ipynb)

## Dataset General Information

| Property | Value |
|----------|-------|
| **Dataset Name** | TUANDROMD (Android Malware Dataset) |
| **Source File** | `Datasets/TUANDROMD.csv` |
| **Total Samples (raw)** | 4,465 |
| **Total Samples (after cleaning)** | 662 (duplicates & missing values removed) |
| **Number of Classes** | 2 |
| **Class Labels** | `goodware` (0), `malware` (1) |
| **Class Distribution** | goodware: 531, malware: 131 |
| **Sample Size** | N/A (tabular numerical features, not images) |
| **Train Samples** | 529 (80%) |
| **Validation Samples** | 66 (10%) |
| **Test Samples** | 67 (10%) |

### Feature Extraction Details

| Property | Value |
|----------|-------|
| **Number of Features** | 241 |
| **Feature Type** | Binary/numerical Android app permissions and API calls |
| **Feature Matrix Dimension** | (662, 241) raw → (662, 241) after StandardScaler |
| **Scaling** | StandardScaler (zero mean, unit variance) |

<details>
<summary><strong>Sample Feature Names (first 25)</strong></summary>

```
ACCESS_ALL_DOWNLOADS
ACCESS_CACHE_FILESYSTEM
ACCESS_CHECKIN_PROPERTIES
ACCESS_COARSE_LOCATION
ACCESS_COARSE_UPDATES
ACCESS_FINE_LOCATION
ACCESS_LOCATION_EXTRA_COMMANDS
ACCESS_MOCK_LOCATION
ACCESS_MTK_MMHW
ACCESS_NETWORK_STATE
ACCESS_PROVIDER
ACCESS_SERVICE
ACCESS_SHARED_DATA
ACCESS_SUPERUSER
ACCESS_SURFACE_FLINGER
ACCESS_WIFI_STATE
activityCalled
ACTIVITY_RECOGNITION
ACCOUNT_MANAGER
ADD_VOICEMAIL
ANT
ANT_ADMIN
AUTHENTICATE_ACCOUNTS
AUTORUN_MANAGER_LICENSE_MANAGER
AUTORUN_MANAGER_LICENSE_SERVICE
```

</details>

<details>
<summary><strong>Sample Feature Names (last 5)</strong></summary>

```
Landroid/telephony/TelephonyManager;->getSimOperator
Landroid/telephony/TelephonyManager;->getSimOperatorName
Landroid/telephony/TelephonyManager;->getSimCountryIso
Landroid/telephony/TelephonyManager;->getSimSerialNumber
Lorg/apache/http/impl/client/DefaultHttpClient;->execute
```

</details>

---

## Model 1: Linear Regression (Manual Gradient Descent)

### Implementation Details

This model implements **linear regression from scratch** using batch gradient descent to predict numeric labels (0 or 1). For classification-style metrics, the regressor output is thresholded at 0.5.

#### Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | Batch Gradient Descent (manual implementation) |
| **Learning Rate** | 0.005 |
| **Number of Epochs** | 2000 (max) |
| **Batch Size** | Full batch (all training samples) |
| **L2 Regularization (λ)** | 1e-4 |
| **Gradient Clipping** | max_grad_norm = 5.0 |
| **Early Stopping** | patience = 50 epochs, min_delta = 1e-8 |

#### Cross-Validation

| Property | Value |
|----------|-------|
| **Cross-Validation Used?** | No (explicit train/val/test split) |
| **Train/Val/Test Ratio** | 80% / 10% / 10% |

### Results on Test Data

| Metric | Value |
|--------|-------|
| **MSE** | 0.066672 |
| **MAE** | 0.186956 |
| **R²** | 0.573659 |
| **Accuracy** | 92.54% |
| **Threshold** | 0.5 |

#### Outputs Provided
- ✅ **Loss Curve** – Train & Validation MSE over epochs (early stopped at epoch 404)
- ✅ **Accuracy** – 92.54%
- ✅ **Confusion Matrix** – Displayed as heatmap
- ✅ **ROC Curve** – With AUC score

---

## Model 2: KNN Regressor

### Implementation Details

This model uses **K-Nearest Neighbors Regressor** from scikit-learn. The regressor predicts a continuous value in [0, 1], which is thresholded at 0.5 for classification metrics.

#### Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| **Algorithm** | KNeighborsRegressor (scikit-learn) |
| **n_neighbors (K)** | 5 |
| **Weights** | uniform (default) |
| **Distance Metric** | Minkowski (p=2, Euclidean) |

#### Cross-Validation

| Property | Value |
|----------|-------|
| **Cross-Validation Used?** | No (explicit train/val/test split) |
| **Train/Val/Test Ratio** | 80% / 10% / 10% |

### Results on Test Data

| Metric | Value |
|--------|-------|
| **MSE** | 0.051940 |
| **MAE** | 0.092537 |
| **R²** | 0.667863 |
| **Accuracy** | 92.54% |
| **Threshold** | 0.5 |

#### Outputs Provided
- ✅ **Loss Curve** – N/A for KNN (no iterative training); hyperparameter was fixed
- ✅ **Accuracy** – 92.54%
- ✅ **Confusion Matrix** – Displayed as heatmap
- ✅ **ROC Curve** – With AUC score

---

# Image Dataset (Image.ipynb)

## Dataset General Information

| Property | Value |
|----------|-------|
| **Dataset Name** | IJCNN 2013 Traffic Sign Recognition (subset) |
| **Source File** | `Datasets/FullIJCNN2013.zip` |
| **Extracted To** | `Datasets/IJCNN2013/FullIJCNN2013` |
| **Total Classes Available** | 43 |
| **Classes Used** | 5 (first 5 class folders: 00, 01, 02, 03, 04) |
| **Total Samples Loaded** | 262 |
| **Max Samples Per Class** | 200 (capped) |
| **Image Size (original)** | Variable |
| **Image Size (after resize)** | 48 × 48 pixels (RGB) |
| **Train Samples** | 188 (≈72%) |
| **Validation Samples** | 21 (≈8%) |
| **Test Samples** | 53 (20%) |

### Class Distribution

| Class | Samples |
|-------|---------|
| 00 | 4 |
| 01 | 79 |
| 02 | 81 |
| 03 | 30 |
| 04 | 68 |

### Feature Extraction Details

| Property | Value |
|----------|-------|
| **Feature Extraction Method** | Flattened RGB pixel intensities |
| **Number of Features** | 6,912 (48 × 48 × 3 channels) |
| **Feature Names** | Pixel values at positions (row, col, channel) |
| **Feature Matrix Dimension** | (262, 6912) |
| **Preprocessing** | Resize to 48×48, normalize to [0,1], then StandardScaler |

---

## Model 1: Logistic Regression (SGD)

### Implementation Details

This model uses **SGDClassifier** from scikit-learn with log-loss (logistic regression) trained via mini-batch stochastic gradient descent.

#### Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | SGD (Stochastic Gradient Descent) |
| **Loss Function** | log_loss (cross-entropy) |
| **Penalty (Regularization)** | L2 |
| **Alpha (L2 strength)** | 1e-4 |
| **Learning Rate Schedule** | optimal |
| **Initial Learning Rate (eta0)** | 0.01 |
| **Number of Epochs** | 24 |
| **Batch Size** | 64 |
| **Warm Start** | True |
| **Random State** | 42 |

#### Cross-Validation

| Property | Value |
|----------|-------|
| **Cross-Validation Used?** | No (explicit train/val/test split) |
| **Train/Val/Test Ratio** | 72% / 8% / 20% (stratified) |

### Results on Test Data

| Metric | Value |
|--------|-------|
| **Accuracy** | 84.91% |
| **ROC AUC (OVR)** | 0.7938 |

#### Outputs Provided
- ✅ **Loss Curve** – Log-loss over 24 epochs (final: 1.3568)
- ✅ **Accuracy** – 84.91%
- ✅ **Confusion Matrix** – 5×5 matrix displayed as heatmap
- ✅ **ROC Curve** – One-vs-Rest ROC for each class with per-class AUC

---

## Model 2: KMeans Classifier

### Implementation Details

This model uses **KMeans clustering** (unsupervised) with cluster-to-label mapping to perform classification. Each cluster is assigned to the majority class label from the training data.

#### Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| **Algorithm** | KMeans (scikit-learn) |
| **Number of Clusters (K)** | 5 (equal to number of classes) |
| **n_init** | 30 (number of initializations) |
| **max_iter** | 400 |
| **Random State** | 42 |

#### "Loss Curve" (Inertia)

An **elbow plot** (inertia vs. K for K=1 to 10) is shown on training data to visualize cluster tightness. Lower inertia indicates tighter clusters.

#### Cross-Validation

| Property | Value |
|----------|-------|
| **Cross-Validation Used?** | No (explicit train/val/test split) |
| **Train/Val/Test Ratio** | 72% / 8% / 20% (stratified) |

### Results on Test Data

| Metric | Value |
|--------|-------|
| **Accuracy** | 30.19% |
| **ROC AUC (OVR)** | 0.5272 |

> **Note:** KMeans is an unsupervised algorithm and is not optimized for classification. The low accuracy reflects this limitation. Pseudo-probabilities for ROC are computed from inverse distances to cluster centroids.

#### Outputs Provided
- ✅ **Loss Curve** – Inertia vs. K (elbow plot)
- ✅ **Accuracy** – 30.19%
- ✅ **Confusion Matrix** – 5×5 matrix displayed as heatmap
- ✅ **ROC Curve** – One-vs-Rest ROC with per-class AUC (based on pseudo-probabilities)

---

## Summary Table

| Notebook | Model | Dataset | Classes | Features | Train/Val/Test | CV? | Accuracy | Loss Curve | CM | ROC |
|----------|-------|---------|---------|----------|----------------|-----|----------|------------|----|----|
| Numerical.ipynb | Linear Regression (GD) | TUANDROMD | 2 | 241 | 529/66/67 | No | 92.54% | ✅ MSE | ✅ | ✅ |
| Numerical.ipynb | KNN Regressor | TUANDROMD | 2 | 241 | 529/66/67 | No | 92.54% | — | ✅ | ✅ |
| Image.ipynb | Logistic Regression (SGD) | IJCNN2013 | 5 | 6,912 | 188/21/53 | No | 84.91% | ✅ Log-loss | ✅ | ✅ |
| Image.ipynb | KMeans Classifier | IJCNN2013 | 5 | 6,912 | 188/21/53 | No | 30.19% | ✅ Inertia | ✅ | ✅ |

---

## How to Run

1. Ensure datasets are in place:
   - `Datasets/TUANDROMD.csv`
   - `Datasets/FullIJCNN2013.zip`

2. Run notebooks in order (top to bottom):
   - **Numerical.ipynb** – for Linear Regression & KNN on tabular data
   - **Image.ipynb** – for Logistic Regression & KMeans on traffic sign images

3. All required outputs (loss curves, accuracy, confusion matrices, ROC curves) are generated automatically.

---

*Generated for Machine Learning Project*

