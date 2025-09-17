import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score  # or use appropriate metric for evaluation
import optuna
import shap
from xgboost import XGBClassifier

# *** 1. Prepare text data and initial TF-IDF features ***
texts = [  # Example corpus (replace with actual dataset)
    "I loved the movie, it was fantastic and thrilling",
    "The film was boring and too long",
    "What an amazing experience, would watch again!",
    "Terrible movie. Waste of time.",
    # ... (more samples)
]
labels = np.array([1, 0, 1, 0])  # Example binary labels (1=positive, 0=negative)

# Convert text to TF-IDF features (sparse matrix)
vectorizer = TfidfVectorizer(max_features=10000)  # limit features for memory, adjust as needed
X_sparse = vectorizer.fit_transform(texts)        # X_sparse is a SciPy sparse matrix
y = labels

# If dealing with a large dataset, ensure memory efficiency by keeping X_sparse sparse.
# Only convert to dense when necessary (e.g., computing SHAP values for a small subset).

# *** 2. Hyperparameter tuning for the base model using Optuna (on a hold-out split or via CV) ***
# Split off a small validation set for tuning (to avoid nested CV for simplicity)
X_tune, X_valid, y_tune, y_valid = train_test_split(X_sparse, y, test_size=0.2, stratify=y, random_state=42)

def objective(trial):
    # Define the hyperparameter search space for XGBoost
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "tree_method": "auto"  # use "hist" or "gpu_hist" if GPU available for speed
    }
    # Train on the tuning split and evaluate on the validation split
    # Note: We convert the small hold-out sets to dense for XGBoost if needed (XGB can accept sparse, but dense is fine here).
    X_tune_dense = X_tune.toarray()   # densify only the small tuning subset
    X_valid_dense = X_valid.toarray() # densify the validation subset
    model = XGBClassifier(**params)
    model.fit(X_tune_dense, y_tune, eval_set=[(X_valid_dense, y_valid)], early_stopping_rounds=20, verbose=False)
    # Evaluate performance (e.g., accuracy on validation)
    preds = model.predict(X_valid_dense)
    accuracy = accuracy_score(y_valid, preds)
    return accuracy  # Optuna will maximize this by default if direction="maximize"

# Run Optuna optimization (with a small number of trials for demonstration; increase for real tuning)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, show_progress_bar=False)
best_params = study.best_params
print("Best base model params found:", best_params)

# ***(Optional) If no separate tuning set, one could perform internal CV in objective instead of train_test_split.***
# For example, use StratifiedKFold inside objective to compute average CV score for each trial.
# This ensures all data contributes to tuning, at cost of extra computation.

# *** 3. Train base model with best hyperparameters using Stratified K-Fold for OOF predictions ***
best_params.update({"use_label_encoder": False, "eval_metric": "logloss"})  # ensure necessary params for XGB
base_model = XGBClassifier(**best_params)

# Prepare arrays to hold out-of-fold predictions and SHAP values
n_samples, n_features = X_sparse.shape
oof_preds = np.zeros(n_samples)                # will store predicted probabilities for positive class (for each sample)
oof_shap = np.zeros((n_samples, n_features))   # will store shap values for each feature of each sample

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_sparse, y), 1):
    X_train_fold = X_sparse[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X_sparse[val_idx]
    y_val_fold = y[val_idx]

    # Train base model on this fold’s training data
    model_fold = XGBClassifier(**best_params)
    model_fold.fit(X_train_fold, y_train_fold)  # XGBoost can accept sparse matrix input

    # OOF predictions for the validation fold (probability of class 1)
    val_pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]  # get probability of positive class
    oof_preds[val_idx] = val_pred_proba

    # Compute SHAP values for the validation fold
    explainer = shap.TreeExplainer(model_fold)
    # Convert validation fold features to dense for SHAP calculation (done fold-by-fold to limit memory usage)
    X_val_fold_dense = X_val_fold.toarray()  
    # For binary classification, shap_values returns a matrix of shape (n_val_samples, n_features)
    shap_values_fold = explainer.shap_values(X_val_fold_dense)
    # If shap_values returns a list (e.g., for multiclass, shap_values would be list of arrays for each class),
    # take the array for the class of interest. For binary XGBoost, we get a single array.
    if isinstance(shap_values_fold, list):
        shap_values_fold = shap_values_fold[1]  # for binary, index 1 is usually the positive class explanation
    # Store the SHAP values in the OOF array
    oof_shap[val_idx, :] = shap_values_fold

    print(f"Fold {fold}: base model accuracy = {accuracy_score(y_val_fold, (val_pred_proba>0.5).astype(int)):.3f}")

# After this loop, we have:
# oof_preds: an array of length n_samples with the out-of-fold predicted probabilities for each training sample.
# oof_shap: an array of shape (n_samples, n_features) with the out-of-fold SHAP values for each feature of each sample.

# *** 4. Augment original features with OOF predictions and SHAP values ***
# Convert OOF arrays to sparse format for efficient hstack with original sparse features
oof_preds_feat = csr_matrix(oof_preds.reshape(-1, 1))  # shape (n_samples, 1)
oof_shap_feat = csr_matrix(oof_shap)                  # shape (n_samples, n_features)

# Create augmented feature matrices:
X_train_aug_P      = hstack([X_sparse, oof_preds_feat])                   # original + P
X_train_aug_SHAP   = hstack([X_sparse, oof_shap_feat])                   # original + SHAP
X_train_aug_P_SHAP = hstack([X_sparse, oof_preds_feat, oof_shap_feat])   # original + P + SHAP

# The above are still sparse matrices. Their dimensions:
print("Original feature count:", X_sparse.shape[1])
print("Augmented (P) feature count:", X_train_aug_P.shape[1])
print("Augmented (SHAP) feature count:", X_train_aug_SHAP.shape[1])
print("Augmented (P+SHAP) feature count:", X_train_aug_P_SHAP.shape[1])
# Note: For binary classification, P adds 1 feature. SHAP adds as many features as original (each feature gets a SHAP value).
# So P+SHAP roughly doubles the feature count (plus one). For text with thousands of features, this is a significant increase.

# *** 5. Train second-stage models on the augmented feature sets ***
# We'll train one model using only P, one using only SHAP, and one using both P+SHAP.
# Reuse the same model type (XGBClassifier) for second stage for consistency. One could also use a simpler model like LogisticRegression.
model_P      = XGBClassifier(**best_params)
model_SHAP   = XGBClassifier(**best_params)
model_P_SHAP = XGBClassifier(**best_params)

model_P.fit(X_train_aug_P, y)         # Train on original features + prediction
model_SHAP.fit(X_train_aug_SHAP, y)   # Train on original features + SHAP values
model_P_SHAP.fit(X_train_aug_P_SHAP, y)  # Train on original + prediction + SHAP (full SFA)

# *** 6. Using the trained pipeline on new/unseen data (e.g., test set) ***
# Suppose we have a test set of texts:
test_texts = [
    "An absolutely fantastic film with great acting", 
    "Not my taste, I found it dull and uninspired"
]
X_test_sparse = vectorizer.transform(test_texts)  # transform test texts to TF-IDF features (sparse)

# Generate base model predictions and SHAP values for test data.
# Option 1: Train a final base model on the entire training set (using best_params), then explain test data with it.
final_base_model = XGBClassifier(**best_params)
final_base_model.fit(X_sparse, y)  # train on all training data
test_pred_proba = final_base_model.predict_proba(X_test_sparse)[:, 1]
# Compute SHAP values for test data using the final base model
explainer_final = shap.TreeExplainer(final_base_model)
X_test_dense = X_test_sparse.toarray()
test_shap_values = explainer_final.shap_values(X_test_dense)
if isinstance(test_shap_values, list):
    test_shap_values = test_shap_values[1]  # take positive-class SHAP values if list is returned
# Convert to sparse and augment test features
test_preds_feat = csr_matrix(test_pred_proba.reshape(-1, 1))
test_shap_feat = csr_matrix(test_shap_values)
X_test_aug_P      = hstack([X_test_sparse, test_preds_feat])
X_test_aug_SHAP   = hstack([X_test_sparse, test_shap_feat])
X_test_aug_P_SHAP = hstack([X_test_sparse, test_preds_feat, test_shap_feat])

# Make final predictions using the second-stage models
pred_labels_P      = model_P.predict(X_test_aug_P)
pred_labels_SHAP   = model_SHAP.predict(X_test_aug_SHAP)
pred_labels_P_SHAP = model_P_SHAP.predict(X_test_aug_P_SHAP)

print("Test Texts:", test_texts)
print("Predictions with P only model:", pred_labels_P)
print("Predictions with SHAP only model:", pred_labels_SHAP)
print("Predictions with P+SHAP model (SFA):", pred_labels_P_SHAP)
