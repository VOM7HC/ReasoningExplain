import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import shap
from xgboost import XGBRegressor

# *** 1. Prepare text data and initial TF-IDF features (example for text regression) ***
# For demonstration - replace with your actual regression dataset
texts = [
    "I loved the movie, it was fantastic and thrilling",
    "The film was boring and too long",
    "What an amazing experience, would watch again!",
    "Terrible movie. Waste of time.",
    "Pretty good overall, some great scenes",
    "Mediocre at best, nothing special",
    "Absolutely brilliant cinematography and acting",
    "Could have been better, disappointed",
]
# Example continuous target values (e.g., ratings from 0-10)
labels = np.array([9.2, 2.5, 8.7, 1.3, 6.5, 4.2, 9.8, 3.1])

# Convert text to TF-IDF features (sparse matrix)
vectorizer = TfidfVectorizer(max_features=10000)
X_sparse = vectorizer.fit_transform(texts)
y = labels

# *** 2. Hyperparameter tuning for regression using Optuna ***
X_tune, X_valid, y_tune, y_valid = train_test_split(
    X_sparse, y, test_size=0.2, random_state=42
)

def objective(trial):
    """Objective function for regression hyperparameter tuning"""
    params = {
        # Regression-specific parameters
        "objective": "reg:squarederror",  # Changed from binary:logistic
        "eval_metric": "rmse",            # Changed from logloss
        
        # Hyperparameter search space
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "tree_method": "auto"
    }
    
    # Convert sparse matrices to dense for this small example
    X_tune_dense = X_tune.toarray()
    X_valid_dense = X_valid.toarray()
    
    # Train regression model
    model = XGBRegressor(**params, random_state=42)
    model.fit(
        X_tune_dense, y_tune,
        eval_set=[(X_valid_dense, y_valid)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Evaluate using RMSE (Root Mean Squared Error)
    preds = model.predict(X_valid_dense)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    
    return rmse  # Optuna will minimize this

# Run Optuna optimization
study = optuna.create_study(direction="minimize")  # Minimize RMSE
study.optimize(objective, n_trials=20, show_progress_bar=False)
best_params = study.best_params
print(f"Best regression params found: {best_params}")
print(f"Best RMSE: {study.best_value:.4f}")

# *** 3. Train base model with best hyperparameters using K-Fold for OOF predictions ***
best_params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
base_model = XGBRegressor(**best_params, random_state=42)

# Prepare arrays for out-of-fold predictions and SHAP values
n_samples, n_features = X_sparse.shape
oof_preds = np.zeros(n_samples)  # Single array for regression predictions
oof_shap = np.zeros((n_samples, n_features))

# Use KFold instead of StratifiedKFold (no stratification needed for regression)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_sparse), 1):
    X_train_fold = X_sparse[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X_sparse[val_idx]
    y_val_fold = y[val_idx]
    
    # Train base model on this fold's training data
    model_fold = XGBRegressor(**best_params, random_state=42)
    model_fold.fit(X_train_fold, y_train_fold)
    
    # OOF predictions - direct continuous values (no probabilities)
    val_pred = model_fold.predict(X_val_fold)
    oof_preds[val_idx] = val_pred
    
    # Compute SHAP values for validation fold
    explainer = shap.TreeExplainer(model_fold)
    X_val_fold_dense = X_val_fold.toarray()
    
    # For regression, shap_values returns a single array (not a list)
    shap_values_fold = explainer.shap_values(X_val_fold_dense)
    oof_shap[val_idx, :] = shap_values_fold
    
    # Evaluate fold performance using regression metrics
    fold_rmse = mean_squared_error(y_val_fold, val_pred, squared=False)
    fold_mae = mean_absolute_error(y_val_fold, val_pred)
    fold_r2 = r2_score(y_val_fold, val_pred)
    
    print(f"Fold {fold}: RMSE={fold_rmse:.3f}, MAE={fold_mae:.3f}, R²={fold_r2:.3f}")

# Overall OOF performance
overall_rmse = mean_squared_error(y, oof_preds, squared=False)
overall_mae = mean_absolute_error(y, oof_preds)
overall_r2 = r2_score(y, oof_preds)
print(f"\nOverall OOF: RMSE={overall_rmse:.3f}, MAE={overall_mae:.3f}, R²={overall_r2:.3f}")

# *** 4. Augment original features with OOF predictions and SHAP values ***
# Convert OOF arrays to sparse format for efficient stacking
oof_preds_feat = csr_matrix(oof_preds.reshape(-1, 1))  # Shape (n_samples, 1)
oof_shap_feat = csr_matrix(oof_shap)                  # Shape (n_samples, n_features)

# Create augmented feature matrices
X_train_aug_P = hstack([X_sparse, oof_preds_feat])         # original + predictions
X_train_aug_SHAP = hstack([X_sparse, oof_shap_feat])       # original + SHAP
X_train_aug_P_SHAP = hstack([X_sparse, oof_preds_feat, oof_shap_feat])  # all

print(f"\nFeature dimensions:")
print(f"Original features: {X_sparse.shape[1]}")
print(f"Augmented (P): {X_train_aug_P.shape[1]}")
print(f"Augmented (SHAP): {X_train_aug_SHAP.shape[1]}")
print(f"Augmented (P+SHAP): {X_train_aug_P_SHAP.shape[1]}")

# *** 5. Train second-stage regression models on augmented features ***
model_P = XGBRegressor(**best_params, random_state=42)
model_SHAP = XGBRegressor(**best_params, random_state=42)
model_P_SHAP = XGBRegressor(**best_params, random_state=42)

model_P.fit(X_train_aug_P, y)
model_SHAP.fit(X_train_aug_SHAP, y)
model_P_SHAP.fit(X_train_aug_P_SHAP, y)

# *** 6. Inference on new test data ***
test_texts = [
    "An absolutely fantastic film with great acting",
    "Not my taste, I found it dull and uninspired",
    "Decent movie with some good moments",
]
X_test_sparse = vectorizer.transform(test_texts)

# Train final base model on all training data
final_base_model = XGBRegressor(**best_params, random_state=42)
final_base_model.fit(X_sparse, y)

# Generate predictions and SHAP values for test data
test_pred = final_base_model.predict(X_test_sparse)

# Compute SHAP values for test data
explainer_final = shap.TreeExplainer(final_base_model)
X_test_dense = X_test_sparse.toarray()
test_shap_values = explainer_final.shap_values(X_test_dense)

# Augment test features
test_preds_feat = csr_matrix(test_pred.reshape(-1, 1))
test_shap_feat = csr_matrix(test_shap_values)
X_test_aug_P = hstack([X_test_sparse, test_preds_feat])
X_test_aug_SHAP = hstack([X_test_sparse, test_shap_feat])
X_test_aug_P_SHAP = hstack([X_test_sparse, test_preds_feat, test_shap_feat])

# Make predictions using second-stage models
pred_base = test_pred
pred_P = model_P.predict(X_test_aug_P)
pred_SHAP = model_SHAP.predict(X_test_aug_SHAP)
pred_P_SHAP = model_P_SHAP.predict(X_test_aug_P_SHAP)

# *** 7. Create final SFA ensemble predictions ***
# Simple arithmetic average for regression
sfa_predictions = (pred_base + pred_P + pred_SHAP + pred_P_SHAP) / 4

# Alternative: Weighted average based on validation performance
# weights = [0.2, 0.3, 0.25, 0.25]  # Tune these based on validation
# sfa_predictions = np.average([pred_base, pred_P, pred_SHAP, pred_P_SHAP], 
#                              axis=0, weights=weights)

print("\n" + "="*50)
print("TEST PREDICTIONS (Continuous Values)")
print("="*50)
for i, text in enumerate(test_texts):
    print(f"\nText: '{text[:50]}...'")
    print(f"Base model:     {pred_base[i]:.2f}")
    print(f"P augmented:    {pred_P[i]:.2f}")
    print(f"SHAP augmented: {pred_SHAP[i]:.2f}")
    print(f"P+SHAP:         {pred_P_SHAP[i]:.2f}")
    print(f"SFA Ensemble:   {sfa_predictions[i]:.2f}")

# *** 8. Helper functions for production use ***

def train_sfa_regression(X_train, y_train, best_params, n_folds=5):
    """Train the complete SFA regression pipeline"""
    n_samples, n_features = X_train.shape
    oof_preds = np.zeros(n_samples)
    oof_shap = np.zeros((n_samples, n_features))
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X_train):
        model = XGBRegressor(**best_params, random_state=42)
        model.fit(X_train[train_idx], y_train[train_idx])
        
        oof_preds[val_idx] = model.predict(X_train[val_idx])
        
        explainer = shap.TreeExplainer(model)
        X_val_dense = X_train[val_idx].toarray() if hasattr(X_train[val_idx], 'toarray') else X_train[val_idx]
        oof_shap[val_idx] = explainer.shap_values(X_val_dense)
    
    # Create augmented features
    oof_preds_feat = csr_matrix(oof_preds.reshape(-1, 1))
    oof_shap_feat = csr_matrix(oof_shap)
    
    X_aug_P = hstack([X_train, oof_preds_feat])
    X_aug_SHAP = hstack([X_train, oof_shap_feat])
    X_aug_P_SHAP = hstack([X_train, oof_preds_feat, oof_shap_feat])
    
    # Train second-stage models
    models = {
        'base': XGBRegressor(**best_params, random_state=42),
        'P': XGBRegressor(**best_params, random_state=42),
        'SHAP': XGBRegressor(**best_params, random_state=42),
        'P_SHAP': XGBRegressor(**best_params, random_state=42)
    }
    
    models['base'].fit(X_train, y_train)
    models['P'].fit(X_aug_P, y_train)
    models['SHAP'].fit(X_aug_SHAP, y_train)
    models['P_SHAP'].fit(X_aug_P_SHAP, y_train)
    
    return models, oof_preds, oof_shap

def predict_sfa_regression(models, X_test, vectorizer=None):
    """Make predictions using trained SFA models"""
    if vectorizer:
        X_test = vectorizer.transform(X_test)
    
    # Base predictions
    base_pred = models['base'].predict(X_test)
    
    # Generate SHAP values
    explainer = shap.TreeExplainer(models['base'])
    X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
    shap_values = explainer.shap_values(X_test_dense)
    
    # Augment features
    pred_feat = csr_matrix(base_pred.reshape(-1, 1))
    shap_feat = csr_matrix(shap_values)
    
    X_aug_P = hstack([X_test, pred_feat])
    X_aug_SHAP = hstack([X_test, shap_feat])
    X_aug_P_SHAP = hstack([X_test, pred_feat, shap_feat])
    
    # Get predictions from all models
    predictions = {
        'base': base_pred,
        'P': models['P'].predict(X_aug_P),
        'SHAP': models['SHAP'].predict(X_aug_SHAP),
        'P_SHAP': models['P_SHAP'].predict(X_aug_P_SHAP)
    }
    
    # Ensemble prediction (simple average)
    ensemble_pred = np.mean([predictions[k] for k in predictions], axis=0)
    
    return ensemble_pred, predictions

# *** 9. Evaluation metrics for regression ***
def evaluate_regression_models(y_true, predictions_dict):
    """Evaluate multiple regression models"""
    results = {}
    
    for name, y_pred in predictions_dict.items():
        results[name] = {
            'RMSE': mean_squared_error(y_true, y_pred, squared=False),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    
    return results