# base.py

from typing import Optional, List, Dict, Tuple, Any, Set, Callable
import numpy as np
import pandas as pd
import os
import json
import requests
from os import getenv
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import random
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def default_output_handler(message: str) -> None:
    """Prints messages without newline."""
    print(message, end='', flush=True)

def interact_with_ollama(
    prompt: Optional[str] = None, 
    messages: Optional[Any] = None,
    image_path: Optional[str] = None,
    model: str = 'openhermes2.5-mistral', 
    stream: bool = False, 
    output_handler: Callable[[str], None] = default_output_handler,
    api_url: Optional[str] = None
) -> str:
    """
    Sends a request to the Ollama API for chat or image generation tasks.
    
    Args:
    prompt (Optional[str]): Text prompt for generating content.
    messages (Optional[Any]): Structured chat messages for dialogues.
    image_path (Optional[str]): Path to an image for image-related operations.
    model (str): Model identifier for the API.
    stream (bool): If True, keeps the connection open for streaming responses.
    output_handler (Callable[[str], None]): Function to handle output messages.
    api_url (Optional[str]): API endpoint URL; if not provided, fetched from environment.

    Returns:
    str: Full response from the API.

    Raises:
    ValueError: If necessary parameters are not correctly provided or API URL is missing.
    """
    if not api_url:
        api_url = getenv('API_URL')
        if not api_url:
            raise ValueError('API_URL is not set. Provide it via the api_url variable or as an environment variable.')

    # Define endpoint based on whether it's a chat or a single generate request
    api_endpoint = f"{api_url}/api/{'chat' if messages else 'generate'}"
    data = {'model': model, 'stream': stream}

    # Populate the request data
    if messages:
        data['messages'] = messages
    elif prompt:
        data['prompt'] = prompt
    else:
        raise ValueError("Either messages for chat or a prompt for generate must be provided.")

    # Send request
    response = requests.post(api_endpoint, json=data, stream=stream)
    if response.status_code != 200:
        output_handler(f"Failed to retrieve data: {response.status_code}")
        return ""

    # Handle streaming or non-streaming responses
    full_response = ""
    if stream:
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line.decode('utf-8'))
                response_part = json_response.get('response', '') or json_response.get('message', {}).get('content', '')
                full_response += response_part
                output_handler(response_part)
                if json_response.get('done', False):
                    break
    else:
        json_response = response.json()
        response_part = json_response.get('response', '') or json_response.get('message', {}).get('content', '')
        full_response += response_part
        output_handler(response_part)

    return full_response, response

def get_text_before_last_underscore(token):
    return token.rsplit('_', 1)[0]

class TextVectorizer:
    """Base class for text vectorization"""
    
    def vectorize(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
        
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ModelBase(ABC):
    """Base class for all models (text and vision)"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.client = None
        
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from model"""
        pass

class TfidfTextVectorizer(TextVectorizer):
    def __init__(self):
        self.vectorizer = None
        
    def vectorize(self, texts: List[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer().fit(texts)
        return self.vectorizer.transform(texts).toarray()
        
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        return cosine_similarity(
            base_vector.reshape(1, -1), comparison_vectors
        ).flatten()

class EmbeddingVectorizer(TextVectorizer):
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
    def vectorize(self, texts: List[str]) -> np.ndarray:
        # Encode texts into semantic embeddings
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        # Use cosine similarity on embeddings
        sims = cosine_similarity(base_vector.reshape(1, -1), comparison_vectors).flatten()
        return sims


class OllamaModel(ModelBase):
    """Ollama model implementation supporting both text and vision"""
    
    def __init__(self, model_name: str, api_url: str):
        super().__init__(model_name, api_url=api_url)
    
    def generate(self, prompt: str) -> str:
        # Note: Implement the interact_with_ollama function from your original code
        # Add PLACEHOLDER comment to keep existing implementation
        text_response, _ = interact_with_ollama(
            model=self.model_name,
            prompt=prompt,
            api_url=self.api_url,
            output_handler=lambda o: o
        )
        return text_response

class BaseSHAP(ABC):
    """Base class for SHAP implementations"""
    
    def __init__(self, 
                 model: ModelBase,
                 vectorizer: Optional[TextVectorizer] = None,
                 debug: bool = False):
        self.model = model
        self.vectorizer = vectorizer
        self.debug = debug
        self.results_df = None
        self.shapley_values = None

    def _debug_print(self, message: str) -> None:
        """Print debug messages if debug mode is enabled"""
        if self.debug:
            print(message)

    def _calculate_baseline(self, content: Any, **kwargs) -> str:
        """Calculate baseline model response"""
        return self.model.generate(**self._prepare_generate_args(content, **kwargs))

    @abstractmethod
    def _prepare_generate_args(self, content: Any, **kwargs) -> Dict:
        """Prepare arguments for model.generate()"""
        pass

    def _generate_random_combinations(self, 
                                    samples: List[Any], 
                                    k: int, 
                                    exclude_combinations_set: Set[Tuple[int, ...]]) -> List[Tuple[List, Tuple[int, ...]]]:
        """
        Generate random combinations efficiently using binary representation
        """
        n = len(samples)
        sampled_combinations_set = set()
        max_attempts = k * 10  # Prevent infinite loops in case of duplicates
        attempts = 0

        while len(sampled_combinations_set) < k and attempts < max_attempts:
            attempts += 1
            rand_int = random.randint(1, 2 ** n - 2)
            bin_str = bin(rand_int)[2:].zfill(n)
            combination = [samples[i] for i in range(n) if bin_str[i] == '1']
            indexes = tuple([i + 1 for i in range(n) if bin_str[i] == '1'])
            if indexes not in exclude_combinations_set and indexes not in sampled_combinations_set:
                sampled_combinations_set.add((tuple(combination), indexes))

        if len(sampled_combinations_set) < k:
            self._debug_print(f"Warning: Could only generate {len(sampled_combinations_set)} unique combinations out of requested {k}")
        return list(sampled_combinations_set)

    def _get_result_per_combination(self, 
                                content: Any, 
                                sampling_ratio: float,
                                max_combinations: Optional[int] = 1000) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
        """
        Get model responses for combinations
        
        Args:
            content: Content to analyze
            sampling_ratio: Ratio of non-essential combinations to sample (0-1)
            max_combinations: Maximum number of combinations (must be >= n for n tokens)
        """
        samples = self._get_samples(content)
        n = len(samples)
        self._debug_print(f"Number of samples: {n}")
        if n > 1000:
            print("Warning: the number of samples is greater than 1000; execution will be slow.")

        # Always start with essential combinations (each missing one sample)
        essential_combinations = []
        essential_combinations_set = set()
        for i in range(n):
            combination = samples[:i] + samples[i + 1:]
            indexes = tuple([j + 1 for j in range(n) if j != i])
            essential_combinations.append((combination, indexes))
            essential_combinations_set.add(indexes)
        
        num_essential = len(essential_combinations)
        self._debug_print(f"Number of essential combinations: {num_essential}")
        if max_combinations is not None and max_combinations < num_essential:
            print(f"Warning: max_combinations ({max_combinations}) is less than the number of essential combinations "
                  f"({num_essential}). Will use all essential combinations despite the limit.")
            self._debug_print("No additional combinations will be added.")
            max_combinations = num_essential
        # Calculate how many additional combinations we can/should generate
        remaining_budget = float('inf')
        if max_combinations is not None:
            remaining_budget = max(0, max_combinations - num_essential)
            self._debug_print(f"Remaining combinations budget after essentials: {remaining_budget}")

        # If using sampling ratio, calculate possible additional combinations without generating them
        if sampling_ratio < 1.0:
            # Get theoretical number of total combinations
            theoretical_total = 2 ** n - 1
            theoretical_additional = theoretical_total - num_essential
            # Calculate desired number based on ratio
            desired_additional = int(theoretical_additional * sampling_ratio)
            # Take minimum of sampling ratio and max_combinations limits
            num_additional = min(desired_additional, remaining_budget)
        else:
            num_additional = remaining_budget

        num_additional = int(num_additional)  # Ensure integer
        self._debug_print(f"Number of additional combinations to sample: {num_additional}")

        # Generate additional random combinations if needed
        additional_combinations = []
        if num_additional > 0:
            additional_combinations = self._generate_random_combinations(
                samples, num_additional, essential_combinations_set
            )
            self._debug_print(f"Number of sampled combinations: {len(additional_combinations)}")
        else:
            self._debug_print("No additional combinations to sample.")

        # Process all combinations
        all_combinations = essential_combinations + additional_combinations
        self._debug_print(f"Total combinations to process: {len(all_combinations)}")

        responses = {}
        for idx, (combination, indexes) in enumerate(tqdm(all_combinations, desc="Processing combinations")):
            self._debug_print(f"\nProcessing combination {idx + 1}/{len(all_combinations)}:")
            self._debug_print(f"Combination: {combination}")
            self._debug_print(f"Indexes: {indexes}")

            args = self._prepare_combination_args(combination, content)
            response = self.model.generate(**args)
            self._debug_print(f"Received response for combination {idx + 1}")

            key = self._get_combination_key(combination, indexes)
            responses[key] = (response, indexes)

        return responses

    def _get_df_per_combination(self, responses: Dict[str, Tuple[str, Tuple[int, ...]]], baseline_text: str) -> pd.DataFrame:
        """Create DataFrame with combination results"""
        df = pd.DataFrame(
            [(key.split('_')[0], response[0], response[1])
             for key, response in responses.items()],
            columns=['Content', 'Response', 'Indexes']
        )

        all_texts = [baseline_text] + df["Response"].tolist()
        vectors = self.vectorizer.vectorize(all_texts)
        base_vector = vectors[0]
        comparison_vectors = vectors[1:]
        
        similarities = self.vectorizer.calculate_similarity(base_vector, comparison_vectors)
        df["Similarity"] = similarities

        return df

    def _calculate_shapley_values(self, df: pd.DataFrame, content: Any) -> Dict[str, float]:
        """
        Compute token-level Shapley values using a two-stage ensemble learning approach (SFA method):
        - Stage 1: Train a first-stage model via K-fold CV on binary coalition vectors (token subsets) with 
          `Similarity` as the target. Collect out-of-fold (OOF) predictions and TreeSHAP values for each coalition.
        - Stage 2: Augment original features with OOF predictions (P) and with SHAP values (SHAP) to form three 
          feature sets: original+P, original+SHAP, original+P+SHAP. Train an XGBoost regressor on each augmented set.
        - For each second-stage model (P, SHAP, P+SHAP), compute SHAP values on the training data and average the 
          token-level contributions across all models.
        - Hyperparameters for all models (base and second-stage) are tuned with Optuna (e.g., 20 trials per model for speed).
        - Returns a normalized dict of token Shapley values (sum to 1), similar to the original output format.
        """
        import numpy as np
        import pandas as pd
        from typing import Any, Dict
        from sklearn.model_selection import KFold, train_test_split
        from sklearn.metrics import mean_squared_error
        import optuna
        import shap
        from xgboost import XGBRegressor

        # Get token list and number of features (tokens)
        samples = self._get_samples(content)  # list of tokens in fixed order
        M = len(samples)

        # Safety checks on DataFrame
        if "Indexes" not in df or "Similarity" not in df:
            raise ValueError('df must have columns "Indexes" (list[int]) and "Similarity" (float).')

        # Build design matrix X (binary presence/absence per coalition) and target y
        def to_binary(indexes):
            mask = np.zeros(M, dtype=int)
            for idx in (indexes or []):
                if 1 <= int(idx) <= M:
                    mask[int(idx) - 1] = 1
            return mask

        X = np.vstack(df["Indexes"].apply(to_binary).values)   # shape [n_samples, M]
        y = df["Similarity"].astype(float).values             # shape [n_samples]

        n_samples = X.shape[0]

        # Helper function for hyperparameter tuning with Optuna
        def tune_model(X_data, y_data, n_trials=20):
            # Split data for tuning (20% validation)
            X_tune, X_val, y_tune, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
            # Convert to dense if needed (for small datasets, dense is fine)
            X_tune_dense = X_tune.toarray() if hasattr(X_tune, 'toarray') else X_tune
            X_val_dense = X_val.toarray() if hasattr(X_val, 'toarray') else X_val

            def objective(trial):
                params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "tree_method": "auto"
                }
                model = XGBRegressor(**params, random_state=42)
                model.fit(
                    X_tune_dense, y_tune,
                    eval_set=[(X_val_dense, y_val)],
                    verbose=False
                )
                preds = model.predict(X_val_dense)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                return rmse

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params
            # Ensure required params are set
            best_params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})
            return best_params

        # --- Stage 1: Tune and train base model with KFold for OOF predictions ---
        best_params_base = tune_model(X, y, n_trials=20)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(n_samples)
        oof_shap = np.zeros((n_samples, M))
        for train_idx, val_idx in kf.split(X):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]
            model_fold = XGBRegressor(**best_params_base, random_state=42)
            model_fold.fit(X_train_fold, y_train_fold, verbose=False)
            # Out-of-fold predictions
            oof_preds[val_idx] = model_fold.predict(X_val_fold)
            # Compute SHAP values for validation fold
            explainer_fold = shap.TreeExplainer(model_fold)
            X_val_dense = X_val_fold if not hasattr(X_val_fold, 'toarray') else X_val_fold.toarray()
            # shap_values returns a single array for regression
            shap_values_fold = explainer_fold.shap_values(X_val_dense)
            oof_shap[val_idx, :] = shap_values_fold

        # --- Stage 2: Augment features with OOF predictions and SHAP, then tune/train second-stage models ---
        # Create augmented feature matrices
        # (use scipy.sparse if memory is a concern; here using dense for simplicity)
        oof_preds_feat = oof_preds.reshape(-1, 1)                    # shape (n_samples, 1)
        X_aug_P = np.hstack([X, oof_preds_feat])                     # original features + OOF prediction
        X_aug_SHAP = np.hstack([X, oof_shap])                        # original + SHAP values
        X_aug_P_SHAP = np.hstack([X, oof_preds_feat, oof_shap])      # original + OOF pred + SHAP values

        # Tune hyperparameters for each second-stage model
        best_params_P = tune_model(X_aug_P, y, n_trials=20)
        best_params_SHAP = tune_model(X_aug_SHAP, y, n_trials=20)
        best_params_P_SHAP = tune_model(X_aug_P_SHAP, y, n_trials=20)

        # Train second-stage models on full training data with tuned params
        model_P = XGBRegressor(**best_params_P, random_state=42)
        model_SHAP = XGBRegressor(**best_params_SHAP, random_state=42)
        model_P_SHAP = XGBRegressor(**best_params_P_SHAP, random_state=42)
        model_P.fit(X_aug_P, y, verbose=False)
        model_SHAP.fit(X_aug_SHAP, y, verbose=False)
        model_P_SHAP.fit(X_aug_P_SHAP, y, verbose=False)

        # --- Compute SHAP values for each second-stage model on the training set ---
        expl_P = shap.TreeExplainer(model_P)
        expl_SHAP = shap.TreeExplainer(model_SHAP)
        expl_P_SHAP = shap.TreeExplainer(model_P_SHAP)
        X_train_dense = X  # original training features (already dense numpy array)
        shap_vals_P = expl_P.shap_values(np.hstack([X_train_dense, oof_preds_feat]))
        shap_vals_SHAP = expl_SHAP.shap_values(np.hstack([X_train_dense, oof_shap]))
        shap_vals_P_SHAP = expl_P_SHAP.shap_values(np.hstack([X_train_dense, oof_preds_feat, oof_shap]))

        # Each shap_vals_* has shape (n_samples, num_features_of_model)
        # We only care about the contributions for original token features (the first M columns in each augmented set)
        phi_mean_P = np.array(shap_vals_P)[:, :M].mean(axis=0)
        phi_mean_SHAP = np.array(shap_vals_SHAP)[:, :M].mean(axis=0)
        phi_mean_P_SHAP = np.array(shap_vals_P_SHAP)[:, :M].mean(axis=0)
        # Average token contributions across the three models
        phi_mean_ensemble = (phi_mean_P + phi_mean_SHAP + phi_mean_P_SHAP) / 3.0

        # Map back to token keys (e.g., "token_1") and normalize the values
        shapley_values_raw = {f"{samples[i]}_{i+1}": float(phi_mean_ensemble[i]) for i in range(M)}

        def normalize_shapley_values(values: Dict[str, float], power: float = 1.0) -> Dict[str, float]:
            min_value = min(values.values()) if values else 0.0
            shifted = {k: v - min_value for k, v in values.items()}
            powered = {k: (v ** power) for k, v in shifted.items()}
            tot = sum(powered.values())
            if tot == 0:
                n = max(len(powered), 1)
                return {k: 1.0 / n for k in powered}
            return {k: v / tot for k, v in powered.items()}

        return normalize_shapley_values(shapley_values_raw)

    @abstractmethod
    def _get_samples(self, content: Any) -> List[Any]:
        """Get samples from content for analysis"""
        pass

    @abstractmethod
    def _prepare_combination_args(self, combination: List[Any], original_content: Any) -> Dict:
        """Prepare model arguments for a combination"""
        pass

    @abstractmethod
    def _get_combination_key(self, combination: List[Any], indexes: Tuple[int, ...]) -> str:
        """Get unique key for combination"""
        pass

    def save_results(self, output_dir: str, metadata: Optional[Dict] = None) -> None:
        """Save analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.results_df is not None:
            self.results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
            
        if self.shapley_values is not None:
            with open(os.path.join(output_dir, "shapley_values.json"), 'w') as f:
                json.dump(self.shapley_values, f, indent=2)
                
        if metadata:
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)