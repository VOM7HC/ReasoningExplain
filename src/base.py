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

def default_output_handler(message: str) -> None:
    """Prints messages without newline."""
    print(message, end='', flush=True)

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
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

class OllamaModel(ModelBase):
    """Ollama model implementation supporting both text and vision"""
    
    def __init__(self, model_name: str, api_url: str):
        super().__init__(model_name, api_url=api_url)
    
    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        # Note: Implement the interact_with_ollama function from your original code
        # Add PLACEHOLDER comment to keep existing implementation
        text_response, _ = interact_with_ollama(
            model=self.model_name,
            prompt=prompt,
            image_path=image_path,
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
        Compute approximate Shapley values using a stacked ensemble surrogate model.
        This replaces the single TreeSHAP surrogate with an ensemble of regressors:
        - A base regressor on binary token coalition vectors
        - Augmented regressors using Out-Of-Fold predictions (P) and SHAP values (SHAP)
        - A final ensemble prediction (average of base, P, SHAP, and P+SHAP models)
        Returns a normalized dict of token importances (sum to 1).
        """
        import numpy as np
        import pandas as pd
        from typing import Any, Dict
        from sklearn.model_selection import KFold
        from xgboost import XGBRegressor
        import shap
        from scipy.sparse import csr_matrix, hstack

        samples = self._get_samples(content)  # list of tokens or features
        M = len(samples)
        # ---- safety checks
        if "Indexes" not in df or "Similarity" not in df:
            raise ValueError('df must have columns "Indexes" (list[int]) and "Similarity" (float).')

        # ---- build design matrix X (binary presence/absence for each coalition) and target y
        def to_binary(indexes):
            mask = np.zeros(M, dtype=int)
            for idx in (indexes or []):
                if 1 <= int(idx) <= M:
                    mask[int(idx) - 1] = 1
            return mask

        X = np.vstack(df["Indexes"].apply(to_binary).values)   # shape [n_coalitions, M]
        y = df["Similarity"].astype(float).values              # shape [n_coalitions]

        # ---- 1. Train base regressor with K-Fold to get OOF predictions and SHAP values
        n_samples = X.shape[0]
        oof_preds = np.zeros(n_samples)
        oof_shap = np.zeros((n_samples, M))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            # Train base model on training fold
            model_fold = XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05,
                                      random_state=42, objective="reg:squarederror")
            model_fold.fit(X[train_idx], y[train_idx])
            # Store out-of-fold predictions for validation fold
            oof_preds[val_idx] = model_fold.predict(X[val_idx])
            # Compute SHAP values for validation fold (using TreeExplainer on the fold model)
            explainer_fold = shap.TreeExplainer(model_fold)
            X_val = X[val_idx]
            X_val_dense = X_val if isinstance(X_val, np.ndarray) else X_val.toarray()
            shap_vals = explainer_fold.shap_values(X_val_dense)
            # shap_values returns a list for multi-output models; for regression we get a single array
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            oof_shap[val_idx, :] = shap_vals

        # ---- 2. Augment original features with OOF predictions (P) and OOF SHAP values (SHAP)
        X_sparse = csr_matrix(X)  # convert to sparse for efficient horizontal stacking
        oof_preds_feat = csr_matrix(oof_preds.reshape(-1, 1))  # shape (n_samples, 1)
        oof_shap_feat = csr_matrix(oof_shap)                  # shape (n_samples, M)
        X_aug_P = hstack([X_sparse, oof_preds_feat])                   # original + OOF prediction
        X_aug_SHAP = hstack([X_sparse, oof_shap_feat])                 # original + OOF SHAP values
        X_aug_P_SHAP = hstack([X_sparse, oof_preds_feat, oof_shap_feat])  # original + pred + SHAP

        # ---- 3. Train ensemble regressors on augmented features (and base on original)
        model_base = XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05,
                                   random_state=42, objective="reg:squarederror")
        model_P = XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05,
                                random_state=42, objective="reg:squarederror")
        model_SHAP = XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05,
                                   random_state=42, objective="reg:squarederror")
        model_P_SHAP = XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05,
                                     random_state=42, objective="reg:squarederror")
        model_base.fit(X, y)
        model_P.fit(X_aug_P, y)
        model_SHAP.fit(X_aug_SHAP, y)
        model_P_SHAP.fit(X_aug_P_SHAP, y)

        # ---- 4. Compute SHAP values for each observed coalition using all models
        explainer_base = shap.TreeExplainer(model_base)
        explainer_P = shap.TreeExplainer(model_P)
        explainer_SHAP = shap.TreeExplainer(model_SHAP)
        explainer_P_SHAP = shap.TreeExplainer(model_P_SHAP)
        # Convert to dense for SHAP (if any sparse matrices)
        X_dense = X if isinstance(X, np.ndarray) else X.toarray()
        X_P_dense = X_aug_P.toarray() if not isinstance(X_aug_P, np.ndarray) else X_aug_P
        X_SHAP_dense = X_aug_SHAP.toarray() if not isinstance(X_aug_SHAP, np.ndarray) else X_aug_SHAP
        X_P_SHAP_dense = X_aug_P_SHAP.toarray() if not isinstance(X_aug_P_SHAP, np.ndarray) else X_aug_P_SHAP
        shap_vals_base = explainer_base.shap_values(X_dense)
        shap_vals_P = explainer_P.shap_values(X_P_dense)
        shap_vals_SHAP = explainer_SHAP.shap_values(X_SHAP_dense)
        shap_vals_P_SHAP = explainer_P_SHAP.shap_values(X_P_SHAP_dense)
        # Ensure outputs are numpy arrays (for single-output XGB, shap returns an array directly)
        shap_vals_base = np.array(shap_vals_base[0] if isinstance(shap_vals_base, list) else shap_vals_base)
        shap_vals_P = np.array(shap_vals_P[0] if isinstance(shap_vals_P, list) else shap_vals_P)
        shap_vals_SHAP = np.array(shap_vals_SHAP[0] if isinstance(shap_vals_SHAP, list) else shap_vals_SHAP)
        shap_vals_P_SHAP = np.array(shap_vals_P_SHAP[0] if isinstance(shap_vals_P_SHAP, list) else shap_vals_P_SHAP)

        # ---- 5. Aggregate SHAP values from ensemble (average contributions of original features)
        phi_base = shap_vals_base.mean(axis=0)               # shap contributions from base model
        phi_P = shap_vals_P[:, :M].mean(axis=0)              # contributions from original features in P model
        phi_SHAP = shap_vals_SHAP[:, :M].mean(axis=0)        # contributions from original features in SHAP model
        phi_P_SHAP = shap_vals_P_SHAP[:, :M].mean(axis=0)    # contributions from original features in P+SHAP model
        phi_mean = (phi_base + phi_P + phi_SHAP + phi_P_SHAP) / 4.0  # ensemble average contribution

        # ---- 6. Normalize Shapley values (min shift, optional power, then L1 normalize to sum=1)
        def normalize_shapley_values(values: Dict[str, float], power: float = 1.0) -> Dict[str, float]:
            min_value = min(values.values()) if values else 0.0
            shifted = {k: v - min_value for k, v in values.items()}
            powered = {k: (v ** power) for k, v in shifted.items()}
            tot = sum(powered.values())
            if tot == 0:
                n = max(len(powered), 1)
                return {k: 1.0 / n for k, v in powered.items()}
            return {k: v / tot for k, v in powered.items()}

        shapley_values_raw = {f"{samples[i]}_{i+1}": float(phi_mean[i]) for i in range(M)}
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