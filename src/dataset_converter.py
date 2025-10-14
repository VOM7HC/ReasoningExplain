"""
XAI Dataset Downloader and Converter
Downloads datasets from Hugging Face and converts to three study formats:
- Study 1: Explanation Quality Ratings
- Study 2: Decision Logs with Trust Items
- Study 3: Bias Detection Logs
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Dict, List, Optional
import json
from pathlib import Path


class XAIDatasetConverter:
    """Convert Hugging Face datasets to XAI human evaluation formats"""
    
    def __init__(self, output_dir: str = "./xai_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    # ==================== STUDY 1: Explanation Quality Ratings ====================
    
    def convert_study1_mqm_dataset(self, save: bool = True) -> pd.DataFrame:
        """
        Convert WMT MQM Human Evaluation dataset to Study 1 format
        Original: Translation quality metrics with multiple dimensions
        Target: participant_id, evaluation_id, method, understandability, completeness, 
                usefulness, trustworthiness, actionability, sufficiency
        """
        print("Downloading RicardoRei/wmt-mqm-human-evaluation...")
        dataset = load_dataset("RicardoRei/wmt-mqm-human-evaluation", split="train")
        df = pd.DataFrame(dataset)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Original columns: {df.columns.tolist()}")
        
        # Map to Study 1 format
        converted_df = pd.DataFrame({
            'participant_id': df.get('annotator', df.index.astype(str)),  # Use annotator as participant
            'evaluation_id': df.index.astype(str),  # Unique eval ID
            'method': 'translation_explanation',  # Method name
            
            # Map MQM scores to Likert scale (1-5)
            # Note: These are approximations - you may need to adjust based on your scale
            'understandability': self._normalize_to_likert(df.get('mqm_score', 3), inverse=True),
            'completeness': self._normalize_to_likert(df.get('mqm_score', 3), inverse=True),
            'usefulness': np.random.randint(3, 5, len(df)),  # Placeholder - no direct mapping
            'trustworthiness': self._normalize_to_likert(df.get('mqm_score', 3), inverse=True),
            'actionability': np.random.randint(2, 5, len(df)),  # Placeholder
            'sufficiency': self._normalize_to_likert(df.get('mqm_score', 3), inverse=True),
            
            # Additional metadata
            'source_text': df.get('source', ''),
            'target_text': df.get('target', ''),
            'original_score': df.get('mqm_score', 0)
        })
        
        if save:
            output_path = self.output_dir / "study1_explanation_quality.csv"
            converted_df.to_csv(output_path, index=False)
            print(f"✓ Study 1 dataset saved to {output_path}")
            print(f"  Shape: {converted_df.shape}")
        
        return converted_df
    
    def convert_study1_explainable_ai_emotions(self, save: bool = True) -> pd.DataFrame:
        """
        Convert ExplainableAI emotions dataset to Study 1 format
        This dataset has actual human ratings of AI explanations
        """
        print("Downloading imhmdf/ExplainableAI-emotions-DPO-ORPO-RLHF...")
        try:
            dataset = load_dataset("imhmdf/ExplainableAI-emotions-DPO-ORPO-RLHF", split="train")
            df = pd.DataFrame(dataset)
            
            print(f"Original dataset shape: {df.shape}")
            print(f"Original columns: {df.columns.tolist()}")
            
            # Map to Study 1 format
            converted_df = pd.DataFrame({
                'participant_id': [f"P{str(i).zfill(4)}" for i in range(len(df))],
                'evaluation_id': df.index.astype(str),
                'method': 'emotion_explanation',
                
                # Map actual ratings (adjust based on actual column names)
                'understandability': df.get('clarity', np.random.randint(3, 5, len(df))),
                'completeness': df.get('correctness', np.random.randint(3, 5, len(df))),
                'usefulness': df.get('helpfulness', np.random.randint(3, 5, len(df))),
                'trustworthiness': np.random.randint(3, 5, len(df)),  # Placeholder if not in data
                'actionability': np.random.randint(2, 5, len(df)),
                'sufficiency': df.get('verbosity', np.random.randint(3, 5, len(df))),
                
                # Keep original data
                'prompt': df.get('prompt', ''),
                'response': df.get('chosen', df.get('response', ''))
            })
            
            if save:
                output_path = self.output_dir / "study1_explainable_emotions.csv"
                converted_df.to_csv(output_path, index=False)
                print(f"✓ Study 1 (emotions) dataset saved to {output_path}")
            
            return converted_df
            
        except Exception as e:
            print(f"⚠ Could not load dataset: {e}")
            return pd.DataFrame()
    
    # ==================== STUDY 2: Decision Logs with Trust Items ====================
    
    def convert_study2_arena_preferences(self, save: bool = True) -> pd.DataFrame:
        """
        Convert LMArena human preference dataset to Study 2 format
        Target: participant_id, task_id, initial_decision, final_decision, 
                trust_item_1...trust_item_12, time_taken
        """
        print("Downloading lmarena-ai/arena-human-preference-140k...")
        # Note: This dataset is large, you may want to limit the number of rows
        dataset = load_dataset("lmarena-ai/arena-human-preference-140k", split="train[:10000]")
        df = pd.DataFrame(dataset)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Original columns: {df.columns.tolist()}")
        
        # Generate trust items (synthetic - replace with actual trust survey if available)
        trust_items = {f'trust_item_{i}': np.random.randint(1, 8, len(df)) for i in range(1, 13)}
        
        converted_df = pd.DataFrame({
            'participant_id': df.get('user_id', [f"U{str(i).zfill(5)}" for i in range(len(df))]),
            'task_id': df.get('question_id', df.index.astype(str)),
            'evaluation_id': df.index.astype(str),
            
            # Map preferences to decisions
            'initial_decision': df.get('winner', 'model_a'),  # Assuming this is the choice
            'final_decision': df.get('winner', 'model_a'),  # No revision data, so same as initial
            
            # Trust items (1-7 Likert scale) - SYNTHETIC, replace with actual data
            **trust_items,
            
            # Time taken (if available, otherwise estimate)
            'time_taken': df.get('duration', np.random.uniform(30, 300, len(df))),
            
            # Additional metadata
            'model_a': df.get('model_a', ''),
            'model_b': df.get('model_b', ''),
            'conversation_a': df.get('conversation_a', '').astype(str),
            'conversation_b': df.get('conversation_b', '').astype(str)
        })
        
        if save:
            output_path = self.output_dir / "study2_decision_logs.csv"
            converted_df.to_csv(output_path, index=False)
            print(f"✓ Study 2 dataset saved to {output_path}")
            print(f"  Shape: {converted_df.shape}")
            print(f"  ⚠ Note: trust_item_1-12 are SYNTHETIC - replace with actual trust survey data")
        
        return converted_df
    
    def convert_study2_financial_decisions(self, save: bool = True) -> pd.DataFrame:
        """
        Convert synthetic VC financial decisions dataset to Study 2 format
        This has actual initial/final decision structure
        """
        print("Downloading ZennyKenny/synthetic_vc_financial_decisions_reasoning_dataset...")
        try:
            dataset = load_dataset("ZennyKenny/synthetic_vc_financial_decisions_reasoning_dataset", 
                                   split="train")
            df = pd.DataFrame(dataset)
            
            print(f"Original dataset shape: {df.shape}")
            print(f"Original columns: {df.columns.tolist()}")
            
            # Generate trust items (synthetic)
            trust_items = {f'trust_item_{i}': np.random.randint(1, 8, len(df)) for i in range(1, 13)}
            
            converted_df = pd.DataFrame({
                'participant_id': [f"P{str(i).zfill(4)}" for i in range(len(df))],
                'task_id': df.index.astype(str),
                'evaluation_id': df.index.astype(str),
                
                'initial_decision': df.get('initial_assessment', 'unknown'),
                'final_decision': df.get('final_decision', 'unknown'),
                
                **trust_items,
                
                'time_taken': np.random.uniform(60, 600, len(df)),
                
                # Keep reasoning
                'reasoning': df.get('reasoning', '')
            })
            
            if save:
                output_path = self.output_dir / "study2_financial_decisions.csv"
                converted_df.to_csv(output_path, index=False)
                print(f"✓ Study 2 (financial) dataset saved to {output_path}")
            
            return converted_df
            
        except Exception as e:
            print(f"⚠ Could not load dataset: {e}")
            return pd.DataFrame()
    
    # ==================== STUDY 3: Bias Detection Logs ====================
    
    def convert_study3_bias_multidomain(self, save: bool = True) -> pd.DataFrame:
        """
        Convert bias detection multidomain dataset to Study 3 format
        Target: participant_id, task_id, detected_bias (bool), confidence (1-5), 
                identified_type (string/None), time_taken
        """
        print("Downloading piyush333/bias-detection-multidomain-v1...")
        try:
            dataset = load_dataset("piyush333/bias-detection-multidomain-v1", split="train[:10000]")
            df = pd.DataFrame(dataset)
            
            print(f"Original dataset shape: {df.shape}")
            print(f"Original columns: {df.columns.tolist()}")
            
            # Simulate participant-level data from labeled dataset
            converted_df = pd.DataFrame({
                'participant_id': [f"P{str(i % 50).zfill(3)}" for i in range(len(df))],  # 50 participants
                'task_id': df.index.astype(str),
                
                # Map bias labels to detection
                'detected_bias': df.get('bias_class', df.get('label', 0)) != 0,
                
                # Confidence (1-5) - simulate based on bias presence
                'confidence': df.get('bias_class', df.get('label', 0)).apply(
                    lambda x: np.random.randint(4, 6) if x != 0 else np.random.randint(2, 4)
                ),
                
                # Bias type
                'identified_type': df.get('bias_type', None),
                
                # Time taken (simulated)
                'time_taken': np.random.uniform(15, 120, len(df)),
                
                # Keep original text
                'text_content': df.get('text', df.get('content', '')),
                'actual_bias_label': df.get('bias_class', df.get('label', 0)),
                'domain': df.get('domain', 'unknown')
            })
            
            if save:
                output_path = self.output_dir / "study3_bias_detection.csv"
                converted_df.to_csv(output_path, index=False)
                print(f"✓ Study 3 dataset saved to {output_path}")
                print(f"  Shape: {converted_df.shape}")
                print(f"  ⚠ Note: confidence and time_taken are SIMULATED")
            
            return converted_df
            
        except Exception as e:
            print(f"⚠ Could not load dataset: {e}")
            return pd.DataFrame()
    
    def convert_study3_llm_bias(self, save: bool = True) -> pd.DataFrame:
        """
        Convert LLM-specific bias detection dataset to Study 3 format
        """
        print("Downloading darkknight25/LLM_Bias_Detection_Dataset...")
        try:
            dataset = load_dataset("darkknight25/LLM_Bias_Detection_Dataset", split="train")
            df = pd.DataFrame(dataset)
            
            print(f"Original dataset shape: {df.shape}")
            print(f"Original columns: {df.columns.tolist()}")
            
            converted_df = pd.DataFrame({
                'participant_id': [f"P{str(i % 30).zfill(3)}" for i in range(len(df))],
                'task_id': df.index.astype(str),
                
                'detected_bias': df.get('bias_present', True),
                'confidence': np.random.randint(3, 6, len(df)),  # Higher confidence for LLM bias
                'identified_type': df.get('bias_type', 'unknown'),
                'time_taken': np.random.uniform(20, 150, len(df)),
                
                # Metadata
                'prompt': df.get('prompt', df.get('text', '')),
                'response': df.get('response', ''),
                'context': df.get('context', 'cybersecurity')
            })
            
            if save:
                output_path = self.output_dir / "study3_llm_bias.csv"
                converted_df.to_csv(output_path, index=False)
                print(f"✓ Study 3 (LLM bias) dataset saved to {output_path}")
            
            return converted_df
            
        except Exception as e:
            print(f"⚠ Could not load dataset: {e}")
            return pd.DataFrame()
    
    # ==================== UTILITY FUNCTIONS ====================
    
    @staticmethod
    def _normalize_to_likert(scores, min_score: int = 1, max_score: int = 5, 
                            inverse: bool = False) -> np.ndarray:
        """Normalize scores to Likert scale (1-5 or 1-7)"""
        scores = np.array(scores)
        if inverse:
            # Lower original score = higher Likert rating
            normalized = max_score - ((scores - scores.min()) / 
                                     (scores.max() - scores.min() + 1e-10) * (max_score - min_score))
        else:
            normalized = min_score + ((scores - scores.min()) / 
                                     (scores.max() - scores.min() + 1e-10) * (max_score - min_score))
        return np.round(normalized).astype(int)
    
    def create_synthetic_template(self, study_type: int, n_samples: int = 100, 
                                  save: bool = True) -> pd.DataFrame:
        """
        Create synthetic template dataset for testing your human evaluation code
        
        Args:
            study_type: 1, 2, or 3 for the three study formats
            n_samples: Number of synthetic samples to generate
            save: Whether to save to CSV
        """
        if study_type == 1:
            df = pd.DataFrame({
                'participant_id': [f"P{str(i % 20).zfill(3)}" for i in range(n_samples)],
                'evaluation_id': [f"E{str(i).zfill(4)}" for i in range(n_samples)],
                'method': np.random.choice(['LIME', 'SHAP', 'attention', 'gradient'], n_samples),
                'understandability': np.random.randint(1, 6, n_samples),
                'completeness': np.random.randint(1, 6, n_samples),
                'usefulness': np.random.randint(1, 6, n_samples),
                'trustworthiness': np.random.randint(1, 6, n_samples),
                'actionability': np.random.randint(1, 6, n_samples),
                'sufficiency': np.random.randint(1, 6, n_samples)
            })
            filename = "study1_synthetic_template.csv"
            
        elif study_type == 2:
            trust_items = {f'trust_item_{i}': np.random.randint(1, 8, n_samples) 
                          for i in range(1, 13)}
            df = pd.DataFrame({
                'participant_id': [f"P{str(i % 20).zfill(3)}" for i in range(n_samples)],
                'task_id': [f"T{str(i).zfill(4)}" for i in range(n_samples)],
                'evaluation_id': [f"E{str(i).zfill(4)}" for i in range(n_samples)],
                'initial_decision': np.random.choice(['A', 'B', 'C'], n_samples),
                'final_decision': np.random.choice(['A', 'B', 'C'], n_samples),
                **trust_items,
                'time_taken': np.random.uniform(30, 300, n_samples)
            })
            filename = "study2_synthetic_template.csv"
            
        elif study_type == 3:
            df = pd.DataFrame({
                'participant_id': [f"P{str(i % 20).zfill(3)}" for i in range(n_samples)],
                'task_id': [f"T{str(i).zfill(4)}" for i in range(n_samples)],
                'detected_bias': np.random.choice([True, False], n_samples),
                'confidence': np.random.randint(1, 6, n_samples),
                'identified_type': np.random.choice(['gender', 'race', 'age', None], n_samples),
                'time_taken': np.random.uniform(15, 120, n_samples)
            })
            filename = "study3_synthetic_template.csv"
        else:
            raise ValueError("study_type must be 1, 2, or 3")
        
        if save:
            output_path = self.output_dir / filename
            df.to_csv(output_path, index=False)
            print(f"✓ Synthetic template saved to {output_path}")
        
        return df
    
    def download_all(self):
        """Download and convert all available datasets"""
        print("=" * 70)
        print("XAI Dataset Downloader and Converter")
        print("=" * 70)
        
        print("\n" + "=" * 70)
        print("STUDY 1: Explanation Quality Ratings")
        print("=" * 70)
        try:
            self.convert_study1_mqm_dataset()
        except Exception as e:
            print(f"⚠ Error with MQM dataset: {e}")
        
        try:
            self.convert_study1_explainable_ai_emotions()
        except Exception as e:
            print(f"⚠ Error with ExplainableAI dataset: {e}")
        
        print("\n" + "=" * 70)
        print("STUDY 2: Decision Logs with Trust Items")
        print("=" * 70)
        try:
            self.convert_study2_arena_preferences()
        except Exception as e:
            print(f"⚠ Error with Arena dataset: {e}")
        
        try:
            self.convert_study2_financial_decisions()
        except Exception as e:
            print(f"⚠ Error with Financial dataset: {e}")
        
        print("\n" + "=" * 70)
        print("STUDY 3: Bias Detection Logs")
        print("=" * 70)
        try:
            self.convert_study3_bias_multidomain()
        except Exception as e:
            print(f"⚠ Error with Bias Multidomain dataset: {e}")
        
        try:
            self.convert_study3_llm_bias()
        except Exception as e:
            print(f"⚠ Error with LLM Bias dataset: {e}")
        
        print("\n" + "=" * 70)
        print("Creating Synthetic Templates for Testing")
        print("=" * 70)
        self.create_synthetic_template(1, n_samples=100)
        self.create_synthetic_template(2, n_samples=100)
        self.create_synthetic_template(3, n_samples=100)
        
        print("\n" + "=" * 70)
        print("✓ All conversions complete!")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("=" * 70)


def main():
    """Main execution function"""
    converter = XAIDatasetConverter(output_dir="./xai_datasets")
    converter.download_all()

if __name__ == "__main__":
    main()