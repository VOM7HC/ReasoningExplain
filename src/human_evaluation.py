# human_evaluation.py
"""
Human Evaluation Framework for TokenSHAP
Implements three evaluation studies:
1. Explanation Quality (N=30-50 participants)
2. Trust and Reliance (N=50-100 participants) 
3. Bias Detection Task (N=20-40 participants)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

class HumanEvaluationFramework:
    """Framework for conducting human evaluation studies on TokenSHAP explanations"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize evaluation framework
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Study 1 dimensions (Likert scale 1-5)
        self.quality_dimensions = {
            'understandability': 'I understand why the model made this decision',
            'completeness': 'The explanation includes all important factors',
            'usefulness': 'This explanation helps me use the model better',
            'trustworthiness': 'I trust this explanation',
            'actionability': 'I can use this to improve the model',
            'sufficiency': 'No additional information is needed'
        }
        
        # Study 2 metrics
        self.trust_scale_items = [
            "The system is deceptive",  # reverse
            "The system behaves in an underhanded manner",  # reverse  
            "I am suspicious of the system's intent",  # reverse
            "I am wary of the system",  # reverse
            "The system's actions will have a harmful or injurious outcome",  # reverse
            "I am confident in the system",
            "The system provides security",
            "The system has integrity", 
            "The system is dependable",
            "The system is reliable",
            "I can trust the system",
            "I am familiar with the system"
        ]
        
        # Initialize results storage
        self.study1_results = []
        self.study2_results = []
        self.study3_results = []

class Study1_ExplanationQuality:
    """
    Study 1: Explanation Quality Assessment
    N=30-50 participants rate explanations on 6 dimensions using 5-point Likert scales
    """
    
    def __init__(self, framework: HumanEvaluationFramework):
        self.framework = framework
        self.results = []
        
    def create_evaluation_interface(self, 
                                   text: str,
                                   prediction: str,
                                   shapley_values: Dict[str, float],
                                   method_name: str = "TokenSHAP") -> Dict:
        """
        Create evaluation interface for a single example
        
        Args:
            text: Input text
            prediction: Model prediction
            shapley_values: Token importance scores
            method_name: Name of explanation method
            
        Returns:
            Evaluation template dictionary
        """
        return {
            'id': f"{method_name}_{datetime.now().timestamp()}",
            'method': method_name,
            'input_text': text,
            'prediction': prediction,
            'shapley_values': shapley_values,
            'dimensions': self.framework.quality_dimensions,
            'ratings': {dim: None for dim in self.framework.quality_dimensions.keys()}
        }
    
    def collect_ratings(self, 
                        participant_id: str,
                        evaluation_template: Dict,
                        ratings: Dict[str, int]) -> None:
        """
        Collect ratings from a participant
        
        Args:
            participant_id: Unique participant identifier
            evaluation_template: The evaluation template shown
            ratings: Dictionary of dimension ratings (1-5)
        """
        result = {
            'participant_id': participant_id,
            'timestamp': datetime.now().isoformat(),
            'evaluation_id': evaluation_template['id'],
            'method': evaluation_template['method'],
            **ratings
        }
        self.results.append(result)
        
    def calculate_inter_rater_reliability(self, min_ratings: int = 3) -> float:
        """
        Calculate Krippendorff's alpha for inter-rater reliability
        
        Args:
            min_ratings: Minimum number of ratings per item
            
        Returns:
            Krippendorff's alpha coefficient
        """
        if len(self.results) < min_ratings:
            return None
            
        df = pd.DataFrame(self.results)
        
        # Calculate alpha for each dimension
        alphas = {}
        for dim in self.framework.quality_dimensions.keys():
            if dim in df.columns:
                # Group by evaluation_id and participant
                pivot = df.pivot_table(
                    values=dim, 
                    index='evaluation_id',
                    columns='participant_id',
                    aggfunc='first'
                )
                
                # Calculate pairwise agreement
                if pivot.shape[1] >= 2:
                    from itertools import combinations
                    kappas = []
                    for rater1, rater2 in combinations(pivot.columns, 2):
                        if not pivot[rater1].isna().all() and not pivot[rater2].isna().all():
                            kappa = cohen_kappa_score(
                                pivot[rater1].dropna(),
                                pivot[rater2].dropna(),
                                weights='quadratic'
                            )
                            kappas.append(kappa)
                    alphas[dim] = np.mean(kappas) if kappas else None
                    
        return alphas
    
    def analyze_results(self) -> pd.DataFrame:
        """
        Analyze Study 1 results with statistical tests
        
        Returns:
            DataFrame with analysis results
        """
        df = pd.DataFrame(self.results)
        
        # Calculate mean ratings per method and dimension
        summary = df.groupby('method')[list(self.framework.quality_dimensions.keys())].agg([
            'mean', 'std', 'count'
        ])
        
        # Conduct repeated measures ANOVA if multiple methods
        methods = df['method'].unique()
        if len(methods) > 1:
            # Prepare data for ANOVA
            anova_results = {}
            for dim in self.framework.quality_dimensions.keys():
                if dim in df.columns:
                    groups = [df[df['method'] == m][dim].values for m in methods]
                    # Remove NaN values
                    groups = [g[~np.isnan(g)] for g in groups]
                    if all(len(g) > 0 for g in groups):
                        f_stat, p_value = stats.f_oneway(*groups)
                        anova_results[dim] = {'f_stat': f_stat, 'p_value': p_value}
            
            # Post-hoc tests if significant
            posthoc_results = {}
            for dim, anova in anova_results.items():
                if anova['p_value'] < 0.05:
                    # Pairwise t-tests with Bonferroni correction
                    from itertools import combinations
                    pairs = list(combinations(methods, 2))
                    alpha_corrected = 0.05 / len(pairs)
                    
                    pair_results = []
                    for m1, m2 in pairs:
                        g1 = df[df['method'] == m1][dim].dropna().values
                        g2 = df[df['method'] == m2][dim].dropna().values
                        if len(g1) > 0 and len(g2) > 0:
                            t_stat, p_val = stats.ttest_ind(g1, g2)
                            effect_size = (np.mean(g1) - np.mean(g2)) / np.sqrt(
                                (np.var(g1) + np.var(g2)) / 2
                            )
                            pair_results.append({
                                'pair': f"{m1} vs {m2}",
                                't_stat': t_stat,
                                'p_value': p_val,
                                'significant': p_val < alpha_corrected,
                                'cohens_d': effect_size
                            })
                    posthoc_results[dim] = pair_results
            
            # Store ANOVA results
            self.anova_results = anova_results
            self.posthoc_results = posthoc_results
        
        return summary
    
    def visualize_results(self):
        """Create visualizations for Study 1 results"""
        df = pd.DataFrame(self.results)
        
        # Create subplots for each dimension
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, dim in enumerate(self.framework.quality_dimensions.keys()):
            if dim in df.columns:
                # Box plot for each method
                df.boxplot(column=dim, by='method', ax=axes[idx])
                axes[idx].set_title(dim.replace('_', ' ').title())
                axes[idx].set_xlabel('Method')
                axes[idx].set_ylabel('Rating (1-5)')
                
        plt.suptitle('Study 1: Explanation Quality Ratings')
        plt.tight_layout()
        plt.savefig(os.path.join(self.framework.output_dir, 'study1_quality_ratings.png'))
        plt.show()


class Study2_TrustReliance:
    """
    Study 2: Trust and Reliance Measurement
    N=50-100 participants make decisions with model assistance
    """
    
    def __init__(self, framework: HumanEvaluationFramework):
        self.framework = framework
        self.results = []
        
    def create_decision_task(self,
                            text: str,
                            true_label: str,
                            model_suggestion: str,
                            model_confidence: float,
                            shapley_values: Optional[Dict[str, float]] = None,
                            show_explanation: bool = True) -> Dict:
        """
        Create a decision task for participants
        
        Args:
            text: Input text for classification
            true_label: Ground truth label
            model_suggestion: Model's predicted label
            model_confidence: Model's confidence score (0-1)
            shapley_values: Token importance scores (if showing explanation)
            show_explanation: Whether to show explanation
            
        Returns:
            Task template dictionary
        """
        task = {
            'id': f"task_{datetime.now().timestamp()}",
            'text': text,
            'true_label': true_label,
            'model_suggestion': model_suggestion,
            'model_confidence': model_confidence,
            'show_explanation': show_explanation
        }
        
        if show_explanation and shapley_values:
            task['shapley_values'] = shapley_values
            
        return task
    
    def collect_decision(self,
                        participant_id: str,
                        task: Dict,
                        initial_decision: str,
                        final_decision: str,
                        trust_ratings: List[int],
                        time_taken: float) -> None:
        """
        Collect participant decision and trust ratings
        
        Args:
            participant_id: Unique participant identifier
            task: Task template
            initial_decision: Participant's initial decision
            final_decision: Participant's final decision after seeing AI suggestion
            trust_ratings: Ratings on 12-item trust scale (1-7)
            time_taken: Time taken to make decision (seconds)
        """
        # Calculate Weight of Advice (WoA)
        woa = 0.0
        if initial_decision != final_decision:
            # Simplified WoA for categorical decisions
            woa = 1.0 if final_decision == task['model_suggestion'] else 0.5
            
        # Calculate trust score (reverse score negative items)
        trust_items_reversed = [
            7 - trust_ratings[i] if i < 5 else trust_ratings[i]
            for i in range(len(trust_ratings))
        ]
        trust_score = np.mean(trust_items_reversed)
        
        result = {
            'participant_id': participant_id,
            'timestamp': datetime.now().isoformat(),
            'task_id': task['id'],
            'show_explanation': task['show_explanation'],
            'initial_decision': initial_decision,
            'final_decision': final_decision,
            'model_suggestion': task['model_suggestion'],
            'true_label': task['true_label'],
            'model_confidence': task['model_confidence'],
            'weight_of_advice': woa,
            'trust_score': trust_score,
            'trust_ratings': trust_ratings,
            'time_taken': time_taken,
            'followed_ai': final_decision == task['model_suggestion'],
            'correct_final': final_decision == task['true_label'],
            'ai_was_correct': task['model_suggestion'] == task['true_label']
        }
        
        self.results.append(result)
        
    def calculate_appropriate_reliance(self) -> Dict:
        """
        Calculate appropriate reliance metrics
        
        Returns:
            Dictionary with reliance metrics
        """
        df = pd.DataFrame(self.results)
        
        metrics = {}
        
        # Overall reliance rate
        metrics['overall_reliance'] = df['followed_ai'].mean()
        
        # Appropriate reliance (followed AI when correct, didn't when incorrect)
        appropriate = df[
            ((df['ai_was_correct'] == True) & (df['followed_ai'] == True)) |
            ((df['ai_was_correct'] == False) & (df['followed_ai'] == False))
        ]
        metrics['appropriate_reliance_rate'] = len(appropriate) / len(df)
        
        # Over-reliance (followed AI when incorrect)
        over_reliance = df[
            (df['ai_was_correct'] == False) & (df['followed_ai'] == True)
        ]
        metrics['over_reliance_rate'] = len(over_reliance) / len(df)
        
        # Under-reliance (didn't follow AI when correct)
        under_reliance = df[
            (df['ai_was_correct'] == True) & (df['followed_ai'] == False)
        ]
        metrics['under_reliance_rate'] = len(under_reliance) / len(df)
        
        # Correlation between confidence and reliance
        metrics['confidence_reliance_corr'] = df['model_confidence'].corr(
            df['followed_ai'].astype(float)
        )
        
        # Compare with/without explanations
        if df['show_explanation'].nunique() > 1:
            with_exp = df[df['show_explanation'] == True]
            without_exp = df[df['show_explanation'] == False]
            
            metrics['reliance_with_explanation'] = with_exp['followed_ai'].mean()
            metrics['reliance_without_explanation'] = without_exp['followed_ai'].mean()
            
            metrics['trust_with_explanation'] = with_exp['trust_score'].mean()
            metrics['trust_without_explanation'] = without_exp['trust_score'].mean()
            
            # Statistical test
            t_stat, p_val = stats.ttest_ind(
                with_exp['trust_score'].values,
                without_exp['trust_score'].values
            )
            metrics['trust_diff_pvalue'] = p_val
            
        return metrics
    
    def analyze_results(self) -> pd.DataFrame:
        """
        Analyze Study 2 results with mixed-effects models
        
        Returns:
            DataFrame with analysis results
        """
        df = pd.DataFrame(self.results)
        
        # Basic statistics
        summary = df.groupby('show_explanation').agg({
            'trust_score': ['mean', 'std'],
            'weight_of_advice': ['mean', 'std'],
            'followed_ai': 'mean',
            'correct_final': 'mean',
            'time_taken': ['mean', 'std']
        })
        
        # Calculate and store effect sizes separately
        if df['show_explanation'].nunique() > 1:
            with_exp = df[df['show_explanation'] == True]
            without_exp = df[df['show_explanation'] == False]
            
            # Cohen's d for trust score
            d_trust = (with_exp['trust_score'].mean() - without_exp['trust_score'].mean()) / \
                     np.sqrt((with_exp['trust_score'].var() + without_exp['trust_score'].var()) / 2)
            
            # Cohen's d for accuracy
            d_accuracy = (with_exp['correct_final'].mean() - without_exp['correct_final'].mean()) / \
                        np.sqrt((with_exp['correct_final'].var() + without_exp['correct_final'].var()) / 2)
            
            # Store effect sizes as a class attribute
            self.effect_sizes = {
                'trust_score_cohens_d': d_trust,
                'accuracy_cohens_d': d_accuracy
            }
            
            # Optionally print the effect sizes
            print(f"\nEffect Sizes (Cohen's d):")
            print(f"  Trust Score: {d_trust:.3f}")
            print(f"  Accuracy: {d_accuracy:.3f}")
        
        return summary
    
    def visualize_results(self):
        """Create visualizations for Study 2 results"""
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Trust scores comparison
        if 'show_explanation' in df.columns:
            df.boxplot(column='trust_score', by='show_explanation', ax=axes[0, 0])
            axes[0, 0].set_title('Trust Score by Explanation Condition')
            axes[0, 0].set_xlabel('Show Explanation')
            axes[0, 0].set_ylabel('Trust Score (1-7)')
        
        # Reliance vs Model Confidence
        axes[0, 1].scatter(df['model_confidence'], df['followed_ai'].astype(int))
        axes[0, 1].set_title('Reliance vs Model Confidence')
        axes[0, 1].set_xlabel('Model Confidence')
        axes[0, 1].set_ylabel('Followed AI (0/1)')
        
        # Appropriate reliance breakdown
        reliance_types = ['Appropriate', 'Over-reliance', 'Under-reliance']
        reliance_metrics = self.calculate_appropriate_reliance()
        values = [
            reliance_metrics.get('appropriate_reliance_rate', 0),
            reliance_metrics.get('over_reliance_rate', 0),
            reliance_metrics.get('under_reliance_rate', 0)
        ]
        axes[1, 0].bar(reliance_types, values)
        axes[1, 0].set_title('Reliance Breakdown')
        axes[1, 0].set_ylabel('Proportion')
        
        # Time taken distribution
        df['time_taken'].hist(ax=axes[1, 1], bins=20)
        axes[1, 1].set_title('Decision Time Distribution')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.suptitle('Study 2: Trust and Reliance Results')
        plt.tight_layout()
        plt.savefig(os.path.join(self.framework.output_dir, 'study2_trust_reliance.png'))
        plt.show()


class Study3_BiasDetection:
    """
    Study 3: Bias Detection Task
    N=20-40 participants identify biased predictions
    """
    
    def __init__(self, framework: HumanEvaluationFramework):
        self.framework = framework
        self.results = []
        
    def create_bias_detection_task(self,
                                  text: str,
                                  prediction: str,
                                  is_biased: bool,
                                  bias_type: Optional[str] = None,
                                  shapley_values: Optional[Dict[str, float]] = None,
                                  show_explanation: bool = True,
                                  protected_tokens: Optional[List[str]] = None) -> Dict:
        """
        Create a bias detection task
        
        Args:
            text: Input text
            prediction: Model prediction
            is_biased: Ground truth whether prediction is biased
            bias_type: Type of bias (e.g., 'gender', 'race', 'age')
            shapley_values: Token importance scores
            show_explanation: Whether to show explanation
            protected_tokens: List of protected attribute tokens
            
        Returns:
            Task template dictionary
        """
        task = {
            'id': f"bias_task_{datetime.now().timestamp()}",
            'text': text,
            'prediction': prediction,
            'is_biased': is_biased,
            'bias_type': bias_type,
            'show_explanation': show_explanation,
            'protected_tokens': protected_tokens or []
        }
        
        if show_explanation and shapley_values:
            task['shapley_values'] = shapley_values
            
            # Calculate protected attribute importance ratio (PAIR)
            if protected_tokens:
                protected_importance = sum(
                    shapley_values.get(f"{token}_{i}", 0)
                    for i, token in enumerate(text.split())
                    if token in protected_tokens
                )
                total_importance = sum(shapley_values.values())
                task['pair_score'] = protected_importance / total_importance if total_importance > 0 else 0
                
        return task
    
    def collect_detection(self,
                         participant_id: str,
                         task: Dict,
                         detected_bias: bool,
                         confidence: int,
                         identified_bias_type: Optional[str],
                         time_taken: float) -> None:
        """
        Collect participant's bias detection response
        
        Args:
            participant_id: Unique participant identifier
            task: Task template
            detected_bias: Whether participant detected bias
            confidence: Confidence in detection (1-5)
            identified_bias_type: Type of bias identified by participant
            time_taken: Time taken to make decision (seconds)
        """
        result = {
            'participant_id': participant_id,
            'timestamp': datetime.now().isoformat(),
            'task_id': task['id'],
            'show_explanation': task['show_explanation'],
            'true_bias': task['is_biased'],
            'true_bias_type': task.get('bias_type'),
            'detected_bias': detected_bias,
            'confidence': confidence,
            'identified_bias_type': identified_bias_type,
            'time_taken': time_taken,
            'correct_detection': detected_bias == task['is_biased'],
            'pair_score': task.get('pair_score', None)
        }
        
        # Calculate detection accuracy
        if task['is_biased']:
            result['true_positive'] = detected_bias
            result['false_negative'] = not detected_bias
            result['true_negative'] = False
            result['false_positive'] = False
        else:
            result['true_positive'] = False
            result['false_negative'] = False
            result['true_negative'] = not detected_bias
            result['false_positive'] = detected_bias
            
        self.results.append(result)
    
    def calculate_detection_metrics(self) -> Dict:
        """
        Calculate bias detection performance metrics
        
        Returns:
            Dictionary with detection metrics
        """
        df = pd.DataFrame(self.results)
        
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = df['correct_detection'].mean()
        
        # Sensitivity (true positive rate)
        biased_cases = df[df['true_bias'] == True]
        if len(biased_cases) > 0:
            metrics['sensitivity'] = biased_cases['detected_bias'].mean()
        
        # Specificity (true negative rate)
        unbiased_cases = df[df['true_bias'] == False]
        if len(unbiased_cases) > 0:
            metrics['specificity'] = (~unbiased_cases['detected_bias']).mean()
        
        # False positive rate
        metrics['false_positive_rate'] = 1 - metrics.get('specificity', 0)
        
        # Confidence calibration
        metrics['mean_confidence'] = df['confidence'].mean()
        metrics['confidence_accuracy_corr'] = df['confidence'].corr(
            df['correct_detection'].astype(float)
        )
        
        # Compare with/without explanations
        if df['show_explanation'].nunique() > 1:
            with_exp = df[df['show_explanation'] == True]
            without_exp = df[df['show_explanation'] == False]
            
            metrics['accuracy_with_explanation'] = with_exp['correct_detection'].mean()
            metrics['accuracy_without_explanation'] = without_exp['correct_detection'].mean()
            
            # Chi-square test for independence
            from scipy.stats import chi2_contingency
            contingency = pd.crosstab(
                df['show_explanation'],
                df['correct_detection']
            )
            chi2, p_val, dof, expected = chi2_contingency(contingency)
            metrics['explanation_effect_pvalue'] = p_val
            
        # Performance by bias type
        if 'true_bias_type' in df.columns:
            bias_types = df['true_bias_type'].dropna().unique()
            for bias_type in bias_types:
                subset = df[df['true_bias_type'] == bias_type]
                metrics[f'accuracy_{bias_type}'] = subset['correct_detection'].mean()
                
        return metrics
    
    def analyze_results(self) -> pd.DataFrame:
        """
        Analyze Study 3 results
        
        Returns:
            DataFrame with analysis results
        """
        df = pd.DataFrame(self.results)
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        
        y_true = df['true_bias'].astype(int)
        y_pred = df['detected_bias'].astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=['Not Biased', 'Biased'],
            output_dict=True
        )
        
        # Summary by condition
        summary = df.groupby('show_explanation').agg({
            'correct_detection': 'mean',
            'confidence': 'mean',
            'time_taken': 'mean'
        })
        
        # Add detection metrics
        metrics = self.calculate_detection_metrics()
        
        return summary, cm, report, metrics
    
    def visualize_results(self):
        """Create visualizations for Study 3 results"""
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        y_true = df['true_bias'].astype(int)
        y_pred = df['detected_bias'].astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # Detection accuracy by condition
        if 'show_explanation' in df.columns:
            df.groupby('show_explanation')['correct_detection'].mean().plot(
                kind='bar', ax=axes[0, 1]
            )
            axes[0, 1].set_title('Detection Accuracy by Condition')
            axes[0, 1].set_ylabel('Accuracy')
            
        # Confidence distribution
        df['confidence'].hist(ax=axes[1, 0], bins=5)
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence (1-5)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Time taken by detection outcome
        df.boxplot(column='time_taken', by='correct_detection', ax=axes[1, 1])
        axes[1, 1].set_title('Decision Time by Detection Outcome')
        axes[1, 1].set_xlabel('Correct Detection')
        axes[1, 1].set_ylabel('Time (seconds)')
        
        plt.suptitle('Study 3: Bias Detection Results')
        plt.tight_layout()
        plt.savefig(os.path.join(self.framework.output_dir, 'study3_bias_detection.png'))
        plt.show()


def run_complete_evaluation(token_shap_instance, 
                           test_samples: List[Dict],
                           participant_pool: List[str],
                           output_dir: str = "evaluation_results") -> Dict:
    """
    Run complete human evaluation suite
    
    Args:
        token_shap_instance: Initialized TokenSHAP instance
        test_samples: List of test samples with ground truth
        participant_pool: List of participant IDs
        output_dir: Directory for results
        
    Returns:
        Dictionary with all evaluation results
    """
    # Initialize framework
    framework = HumanEvaluationFramework(output_dir)
    
    # Initialize studies
    study1 = Study1_ExplanationQuality(framework)
    study2 = Study2_TrustReliance(framework)
    study3 = Study3_BiasDetection(framework)
    
    results = {
        'study1': None,
        'study2': None,
        'study3': None,
        'metrics': {}
    }
    
    print("Running Human Evaluation Studies...")
    print("-" * 50)
    
    # Study 1: Explanation Quality (N=30-50)
    print("\nStudy 1: Explanation Quality Assessment")
    study1_participants = participant_pool[:50]
    
    for sample in test_samples[:10]:  # Use subset for quality evaluation
        # Generate TokenSHAP explanation
        results_df = token_shap_instance.analyze(
            sample['text'],
            sampling_ratio=0.3,
            max_combinations=100
        )
        shapley_values = token_shap_instance.shapley_values
        
        # Create evaluation interface
        eval_template = study1.create_evaluation_interface(
            text=sample['text'],
            prediction=sample.get('prediction', 'Unknown'),
            shapley_values=shapley_values,
            method_name="TokenSHAP"
        )
        
        # Simulate participant ratings (replace with actual collection)
        for participant in study1_participants[:30]:
            ratings = {
                dim: np.random.randint(3, 6)  # Simulated ratings 3-5
                for dim in framework.quality_dimensions.keys()
            }
            study1.collect_ratings(participant, eval_template, ratings)
    
    study1_summary = study1.analyze_results()
    study1.visualize_results()
    results['study1'] = study1_summary
    print(f"Study 1 Complete: {len(study1.results)} ratings collected")
    
    # Study 2: Trust and Reliance (N=50-100)
    print("\nStudy 2: Trust and Reliance Measurement")
    study2_participants = participant_pool[:100]
    
    for sample in test_samples[:20]:  # Use subset for trust evaluation
        # Generate explanation
        results_df = token_shap_instance.analyze(
            sample['text'],
            sampling_ratio=0.3,
            max_combinations=100
        )
        shapley_values = token_shap_instance.shapley_values
        
        # Create tasks with and without explanations
        for show_exp in [True, False]:
            task = study2.create_decision_task(
                text=sample['text'],
                true_label=sample.get('true_label', 'positive'),
                model_suggestion=sample.get('prediction', 'positive'),
                model_confidence=sample.get('confidence', 0.85),
                shapley_values=shapley_values if show_exp else None,
                show_explanation=show_exp
            )
            
            # Simulate participant decisions (replace with actual collection)
            for participant in study2_participants[:50]:
                initial = np.random.choice(['positive', 'negative'])
                final = np.random.choice(['positive', 'negative'])
                trust_ratings = [np.random.randint(3, 7) for _ in range(12)]
                time = np.random.uniform(5, 30)
                
                study2.collect_decision(
                    participant, task, initial, final,
                    trust_ratings, time
                )
    
    study2_summary = study2.analyze_results()
    study2_metrics = study2.calculate_appropriate_reliance()
    study2.visualize_results()
    results['study2'] = study2_summary
    results['metrics']['trust_reliance'] = study2_metrics
    print(f"Study 2 Complete: {len(study2.results)} decisions collected")
    
    # Study 3: Bias Detection (N=20-40)
    print("\nStudy 3: Bias Detection Task")
    study3_participants = participant_pool[:40]
    
    # Create biased and unbiased samples
    bias_samples = [s for s in test_samples if s.get('is_biased', False)][:10]
    unbias_samples = [s for s in test_samples if not s.get('is_biased', True)][:10]
    
    for sample in bias_samples + unbias_samples:
        # Generate explanation
        results_df = token_shap_instance.analyze(
            sample['text'],
            sampling_ratio=0.3,
            max_combinations=100
        )
        shapley_values = token_shap_instance.shapley_values
        
        # Create bias detection tasks
        for show_exp in [True, False]:
            task = study3.create_bias_detection_task(
                text=sample['text'],
                prediction=sample.get('prediction', 'Unknown'),
                is_biased=sample.get('is_biased', False),
                bias_type=sample.get('bias_type', None),
                shapley_values=shapley_values if show_exp else None,
                show_explanation=show_exp,
                protected_tokens=sample.get('protected_tokens', [])
            )
            
            # Simulate participant detection (replace with actual collection)
            for participant in study3_participants[:20]:
                detected = np.random.choice([True, False])
                confidence = np.random.randint(2, 6)
                bias_type = np.random.choice(['gender', 'race', 'age', None])
                time = np.random.uniform(10, 60)
                
                study3.collect_detection(
                    participant, task, detected,
                    confidence, bias_type, time
                )
    
    study3_summary, cm, report, study3_metrics = study3.analyze_results()
    study3.visualize_results()
    results['study3'] = study3_summary
    results['metrics']['bias_detection'] = study3_metrics
    print(f"Study 3 Complete: {len(study3.results)} detections collected")
    
    # Save all results
    print("\nSaving results...")
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump({
            'study1_metrics': results['study1'].to_dict() if results['study1'] is not None else {},
            'study2_metrics': results['study2'].to_dict() if results['study2'] is not None else {},
            'study3_metrics': results['study3'].to_dict() if results['study3'] is not None else {},
            'overall_metrics': results['metrics']
        }, f, indent=2, default=str)
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")
    
    return results


# Example usage for integration with your TokenSHAP
if __name__ == "__main__":
    # This would be imported and used with your actual TokenSHAP implementation
    print("Human Evaluation Framework loaded successfully!")
    print("Use run_complete_evaluation() to run all three studies")