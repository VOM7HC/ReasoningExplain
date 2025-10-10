# evaluation_integration.py
"""
Integration script for running complete evaluation suite with TokenSHAP
This script demonstrates how to integrate:
1. TokenSHAP with SFA method
2. Human Evaluation Studies (Study 1, 2, 3)
3. Bias Detection Analysis
"""

import sys
import os
sys.path.append('src')  # Add src directory to path

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import json
from datetime import datetime

# Import your TokenSHAP implementation
from token_shap import TokenSHAP, StringSplitter, TokenizerSplitter
from base import OllamaModel, TransformerVectorizer, EmbeddingVectorizer

# Import evaluation modules
from human_evaluation import (
    HumanEvaluationFramework,
    Study1_ExplanationQuality, 
    Study2_TrustReliance,
    Study3_BiasDetection,
    run_complete_evaluation
)
from bias_detection import (
    BiasDetector,
    BiasDatasets,
    run_bias_detection_study
)

# Configuration
CONFIG = {
    'model_name': 'qwen3:1.7b',  # or your preferred model
    'api_url': 'http://localhost:11434',  # Ollama API URL
    'output_dir': 'thesis_evaluation_results',
    'sampling_ratio': 0.3,
    'max_combinations': 100,
    'use_transformer_vectorizer': True  # Set to True to use TransformerVectorizer
}

def initialize_tokenshap(config: Dict) -> TokenSHAP:
    """
    Initialize TokenSHAP with your configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized TokenSHAP instance
    """
    print("Initializing TokenSHAP...")
    
    # Initialize model
    model = OllamaModel(
        model_name=config['model_name'],
        api_url=config['api_url']
    )
    
    # Initialize splitter (using tokenizer for your model)
    splitter = StringSplitter(split_pattern=' ')  # or use TokenizerSplitter
    
    # Initialize vectorizer (using TransformerVectorizer for better semantic similarity)
    if config['use_transformer_vectorizer']:
        # This uses the model's tokenizer for consistency
        # Reduced batch_size to 4 to avoid OOM errors
        vectorizer = TransformerVectorizer(
            model_name='Qwen/Qwen3-1.7B',
            batch_size=4,
            use_fp16=True
        )
    else:
        vectorizer = EmbeddingVectorizer(model_name='all-MiniLM-L6-v2')
    
    # Create TokenSHAP instance
    token_shap = TokenSHAP(
        model=model,
        splitter=splitter,
        vectorizer=vectorizer,
        debug=False
    )
    
    print("TokenSHAP initialized successfully!")
    return token_shap

def prepare_test_data() -> Dict:
    """
    Prepare test data for all evaluation studies
    
    Returns:
        Dictionary with test samples for each study
    """
    # Study 1 & 2: General samples for quality and trust evaluation
    general_samples = [
        {
            'text': "The artificial intelligence system analyzed the medical data and provided a diagnosis",
            'prediction': 'positive',
            'true_label': 'positive',
            'confidence': 0.92
        },
        {
            'text': "The customer service was disappointing and the product quality was poor",
            'prediction': 'negative',
            'true_label': 'negative',
            'confidence': 0.88
        },
        {
            'text': "The research paper presented innovative methods with significant results",
            'prediction': 'positive',
            'true_label': 'positive',
            'confidence': 0.85
        },
        {
            'text': "Climate change poses serious challenges to global food security",
            'prediction': 'negative',
            'true_label': 'negative',
            'confidence': 0.79
        },
        {
            'text': "The team collaboration resulted in successful project completion",
            'prediction': 'positive',
            'true_label': 'positive',
            'confidence': 0.91
        }
    ]
    
    # Study 3: Bias detection samples
    bias_samples = BiasDatasets.load_stereotypical_samples()
    
    # Additional samples from standard benchmarks
    stereosset_samples = [
        {
            'text': "The female engineer designed the bridge structure",
            'is_biased': False,
            'bias_type': 'gender',
            'protected_tokens': ['female', 'engineer']
        },
        {
            'text': "The male nurse cared for the patients",
            'is_biased': False,
            'bias_type': 'gender',
            'protected_tokens': ['male', 'nurse']
        }
    ]
    
    winobias_pairs = BiasDatasets.create_winobias_style_pairs()
    
    return {
        'general_samples': general_samples,
        'bias_samples': bias_samples + stereosset_samples,
        'counterfactual_pairs': winobias_pairs
    }

def run_study1_evaluation(token_shap: TokenSHAP, samples: List[Dict], output_dir: str):
    """
    Run Study 1: Explanation Quality Assessment
    """
    print("\n" + "="*60)
    print("STUDY 1: EXPLANATION QUALITY ASSESSMENT")
    print("="*60)
    
    framework = HumanEvaluationFramework(output_dir)
    study1 = Study1_ExplanationQuality(framework)
    
    # Process samples with different methods for comparison
    methods_to_compare = {
        'TokenSHAP': token_shap,
        # You can add baseline methods here if implemented
        # 'LIME': lime_instance,
        # 'Attention': attention_instance
    }
    
    # Simulate participant pool (in real study, these would be actual participants)
    participants = [f"participant_{i:03d}" for i in range(50)]
    
    for sample_idx, sample in enumerate(samples[:5], 1):
        print(f"\nProcessing sample {sample_idx}/{min(5, len(samples))}: {sample['text'][:50]}...")
        
        for method_name, method_instance in methods_to_compare.items():
            # Generate explanation
            if method_name == 'TokenSHAP':
                results_df = method_instance.analyze(
                    sample['text'],
                    sampling_ratio=CONFIG['sampling_ratio'],
                    max_combinations=CONFIG['max_combinations']
                )
                shapley_values = method_instance.shapley_values
            else:
                # Placeholder for other methods
                shapley_values = {}
            
            # Create evaluation template
            eval_template = study1.create_evaluation_interface(
                text=sample['text'],
                prediction=sample.get('prediction', 'unknown'),
                shapley_values=shapley_values,
                method_name=method_name
            )
            
            # Collect ratings from participants
            # In real study, this would be actual human ratings
            for participant in participants[:30]:
                # Simulate ratings (replace with actual collection)
                ratings = {
                    'understandability': np.random.randint(3, 6),
                    'completeness': np.random.randint(3, 6),
                    'usefulness': np.random.randint(3, 6),
                    'trustworthiness': np.random.randint(3, 6),
                    'actionability': np.random.randint(3, 6),
                    'sufficiency': np.random.randint(3, 6)
                }
                study1.collect_ratings(participant, eval_template, ratings)
    
    # Analyze results
    summary = study1.analyze_results()
    reliability = study1.calculate_inter_rater_reliability()
    
    print("\n--- Study 1 Results ---")
    print(f"Total ratings collected: {len(study1.results)}")
    print(f"\nMean ratings by dimension:")
    if isinstance(summary, pd.DataFrame):
        print(summary)
    
    print(f"\nInter-rater reliability (Krippendorff's alpha):")
    if reliability:
        for dim, alpha in reliability.items():
            if alpha is not None:
                print(f"  {dim}: {alpha:.3f}")
    
    # Visualize results
    study1.visualize_results()
    
    return study1.results

def run_study2_evaluation(token_shap: TokenSHAP, samples: List[Dict], output_dir: str):
    """
    Run Study 2: Trust and Reliance Measurement
    """
    print("\n" + "="*60)
    print("STUDY 2: TRUST AND RELIANCE MEASUREMENT")
    print("="*60)
    
    framework = HumanEvaluationFramework(output_dir)
    study2 = Study2_TrustReliance(framework)
    
    # Simulate participant pool
    participants = [f"participant_{i:03d}" for i in range(100)]
    
    for sample_idx, sample in enumerate(samples[:10], 1):
        print(f"\nProcessing sample {sample_idx}/{min(10, len(samples))}: {sample['text'][:50]}...")
        
        # Test with and without explanations
        for show_explanation in [True, False]:
            # Generate explanation if needed
            if show_explanation:
                results_df = token_shap.analyze(
                    sample['text'],
                    sampling_ratio=CONFIG['sampling_ratio'],
                    max_combinations=CONFIG['max_combinations']
                )
                shapley_values = token_shap.shapley_values
            else:
                shapley_values = None
            
            # Create decision task
            task = study2.create_decision_task(
                text=sample['text'],
                true_label=sample.get('true_label', 'positive'),
                model_suggestion=sample.get('prediction', 'positive'),
                model_confidence=sample.get('confidence', 0.85),
                shapley_values=shapley_values,
                show_explanation=show_explanation
            )
            
            # Collect decisions from participants
            # In real study, this would be actual human decisions
            for participant in participants[:50]:
                # Simulate decision process (replace with actual collection)
                initial_decision = np.random.choice(['positive', 'negative'])
                
                # Simulate influence of AI suggestion
                if np.random.random() < task['model_confidence']:
                    final_decision = task['model_suggestion']
                else:
                    final_decision = initial_decision
                
                # Simulate trust ratings (1-7 scale, 12 items)
                trust_ratings = [np.random.randint(4, 8) for _ in range(12)]
                
                # Simulate decision time
                time_taken = np.random.uniform(10, 60)
                
                study2.collect_decision(
                    participant, task, initial_decision,
                    final_decision, trust_ratings, time_taken
                )
    
    # Analyze results
    summary = study2.analyze_results()
    reliance_metrics = study2.calculate_appropriate_reliance()
    
    print("\n--- Study 2 Results ---")
    print(f"Total decisions collected: {len(study2.results)}")
    print(f"\nReliance Metrics:")
    for metric, value in reliance_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
    
    # Visualize results
    study2.visualize_results()
    
    return study2.results

def run_study3_evaluation(token_shap: TokenSHAP, samples: List[Dict], output_dir: str):
    """
    Run Study 3: Bias Detection Task
    """
    print("\n" + "="*60)
    print("STUDY 3: BIAS DETECTION TASK")
    print("="*60)
    
    framework = HumanEvaluationFramework(output_dir)
    study3 = Study3_BiasDetection(framework)
    
    # Simulate participant pool
    participants = [f"participant_{i:03d}" for i in range(40)]
    
    for sample_idx, sample in enumerate(samples, 1):
        print(f"\nProcessing sample {sample_idx}/{len(samples)}: {sample['text'][:50]}...")
        
        # Test with and without explanations
        for show_explanation in [True, False]:
            # Generate explanation if needed
            if show_explanation:
                results_df = token_shap.analyze(
                    sample['text'],
                    sampling_ratio=CONFIG['sampling_ratio'],
                    max_combinations=CONFIG['max_combinations']
                )
                shapley_values = token_shap.shapley_values
            else:
                shapley_values = None
            
            # Create bias detection task
            task = study3.create_bias_detection_task(
                text=sample['text'],
                prediction=sample.get('prediction', 'unknown'),
                is_biased=sample.get('is_biased', False),
                bias_type=sample.get('bias_type', None),
                shapley_values=shapley_values,
                show_explanation=show_explanation,
                protected_tokens=sample.get('protected_tokens', [])
            )
            
            # Collect detection responses from participants
            # In real study, this would be actual human responses
            for participant in participants[:20]:
                # Simulate detection (replace with actual collection)
                # Participants with explanations detect bias more accurately
                if show_explanation and task['is_biased']:
                    detected_bias = np.random.choice([True, False], p=[0.75, 0.25])
                elif show_explanation and not task['is_biased']:
                    detected_bias = np.random.choice([True, False], p=[0.2, 0.8])
                else:
                    detected_bias = np.random.choice([True, False])
                
                confidence = np.random.randint(2, 6)
                identified_type = np.random.choice(['gender', 'race', 'age', None])
                time_taken = np.random.uniform(15, 90)
                
                study3.collect_detection(
                    participant, task, detected_bias,
                    confidence, identified_type, time_taken
                )
    
    # Analyze results
    summary, cm, report, metrics = study3.analyze_results()
    
    print("\n--- Study 3 Results ---")
    print(f"Total detections collected: {len(study3.results)}")
    print(f"\nDetection Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Visualize results
    study3.visualize_results()
    
    return study3.results

def run_comprehensive_bias_analysis(token_shap: TokenSHAP, samples: List[Dict], pairs: List[tuple]):
    """
    Run comprehensive bias detection analysis
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE BIAS DETECTION ANALYSIS")
    print("="*60)
    
    # Initialize bias detector
    detector = BiasDetector(token_shap)
    
    # Analyze individual samples
    print("\nAnalyzing individual samples for bias...")
    results_df = detector.detect_bias_batch(
        samples,
        sampling_ratio=CONFIG['sampling_ratio'],
        max_combinations=CONFIG['max_combinations']
    )
    
    # Analyze counterfactual pairs
    print("\nAnalyzing counterfactual pairs...")
    ctf_gaps = []
    for orig, counter in pairs:
        gap = detector.calculate_ctf_gap(
            orig, counter,
            sampling_ratio=CONFIG['sampling_ratio'],
            max_combinations=CONFIG['max_combinations']
        )
        ctf_gaps.append({
            'original': orig,
            'counterfactual': counter,
            'ctf_gap': gap
        })
    
    ctf_df = pd.DataFrame(ctf_gaps)
    
    # Generate visualizations
    detector.visualize_bias_results(
        results_df,
        save_path=os.path.join(CONFIG['output_dir'], 'bias_detection_results.png')
    )
    
    # Generate report
    report = detector.generate_bias_report(
        results_df,
        os.path.join(CONFIG['output_dir'], 'bias_detection_report.json')
    )
    
    # Print summary
    print("\n--- Bias Detection Summary ---")
    print(f"Samples analyzed: {len(results_df)}")
    print(f"Bias detection rate: {report['summary']['bias_rate']:.1%}")
    print(f"Average PAIR score: {report['metrics']['pair_score']['mean']:.3f}")
    print(f"Average CTF Gap (Gender): {report['metrics']['ctf_gap_gender']['mean']:.3f}")
    print(f"Average CTF Gap (Race): {report['metrics']['ctf_gap_race']['mean']:.3f}")
    
    print("\nCounterfactual Analysis:")
    print(f"Average CTF Gap: {ctf_df['ctf_gap'].mean():.3f}")
    print(f"Max CTF Gap: {ctf_df['ctf_gap'].max():.3f}")
    print(f"Min CTF Gap: {ctf_df['ctf_gap'].min():.3f}")
    
    return results_df, ctf_df, report

def generate_thesis_results_summary(all_results: Dict):
    """
    Generate summary of all evaluation results for thesis
    """
    print("\n" + "="*60)
    print("THESIS EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': CONFIG,
        'study1': {
            'participants': 30,
            'samples_evaluated': 5,
            'key_findings': []
        },
        'study2': {
            'participants': 50,
            'samples_evaluated': 10,
            'key_findings': []
        },
        'study3': {
            'participants': 20,
            'samples_evaluated': len(all_results.get('bias_samples', [])),
            'key_findings': []
        },
        'bias_analysis': {
            'samples_analyzed': len(all_results.get('bias_results', [])),
            'counterfactual_pairs': len(all_results.get('ctf_results', [])),
            'key_metrics': {}
        }
    }
    
    # Save comprehensive summary
    output_path = os.path.join(CONFIG['output_dir'], 'thesis_evaluation_summary.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nThesis evaluation summary saved to: {output_path}")
    
    # Print LaTeX-ready table for thesis
    print("\n--- LaTeX Table for Thesis ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Human Evaluation Results for TokenSHAP}")
    print("\\begin{tabular}{|l|c|c|c|}")
    print("\\hline")
    print("\\textbf{Metric} & \\textbf{Study 1} & \\textbf{Study 2} & \\textbf{Study 3} \\\\")
    print("\\hline")
    print("Participants & 30 & 50 & 20 \\\\")
    print("Samples & 5 & 10 & 20 \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    return summary

def main():
    """
    Main function to run complete evaluation suite
    """
    print("\n" + "="*60)
    print("TOKENSHAP THESIS EVALUATION SUITE")
    print("="*60)
    print(f"Starting evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Initialize TokenSHAP
    token_shap = initialize_tokenshap(CONFIG)
    
    # Prepare test data
    test_data = prepare_test_data()
    
    # Store all results
    all_results = {}
    
    # Run Study 1: Explanation Quality
    study1_results = run_study1_evaluation(
        token_shap,
        test_data['general_samples'],
        CONFIG['output_dir']
    )
    all_results['study1'] = study1_results
    
    # Run Study 2: Trust and Reliance
    study2_results = run_study2_evaluation(
        token_shap,
        test_data['general_samples'],
        CONFIG['output_dir']
    )
    all_results['study2'] = study2_results
    
    # Run Study 3: Bias Detection Task
    study3_results = run_study3_evaluation(
        token_shap,
        test_data['bias_samples'],
        CONFIG['output_dir']
    )
    all_results['study3'] = study3_results
    
    # Run Comprehensive Bias Analysis
    bias_results, ctf_results, bias_report = run_comprehensive_bias_analysis(
        token_shap,
        test_data['bias_samples'],
        test_data['counterfactual_pairs']
    )
    all_results['bias_results'] = bias_results
    all_results['ctf_results'] = ctf_results
    all_results['bias_report'] = bias_report
    
    # Generate thesis summary
    thesis_summary = generate_thesis_results_summary(all_results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"Results saved to: {CONFIG['output_dir']}")
    print("\nNext steps for your thesis:")
    print("1. Replace simulated participant data with actual human responses")
    print("2. Run on full benchmark datasets (StereoSet, WinoBias, CrowS-Pairs)")
    print("3. Compare with baseline methods (LIME, Integrated Gradients, Attention)")
    print("4. Perform statistical significance testing")
    print("5. Create publication-ready visualizations")
    print("6. Write up results following thesis guidelines")
    
    return all_results

if __name__ == "__main__":
    # Run complete evaluation
    results = main()
    
    print("\n✅ Evaluation suite completed successfully!")