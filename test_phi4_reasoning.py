#!/usr/bin/env python3

"""
Test script for Phi4 reasoning with ReasoningSHAP
"""

import os
import sys
sys.path.append('/home/khoi/Project/ReasoningExplain/src')

from reasoning_shap import ReasoningSHAP
from base import OllamaModel, TfidfTextVectorizer

def test_phi4_reasoning():
    """Test the ReasoningSHAP with Phi4 reasoning model"""

    # Set up Ollama API
    api_url = os.getenv('API_URL', 'http://localhost:11434')
    model_name = 'phi4-reasoning:latest'  # Make sure this model is available in Ollama

    print(f"Testing with model: {model_name}")
    print(f"API URL: {api_url}")

    # Initialize components
    vectorizer = TfidfTextVectorizer()
    model = OllamaModel(model_name=model_name, api_url=api_url)

    # Create analyzer
    analyzer = ReasoningSHAP(
        model=model,
        vectorizer=vectorizer,
        debug=True
    )

    # Simple math problem for testing
    problem = """
    A train leaves Station A at 9:00 AM traveling at 60 mph toward Station B.
    Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A.
    The distance between the stations is 280 miles.
    At what time will the trains meet?
    """

    print("\n" + "="*60)
    print("TESTING PHI4 REASONING STEP GENERATION")
    print("="*60)
    print(f"Problem: {problem.strip()}")

    try:
        # Test step generation
        print("\nGenerating reasoning steps...")
        steps = analyzer.generate_reasoning_steps(problem, num_steps=5)

        print(f"\nGenerated {len(steps)} steps:")
        for step in steps:
            print(f"\n{step}")
            print("-" * 40)

        if len(steps) > 0:
            print("\n✅ Step generation successful!")

            # Test analysis with reduced combinations for faster execution
            print("\nTesting analysis with generated steps...")
            results = analyzer.analyze_reasoning(
                problem=problem,
                steps=steps,
                sampling_ratio=0.05,  # Very small sample for quick test
                max_combinations=10   # Small number for quick test
            )

            print("\n✅ Analysis completed successfully!")
            print(f"Results shape: {results.shape}")

        else:
            print("\n❌ No steps were generated!")

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_phi4_reasoning()