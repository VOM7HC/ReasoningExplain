# reasoning_shap.py

from typing import List, Dict, Optional, Tuple, Any, Set
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from base import BaseSHAP, TextVectorizer, ModelBase, TfidfTextVectorizer, OllamaModel

class ReasoningStep:
    """Represents a single step in reasoning process"""
    def __init__(self, step_number: int, description: str, content: str):
        self.step_number = step_number
        self.description = description
        self.content = content
        
    def __str__(self):
        return f"Step {self.step_number}: {self.description}\n{self.content}"

class ReasoningSHAP(BaseSHAP):
    """Analyzes importance of reasoning steps using SHAP values"""
    
    def __init__(self, 
                 model: ModelBase,
                 vectorizer: Optional[TextVectorizer] = None,
                 debug: bool = False):
        super().__init__(model=model, vectorizer=vectorizer, debug=debug)
        self.reasoning_steps = []
        self.problem_statement = ""
        
    def _prepare_generate_args(self, content: str, **kwargs) -> Dict:
        """Prepare arguments for model.generate()"""
        return {"prompt": content}
    
    def _format_reasoning_chain(self, steps: List[ReasoningStep], problem: str) -> str:
        """Format a chain of reasoning steps into a prompt"""
        prompt_parts = [f"Problem: {problem}\n\nLet's solve this step by step:"]
        
        for step in steps:
            prompt_parts.append(f"\nStep {step.step_number}: {step.description}")
            prompt_parts.append(step.content)
            
        prompt_parts.append("\n\nBased on the above reasoning, what is the final answer?")
        return '\n'.join(prompt_parts)
    
    def _get_samples(self, content: Dict[str, Any]) -> List[ReasoningStep]:
        """Get reasoning steps from structured content"""
        if isinstance(content, dict):
            return content.get('steps', [])
        return []
    
    def _prepare_combination_args(self, combination: List[ReasoningStep], 
                                 original_content: Dict[str, Any]) -> Dict:
        """Prepare model arguments for a combination of reasoning steps"""
        problem = original_content.get('problem', self.problem_statement)
        prompt = self._format_reasoning_chain(combination, problem)
        return {"prompt": prompt}
    
    def _get_combination_key(self, combination: List[ReasoningStep], 
                           indexes: Tuple[int, ...]) -> str:
        """Get unique key for combination"""
        step_nums = [step.step_number for step in combination]
        return f"steps_{','.join(map(str, step_nums))}_indexes_{','.join(map(str, indexes))}"
    
    def generate_reasoning_steps(self, problem: str, num_steps: int = 5) -> List[ReasoningStep]:
        """Generate reasoning steps for a problem using the model"""
        self.problem_statement = problem
        
        prompt = (f"Problem: {problem}\n\n"
                 f"Please solve this problem in exactly {num_steps} clear, logical steps. "
                 f"Format each step as:\n"
                 f"Step [number]: [brief description]\n"
                 f"[detailed reasoning]\n")
        
        response = self.model.generate(prompt)
        return self._parse_reasoning_response(response)
    
    def _parse_reasoning_response(self, response: str) -> List[ReasoningStep]:
        """Parse model response into ReasoningStep objects"""
        steps = []
        
        # Simple parsing - split by "Step" pattern
        step_pattern = r'Step\s+(\d+):\s*([^\n]+)\n((?:(?!Step\s+\d+:).)*)'
        matches = re.finditer(step_pattern, response, re.DOTALL)
        
        for match in matches:
            step_num = int(match.group(1))
            description = match.group(2).strip()
            content = match.group(3).strip()
            steps.append(ReasoningStep(step_num, description, content))
            
        self.reasoning_steps = steps
        return steps
    
    def analyze_reasoning(self, 
                         problem: str,
                         steps: Optional[List[ReasoningStep]] = None,
                         sampling_ratio: float = 0.1,
                         max_combinations: Optional[int] = 100,
                         auto_generate_steps: bool = True,
                         num_steps: int = 5) -> pd.DataFrame:
        """
        Analyze the importance of reasoning steps in solving a problem
        
        This method inherits from BaseSHAP._calculate_shapley_values which includes:
        - Optuna hyperparameter tuning (20 trials per model)
        - K-Fold cross-validation for OOF predictions
        - Three augmented models (P, SHAP, P+SHAP)
        - Ensemble averaging
        """
        self.problem_statement = problem
        
        # Get or generate reasoning steps
        if steps is None and auto_generate_steps:
            self._debug_print("Generating reasoning steps...")
            steps = self.generate_reasoning_steps(problem, num_steps)
        elif steps is not None:
            self.reasoning_steps = steps
        else:
            raise ValueError("No steps provided and auto_generate_steps is False")
        
        if not steps:
            raise ValueError("No reasoning steps to analyze")
            
        self._debug_print(f"Analyzing {len(steps)} reasoning steps")
        
        # Prepare content for analysis
        content = {
            'problem': problem,
            'steps': steps
        }
        
        # Calculate baseline (full reasoning chain)
        full_chain_prompt = self._format_reasoning_chain(steps, problem)
        self.baseline_text = self._calculate_baseline(full_chain_prompt)
        self._debug_print("Baseline response calculated")
        
        # Get responses for different combinations of steps
        responses = self._get_result_per_combination(
            content,
            sampling_ratio=sampling_ratio,
            max_combinations=max_combinations
        )
        
        # Create results DataFrame
        self.results_df = self._get_df_per_combination(responses, self.baseline_text)
        
        # Calculate Shapley values using inherited method with Optuna, K-fold, etc.
        self.shapley_values = self._calculate_shapley_values(self.results_df, content)
        
        # Print results
        self.print_results()
        
        return self.results_df
    
    def print_results(self):
        """Print analysis results"""
        if not hasattr(self, 'shapley_values'):
            return
        
        print("\n" + "="*60)
        print("REASONING STEP IMPORTANCE ANALYSIS")
        print("="*60)
        print(f"\nProblem: {self.problem_statement}\n")
        print(f"Total steps analyzed: {len(self.reasoning_steps)}")
        print("-"*60)
        
        # Sort steps by importance
        sorted_values = sorted(self.shapley_values.items(), key=lambda x: x[1], reverse=True)
        
        for key, value in sorted_values:
            # Extract step number from key
            match = re.search(r'(\d+)', key)
            if match:
                step_num = int(match.group(1))
                step = next((s for s in self.reasoning_steps 
                           if s.step_number == step_num), None)
                if step:
                    print(f"\nStep {step_num}: {step.description}")
                    print(f"Shapley Value: {value:.4f}")
                    print(f"Content: {step.content[:100]}...")
        
        print("\n" + "="*60)
    
    def plot_importance(self):
        """Simple bar plot of step importance"""
        if not hasattr(self, 'shapley_values'):
            return
        
        # Extract step numbers and values
        step_data = []
        for key, value in self.shapley_values.items():
            match = re.search(r'(\d+)', key)
            if match:
                step_num = int(match.group(1))
                step_data.append((step_num, value))
        
        step_data.sort(key=lambda x: x[0])
        
        if step_data:
            steps, values = zip(*step_data)
            
            plt.figure(figsize=(10, 6))
            plt.bar([f"Step {s}" for s in steps], values)
            plt.xlabel('Reasoning Step')
            plt.ylabel('Shapley Value')
            plt.title('Reasoning Step Importance')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


def main():
    """Main function to run reasoning analysis"""
    import os
    
    # Set up Ollama API
    api_url = os.getenv('API_URL', 'http://localhost:11434')
    model_name = 'phi4-reasoning:latest'  # or any model you have in Ollama
    
    # Initialize components
    vectorizer = TfidfTextVectorizer()
    model = OllamaModel(model_name=model_name, api_url=api_url)
    
    # Create analyzer
    analyzer = ReasoningSHAP(
        model=model,
        vectorizer=vectorizer,
        debug=True
    )
    
    # Example problem
    problem = """
    A train leaves Station A at 9:00 AM traveling at 60 mph toward Station B.
    Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A.
    The distance between the stations is 280 miles.
    At what time will the trains meet?
    """
    
    # You can provide manual steps or let it auto-generate
    manual_steps = [
        ReasoningStep(1, "Define variables", 
                     "Let t = time in hours after 9:00 AM when trains meet"),
        ReasoningStep(2, "Calculate Train A's distance",
                     "Train A travels for t hours at 60 mph, covering 60t miles"),
        ReasoningStep(3, "Calculate Train B's distance",
                     "Train B starts 1 hour later, travels (t-1) hours at 80 mph, covering 80(t-1) miles"),
        ReasoningStep(4, "Set up equation",
                     "Total distance = 280 miles, so: 60t + 80(t-1) = 280"),
        ReasoningStep(5, "Solve equation",
                     "60t + 80t - 80 = 280; 140t = 360; t = 2.57 hours = 11:34 AM")
    ]
    
    # Run analysis with manual steps
    print("Analyzing with manual steps...")
    results = analyzer.analyze_reasoning(
        problem=problem,
        steps=manual_steps,
        sampling_ratio=0.1,
        max_combinations=50
    )
    
    # Plot results
    analyzer.plot_importance()
    
    # Or run with auto-generated steps
    print("\n\nAnalyzing with auto-generated steps...")
    analyzer2 = ReasoningSHAP(
        model=model,
        vectorizer=vectorizer,
        debug=False
    )
    
    results2 = analyzer2.analyze_reasoning(
        problem=problem,
        steps=None,
        auto_generate_steps=True,
        num_steps=5,
        sampling_ratio=0.1,
        max_combinations=50
    )
    
    # Save results
    analyzer.save_results("reasoning_analysis_results")


if __name__ == "__main__":
    main()