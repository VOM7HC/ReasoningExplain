# reasoning_shap.py

from typing import List, Dict, Optional, Tuple, Any, Set
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from base import BaseSHAP, TextVectorizer, ModelBase, TfidfTextVectorizer, OllamaModel, EmbeddingVectorizer, TransformerVectorizer

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
        """Generate reasoning steps for a problem using Phi4 reasoning model with <think> tags"""
        self.problem_statement = problem

        # Prompt designed for Phi4 reasoning to generate structured thinking
        prompt = (f"Please solve this problem step by step. Show your reasoning process in <think> tags:\n\n"
                 f"Problem: {problem}\n\n"
                 f"Please think through this problem carefully and provide exactly {num_steps} clear steps. "
                 f"Use <think> tags to show your reasoning process, then format your final answer as steps.")

        response = self.model.generate(prompt)
        return self._parse_reasoning_response(response)
    
    def _parse_reasoning_response(self, response: str) -> List[ReasoningStep]:
        """Parse model response into ReasoningStep objects, extracting from <think> tags"""
        steps = []

        # First try to extract content from <think> tags
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, response, re.DOTALL)

        if think_matches:
            # Parse reasoning from think tags
            think_content = ' '.join(think_matches)
            # Look for step patterns within think content or after think tags
            full_content = think_content + "\n" + response
        else:
            # Fallback to original response if no think tags found
            full_content = response

        # Parse steps from the content - try multiple patterns
        step_patterns = [
            r'Step\s+(\d+):\s*([^\n]+)\n((?:(?!Step\s+\d+:).)*)',  # Original pattern
            r'(\d+)\.\s*([^\n]+)\n((?:(?!\d+\.).)*)',              # Numbered list pattern
            r'Step\s+(\d+):\s*([^\n]+)(.*?)(?=Step\s+\d+:|$)',     # Lookahead pattern
        ]

        for pattern in step_patterns:
            matches = re.finditer(pattern, full_content, re.DOTALL)
            for match in matches:
                step_num = int(match.group(1))
                description = match.group(2).strip()
                content = match.group(3).strip() if len(match.groups()) > 2 else description

                # Avoid duplicate steps
                if not any(s.step_number == step_num for s in steps):
                    steps.append(ReasoningStep(step_num, description, content))

            if steps:  # If we found steps with this pattern, use them
                break

        # If no structured steps found, create steps from sentences/paragraphs
        if not steps:
            self._debug_print("No structured steps found, creating from content")
            sentences = [s.strip() for s in full_content.split('.') if s.strip()]
            for i, sentence in enumerate(sentences[:5], 1):  # Limit to 5 steps
                if sentence:
                    steps.append(ReasoningStep(i, f"Reasoning step {i}", sentence))

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
                    print(f"Content: {step.content}")
        
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
    model_name = 'qwen3:4b'  # or any model you have in Ollama
    
    # Initialize components
    #vectorizer = TfidfTextVectorizer()
    #vectorizer = EmbeddingVectorizer(model_name='all-MiniLM-L6-v2')  # Example embedding model
    vectorizer = TransformerVectorizer(model_name='Qwen/Qwen3:4b')  # Using the same model for embeddings
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

    # Use Phi4 reasoning to auto-generate steps with <think> tags
    print("Analyzing with Phi4 auto-generated reasoning steps...")
    results = analyzer.analyze_reasoning(
        problem=problem,
        steps=None,
        auto_generate_steps=True,
        num_steps=5,
        sampling_ratio=0.1,
        max_combinations=50
    )

    # Plot results
    analyzer.plot_importance()

    # Save results
    analyzer.save_results("reasoning_analysis_results")


if __name__ == "__main__":
    main()