import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import re

class MMLUEvaluator:
    def __init__(self, model_name: str = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.subjects = ['human_aging', 'nutrition', 'world_religions', 'prehistory', 'sociology']

        # self.subjects = [
        #     'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        #     'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
        #     'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
        #     'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
        #     'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
        #     'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        #     'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
        #     'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
        #     'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
        #     'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing',
        #     'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
        #     'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
        #     'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
        #     'sociology', 'us_foreign_policy', 'virology', 'world_religions'
        # ]
        
    def load_model(self):
        print(f"Loading model: {self.model_name}")
        cache_dir: str = "[CACHE_DIR]"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                        cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model_config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            cache_dir = cache_dir,
            low_cpu_mem_usage=True,
            torch_dtype=model_config.torch_dtype, 
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        
        if self.device != "auto":
            self.model = self.model.to(self.device)
            
        self.model.eval()
        print("Model loaded successfully!")
        
    def format_question(self, question: str, choices: List[str]) -> str:
        choice_labels = ['A', 'B', 'C', 'D']
        formatted_choices = '\n'.join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
        
        prompt = f"""Answer the following multiple choice question by selecting the most appropriate option.

Question: {question}

Options:
{formatted_choices}

Answer: """
        return prompt
    
    def get_model_prediction(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        return response.strip()
    
    def extract_answer(self, response: str) -> str:
        match = re.search(r'\b([ABCD])\b', response.upper())
        if match:
            return match.group(1)
        
        if response and response[0].upper() in ['A', 'B', 'C', 'D']:
            return response[0].upper()
            
        return 'A'
    
    def evaluate_subject(self, subject: str, max_samples: int = None) -> Dict:
        print(f"Evaluating subject: {subject}")
        
        dataset = load_dataset("cais/mmlu", subject)
        test_data = dataset['test']
        
        if max_samples:
            test_data = test_data.select(range(min(max_samples, len(test_data))))
        
        correct = 0
        total = len(test_data)
        predictions = []
        
        for i, example in enumerate(tqdm(test_data, desc=f"Processing {subject}")):
            question = example['question']
            choices = example['choices']
            correct_answer_idx = example['answer']
            correct_answer = ['A', 'B', 'C', 'D'][correct_answer_idx]
            
            prompt = self.format_question(question, choices)
            response = self.get_model_prediction(prompt)
            predicted_answer = self.extract_answer(response)
            
            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct += 1
                
            predictions.append({
                'question': question,
                'choices': choices,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'model_response': response
            })
        
        accuracy = correct / total
        
        return {
            'subject': subject,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions
        }
    
    def evaluate_all_subjects(self, max_samples_per_subject: int = None, save_results: bool = True) -> Dict:
        if self.model is None:
            self.load_model()
            
        all_results = {}
        subject_accuracies = {}
        
        for subject in self.subjects:
            try:
                result = self.evaluate_subject(subject, max_samples_per_subject)
                all_results[subject] = result
                subject_accuracies[subject] = result['accuracy']
                print(f"{subject}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
            except Exception as e:
                print(f"Error evaluating {subject}: {str(e)}")
                continue
        
        overall_accuracy = np.mean(list(subject_accuracies.values()))
        
        # stem_subjects = [s for s in subject_accuracies.keys() if any(x in s for x in ['mathematics', 'physics', 'chemistry', 'biology', 'computer_science'])]
        humanities_subjects = [s for s in subject_accuracies.keys() if any(x in s for x in ['history', 'philosophy', 'law', 'world_religions', 'prehistory'])]
        social_sciences_subjects = [s for s in subject_accuracies.keys() if any(x in s for x in ['psychology', 'sociology', 'economics', 'government_and_politics', 'international_law', 'us_foreign_policy'])]
        # other_subjects = [s for s in subject_accuracies.keys() if s not in stem_subjects + humanities_subjects + social_sciences_subjects]
        
        category_results = {}
        # if stem_subjects:
        #     category_results['STEM'] = np.mean([subject_accuracies[s] for s in stem_subjects])
        if humanities_subjects:
            category_results['Humanities'] = np.mean([subject_accuracies[s] for s in humanities_subjects])
        if social_sciences_subjects:
            category_results['Social Sciences'] = np.mean([subject_accuracies[s] for s in social_sciences_subjects])
        # if other_subjects:
        #     category_results['Other'] = np.mean([subject_accuracies[s] for s in other_subjects])
        
        final_results = {
            'model_name': self.model_name,
            'overall_accuracy': overall_accuracy,
            'category_accuracies': category_results,
            'subject_accuracies': subject_accuracies,
            'detailed_results': all_results
        }
        
        # if save_results:
        #     output_file = f"mmlu_results_{self.model_name.replace('/', '_')}.json"
        #     with open(output_file, 'w', encoding='utf-8') as f:
        #         json.dump(final_results, f, ensure_ascii=False, indent=2)
        #     print(f"Results saved to {output_file}")
        
        return final_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama 3.3 70B on MMLU dataset")
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
                       help="Model name to evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples per subject (for testing)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--subjects", nargs='+', default=None,
                       help="Specific subjects to evaluate (default: all)")
    
    args = parser.parse_args()
    
    evaluator = MMLUEvaluator(model_name=args.model_name, device=args.device)
    
    if args.subjects:
        evaluator.subjects = args.subjects
    
    results = evaluator.evaluate_all_subjects(max_samples_per_subject=args.max_samples)

if __name__ == "__main__":
    main()