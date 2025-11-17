import json
import logging
import torch
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM, apply_chat_template
from dataclasses import dataclass, field
from accelerate import Accelerator
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from vllm import LLM


def llm_model_upload_vllm(train_mode: str, cache_dir: str = "[CACHE_DIR]", max_len: int = 2048+256):
    if 'llama-3.1' in train_mode:
        model_name = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    elif 'llama-3.3' in train_mode:
        model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = max_len
        model = LLM(model_name, 
                    download_dir = cache_dir, 
                    dtype=torch.bfloat16, 
                    quantization="bitsandbytes", 
                    load_format="bitsandbytes", 
                    max_model_len=2048+256,
                    distributed_executor_backend="ray")
        # import pdb; pdb.set_trace()
        # model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, device_map="auto")
        # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        # torch_dtype=torch.float16,
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # model.config.pad_token_id = tokenizer.pad_token_id

        # model.config.pad_token_id = tokenizer.pad_token_id
        # Encode input text
        return model, tokenizer

    except Exception as e:
        import pdb; pdb.set_trace()
        return f"An error occurred: {str(e)}"


def llm_model_upload(train_mode: str, cache_dir: str = "[CACHE_DIR]", max_len: int = 2048+256):
    if 'llama-3.1' in train_mode:
        model_name = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    elif 'llama-3.3' in train_mode:
        model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir)

        model_config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir = cache_dir,
            low_cpu_mem_usage=True,
            torch_dtype=model_config.torch_dtype, 
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        
        # model.config.use_cache = False
        # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        # torch_dtype=torch.float16,
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id

        # Encode input text
        return model, model, tokenizer

    except Exception as e:
        import pdb; pdb.set_trace()
        return f"An error occurred: {str(e)}"


def peft_lm_model_upload(base_model, tokenizer, train_mode, cache_dir = "[CACHE_DIR]", max_len = 2048+256):
    if 'llama-3.1' in train_mode:
        model_name = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    elif 'llama-3.3' in train_mode:
        model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

    try:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = prepare_model_for_kbit_training(base_model)
        # base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, device_map="auto")
        # base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, device_map={"":"cuda:1"})
        lora_model = get_peft_model(base_model, peft_config)
        return lora_model, base_model, tokenizer

    except Exception as e:
        import pdb; pdb.set_trace()
        return f"An error occurred: {str(e)}"

def peft_lm_model_upload_vllm(train_mode, cache_dir = "[CACHE_DIR]", max_len = 2048+256):
    if 'llama-3.1' in train_mode:
        model_name = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
    elif 'llama-3.3' in train_mode:
        model_name = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_len
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if os.path.exists("./lora_model") and len(os.listdir("./lora_model")) > 0:
        # base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, device_map={"":"cuda:1"})
        base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, device_map="auto")
        # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        lora_model = PeftModel.from_pretrained(
            base_model,
            "./lora_model"
        )
        return lora_model, base_model, tokenizer

    try:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, device_map="auto")
        # base_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir, device_map={"":"cuda:1"})
        lora_model = get_peft_model(base_model, peft_config)
        return lora_model, base_model, tokenizer

    except Exception as e:
        return f"An error occurred: {str(e)}"


def llm_inference_old(model, tokenizer, train_mode: str, input_text: str, max_input_len: int = 2048, max_new_tokens: int = 256, temperature: float = 1.0):
    """
    Performs inference using a Llama3 model. VLLM inference

    Parameters:
        train_mode (str): Name or path to the pre-trained Llama3 model.
        input_text (str): The input text for the model.
        max_length (int): Maximum length of the generated text.
        temperature (float): Sampling temperature; higher values produce more diverse outputs.

    Returns:
        str: The generated text.
    """
    try:
        input_text = tokenizer.apply_chat_template(conversation=[{'role': 'user', 'content': input_text}], add_generation_prompt=True, tokenize=False)

        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to('cuda')
        attention_mask = inputs.attention_mask.to('cuda')  # Create attention mask

        # Generate response
        outputs = model.generate(
            input_ids,
            temperature=temperature,
            num_return_sequences=1,
            do_sample=True,
            # max_length=max_input_len+max_new_tokens,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.5
        )

        # Decode and return the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text.split('assistant\n\n')[1]
        return generated_text

    except Exception as e:
        return f"An error occurred: {str(e)}"

