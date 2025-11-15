import os
import gc
import torch
from dataclasses import dataclass, field
from typing import Optional
import wandb

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from trl import DPOConfig, DPOTrainer, extract_prompt, KTOConfig, KTOTrainer, unpair_preference_dataset, apply_chat_template, SFTConfig, SFTTrainer
import pickle



def DPO_train(total_records, preference_data, model, base_model, train_mode, tokenizer, cache_dir, max_len=2048+256):
    def get_paired_dataset_dpo(preference_data) -> Dataset:
        dataset = Dataset.from_dict(preference_data)
        dataset = dataset.map(extract_prompt)
        return dataset
    
    # Training steps & epochs
    save_steps = 50
    output_dir = "./lora_model"
    report_to = "wandb"
    seed = 42
    set_seed(seed)

    model.config.use_cache = False

    for i_name in range(100):
        if os.path.exists(f'./sim_training_data_dpo_{i_name}.pkl'):
            continue
        with open(f'./total_data_dpo_{i_name}.pkl', 'wb') as f:
            pickle.dump(total_records, f)
        with open(f'./sim_training_data_dpo_{i_name}.pkl', 'wb') as f:
            pickle.dump(preference_data, f)
        run_name = f"dpo_llama3.3-no-ds-new_data_no_reason_{i_name}"
        break
    
    for i_ in range(5):
        try:
            wandb.init(project="huggingface", entity="wandb-reverie", name=run_name)
            break
        except:
            continue

    try:
        # Load and filter train dataset
        train_dataset = get_paired_dataset_dpo(preference_data)
        eval_dataset = None

        # Configure DPO
        training_args = DPOConfig(
            per_device_train_batch_size=1,
            num_train_epochs=2,
            logging_steps=1,
            # save_steps=50,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            learning_rate=5e-5,
            logging_dir="./custom_logs",
            output_dir=output_dir,
            report_to=report_to,
            beta=0.1,
            bf16=True,  # Whether you want BF16 training
            max_prompt_length=2048,
            max_length=max_len,
            remove_unused_columns=False,
            run_name=run_name,
            seed=seed
        )

        # Check parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {total_params}")
        print(f"Number of trainable parameters: {trainable_params}")
        print(f"Ratio of trainable parameters: {100 * trainable_params / total_params:.4f}%")

        
        # Initialize DPOTrainer

        gc.collect()
        torch.cuda.empty_cache()

        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer
        )
            
        # Train
        dpo_trainer.train()

        # Save model and tokenizer
        dpo_trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    except:
        pass

    gc.collect()
    torch.cuda.empty_cache()

    model.eval()
    model.config.use_cache = True

    wandb.finish()

    return model, base_model, tokenizer

def KTO_train(total_records, preference_data, model, base_model, train_mode, tokenizer, cache_dir, max_len=2048+256):
    def filter_abstention(example):
        return 'No Recommendation' not in example['completion'][0]['content']
    
    def get_unpaired_dataset(preference_data) -> Dataset:
        preference_data.pop('score')
        dataset = Dataset.from_dict(preference_data)
        dataset = dataset.map(extract_prompt)
        dataset = unpair_preference_dataset(dataset)
        # abstention filtering part
        dataset = dataset.filter(filter_abstention)
        dataset = dataset.shuffle(seed=42)
        return dataset

    # Training steps & epochs
    save_steps = 50
    output_dir = "./lora_model"
    report_to = "wandb"
    seed = 42
    set_seed(seed)

    model.config.use_cache = False

    for i_name in range(100):
        if os.path.exists(f'./sim_training_data_kto_{i_name}.pkl'):
            continue
        with open(f'./total_data_kto_{i_name}.pkl', 'wb') as f:
            pickle.dump(total_records, f)
        with open(f'./sim_training_data_kto_{i_name}.pkl', 'wb') as f:
            pickle.dump(preference_data, f)
        run_name = f"kto_llama3.3-no-ds-new_data_no_reason_{i_name}"
        break
    
    wandb.init(project="huggingface", entity="wandb-reverie", name=run_name)

    train_dataset = get_unpaired_dataset(preference_data)
    eval_dataset = None


    training_args = KTOConfig(
        #deepspeed="./src/ds_config.json",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        logging_steps=1,
        save_steps=save_steps,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        logging_dir="./custom_logs",
        output_dir=output_dir,
        report_to=report_to,
        beta=0.1,
        bf16=True,
        max_prompt_length=2048,
        max_length=max_len,
        remove_unused_columns=False,
        run_name=run_name,
        seed=seed,
    )

    # see number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    # see number of trainable parameters
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"ratio of trainable parameters: {100*sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters())}%")
    
    gc.collect()
    torch.cuda.empty_cache()

    kto_trainer = KTOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    # 6. train
    kto_trainer.train()
    kto_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    gc.collect()
    torch.cuda.empty_cache()

    model.eval()

    wandb.finish()
    model.config.use_cache = True

    return model, base_model, tokenizer



def SFT_train(total_records, preference_data, model, base_model, train_mode, tokenizer, cache_dir, max_len=2048+256):
    def get_dataset_sft(preference_data):
        new_dict = {'messages': []}
        for k_ in ['chosen', 'rejected']:
            for i_ in range(len(preference_data[k_])):
                if k_ == 'rejected':
                    # continue ## without rejected suggestion option
                    preference_data[k_][i_][1]['content'] = preference_data[k_][i_][1]['content'] + '\nCreate a bad answer and generate a reason why it is bad.'
                new_dict['messages'].append(preference_data[k_][i_])
        
        dataset = Dataset.from_dict(new_dict)
        dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
        dataset = dataset.shuffle(seed=42)
        return dataset

    # Training steps & epochs
    save_steps = 50
    output_dir = "./lora_model"
    report_to = "wandb"
    seed = 42
    set_seed(seed)

    model.config.use_cache = False
    
    for i_name in range(100):
        if os.path.exists(f'./sim_training_data_sft_{i_name}.pkl'):
            continue
        with open(f'./total_data_sft_{i_name}.pkl', 'wb') as f:
            pickle.dump(total_records, f)
        with open(f'./sim_training_data_sft_{i_name}.pkl', 'wb') as f:
            pickle.dump(preference_data, f)
        run_name = f"sft_llama3.3-no-ds-new_data_no_reason_{i_name}"
        break

    wandb.init(project="huggingface", entity="wandb-reverie", name=run_name)

    train_dataset = get_dataset_sft(preference_data)
    eval_dataset = None

    training_args = SFTConfig(
        #deepspeed="./src/ds_config.json",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=save_steps,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        logging_dir="./custom_logs",
        output_dir=output_dir,
        report_to=report_to,
        bf16=True,
        max_seq_length=max_len,
        remove_unused_columns=True,
        run_name=run_name,
        seed=seed,
    )

    # see number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    # see number of trainable parameters
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"ratio of trainable parameters: {100*sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters())}%")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    sft_trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    # 6. train
    sft_trainer.train()
    sft_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    gc.collect()
    torch.cuda.empty_cache()

    model.eval()

    wandb.finish()
    model.config.use_cache = True

    return model, base_model, tokenizer


def SFT_divide_train(total_records, preference_data, model, base_model, train_mode, tokenizer, cache_dir, max_len=2048+256):
    def get_dataset_sft(preference_data):
        new_dict = {'messages': []}
        for k_ in ['chosen', 'rejected']:
            for i_ in range(len(preference_data[k_])):
                if k_ == 'rejected':
                    continue ## without rejected suggestion option
                    preference_data[k_][i_][1]['content'] = preference_data[k_][i_][1]['content'] + '\nCreate a bad answer and generate a reason why it is bad.'
                new_dict['messages'].append(preference_data[k_][i_])
        
        dataset = Dataset.from_dict(new_dict)
        dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
        dataset = dataset.shuffle(seed=42)
        return dataset

    # Training steps & epochs
    save_steps = 50
    output_dir = "./lora_model"
    report_to = "wandb"
    seed = 42
    set_seed(seed)

    model.config.use_cache = False
    
    for i_name in range(100):
        if os.path.exists(f'./sim_training_data_sft_{i_name}.pkl'):
            continue
        with open(f'./total_data_sft_{i_name}.pkl', 'wb') as f:
            pickle.dump(total_records, f)
        with open(f'./sim_training_data_sft_{i_name}.pkl', 'wb') as f:
            pickle.dump(preference_data, f)
        run_name = f"sft_llama3.3-no-ds-new_data_no_reason_{i_name}"
        break

    wandb.init(project="huggingface", entity="wandb-reverie", name=run_name)

    train_dataset = get_dataset_sft(preference_data)
    eval_dataset = None

    training_args = SFTConfig(
        #deepspeed="./src/ds_config.json",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=save_steps,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        logging_dir="./custom_logs",
        output_dir=output_dir,
        report_to=report_to,
        bf16=True,
        max_seq_length=max_len,
        remove_unused_columns=True,
        run_name=run_name,
        seed=seed,
    )

    # see number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    # see number of trainable parameters
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"ratio of trainable parameters: {100*sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters())}%")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    sft_trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    # 6. train
    sft_trainer.train()
    sft_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    gc.collect()
    torch.cuda.empty_cache()

    model.eval()

    wandb.finish()
    model.config.use_cache = True

    return model, base_model, tokenizer