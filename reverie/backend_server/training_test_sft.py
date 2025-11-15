import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

from trl import SFTConfig, SFTTrainer, extract_prompt, apply_chat_template

@dataclass
class ScriptArguments:
    """
    The arguments for the SFT training script.
    """

    # data parameters
    # beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for SFT loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        metadata={"help": "the location of the SFT model name or path"},
    )
    cache_dir: Optional[str] = field(
        default="[CACHE_DIR]",
        metadata={"help": "the cache directory for the model"},
    )
    train_data_dir: Optional[str] = field(
        default="./sim_training_data.pkl",
        metadata={"help": "the directory of the training data"},
    )
    eval_data_dir: Optional[str] = field(
        default="./data/agent_training_eval.json",
        metadata={"help": "the directory of the eval data"},
    )
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=10, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    logging_dir: Optional[str] = field(default="./custom_logs", metadata={"help": "the logging directory"}) 
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=32, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=2500, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=3000, metadata={"help": "the maximum sequence length"})
    #max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    #logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    epochs: Optional[int] = field(default=3, metadata={"help": "the number of epochs"})
    save_steps: Optional[int] = field(default=50, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=20, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results_no_reason", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

def get_paired_dataset(data_dir = "./sim_training_data.pkl", tokenizer=""):
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': list[str],
        'chosen': list[str],
        'rejected': list[str],
    }
    """
    # dataset = load_dataset("json", data_files=data_dir, split="train")
    import pickle
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    data.pop('score')

    new_dict = {'messages': []}
    for k_ in ['chosen', 'rejected']:
        for i_ in range(len(data[k_])):
            if k_ == 'rejected':
                data[k_][i_][1]['content'] = data[k_][i_][1]['content'] + '\nCreate a bad answer and generate a reason why it is bad.'
            new_dict['messages'].append(data[k_][i_])
    
    dataset = Dataset.from_dict(new_dict)
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    dataset = dataset.shuffle(seed=42)
    return dataset
 

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)
    model_config = AutoConfig.from_pretrained(
        script_args.model_name_or_path,
        cache_dir=script_args.cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        cache_dir=script_args.cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype=model_config.torch_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        cache_dir=script_args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = get_paired_dataset(data_dir=script_args.train_data_dir, tokenizer=tokenizer)

    eval_dataset = None


    training_args = SFTConfig(
        #deepspeed="./src/ds_config.json",
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        # eval_strategy="steps",
        # eval_steps=script_args.eval_steps,
        # generate_during_eval=True,
        logging_dir=script_args.logging_dir,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        # lr_scheduler_type=script_args.lr_scheduler_type,
        # warmup_steps=script_args.warmup_steps,
        # beta=script_args.beta,
        #optim=script_args.optimizer_type,
        bf16=True,
        # max_prompt_length=script_args.max_prompt_length,
        max_seq_length=script_args.max_length,
        remove_unused_columns=True,
        run_name="sft_llama3.3-no-ds-new_data_no_reason",
        #gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
    )
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    lora_model = get_peft_model(model, peft_config)

    # see number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    # see number of trainable parameters
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"ratio of trainable parameters: {100*sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters())}%")

    import pdb; pdb.set_trace()

    sft_trainer = SFTTrainer(
        lora_model,
        # ref_model=None,
        args=training_args,
        #beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        # peft_config=peft_config,
        #data_collator=DebugCollator(tokenizer)
    )
    # 6. train
    sft_trainer.train()
    sft_trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)