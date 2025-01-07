# Based on a script from: https://github.com/huggingface/trl/issues/1303
# Run this with DDP with "accelerate launch test_scripts/test_ddp.py"
import os
import sys
import torch
import wandb
from dataclasses import dataclass, field

from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, PreTrainedTokenizer
from peft import LoraConfig
from trl import ORPOConfig, ORPOTrainer, ModelConfig
from accelerate import PartialState
import logging

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('env', raise_error_if_not_found=True))

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

wandb.login(key=os.getenv('WANDB_API_KEY'))
os.environ["WANDB_PROJECT"]="AIMO"
os.environ['WANDB_WATCH']='false'
os.environ["WANDB_LOG_MODEL"]='false'


# Helper Functions
def get_tokenizer(model_args: ModelConfig) -> PreTrainedTokenizer:
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = 'right'
    return tokenizer

def format_data(current_row):

    current_row['chosen'] = current_row['chosen'] + tokenizer.eos_token # tokenizer.apply_chat_template(current_row['chosen'], tokenize=False)
    current_row['rejected'] = current_row['rejected'] + tokenizer.eos_token # tokenizer.apply_chat_template(current_row['rejected'], tokenize=False)

    return current_row


## Setting up Argument Parsers
# Parser for Dataset
@dataclass
class ScriptArgument:
    dataset: str = field(
        default="Hrushi/ORPO-Gemma-7b-Part-1a",
        metadata={"help": "The name/repo_id of the dataset to use."},
    )

# Setup the parser
arg_parser = HfArgumentParser((ScriptArgument, ModelConfig, ORPOConfig))
args, model_args, orpo_args = arg_parser.parse_yaml_file('orpo_config_full.yaml')
# args, model_args, orpo_args = arg_parser.parse_args_into_dataclasses()

# log_level = orpo_args.get_process_log_level()
log_level = 20
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process the small summary:
logger.info(f"Model parameters {model_args}")
logger.info(f"Data parameters {args}")
logger.info(f"Training/evaluation parameters {orpo_args}")


# Setup
## Setup bnb config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=model_args.load_in_4bit,
    bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
    bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
    bnb_4bit_compute_dtype=model_args.torch_dtype,
)

## Setup Tokenizer
logger.info(f"Loading Tokenizer for {model_args.model_name_or_path}")
tokenizer = get_tokenizer(model_args)
logger.info(f"Tokenizer Loaded for {model_args.model_name_or_path}")

## Setup Model
logger.info(f"Loading Model for {model_args.model_name_or_path}")
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_args.model_name_or_path,
    revision=model_args.model_revision,
    trust_remote_code=model_args.trust_remote_code,
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    use_cache=False if orpo_args.gradient_checkpointing else True,
    device_map={'':PartialState().process_index},
    quantization_config=bnb_config,
)
logger.info(f"Model Loaded for {model_args.model_name_or_path}")


## Setup Data
logger.info(f"Loading Dataset {args.dataset}")
dataset = load_dataset(args.dataset)['train']
training_dataset = dataset.map(
    format_data,
    num_proc=os.cpu_count(),
)
logger.info(f"Dataset Loaded {args.dataset}")
logger.info(f"Training on {len(training_dataset)} samples")

## LoRA Config
lora_config = LoraConfig(
    r=model_args.lora_r,
    lora_alpha=model_args.lora_alpha,
    lora_dropout=model_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=model_args.lora_target_modules,
)

## Setup Trainer
orpo_args.dataset_num_proc = os.cpu_count()
orpo_args.output_dir = os.path.join('Training_Outputs', f"AIMO-ORPO-DDP-{model_args.model_name_or_path.split('/')[-1].replace('.', '_')}")
orpo_args.run_name = f"AIMO-ORPO-DDP-{model_args.model_name_or_path.split('/')[-1].replace('.', '_')}-{orpo_args.num_train_epochs}Epoch-r{model_args.lora_r}-a{model_args.lora_alpha}"
orpo_args.bf16=model_args.torch_dtype == 'bfloat16'
orpo_args.fp16=model_args.torch_dtype == 'float16'
orpo_args.hub_model_id = f"AIMO-ORPO-{model_args.model_name_or_path.split('/')[-1].replace('.', '_')}-{orpo_args.num_train_epochs}Epoch-r{model_args.lora_r}-a{model_args.lora_alpha}"
orpo_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

logger.info(f"Training with {orpo_args.num_train_epochs} epochs")
trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=training_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config
)

# Train
trainer.train()

# # After Training
# SAVE_FPATH = f"AIMO_Finetuned_Models/{orpo_args.run_name}"
# logger.info(f"Saving Model to {SAVE_FPATH}")
# model.save_pretrained(SAVE_FPATH)
# logger.info(f"Saving Tokenizer to {SAVE_FPATH}")
# tokenizer.save_pretrained(SAVE_FPATH)

commit_message = f"Finetuned {model_args.model_name_or_path} with ORPO for {orpo_args.num_train_epochs} epoch(s) with r={model_args.lora_r} and alpha={model_args.lora_alpha}"
logger.info(f"Pushing to HuggingFace Hub with commit message {commit_message}")
trainer.push_to_hub(commit_message=commit_message)
logger.info("Pushed to HuggingFace Hub")