# Based on a script from: https://github.com/huggingface/trl/issues/1303
# Run this with DDP with "accelerate launch test_scripts/test_ddp.py"
import os
import sys
import wandb
import torch
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, PreTrainedTokenizer
from peft import LoraConfig, PeftModelForCausalLM, PeftModel
from trl import ModelConfig, DPOConfig, DPOTrainer
from accelerate import PartialState
import logging
from typing import Dict
from huggingface_hub import HfFileSystem, hf_hub_download
# from contextlib import nullcontext

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
def get_tokenizer(args) -> PreTrainedTokenizer:
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.tokenizer_name_or_path,
        # trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = 'right'
    return tokenizer

def return_prompt_and_responses(samples) -> Dict[str, str]:
    return {
        "prompt": samples["prompt"],
        "chosen": samples["chosen"] + tokenizer.eos_token,
        "rejected": samples["rejected"] + tokenizer.eos_token,
    }

# Download adapter
def setup_adapter(adapter_repo_id:str, adapter_path:str):
    
    if not os.path.exists(adapter_path):
        os.mkdir(adapter_path)
    
    hfs = HfFileSystem()

    for f_info in hfs.listdir(adapter_repo_id):
        if f_info['type'] == 'directory':
            continue

        fname = f_info['name'].split('/')[-1]
        hf_hub_download(repo_id=adapter_repo_id, filename=fname, local_dir=adapter_path)
    

## Setting up Argument Parsers
# Parser for Dataset
@dataclass
class ScriptArgument:
    dataset: str = field(
        default="Hrushi/SFT-NCERT-TrainingData-Part1",
        metadata={"help": "The name/repo_id of the dataset to use."},
    )

    tokenizer_name_or_path: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        metadata={"help": "The name/repo_id of the Tokenizer to use."},
    )

    adapter_repo_id: str = field(
        default="Hrushi/AIMO-SFT-DDP-deepseek-math-7b-base-N141128-E1-R128-A256",
        metadata={"help": "The name/repo_id of the LoRA Adapter to use."},
    )

    adapter_path: str = field(
        default="Hrushi/DeepseekMath-Base-Adapter",
        metadata={"help": "Local folder path of the LoRA adapter to use."},
    )

# Setup the parser
arg_parser = HfArgumentParser((ScriptArgument, ModelConfig, DPOConfig))
args, model_args, dpo_args = arg_parser.parse_yaml_file('dpo_config.yaml')

log_level = 20
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
tqdm.pandas()


# # Log on each process the small summary:
# logger.info(f"Model parameters {model_args}")
# logger.info(f"Data parameters {args}")
# logger.info(f"Training/evaluation parameters {dpo_args}")


# Setup
## Setup bnb config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=model_args.load_in_4bit,
    bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
    bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
    bnb_4bit_compute_dtype=model_args.torch_dtype,
)

## Setup Tokenizer
logger.info(f"Loading Tokenizer for {args.tokenizer_name_or_path}")
tokenizer = get_tokenizer(args)
logger.info(f"Tokenizer Loaded for {args.tokenizer_name_or_path}")

## Setup Model
logger.info(f"Loading Model for {model_args.model_name_or_path}")

# Load the base Model
model_torch_dtype = 'auto'
if model_args.torch_dtype == 'float16':
    model_torch_dtype = torch.float16
elif model_args.torch_dtype == 'bfloat16':
    model_torch_dtype = torch.bfloat16
    
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_args.model_name_or_path,
    revision=model_args.model_revision,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    use_cache=False if dpo_args.gradient_checkpointing else True,
    device_map={'':PartialState().process_index},
    quantization_config=bnb_config,
)
logger.info(f"Base Model Loaded - {model_args.model_name_or_path}")

# # Check if the Adapter is download, if not download
# logger.info(f"Setting up the LoRA Adapter...")
# setup_adapter(adapter_repo_id=args.adapter_repo_id, adapter_path=args.adapter_path)
# logger.info(f"Setup of LoRA Adapter completed")

# # Load the trainer adapater
# lora_model = PeftModel.from_pretrained(
#     model,
#     args.adapter_repo_id,
#     is_trainable=True,
#     adapter_name="trainer",
# )
# logger.info(f"Trainer Adapter Loaded - {args.adapter_repo_id}")

# # Load the reference adapter
# lora_model.load_adapter(args.adapter_repo_id, adapter_name="reference")
# logger.info(f"Reference Adapter Loaded - {args.adapter_repo_id}")


## Setup Data
logger.info(f"Loading Dataset {args.dataset}")
training_dataset = load_dataset(args.dataset)['train']
original_columns = training_dataset.column_names
training_dataset = training_dataset.map(return_prompt_and_responses, batched=False, remove_columns=original_columns, num_proc=os.cpu_count())
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
FT_MODEL_NAME = f"AIMO-DPO-DDP-{model_args.model_name_or_path.split('/')[-1]}"
MODEL_ID = f"{FT_MODEL_NAME}-N{training_dataset.num_rows}-E{dpo_args.num_train_epochs}-R{model_args.lora_r}-A{model_args.lora_alpha}"
dpo_args.dataset_num_proc = os.cpu_count()
dpo_args.output_dir = os.path.join('Training_Outputs', FT_MODEL_NAME)
dpo_args.run_name = MODEL_ID
dpo_args.bf16=model_args.torch_dtype == 'bfloat16'
dpo_args.fp16=model_args.torch_dtype == 'float16'
dpo_args.hub_model_id = MODEL_ID
dpo_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

logger.info(f"Training with {dpo_args.num_train_epochs} epochs")
trainer = DPOTrainer(
    model=model,
    args=dpo_args,
    train_dataset=training_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

# Train
trainer.train()

# Push to HuggingFace Hub
commit_message = f"Finetuned {model_args.model_name_or_path} with DPO for {dpo_args.num_train_epochs} epoch(s) with r={model_args.lora_r} and alpha={model_args.lora_alpha}"
logger.info(f"Pushing to HuggingFace Hub with commit message {commit_message}")
trainer.push_to_hub(commit_message=commit_message)
logger.info("Pushed to HuggingFace Hub")
