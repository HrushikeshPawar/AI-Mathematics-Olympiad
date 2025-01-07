# Based on a script from: https://github.com/huggingface/trl/issues/1303
# Run this with DDP with "accelerate launch test_scripts/test_ddp.py"
import os
import sys
import wandb
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, PreTrainedTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer, ModelConfig, DataCollatorForCompletionOnlyLM
from accelerate import PartialState
import logging
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


def formatting_prompts_func(examples):
    prompt = examples['prompt']
    completion = examples['completion']
    text = f"{prompt}{completion}" + tokenizer.eos_token
    return {"text": text}


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

# Setup the parser
arg_parser = HfArgumentParser((ScriptArgument, ModelConfig, SFTConfig))
args, model_args, sft_args = arg_parser.parse_yaml_file('sft_config_full.yaml')

log_level = 20
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
tqdm.pandas()


# Log on each process the small summary:
logger.info(f"Model parameters {model_args}")
logger.info(f"Data parameters {args}")
logger.info(f"Training/evaluation parameters {sft_args}")


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
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_args.model_name_or_path,
    revision=model_args.model_revision,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    use_cache=False if sft_args.gradient_checkpointing else True,
    device_map={'':PartialState().process_index},
    quantization_config=bnb_config,
)
logger.info(f"Model Loaded for {model_args.model_name_or_path}")


## Setup Data
logger.info(f"Loading Dataset {args.dataset}")
training_dataset = load_dataset(args.dataset)['train']
training_dataset = training_dataset.map(formatting_prompts_func, batched=False)
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
FT_MODEL_NAME = f"AIMO-SFT-DDP-{model_args.model_name_or_path.split('/')[-1]}"
sft_args.dataset_num_proc = os.cpu_count()
sft_args.output_dir = os.path.join('Training_Outputs', FT_MODEL_NAME)
sft_args.run_name = f"{FT_MODEL_NAME}-N{training_dataset.num_rows}-E{sft_args.num_train_epochs}-R{model_args.lora_r}-A{model_args.lora_alpha}"
sft_args.bf16=model_args.torch_dtype == 'bfloat16'
sft_args.fp16=model_args.torch_dtype == 'float16'
sft_args.hub_model_id = f"{FT_MODEL_NAME}-N{training_dataset.num_rows}-E{sft_args.num_train_epochs}-R{model_args.lora_r}-A{model_args.lora_alpha}"
sft_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
response_template_ids = tokenizer.encode(">\n<start_of_solution>", add_special_tokens=False)[2:]
logger.info(f"Training with {sft_args.num_train_epochs} epochs")
trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=training_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response_template_ids),
    formatting_func=formatting_prompts_func
)

# Train
trainer.train()

# Push to HuggingFace Hub
commit_message = f"Finetuned {model_args.model_name_or_path} with SFT for {sft_args.num_train_epochs} epoch(s) with r={model_args.lora_r} and alpha={model_args.lora_alpha}"
logger.info(f"Pushing to HuggingFace Hub with commit message {commit_message}")
trainer.push_to_hub(commit_message=commit_message)
logger.info("Pushed to HuggingFace Hub")