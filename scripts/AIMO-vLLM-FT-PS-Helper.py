from dotenv import load_dotenv
import os
import re
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
from io import StringIO
from contextlib import redirect_stdout
from time import perf_counter
from huggingface_hub import HfFileSystem, hf_hub_download
import jsonlines
from vllm import LLM, SamplingParams
import pandas as pd
from typing import Tuple, Optional, List
from collections import Counter
from copy import deepcopy
from sympy.parsing import parse_expr
from sympy.parsing.latex import parse_latex
import signal
import functools
from time import sleep
import json
from termcolor import colored
from datetime import timedelta

tqdm.pandas()
load_dotenv('env')

## Helper Functions
class TimeoutExpired(Exception):
  """Custom exception for timeout"""
  pass

def timeout(seconds=10, error_message='Timeout'):
  """Decorator to timeout a function after a certain number of seconds.

  Args:
      seconds: The maximum number of seconds allowed for the function to run.
      error_message: The message to raise in case of timeout.

  Returns:
      The result of the decorated function if it finishes within the timeout,
      otherwise raises TimeoutExpired with the provided error message.
  """

  def decorator(func):
    @functools.wraps(func)  # Preserve function metadata
    def wrapper(*args, **kwargs):
      def _handle_timeout(signum, frame):
        raise TimeoutExpired(error_message)

      # Set the signal handler and a timer
      signal.signal(signal.SIGALRM, _handle_timeout)
      signal.alarm(seconds)

      try:
        result = func(*args, **kwargs)
      finally:
        # Cancel the alarm if the function finishes before timeout
        signal.alarm(0)
      return result
    return wrapper
  return decorator

@timeout(seconds=60, error_message="Code took too long!")
def run_code_v2(code:str) -> Tuple[str, Optional[str], Optional[str]]:
    with redirect_stdout(StringIO()) as f:
        try:
            exec(code)
            error_name = None
            error_msg = None
        except Exception as e:
            error_name = str(e.__class__.__name__)
            error_msg = str(e)
    return f.getvalue().strip(), error_name, error_msg

def get_code(output_str:str) -> str:
    return re.search(r"<code_block>(.*?)</code_block>", output_str, re.DOTALL).group(1).strip()

def get_final_answer(output_str:str) -> str:
    return re.search(r"<final_answer>(.*?)</final_answer>", output_str, re.DOTALL).group(1).strip()

def post_processing_v2(raw_output:str, input_text:str, end_reason:str):
    error_name, error_msg = None, None
    final_answer = None
    
    if end_reason == '</final_answer>':
        try:
            final_answer = get_final_answer(raw_output)
        except AttributeError:
            error_name = "FABlockSyntanxError"
            error_msg = None
        input_text += raw_output
    else:
        final_answer = None
    
    if end_reason == '<code_output_block>':
        # Get code block
        try:
            code = get_code(raw_output)
            # Runt the code
            code_output, error_name, error_msg = run_code_v2(code)
            input_text += f"{raw_output}{code_output}</code_output_block>\n"
        except AttributeError:
            error_name = "CodeBlockSyntanxError"
            error_msg = None
            input_text += raw_output
    
    elif end_reason == '<end_of_step>':
        input_text += raw_output + "\n"
        
    elif end_reason == "length":
        input_text += raw_output
        
    return input_text, final_answer, error_name, error_msg

def post_process_math_expr(raw_fa:str) -> str:
    
    if 'text' in rf"{raw_fa}" or 'text' in raw_fa or 'ext\{' in raw_fa:
        return raw_fa
    
    try:
        new_fas = parse_expr(raw_fa)
    except Exception:
        try:
            new_fas = parse_latex(raw_fa)
        except Exception:
            new_fas = raw_fa
    
    try:
        new_fas = str(new_fas)
    except Exception:
        new_fas = None

    return new_fas

class Problem:
    
    def __init__(self, exp_id:str, problem:str, ofa:str, max_solutions:int, json_folder_path:str, load_from_file:bool=False):
        
        self.exp_id = exp_id
        self.problem = problem
        self.ofa = post_process_math_expr(ofa)
        self.max_solutions = max_solutions
        self.json_save_path = os.path.join(json_folder_path, f"{self.exp_id}.json")
        self.open_threads = []
        self.open_thread_rounds = []
        self.new_open_threads = []
        self.new_open_thread_rounds = []
        
        if load_from_file:
            with open(self.json_save_path, 'r') as f:
                data = json.load(f)
            
            assert problem == data['Problem'], f"ExpID\t: {self.exp_id}\nProblem (JSON)\t: {problem}\n\nProblem (DF)\t: {data['Problem']}"
            assert post_process_math_expr(ofa) == post_process_math_expr(data['OFA']), f"ExpID\t: {self.exp_id}\nOFA (JSON)\t: {post_process_math_expr(ofa)}\n\nOFA (DF)\t: {post_process_math_expr(data['OFA'])}"
            
            self.fas = [post_process_math_expr(x) for x in data['FAS']]
            self.fa = post_process_math_expr(data['FA'])
            self.closed_threads = data['ClosedThreads']
            self.update_final_answer()
        else:
            self.fas = []
            self.fa = None
            self.solved_correctly = None
            self.solved_once = None
            self.solved_top3 = None
            self.new_open_threads = []
            self.new_open_thread_rounds = []
            self.closed_threads = []
            self.completed = False

    def generate_input_prompt(self, tokenizer:AutoTokenizer, system_prompt:str) -> str:
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "problem", "content": self.problem}
        ]
        
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).replace(tokenizer.bos_token, '')
    
    def get_open_threads(self, tokenizer:AutoTokenizer, system_prompt:str) -> Tuple[List[str], List[int], List[str]]:
        
        missing_threads = self.max_solutions - len(self.open_threads) - len(self.closed_threads)
        
        if missing_threads > 0:
            self.open_threads.extend([self.generate_input_prompt(tokenizer=tokenizer, system_prompt=system_prompt) for _ in range(missing_threads)])
            self.open_thread_rounds.extend([0 for _ in range(missing_threads)])
        
        return deepcopy(self.open_threads), deepcopy(self.open_thread_rounds), [self.exp_id for _ in range(len(self.open_threads))]
    
    def set_open_threads(self):
        self.open_threads = deepcopy(self.new_open_threads)
        self.open_thread_rounds = deepcopy(self.new_open_thread_rounds)
        
        self.new_open_threads = []
        self.new_open_thread_rounds = []
    
    def add_new_open_thread(self, open_thread:str, round_counter:int):
        if round_counter <= MAX_ROUND_ALLOWED:
            self.new_open_threads.append(open_thread)
            self.new_open_thread_rounds.append(round_counter)
    
    def add_closed_thread(self, closed_thread:str):
        self.closed_threads.append(closed_thread)
    
    def add_final_answer(self, fa:str):
        self.fas.append(post_process_math_expr(fa))
        self.update_final_answer()
    
    def update_final_answer(self):
        self.fa = Counter(self.fas).most_common(n=1)[0][0]
        self.solved_top3 = self.ofa in [x for x, _  in Counter(self.fas).most_common(n=3)]
        self.solved_once = self.ofa in self.fas
        
        if len(self.fas) == self.max_solutions:
            self.solved_correctly = self.fa == self.ofa
            self.completed = True
        else:
            self.solved_correctly = self.fa == self.ofa
            self.completed = False
    
    def __str__(self) -> str:
        info = f"Problem\t\t: {self.problem}\n"
        info += f"Gold Answer\t: {self.ofa}\n"
        
        if self.fa is not None:
            info += f"Model Answer\t: {self.fa}\n"
        
        if len(self.fas) > 0:
            info += f"\nSolutions Collected\t: {len(self.fas)}"
            info += f"\nUnique Answers\t\t: | {' | '.join([f'{x} : {y}' for x, y in dict(sorted(Counter(self.fas).items(), key=lambda x: x[1], reverse=True)).items()])} |"
        
        if self.solved_correctly is not None:
            info += f"\nSolved Correctly\t: {self.solved_correctly}\nSolved Once\t\t: {self.solved_once}\nSolved in Top3\t\t: {self.solved_top3}\n"
        
        return info
    
    def __repr__(self) -> str:
        
        info = "Problem("
        info += f"exp_id: {self.exp_id}, "
        info += f"problem: {self.problem}, "
        info += f"ofa: {self.ofa}, "
        info += f"max_solutions: {self.max_solutions}, "
        info += f"fa: {self.fa}, "
        info += f"solved_correctly: {self.solved_correctly}, "
        info += f"solved_once: {self.solved_once}, "
        info += f"solved_top3: {self.solved_top3}, "
        info += f"solved_once: {self.solved_once}, "
        info += f"open_threads: {self.open_threads}, "
        info += f"closed_threads: {self.closed_threads}, "
        info += f"completed: {self.completed}"
        info += ")"
        
        return info
    
    def save_as_json(self):
        json_data = {
            "ExpID": self.exp_id,
            "Problem": self.problem,
            "OFA": self.ofa,
            "FA": self.fa,
            "FAS": self.fas,
            "ClosedThreads": self.closed_threads,
        }
        
        with open(self.json_save_path, 'w') as f:
            json.dump(json_data, f, indent=4)

def generate_open_threads(open_problems:dict, BATCH:int) -> Tuple[list, list, list]:
    GLOBAL_OPEN_THREADS = []
    GLOBAL_OPEN_THREAD_IDS = []
    GLOBAL_OPEN_THREAD_ROUND_COUNTERS = []
    
    for exp_id in open_problems:
        if not open_problems[exp_id].completed:
            open_problems[exp_id].set_open_threads()
            open_threads, round_counters, open_threads_ids = open_problems[exp_id].get_open_threads(tokenizer=tokenizer, system_prompt=system_prompt)
            GLOBAL_OPEN_THREADS.extend(open_threads)
            GLOBAL_OPEN_THREAD_IDS.extend(open_threads_ids)
            GLOBAL_OPEN_THREAD_ROUND_COUNTERS.extend(round_counters)

        if len(GLOBAL_OPEN_THREAD_ROUND_COUNTERS) >= BATCH:
            break
    return GLOBAL_OPEN_THREADS, GLOBAL_OPEN_THREAD_IDS, GLOBAL_OPEN_THREAD_ROUND_COUNTERS

def get_stats(open_problems:dict, closed_problems:int=0) -> dict:
    stats = {
        'nattempted': 0,
        'ncorrect': 0,
        'nwrong': 0,
        'ncorrect_top3': 0,
        'ncorrect_once': 0,
        'ncompleted': 0,
        'total_threads_collected': 0,
        'total_correct_threads': 0,
        'total_wrong_threads': 0,
    }

    for exp_id in open_problems:        
        if len(open_problems[exp_id].fas) == 0:
            continue
        
        stats['nattempted'] += 1
        
        if open_problems[exp_id].solved_correctly:
            stats['ncorrect'] += 1
        else:
            stats['nwrong'] += 1
            
        if open_problems[exp_id].solved_top3:
            stats['ncorrect_top3'] += 1
        
        if open_problems[exp_id].solved_once:
            stats['ncorrect_once'] += 1
        
        stats['total_threads_collected'] += len(open_problems[exp_id].closed_threads)
        stats['total_correct_threads'] += Counter(open_problems[exp_id].fas)[open_problems[exp_id].ofa]
        stats['total_wrong_threads'] += (len(open_problems[exp_id].closed_threads) - Counter(open_problems[exp_id].fas)[open_problems[exp_id].ofa])
        
        if open_problems[exp_id].completed:
            stats['ncompleted'] += 1
    
    return stats, stats['ncompleted'] - closed_problems


## GLOBAL VARIABLES
print("\n\n\n\nSetting up the Variables\n")
MODEL_PATH = "Hrushi/AIMO-SFT-DDP-Merged-Deepseek-Math-7B-RL-N141128-E1-R16-A64"

## Problem Solving variables
MAX_TOKENS = 4096
MAX_NEW_TOKENS = 512
MAX_SOLUTIONS = 5
MAX_ROUND_ALLOWED = 20
JSON_SAVE_FOLDER = "FT-Solutions"
BATCH = 10_000
COMPLETED_PROBLEMS = 0

TEMPERATURE = 1
TOP_P = 1
TOP_K = 100
STOP = ['<code_output_block>', '</final_answer>', '<end_of_step>']


## Model Setup
print("\nSetting up the vLLM\n")
llm = LLM(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    kv_cache_dtype='auto',
    dtype='float16',
    gpu_memory_utilization=0.95,
    tensor_parallel_size=torch.cuda.device_count(),
    disable_log_stats=True,
    device='cuda',
    swap_space=8,
    enable_prefix_caching=True
)

print("\nSetting up the Tokenizer\n")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("\nSetting up the System Prompt\n")
system_prompt = """Your are a high school student appearing for your math exam.
You will be given a math problem to solve, and your job is to write the detailed step by step solution, with good mathematical reasoning and using python code for calculations, simplifications, solving equations, etc.

Format help and instructions:
- Problem has been given enclosed in <math_problem></math_problem> tags.
- Every step must be written within <start_of_step><end_of_step> tags.
- Code needs to be always written inside <code_block></code_block> tags - No backticks or triple backticks.
- Code output will be given in <code_output_block></code_output_block> tags."""

## Problem Solving
sampling_params = SamplingParams(
    n=1,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    max_tokens=MAX_NEW_TOKENS,
    skip_special_tokens=False,
    spaces_between_special_tokens=False,
    stop=STOP + [tokenizer.eos_token],
    include_stop_str_in_output=True,
    truncate_prompt_tokens=MAX_TOKENS,
)

print("\nSetting up the Problem Set\n")
df = pd.read_parquet('AIMO-Math-Problems.parquet')

open_problems = {}
fnames = os.listdir(JSON_SAVE_FOLDER)
fnames = [x.split('.')[0] for x in fnames ]
for _, row in tqdm(df.iterrows(), total=len(df), desc='Collecting Open Problems'):
    load_from_file = row.ExpID in fnames        
    try:
        open_problems[row.ExpID] = Problem(exp_id=row.ExpID, problem=row.Problem, ofa=row.FinalAnswer, max_solutions=MAX_SOLUTIONS, json_folder_path=JSON_SAVE_FOLDER, load_from_file=load_from_file)
    except AssertionError:
        continue
print(f"\n{len(open_problems)} Total Problems!\n")

GLOBAL_OPEN_THREADS, GLOBAL_OPEN_THREAD_IDS, GLOBAL_OPEN_THREAD_ROUND_COUNTERS = generate_open_threads(open_problems, BATCH)
pbar = tqdm(total=len(open_problems), desc='Open Problems')
while len(GLOBAL_OPEN_THREADS) > 0:
    responses = llm.generate(GLOBAL_OPEN_THREADS, sampling_params=sampling_params)
    new_open_threads = []
    new_open_thread_ids = []
    for kdx, resp in enumerate(tqdm(responses, desc='Processing Responses')):
        exp_id = GLOBAL_OPEN_THREAD_IDS[kdx]
        old_input_text = resp.prompt
        output = resp.outputs[0]
        end_reason = output.stop_reason

        if end_reason is None and output.finish_reason == 'length':
            end_reason = 'length'

        output_text = output.text
        next_input_text, final_answer, error_name, error_msg = post_processing_v2(output_text, old_input_text, end_reason)

        if final_answer is not None:
            open_problems[exp_id].add_final_answer(final_answer)
            open_problems[exp_id].add_closed_thread(next_input_text)
            open_problems[exp_id].save_as_json()
        else:
            if old_input_text != next_input_text:
                if len(tokenizer.tokenize(next_input_text)) < MAX_TOKENS:
                    open_problems[exp_id].add_new_open_thread(open_thread=next_input_text, round_counter=GLOBAL_OPEN_THREAD_ROUND_COUNTERS[kdx] + 1)

    GLOBAL_OPEN_THREADS, GLOBAL_OPEN_THREAD_IDS, GLOBAL_OPEN_THREAD_ROUND_COUNTERS = generate_open_threads(open_problems, BATCH)

    stats, tqdm_update = get_stats(open_problems, COMPLETED_PROBLEMS)
    COMPLETED_PROBLEMS += tqdm_update
    pbar.update(tqdm_update)

    # Metrics for Ledger
    nattempted_str = f"{stats['nattempted']: >6} ({round(stats['nattempted'] / len(open_problems) * 100, 2): >5}%)"
    ncompleted_str = f"{stats['ncompleted']: >6} ({round(stats['ncompleted'] / len(open_problems) * 100, 2): >5}%)"
    ncorrect_str = f"{stats['ncorrect']: >6} ({round(stats['ncorrect'] / stats['nattempted'] * 100, 2): >5}%)"
    nwrong_str = f"{stats['nwrong']: >6} ({round(stats['nwrong'] / stats['nattempted'] * 100, 2): >5}%)"
    ncorrect_top3_str = f"{stats['ncorrect_top3']: >6} ({round(stats['ncorrect_top3'] / stats['nattempted'] * 100, 2): >5}%)"
    ncorrect_once_str = f"{stats['ncorrect_once']: >6} ({round(stats['ncorrect_once'] / stats['nattempted'] * 100, 2): >5}%)"
    total_threads_collected_str = f"{stats['total_threads_collected']: >6}"
    total_threads_collected_str = f"{total_threads_collected_str: <15}"
    total_correct_threads_str = f"{stats['total_correct_threads']: >6} ({round(stats['total_correct_threads'] / stats['total_threads_collected'] * 100, 2): >5}%)"
    total_wrong_threads_str = f"{stats['total_wrong_threads']: >6} ({round(stats['total_wrong_threads'] / stats['total_threads_collected'] * 100, 2): >5}%)"
    nattempted_str = f"{stats['nattempted']: >6} ({round(stats['nattempted'] / len(open_problems) * 100, 2): >5}%)"
    time_remaining = remaining = (pbar.total - pbar.n) / pbar.format_dict["rate"] if pbar.format_dict["rate"] and pbar.total else 0
    time_elapsed = pbar.format_dict['elapsed']

    print(f"\n{'++' * 100}\n")
    print("Running Ledger - " + colored(str(timedelta(seconds=int(time_elapsed))) + '\u2191', 'green') +" - " + colored(str(timedelta(seconds=int(time_remaining))) + '\u2193', 'red') + ":")
    print(f"\tTotal Problems Attempted\t: {colored(nattempted_str, 'black', 'on_yellow')}")
    print(f"\tTotal Problems Completed\t: {colored(ncompleted_str, 'black', 'on_yellow')}")
    print(f"\tProblems Solved Correctly\t: {colored(ncorrect_str, 'green')}")
    print(f"\tProblems Solved Incorrectly\t: {colored(nwrong_str, 'red')}")
    print(f"\tProblems Solved Correctly (Top3): {colored(ncorrect_top3_str, 'blue')}")
    print(f"\tProblems Solved Correctly (Once): {colored(ncorrect_once_str, 'blue')}")
    print(f"\tTotal Threads Collected\t\t: {colored(total_threads_collected_str, 'black', 'on_yellow')}")
    print(f"\tCorrect Threads Generated\t: {colored(total_correct_threads_str, 'green')}")
    print(f"\tIncorrect Threads Generated\t: {colored(total_wrong_threads_str, 'red')}")
    print(f"\n{'++' * 100}\n")

pbar.close()

