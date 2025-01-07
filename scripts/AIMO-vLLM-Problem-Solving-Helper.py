from dotenv import load_dotenv
import os
import re
import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from io import StringIO
from contextlib import redirect_stdout
import jsonlines
from vllm import LLM, SamplingParams
import pandas as pd
from typing import Tuple, Optional
from collections import Counter
from sympy.parsing import parse_expr
from sympy.parsing.latex import parse_latex
from termcolor import colored
from datetime import timedelta
import signal
import functools

tqdm.pandas()
load_dotenv('env')


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
    
    if end_reason == '</final_answer>':
        try:
            final_answer = get_final_answer(raw_output)
        except AttributeError:
            error_name = "FABlockSyntanxError"
            error_msg = None
            final_answer = None
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
        
    elif end_reason == "length":
        input_text += raw_output
        
    return input_text, final_answer, error_name, error_msg

def post_process_math_expr(raw_fa:str) -> str:
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



MODEL_ID = "Hrushi/AIMO-SFT-DDP-Merged-Deepseek-Math-7B-Base-N141128-E1-R128-A256"

print('Loading the Model')
llm = LLM(
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    # kv_cache_dtype='fp8',
    dtype='float16',
    gpu_memory_utilization=0.95,
    tensor_parallel_size=torch.cuda.device_count(),
    # trust_remote_code=True,
    # adapter_name_or_path=LORA_ADAPTER_ID,
    swap_space=50,
    enable_prefix_caching=True,
    disable_log_stats=True,
    # enable_chunked_prefill=True,
    # max_num_batched_tokens=8192,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


df = pd.read_parquet('AIMO-Math-Problems.parquet')

sampling_params = SamplingParams(
    n=5,
    temperature=0.2,
    top_p=1,
    top_k=100,
    max_tokens=1024,
    skip_special_tokens=False,
    spaces_between_special_tokens=False,
    stop=['<code_output_block>', '</final_answer>'],
    include_stop_str_in_output=True
)

system_prompt = """Your are a high school student appearing for your math exam.
You will be given a math problem to solve, and your job is to write the detailed step by step solution, with good mathematical reasoning and using python code for calculations, simplifications, solving equations, etc.

Format help and instructions:
- Problem has been given enclosed in <math_problem></math_problem> tags.
- Every step must be written within <start_of_step><end_of_step> tags.
- Code needs to be always written inside <code_block></code_block> tags - No backticks or triple backticks.
- Code output will be given in <code_output_block></code_output_block> tags

Every calculation should be performed using writing supporting python code."""

nattempted = 0
ncorrect = 0
nwrong = 0
ncorrect_top3 = 0
total_threads_collected = 0
total_correct_threads = 0
total_wrong_threads = 0
BATCH = 10000
ROOT_DIR = os.path.join('MATH_Inf_Outputs', 'AIMO-Math-Problems')
nbatch = 1
print('\n\n\n')

with tqdm(range(0, len(df), BATCH)) as pbar:
    for idx in pbar:
        open_threads = []
        closed_threads_ids = []
        closed_threads = []
        open_thread_ids = []
        rounds = 1
        sampling_params.n = 5
        final_answers = {}
        SAVED_FILES = False
        
        print(f"Batch {nbatch:0>3}")
        for jdx in tqdm(df.index.values[idx:idx+BATCH], desc='Getting Problems'):
            exp_id = df.ExpID[jdx]
            problem = df.Problem[jdx]
            ofa = post_process_math_expr(str(df.FinalAnswer[jdx]))
            nattempted += 1

            final_answers[exp_id] = {
                'ofa': ofa,
                'fas': [],
                'fa': None
            }

            save_fpath = os.path.join(ROOT_DIR, f'{exp_id}.jsonl')
            if os.path.exists(save_fpath):
                sol_df = pd.read_json(save_fpath, lines=True, orient='records')
                final_answers[exp_id]['fas'] = [str(x) for x in sol_df['final_answer'].values.tolist()]
                closed_threads.extend(sol_df['solution'].values.tolist())
                closed_threads_ids.extend([exp_id for _ in range(len(sol_df))])
                SAVED_FILES = True

            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "problem", "content": problem}
                ]

                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).replace(tokenizer.bos_token, '')
                open_threads.append(input_text)
                open_thread_ids.append(df.ExpID[jdx])

        while len(open_threads) > 0:
            print(f"\n{'==' * 50}\n")
            print(f'Performing Round: {rounds:0>2}')
            llm.llm_engine.scheduler.free_finished_seq_groups()
            responses = llm.generate(open_threads, sampling_params=sampling_params)
            new_open_threads = []
            new_open_thread_ids = []
            for kdx, resp in enumerate(responses):
                old_input_text = resp.prompt
                for out in resp.outputs:
                    end_reason = out.stop_reason

                    if end_reason is None and out.finish_reason == 'length':
                        end_reason = 'length'

                    output_text = out.text
                    next_input_text, final_answer, error_name, error_msg = post_processing_v2(output_text, old_input_text, end_reason)

                    if final_answer is not None:
                        resp_details = {
                            'solution': next_input_text,
                            'final_answer': final_answer,
                            'error_name': error_name,
                            'error_msg': error_msg,
                        }
                        closed_threads.append(resp_details)
                        closed_threads_ids.append(open_thread_ids[kdx])
                        final_answers[open_thread_ids[kdx]]['fas'].append(final_answer)

                    elif error_name is not None:
                        resp_details = {
                            'solution': next_input_text,
                            'final_answer': final_answer,
                            'error_name': error_name,
                            'error_msg': error_msg,
                        }
                        closed_threads.append(resp_details)
                        closed_threads_ids.append(open_thread_ids[kdx])
                    else:
                        assert old_input_text != next_input_text
                        if len(tokenizer.tokenize(next_input_text)) < 4096:
                            new_open_threads.append(next_input_text)
                            new_open_thread_ids.append(open_thread_ids[kdx])

            sampling_params.n = 1

            open_threads = []
            open_thread_ids = []
            for kdx, thread in enumerate(new_open_threads):
                if thread not in open_threads:
                    open_threads.append(thread)
                    open_thread_ids.append(new_open_thread_ids[kdx])

            print(f"Total Closed Threads: {len(closed_threads)}")
            print(f"Current Open Threads: {len(open_threads)}")
            # print(f"Top 3 Answers       : {Counter(final_answers).most_common(n=3)}")
            rounds += 1

            if rounds > 20:
                break

        if not SAVED_FILES:
            fas_details = {}
            for kdx in tqdm(range(len(closed_threads)), desc='Collecting threads per problem'):
                if closed_threads_ids[kdx] in fas_details:
                    fas_details[closed_threads_ids[kdx]].append(closed_threads[kdx])
                else:
                    fas_details[closed_threads_ids[kdx]] = [closed_threads[kdx]]

            for exp_id in tqdm(fas_details, desc='Saving Closed Threads per problem'):
                save_fpath = os.path.join(ROOT_DIR, f"{exp_id}.jsonl")
                with jsonlines.open(save_fpath, mode='w') as f:
                    f.write_all(fas_details[exp_id])
        
        correct_threads = 0
        for exp_id in tqdm(final_answers, desc='Processing FAs'):
            final_answers[exp_id]['fas'] = [post_process_math_expr(x) for x in final_answers[exp_id]['fas']]
            
            if len(final_answers[exp_id]['fas']) > 1:
                final_answers[exp_id]['fa'], _ = Counter(final_answers[exp_id]['fas']).most_common(n=1)[0]
            else:
                final_answers[exp_id]['fa'] = None

            if final_answers[exp_id]['ofa'] == final_answers[exp_id]['fa']:
                ncorrect += 1
            else:
                nwrong += 1
            if final_answers[exp_id]['ofa'] in [x for x, _  in Counter(final_answers[exp_id]['fas']).most_common(n=3)]:
                ncorrect_top3 += 1
                correct_threads += Counter(final_answers[exp_id]['fas'])[final_answers[exp_id]['ofa']]

        incorrect_threads = len(closed_threads) - correct_threads
        total_threads_collected += len(closed_threads)
        total_correct_threads += correct_threads
        total_wrong_threads += incorrect_threads

        # Metrics for Ledger
        nattempted_str = f"{nattempted: >6} ({round(nattempted / len(df) * 100, 2): >5}%)"
        ncorrect_str = f"{ncorrect: >6} ({round(ncorrect / nattempted * 100, 2): >5}%)"
        nwrong_str = f"{nwrong: >6} ({round(nwrong / nattempted * 100, 2): >5}%)"
        ncorrect_top3_str = f"{ncorrect_top3: >6} ({round(ncorrect_top3 / nattempted * 100, 2): >5}%)"
        total_threads_collected_str = f"{total_threads_collected: >6}"
        total_threads_collected_str = f"{total_threads_collected_str: <15}"
        total_correct_threads_str = f"{total_correct_threads: >6} ({round(total_correct_threads / total_threads_collected * 100, 2): >5}%)"
        total_wrong_threads_str = f"{total_wrong_threads: >6} ({round(total_wrong_threads / total_threads_collected * 100, 2): >5}%)"
        nattempted_str = f"{nattempted: >6} ({round(nattempted / len(df) * 100, 2): >5}%)"
        time_remaining = remaining = (pbar.total - pbar.n) / pbar.format_dict["rate"] if pbar.format_dict["rate"] and pbar.total else 0
        time_elapsed = pbar.format_dict['elapsed']

        print(f"\n{'++' * 100}\n")
        print("Running Ledger - " + colored(str(timedelta(seconds=int(time_elapsed))) + '\u2191', 'green') +" - " + colored(str(timedelta(seconds=int(time_remaining))) + '\u2193', 'red') + ":")
        print(f"\tTotal Problems Attempted\t: {colored(nattempted_str, 'black', 'on_yellow')}")
        print(f"\tProblems Solved Correctly\t: {colored(ncorrect_str, 'green')}")
        print(f"\tProblems Solved Incorrectly\t: {colored(nwrong_str, 'red')}")
        print(f"\tProblems Solved Correctly (Top3): {colored(ncorrect_top3_str, 'blue')}")
        print(f"\tTotal Threads Collected\t\t: {colored(total_threads_collected_str, 'black', 'on_yellow')}")
        print(f"\tCorrect Threads Generated\t: {colored(total_correct_threads_str, 'green')}")
        print(f"\tIncorrect Threads Generated\t: {colored(total_wrong_threads_str, 'red')}")
        print(f"\n{'++' * 100}\n")
        nbatch += 1


# %%

fas_details = {}
for kdx in tqdm(range(len(closed_threads)), desc='Saving Closed Threads'):
    #save_fpath = os.path.join(ROOT_DIR, f"{closed_threads_ids[kdx]}.jsonl")
    
    if closed_threads_ids[kdx] in fas_details:
        fas_details[closed_threads_ids[kdx]].append(closed_threads[kdx])
    else:
        fas_details[closed_threads_ids[kdx]] = [closed_threads[kdx]]
    
    # if os.path.exists(save_fpath):
    #     with jsonlines.open(save_fpath, mode='a') as f:
    #         f.write(closed_threads[kdx])
    # else:                
    #     with jsonlines.open(save_fpath, mode='w') as f:
    #         f.write(closed_threads[kdx])

# %%
fas_details['Exp03999']


