# IMPORTS
import os
from tqdm.auto import tqdm
import json
from dotenv import load_dotenv, find_dotenv
import re
import pandas as pd

import google.auth
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold,
    GenerationResponse
)
import asyncio
from asynciolimiter import Limiter
from tqdm.asyncio import tqdm_asyncio
import jsonlines

# Setup
tqdm.pandas()
print(f"Loaded Environment Variables: {load_dotenv(find_dotenv())}")

# Setting Up LLM
def llm_setup(
        model_name:str='gemini-1.5-flash-001',
        max_output_tokens:int=8192,
        temperature:float=0.2,
        top_p:float=1.0,
        top_k:int=40,
        stop_sequences:list=[],
):
    # Set up the Gemini Model
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    generation_config = GenerationConfig(
        candidate_count=1,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_sequences=stop_sequences,
        response_mime_type="text/plain",
    )

    gemini_flash_model = GenerativeModel(
        model_name=model_name,
        safety_settings=safety_settings,
        generation_config=generation_config
    )

    return gemini_flash_model


# Perform Async Calls
async def get_raw_output_from_llm(
    df: pd.DataFrame,
    llm: GenerativeModel,
    interim_results_df: pd.DataFrame,
    interim_fpath: str,
    prompt_col: str = "Prompt",
    exp_id_col: str = "Exp_ID",
    save_every: int = 100,
    rpm: int = 200,
):
    # Setup Limiter
    rate_limiter = Limiter(rpm // 60)

    ## Async Function Call to Model ##
    async def gemini_async_call(text: str, exp_id: str):
        # with semaphore:
        await rate_limiter.wait()
        try:
            return await llm.generate_content_async(text), exp_id
        except Exception:
            return None, exp_id

    ## Main Function ##
    tasks = []
    # too_long_counter = 0
    interim_results = []
    for _, current_row in tqdm(df.iterrows(), total=len(df), desc="Creating Tasks"):
        if current_row[exp_id_col] in interim_results_df.exp_id.values:
            continue

        input_prompt = current_row[prompt_col]
        exp_id = current_row[exp_id_col]
        tasks.append(asyncio.create_task(gemini_async_call(input_prompt, exp_id)))

    # print(f"Too Long Counter: {too_long_counter}")
    print(f"{len(tasks)} Tasks Created")
    counter = 0
    safety_counter = 0
    input_not_processed_counter = 0
    for done in tqdm_asyncio.as_completed(tasks, ncols=100, desc="Waiting for Tasks:", leave=True):
        try:
            result, exp_id = await done
            result: GenerationResponse
            counter += 1

            if result is None:
                raise Exception

            try:
                interim_results.append(
                    {
                        "exp_id": exp_id,
                        "text": result.text,
                        "input_tokens": result._raw_response.usage_metadata.prompt_token_count,
                        "output_tokens": result._raw_response.usage_metadata.candidates_token_count,
                    }
                )

            except Exception:
                safety_counter += 1
                interim_results.append(
                    {
                        "exp_id": exp_id,
                        "text": None,
                        "input_tokens": None,
                        "output_tokens": None,
                    }
                )
                continue

        except Exception:
            # print(f"Error at index {idx} - input not processed")
            input_not_processed_counter += 1
            interim_results.append(
                {
                    "exp_id": exp_id,
                    "text": None,
                    "input_tokens": None,
                    "output_tokens": None,
                }
            )

        # Save to file
        if counter % save_every == 0:
            with jsonlines.open(interim_fpath, "a") as writer:
                writer.write_all(interim_results)
            interim_results = []

    print(f"Number of Safety Blocks: {safety_counter}")
    print(f"Number of Input not processed: {input_not_processed_counter}")

    with jsonlines.open(interim_fpath, "a") as writer:
        writer.write_all(interim_results)

if __name__ == "__main__":
    # Setup Credentials
    credentials, project_id = google.auth.default()
    LOCATION = os.getenv("GCP_LOCATION")
    vertexai.init(project=os.getenv("GCP_PROJECT"), location=LOCATION, credentials=credentials)

    # Setup LLM
    llm = llm_setup()

    # Run
    intermedate_df = asyncio.run(
    get_raw_output_from_llm(
        df=df,
        llm=gemini_flash_model,
        interim_results_df=interim_results_df,
        interim_fpath=interim_fpath, 
        prompt_col='Prompt',
        exp_id_col='Exp_ID',
        save_every=10,
        rpm=180,
    )
)

