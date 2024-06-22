from importlib import reload
import os
import time

from openai import OpenAI
import pandas as pd

# copilot: OK Tab, Dismiss Esc, Pane: Alt+Enter

import config as cf
import prompts as pr
import utils as ut

if __name__ == "__main__":
    oa_client = OpenAI()
    active_models = cf.oai_models
    # dataframe to store response times
    resp_times = {"poem_id": [],
                  "gpt-3.5-turbo": [],
                  "gpt-4": [],"gpt-4-turbo": [],
                  "gpt-4o": [],
                  "call_type": []}
    resp_time_df = pd.DataFrame(resp_times)
    resp_time_df = resp_time_df.astype(
        {"poem_id": "int64", "gpt-3.5-turbo": "float64", "gpt-4": "float64",
        "gpt-4-turbo": "float64", "gpt-4o": "float64", "call_type": "category"})

    for model in active_models:
        for fn in sorted(os.listdir(cf.corpus_dir))[0:1]:
            print("- Start poem:", fn)
            poem_text = ut.get_poem_text_by_fn(os.path.join(cf.corpus_dir, fn))
            # humor response
            print("  - Humor response", fn)
            humor_prompt = pr.general_prompt + pr.gsep + poem_text
            t1 = time.time()
            humor_completion = oa_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": humor_prompt},
                ],
                temperature=cf.oai_config["temperature"],
                top_p=cf.oai_config["top_p"]
            )
            td = 1000 * (time.time() - t1)
            resp_time_row = pd.DataFrame({"poem_id": fn, model: td, "call_type": "humor"}, index=[0])
            resp_time_df = pd.concat([resp_time_df, resp_time_row], ignore_index=True)
            # poem knowledge response
            known_text_prompt = pr.complete_poem_prompt + pr.gsep + poem_text
            # author knowledge response
            author_prompt = pr.poem_author_prompt + pr.gsep + poem_text

