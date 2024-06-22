from importlib import reload
import os
import re
import time

from openai import OpenAI
import pandas as pd

# copilot: OK Tab, Dismiss Esc, Pane: Alt+Enter

import config as cf
import prompts as pr
import utils as ut

if __name__ == "__main__":
    reload(cf)
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

    # main loop
    for model in active_models:
        for fn in sorted(os.listdir(cf.corpus_dir))[0:5]:
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
            humor_resp = humor_completion.choices[0].message.content
            resp_time_row = pd.DataFrame({"poem_id": fn, model: td, "call_type": "humor"}, index=[0])
            resp_time_df = pd.concat([resp_time_df, resp_time_row], ignore_index=True)
            # figure out output file name
            resp_fn = cf.response_filename_tpl.format(poem_id=fn.replace(".txt", ""), model=model.replace(".", ""))
            techo_dir = os.path.join(cf.response_dir, re.sub(r"-.*", "", model))
            out_dir = os.path.join(techo_dir, model.replace(".", ""))
            out_fn = os.path.join(out_dir, resp_fn)
            # write response to file
            with open(out_fn, mode="w") as f:
                f.write(humor_resp)
            # poem knowledge response
            #known_text_prompt = pr.complete_poem_prompt + pr.gsep + poem_text
            # author knowledge response
            #author_prompt = pr.poem_author_prompt + pr.gsep + poem_text

