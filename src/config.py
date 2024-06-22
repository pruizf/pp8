"""Configuration"""

import os

# IO
corpus_dir = "../corpus"
metadata_file = os.path.join(corpus_dir, "metadata.tsv")

response_dir = "../outputs/model_responses"
response_filename_tpl = "humor_{poem_id}_{model}.txt"
response_filename_tpl_js = "humor_{poem_id}_{model}.json"
completion_filename_tpl = "completion_{poem_id}_{model}.txt"
completion_filename_tpl_js = "completion_{poem_id}_{model}.json"
author_filename_tpl = "author_{poem_id}_{model}.txt"
author_filename_tpl_js = "author_{poem_id}_{model}.json"


resp_time_dir = "../logs"
#resp_time_pkl = os.path.join(resp_time_dir, "response_times.pkl")
resp_time_df = os.path.join(resp_time_dir, "response_times_df.tsv")

call_types = {"humor": "humor", "completion": "completion", "author": "author"}

# Open AI
oai_config = {
  "temperature": 1,
  "top_p": 1
}

oai_models = ["gpt-3.5-turbo"]  # , "gpt-4", "gpt-4-turbo", "gpt-4o"]
