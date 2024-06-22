"""Configuration"""

import os

# IO
corpus_dir = "../corpus"
metadata_file = os.path.join(corpus_dir, "metadata.tsv")
response_dir = "../outputs/model_responses"
response_filename_tpl = "response_{poem_id}_{model}.txt"
response_filename_tpl_js = "response_{poem_id}_{model}.json"

# Open AI
oai_config = {
  "temperature": 1,
  "top_p": 1
}

oai_models = ["gpt-3.5-turbo"]  # , "gpt-4", "gpt-4-turbo", "gpt-4o"]
