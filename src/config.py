"""Configuration"""

import os

# IO --------------------------------------------------------------------------
corpus_dir = "../corpus"
metadata_file = os.path.join(corpus_dir, "metadata.tsv")
log_dir = "../logs"

response_dir = "../outputs/model_responses"
response_filename_tpl = "humor_{poem_id}_{model}_{choiceNbr}.txt"
response_filename_tpl_js = "humor_{poem_id}_{model}_{choiceNbr}.json"
response_filename_tpl_js_full = "humor_{poem_id}_{model}.json"
continuation_filename_tpl = "continuation_{poem_id}_{model}.txt"
continuation_filename_tpl_js = "continuation_{poem_id}_{model}.json"
author_filename_tpl = "author_{poem_id}_{model}.txt"
author_filename_tpl_js = "author_{poem_id}_{model}.json"
# prefix for full completion responses
full_completion_pfx = "full_"

resp_time_dir = "../logs"
#resp_time_pkl = os.path.join(resp_time_dir, "response_times.pkl")
resp_time_df = os.path.join(resp_time_dir, "response_times_df.tsv")

call_types = {"humor": "humor", "continuation": "continuation", "author": "author"}

model_list = ["gpt-4o-mini"]#, "mistral-small", "mistral-large-latest_", "gpt-3.5-turbo", "gpt-4o"]
model_list_for_clf = ["gpt-4o-mini",
                      "mistral-small", "mistral-large-latest",
                      "gpt-3.5-turbo", "gpt-4o", "gpt-3.5",
                      "claude-3-5-haiku-latest", "claude-3-5-sonnet-latest",
                      "deepseek-chat", "gpt-41",
                      "gemini-15-pro", "gemini-20-flash", "mistral-medium"]
model_list_for_postpro = ["gpt-4o-mini"]


# Open AI ---------------------------------------------------------------------
#oai_models = ["gpt-3.5-turbo", "gpt-4o"]  # , "gpt-4", "gpt-4-turbo", "gpt-4o"]
oai_models = ["gpt-4.1"]  # , "gpt-4", "gpt-4-turbo", "gpt-4o"]
oai_config = {
  "temperature": 1,
  "top_p": 1,
  "seed": 14,
  "number_of_completions_humor" : 5,
  "number_of_completions_general" : 1
}

# Mistral ---------------------------------------------------------------------
mistral_models = ["mistral-medium"]#, "mistral-small", "mistral-large-latest", "gpt-3.5-turbo", "gpt-4o"]
#mistral_models = ["mistral-large-latest"]#, "mistral-small", "mistral-large-latest", "gpt-3.5-turbo", "gpt-4o"]
mistral_config = {
  "temperature": 0.7,
  "top_p": 1,
  #"seed": 14
  "number_of_completions_humor" : 5,
  "number_of_completions_general" : 1,
  "random_seed": 42
}

# Anthropic -------------------------------------------------------------------
anthropic_models_actual_versions = {'claude-3-5-sonnet-latest': 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-latest': 'claude-3-5-haiku-20241022'}
#anthropic_models = ['claude-sonnet-latest', 'claude-3-5-haiku-latest']
#anthropic_models = ['claude-sonnet-4-20250514']
#anthropic_models = ['claude-3-5-haiku-latest']
anthropic_models = ['claude-3-5-sonnet-latest']
anthropic_config = {
  "temperature": 1,
  "max_tokens": 500,
}

# Deepseek ---------------------------------------------------------------------
# chat is https://api-docs.deepseek.com/news/news250325 and reasoner is https://api-docs.deepseek.com/news/news250528
# DeepSeek-V3-0324 and DeepSeek-R1-0528 
deepseek_models = ["deepseek-reasoner"]
#deepseek_models = ["deepseek-chat", "deepseek-reasoner"]
deepseek_config = {
  "temperature": 1,
  "top_p": 1,
  #"seed": 14
  "number_of_completions_humor" : 1,
  "number_of_completions_general" : 1
}

# Gemini ---------------------------------------------------------------------
#gemini_models = ["gemini-2.0-flash"]
gemini_models = ["gemini-1.5-pro"]
gemini_config = {
  "temperature": 0.9,
  #"top_p": 1,
  #"seed": 14
  "number_of_completions_humor" : 5,
  "number_of_completions_general" : 1
}


# Analysis --------------------------------------------------------------------

stylo_dir = "../ana/stylo"
stylo_corpus_dir = os.path.join(stylo_dir, "gpt")
# number of choices the text of which we copy to files analyzed with textometry
# (Stylo, TXM etc.)
max_choices_for_textometry = 3
error_analysis_dir = "../ana/ana"
judgements_orig = ("sí", "no", "incierto")

plot_dir = "../ana/img"
clf_plot_dir = '../outputs/plots'
plot_font_family = "Arial"

rdm_seed = 42