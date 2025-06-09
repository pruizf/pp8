"""Prepare Open AI responses into Stylo oppose format"""

import config as cf
from importlib import reload
import os
import utils as ut

#models_to_use = ["gpt-3.5-turbo", "gpt-4o"]
models_to_use = ["gpt-4o-mini", "mistral-small", "mistral-large-latest"]

if __name__ == '__main__':
  reload(cf)
  reload(ut)
  for mod in models_to_use:
    assert mod in cf.model_list, f"Model {mod} not found in the model list."
  if not os.path.exists(cf.stylo_corpus_dir):
    os.makedirs(cf.stylo_corpus_dir)
  for mod_name in models_to_use:
    print("- Model:", mod_name)
    mod_dir = os.path.join(cf.stylo_corpus_dir,
                           ut.clean_model_name(mod_name))
    if not os.path.exists(mod_dir):
      os.makedirs(mod_dir)
    techno_resp_dir = os.path.join(cf.response_dir, "gpt") if mod_name.startswith("gpt") \
      else os.path.join(cf.response_dir, "mistral")
    resp_dir = os.path.join(techno_resp_dir, ut.clean_model_name(mod_name))
    ut.message_to_stylo_for_dir(resp_dir, mod_dir, cf.metadata_file, max_choices=3)
