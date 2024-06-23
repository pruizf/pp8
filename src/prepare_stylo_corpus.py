"""Prepare Open AI responses into Stylo oppose format"""

import config as cf
from importlib import reload
import os
import utils as ut

models_to_use = ["gpt-3.5-turbo", "gpt-4o"]
for mod in models_to_use:
  assert mod in cf.oai_models, f"Model {mod} not found in the OpenAI models list."

if __name__ == '__main__':
  reload(cf)
  reload(ut)
  if not os.path.exists(cf.stylo_corpus_dir):
    os.makedirs(cf.stylo_corpus_dir)
  mod_dir = os.path.join(cf.stylo_corpus_dir,
                         ut.clean_model_name(models_to_use[0]))
  if not os.path.exists(mod_dir):
    os.makedirs(mod_dir)
  techno_resp_dir = os.path.join(cf.response_dir, "gpt")
  resp_dir = os.path.join(techno_resp_dir, ut.clean_model_name(models_to_use[0]))
  ut.message_to_stylo_for_dir(resp_dir, mod_dir, cf.metadata_file)