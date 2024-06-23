"""Utilities"""

import config as cf
import json
import os
import re

import pandas as pd

def clean_model_name(model_name):
  """Clean the model name for use in filenames."""
  return model_name.replace(".", "")

def get_poem_text_by_fn(fn):
  """
  Get poem text by filename.
  Args:
      fn (str): filename
  Returns:
      str: poem text
  """
  with open(fn, "r") as f:
    return f.read().strip()


def get_humor_message_from_resp(fname):
  """
  Get the humor explanation message from Open AI response.
  Args:
      resp (fname): response file name (processed with `open_ai_client.write_resp_message_to_file`)
  Returns:
      str: humor explanation
  """
  with open(fname, "r") as f:
    resp = f.read().strip()
  return json.loads(resp)["reason"]


def process_message_for_stylo(fname, md):
  """Based on the metadata at `md`, figure out which corpus the
  message belongs to for Stylo (primary or secondary)"""
  #breakpoint()
  msg_txt = get_humor_message_from_resp(fname)
  # figure out which corpus the message belongs to
  # based on the metadata
  file_id = int(re.sub(r"_[^\n]+$", "",
    os.path.splitext(os.path.basename(fname))[0].replace("humor_", "")))
  has_humor = md.loc[md["id"] == file_id, "comic"].values[0]
  corpus_type = "primary" if bool(has_humor) else "secondary"
  return msg_txt, corpus_type


def message_to_stylo_for_dir(msgdir, stylo_dir, md_file):
  """Process a directory of responses into Stylo format."""
  md_df = pd.read_csv(md_file, sep="\t")
  # breakpoint()
  out_primary_list = []
  out_secondary_list = []
  for fname in sorted(os.listdir(msgdir)):
    if not fname.startswith("humor"):
      continue
    msg_txt, corpus_type = process_message_for_stylo(
      os.path.join(msgdir, fname), md_df)
    if corpus_type == "primary":
      out_primary_list.append(msg_txt)
    else:
      out_secondary_list.append(msg_txt)

  primary_dir = os.path.join(stylo_dir, "primary_set")
  secondary_dir = os.path.join(stylo_dir, "secondary_set")
  for dname in primary_dir, secondary_dir:
    if not os.path.exists(dname):
      os.makedirs(dname)
  dir_series = primary_dir, secondary_dir
  out_fn_series = "humor_true.txt", "humor_false.txt"
  out_list_series = out_primary_list, out_secondary_list
  for dname, fname, out_list in zip(dir_series, out_fn_series, out_list_series):
    with open(os.path.join(dname, fname), "w") as f:
      f.write("\n".join(out_list))
