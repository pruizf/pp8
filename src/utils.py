"""Utilities"""

#TODO label rewriting could be simplified

from collections import Counter
import json
import os
import re

import pandas as pd
import roman

import config as cf

def clean_model_name(model_name):
  """
  Clean the model name for use in filenames.
  """
  return model_name.replace(".", "")

def get_poem_text_by_fn(fn):
  """
  Get poem text by filename.
  Args:
      fn (str): filename
  Returns:
      str: poem text
  """
  with open(fn, "r", encoding="utf-8") as f:
    return f.read().strip()

def get_poem_text_by_id(txtid, corpus_dir=cf.corpus_dir):
  """
  Get poem text by ID.
  Args:
      txtid (int): ID for poem
      corpus_dir (str): Path for corpus directory
  Returns:
      str: poem text
  """
  with open(os.path.join(corpus_dir, f"{str.zfill(str(txtid),4)}.txt"), "r") as f:
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


def get_current_date_hms():
  """
  Get the current date formatted as a string.
  """
  from datetime import datetime
  return datetime.now().strftime('%H:%M:%S')


# Stylo part ------------------------------------------------------------------

def process_message_for_stylo(fname, md):
  """Based on the metadata at `md`, figure out which corpus the
  message belongs to for Stylo oppose (primary or secondary)"""
  #breakpoint()
  msg_txt = get_humor_message_from_resp(fname)
  # figure out which corpus the message belongs to
  # based on the metadata
  file_id = int(re.sub(r"_[^\n]+$", "",
    os.path.splitext(os.path.basename(fname))[0].replace("humor_", "")))
  has_humor = md.loc[md["id"] == file_id, "comic"].values[0]
  corpus_type = "primary" if bool(has_humor) else "secondary"
  return msg_txt, corpus_type


def message_to_stylo_for_dir(msgdir, stylo_dir, md_file, max_choices=3):
  """Process a directory of responses into Stylo oppose() format."""
  md_df = pd.read_csv(md_file, sep="\t")
  # breakpoint()
  out_primary_list = []
  out_secondary_list = []
  for fname in sorted(os.listdir(msgdir)):
    if not fname.startswith("humor"):
      continue
    choice_nbr = int(re.search(r"_(\d+)\.", fname).group(1))
    assert choice_nbr is not None, f"Choice number not found in {fname}"
    if choice_nbr > max_choices:
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


def clean_century(st):
  clean_st = re.sub(re.compile(r"^\s*Siglo\s*", re.I), "", st)
  # remove ordinal suffixes
  clean_st = re.sub(r"^(.*?)(st|nd|rd|th)$", r"\1", clean_st)
  return clean_st


# Evaluation ------------------------------------------------------------------

def normalize_judgement(jmt):
  if jmt == "incierta":
    return "incierto"
  else:
    assert jmt in cf.judgements_orig, f"Judgement {jmt} not in possible original judgements."
    return jmt.lower()

def get_author_info_for_dir(dname):
  """Get author info from a directory of responses."""
  infos = {}
  for fname in sorted(os.listdir(dname)):
    if not fname.startswith("author"):
      continue
    with open(os.path.join(dname, fname), "r") as f:
      try:
        auth_info = json.load(f)
      except json.decoder.JSONDecodeError as e:
        with open(os.path.join(dname, fname), "r") as f:
          clean_text = re.sub(re.compile(r"^.*\{", re.DOTALL), r"{", f.read())
          clean_text = re.sub(re.compile(r"\}.*$", re.DOTALL), r"}", clean_text)
          auth_info = json.loads(clean_text)
      # For Gemini responses, auth_info is a list of dicts
      if type(auth_info) is list:
        auth_info = auth_info[0]
      au_name = auth_info["author"].strip()
      # if "Juana" in au_name:
      #   breakpoint()
      century = clean_century(str(auth_info["century"]).strip())
      # assign majority class if 
      if century.lower() in ("desconocido", "no disponible"):
        print(f"  - Warning: unknown century for {au_name} in {fname}, assign 19")
        century = 19
      try:
        century = int(century)
      except ValueError:
        try:
          century = int(roman.fromRoman(century))
          #print(f"Error with century conversion: {e}")
        except Exception as e:
          try:
            century = int(century)
          except Exception as e:
            # get only first numeral
            first_century = re.sub(r"[\s-].*", "", century)
            if first_century.isdigit():
              century = int(first_century)
            else:
              century = int(roman.fromRoman(first_century))
          #print(f"Fixed error with century conversion: {e}")
    infos[os.path.basename(fname)] = [au_name, century]
    #print(auth_info)
  return infos

def group_judgement_by_prefix(dname, max_choices=cf.max_choices_for_textometry):
  """
  Get judgements grouped by the filename part before choice number).
  So far only used for analyses. Code in `get_judgement_info_for_dir()`
  already does what this function does mostly.
  Args:
      dname (str): directory name
      max_choices (int): maximum number of choices to consider
  Returns:
      dict: dictionary with judgements grouped by prefix
  """
  judgements = {}
  for fname in sorted(os.listdir(dname)):
    if not fname.startswith("humor"):
      continue
    with open(os.path.join(dname, fname), "r") as f:
      humor_info = json.load(f)
      judgement = humor_info["judgement"].strip()
    prefix = re.sub(r"_\d+\..*$", "", fname)
    if prefix not in judgements:
      judgements[prefix] = []
    if len(judgements[prefix]) < max_choices:
      judgements[prefix].append(judgement.lower())
  return judgements


def choose_among_disagreeing_judgements(jd):
  """
  Choose a judgement among disagreeing judgements for completions
  for the same poem humor prompt.

  Args:
    jd (dict): Dictionary with judgements for each completion choice

  Returns:
    dict: Dictionary with a chosen judgement per poem id
  """
  chosen_jmts = {}
  for ke, va in sorted(jd.items()):
    if len(set(va)) != 1:
      # sort by value count in descending order and get first value
      # no ties posible so far cos binary classif with odd nbr of votes
      #TODO review this if have more than two classes
      chosen_jmt = sorted(Counter(va).items(), key=lambda x: -x[-1])[0][0]
      chosen_jmts[ke] = chosen_jmt.lower()
    else:
      chosen_jmts[ke] = va[0] #if va[0] != "incierto" else "no"
  return chosen_jmts


def get_judgement_info_for_dir(dname, max_choices=cf.max_choices_for_textometry):
  """
  Get humor true/false judgement from a directory of responses.
  """
  #TODO can `group_judgements_by_prefix()` above be integrated here?
  judgements_by_prefix = {}
  for fname in sorted(os.listdir(dname)):
    if not fname.startswith("humor"):
      continue
    prefix = re.sub(r"_\d\..*$", "", fname)
    judgements_by_prefix.setdefault(prefix, [])
    with open(os.path.join(dname, fname), "r", encoding="utf-8") as f:
      try:
        humor_info = json.load(f)
      except json.decoder.JSONDecodeError as e:
        with open(os.path.join(dname, fname), "r", encoding="utf-8") as f:
          clean_text = re.sub(re.compile(r"^.*\{", re.DOTALL), r"{", f.read())
          clean_text = re.sub(re.compile(r"\}.*$", re.DOTALL), r"}", clean_text)
          humor_info = json.loads(clean_text)
      # if humor_info is list take first element
      if type(humor_info) is list:
        humor_info = humor_info[0]
      judgement = normalize_judgement(humor_info["judgement"].lower().strip())
      assert judgement in cf.judgements_orig, f"Judgement {judgement} not in possible original judgements."
      #TODO make configurable in config module and with keyword argument
      #TODO treat uncertain resposes, so that can evaluate 3-way classification
      judgement_norm = "no" if judgement.lower() == "incierto" else judgement.lower()
      if len(judgements_by_prefix[prefix]) < max_choices:
        # judgements[prefix].append(judgement)
        judgements_by_prefix[prefix].append(judgement_norm)
  # analyze judgements
  # for ke, va in judgements_by_prefix.items():
  #   if len(set(va)) != 1:
  #     print(f"  - Diverging judgements for {ke}: {repr(va)}")
  #breakpoint()
  judgements_postpro = choose_among_disagreeing_judgements(judgements_by_prefix)
  return judgements_postpro, judgements_by_prefix


def clean_up_json_response(s):
  """
  Clean up a JSON response string by removing non-json content
  
  Args:
      s (str): The JSON response string to clean.
  
  Returns:
      str: The cleaned JSON response string.
  """
  # remove the prefix and suffix that are not part of the JSON
  s = s.replace("Aquí tienes la respuesta en JSON:\n\n", "")
  s = s.replace("Aquí está mi respuesta en JSON:\n\n", "")
  s = re.sub(r"^[^{]+\{", "{", s, flags=re.DOTALL)
  return s

def postprocess_full_into_individual_responses(cf, dir_to_postpro, model, model_type="gpt"):
  """
  Post-process the full response into individual responses.

  Args:
      cf (module): The configuration module.
      dir_to_postpro (str): Name of the directory with responses to post-process.
      model (str): The model used to generate the response.

  Returns:
      None
  """
  # read response to postprocess
  for fn in sorted(os.listdir(dir_to_postpro)):
    if not fn.startswith("full_humor"):
      continue
    ffn = os.path.join(dir_to_postpro, fn)
    with open(ffn, "r") as f:
      full_resp = json.load(f)
    
    # get the related txt name (poem ID from the original filename)
    # needed for code that writes individual responses
    example_number= re.search(r"_(\d{2,})_", fn).group(1)
    assert len(example_number) > 0
    txt_for_resp = [x for x in os.listdir(cf.corpus_dir) if re.search(rf"{example_number}.txt", x)][0]  

    # write individual responses
    if "claude" in model:
      jresp = json.loads(clean_up_json_response(full_resp["content"][0]["text"]))
      judgement_orig = jresp["judgement"].lower().strip()
      judgement = normalize_judgement(judgement_orig)
      assert judgement in cf.judgements_orig, f"Judgement {judgement} not in possible original judgements."
      reason = jresp["reason"].strip()
      out_json = {}
      out_json["judgement"] = judgement
      out_json["reason"] = reason
      choice_number_re = re.search(r"full_humor_\d{4}_.+_(\d)\.json", fn)
      #breakpoint()
      if choice_number_re is None:
        choice_number = 1
      else:
        #breakpoint()
        choice_number = int(choice_number_re.group(1))
      resp_fn = cf.response_filename_tpl_js.format(
        poem_id=txt_for_resp.replace(".txt", ""), model=model.replace(".", ""), choiceNbr=choice_number)
      resp_fn = os.path.join(cf.response_dir + os.sep + model_type, model.replace(".", ""), resp_fn)
      with open(resp_fn, "w") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    elif "deepseek" in model:
      jresp = json.loads(clean_up_json_response(full_resp["choices"][0]["message"]["content"]))
      judgement_orig = jresp["judgement"].lower().strip()
      judgement = normalize_judgement(judgement_orig)
      assert judgement in cf.judgements_orig, f"Judgement {judgement} not in possible original judgements."
      reason = jresp["reason"].strip()
      out_json = {}
      out_json["judgement"] = judgement
      out_json["reason"] = reason
      choice_number_re = re.search(r"full_humor_\d{4}_.+_(\d)\.json", fn)
      #breakpoint()
      if choice_number_re is None:
        choice_number = 1
      else:
        #breakpoint()
        choice_number = int(choice_number_re.group(1))
      resp_fn = cf.response_filename_tpl_js.format(
        poem_id=txt_for_resp.replace(".txt", ""), model=model.replace(".", ""), choiceNbr=choice_number)
      resp_fn = os.path.join(cf.response_dir + os.sep + model_type, model.replace(".", ""), resp_fn)
      with open(resp_fn, "w") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    elif "gemini" in model:
      #[cand.content.parts[0].text for cand in completion.candidates]
      #jresp = json.loads(resp)
      for cidx, cand in enumerate(full_resp["candidates"]):
        jresp = json.loads(cand["content"]["parts"][0]["text"])[0]
        judgement_orig = jresp["judgement"].lower().strip()
        judgement = normalize_judgement(judgement_orig)
        assert judgement in cf.judgements_orig, f"Judgement {judgement} not in possible original judgements."
        reason = jresp["reason"].strip()
        out_json = {}
        out_json["judgement"] = judgement
        out_json["reason"] = reason
        # write out
        resp_fn = cf.response_filename_tpl_js.format(
          poem_id=txt_for_resp.replace(".txt", ""), model=model.replace(".", ""), choiceNbr=cidx+1)
        resp_fn = os.path.join(cf.response_dir + os.sep + model_type, model.replace(".", ""), resp_fn)
        with open(resp_fn, "w") as f:
          json.dump(out_json, f, indent=2, ensure_ascii=False)
      pass
    else:
      for idx, resp in enumerate(full_resp["choices"]):
        # extract content to write to individual response files
        jresp = json.loads(resp["message"]["content"])
        judgement_orig = jresp["judgement"].lower().strip()
        judgement = normalize_judgement(judgement_orig)
        assert judgement in cf.judgements_orig, f"Judgement {judgement} not in possible original judgements."
        reason = jresp["reason"].strip()
        out_json = {}
        out_json["judgement"] = judgement
        out_json["reason"] = reason
        # write out
        resp_fn = cf.response_filename_tpl_js.format(
          poem_id=txt_for_resp.replace(".txt", ""), model=model.replace(".", ""), choiceNbr=idx+1)
        resp_fn = os.path.join(cf.response_dir + os.sep + model_type, model.replace(".", ""), resp_fn)
        with open(resp_fn, "w") as f:
          json.dump(out_json, f, indent=2, ensure_ascii=False)
