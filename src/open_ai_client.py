"""Open AI client to see humor judgments, poem continuation, and author knowledge"""

from importlib import reload
import json
import os
import re
import time

from openai import OpenAI
from openai.types.chat import ChatCompletion
import pandas as pd


import config as cf
import prompts as pr
import utils as ut


def get_openai_response(oa_client, model, prompt, cf, call_type):
  """
  Returns Open AI response and response time.

  Args:
      oa_client (openai.OpenAI): The OpenAI client.
      model (str): The model to use for generating the response.
      prompt: The prompt to use for generating the response.
      cf (module): The configuration module.

  Returns:
      tuple: A tuple containing the humor response and the response time in seconds.
  """
  t1 = time.time()
  completion = oa_client.chat.completions.create(
    model=model,
    messages=[
      {"role": "user", "content": prompt},
    ],
    temperature=cf.oai_config["temperature"],
    top_p=cf.oai_config["top_p"],
    response_format={"type": "json_object"},
    #seed=cf.oai_config["seed"]
    n=cf.oai_config["number_of_completions_humor"] \
      if call_type == "humor" else cf.oai_config["number_of_completions_general"]
  )
  td = 1000 * (time.time() - t1)
  #breakpoint()
  resps = [resp.message.content for resp in completion.choices]
  return completion, resps, td


def log_response_time(rtdf, fn, model, td, call_type):
  """
  Log the response time for a poem and model.

  Args:
      rtdf (pandas.DataFrame): The response time dataframe.
      fn (str): The filename of the poem.
      model (str): The model used to generate the response.
      td (float): The response time in milliseconds.
      call_type: The type of call (humor, continuation, author).
  """
  row_for_poem = rtdf.loc[
    (rtdf["poem_id"] == int(fn.replace(".txt", ""))) &
    (rtdf["call_type"] == call_type)]
  if not row_for_poem.empty:
    rtdf.loc[
      (rtdf["poem_id"] == int(fn.replace(".txt", ""))) &
      (rtdf["call_type"] == call_type), model] = td
  else:
    rtdf = pd.concat([rtdf, pd.DataFrame(
      {"poem_id": fn.replace(".txt", ""),
       model: td, "call_type": call_type}, index=[0])],
      ignore_index=True)
  return rtdf


def write_resp_message_to_file(cf, fn, tpl, model, resps):
  """
  Write the completion message from the response to a file.

  Args:
      cf (module): The configuration module.
      fn (str): The filename of the poem.
      tpl (str): Formatted string with template for output filename.
        Template format is: prefix_{poem_id}_{model}_{choiceNbr}.txt,
        where the prefix describes the type of response
        (humor judgment, poem completion, author, etc.),
        and choiceNbr identifies different completion choices.
      model (str): The model used to generate the response.
      resps (list): List of humor responses from the model.

  Returns:
      None
  """
  # figure out output file name
  for idx, resp in enumerate(resps):
    resp_fn = tpl.format(
      poem_id=fn.replace(".txt", ""),
      model=model.replace(".", ""),
      choiceNbr=idx+1)
    techno_dir = os.path.join(
      cf.response_dir, re.sub(r"-.*", "", model))
    out_dir = os.path.join(techno_dir, model.replace(".", ""))
    out_fn = os.path.join(out_dir, resp_fn)
    # write response to file
    with open(out_fn, mode="w") as f:
      f.write(resp)


def write_completion_to_file(cf, fn, tpl, model, comp: ChatCompletion):
  """
  Write the response to a file.

  Args:
      cf (module): The configuration module.
      fn (str): The filename of the poem.
      tpl (str): Formatted string with template for output filename.
        Template format is: prefix_{poem_id}_{model}.txt,
        where the prefix describes the type of response
        (humor judgment, poem completion, author, etc.).
      model (str): The model used to generate the response.
      comp (ChatCompletion): Entire completion response.

  Returns:
      None
  """
  # figure out output file name
  resp_fn = tpl.format(
    poem_id=fn.replace(".txt", ""),
    model=model.replace(".", ""))
  techno_dir = os.path.join(
    cf.response_dir, re.sub(r"-.*", "", model))
  out_dir = os.path.join(techno_dir, model.replace(".", ""))
  out_fn = os.path.join(out_dir, resp_fn)
  # write response to file
  comp_txt = comp.model_dump_json()
  jso = json.loads(comp_txt)
  with open(out_fn, mode="w") as f:
    json.dump(jso, f, indent=2)


def log_prompt_to_file(cf, fn, tpl, model, prompt):
  """
  Log prompt to file (used for verifications)

  Args:
      cf (module): The configuration module.
      fn (str): The filename of the poem.
      tpl (str): Formatted string with template for output filename.
        Template format is: prefix_{poem_id}_{model}.txt,
        where the prefix describes the type of response
        (humor judgment, poem completion, author, etc.).
      model (str): The model used to generate the response.
      prompt (str): The prompt to log.

  Returns:
      None
  """
  # figure out output file name
  resp_fn = tpl.format(
    poem_id=fn.replace(".txt", ""),
    model=model.replace(".", ""))
  out_fn = os.path.join(cf.log_dir, resp_fn.replace(".json", ".txt"))
  # write response to file
  with open(out_fn, mode="w") as f:
    f.write(prompt)


def prepare_poem_for_completion_prompt(poem_text, n=4):
  """
  Prepare the poem for the completion prompt by returning only
  the title and first n lines (default is 4).

  Args:
      poem_text (str): The text of the poem.
      n (int): The number of lines to keep. Default is 4.

  Returns:
      str: The title plus first n lines (default 4) of the poem
  """
  lines = poem_text.split("\n")
  lines_no_blanks = [line for line in lines if len(line.strip()) > 0]
  title = lines_no_blanks[0].strip()
  keep = lines_no_blanks[1:n+1]
  return title + "\n\n" + "\n".join(keep)


def process_openai_response(oa_client, model, cf, fn, poem_text, call_type):
  """
  Process the OpenAI response for a given text and write it to a file;
  prompt and template are defined based on the call type (humor, completion, author).
  Also log the response time.

  Args:
      oa_client (openai.OpenAI): The OpenAI client.
      model (str): The model to use for generating the response.
      cf (module): The configuration module.
      fn (str): The filename of the poem.
      poem_text (str): The text of the poem.
      call_type (str): The type of call (humor, continuation, author).

  Returns:
      pandas.DataFrame: The updated response time dataframe.
  """
  print(f"  - {call_type.capitalize()} response", fn)
  assert call_type in cf.call_types

  # Define prompt and template based on call type
  if call_type == "humor":
    prompt = pr.general_prompt_json + pr.gsep + poem_text
    tpl = cf.response_filename_tpl_js
    tpl_full = cf.response_filename_tpl_js_full
  elif call_type == "continuation":
    poem_text = prepare_poem_for_completion_prompt(poem_text)
    prompt = pr.poem_continuation_prompt_json + pr.gsep + poem_text
    tpl = cf.continuation_filename_tpl_js
    tpl_full = tpl
  else:
    prompt = pr.poem_author_prompt_json + pr.gsep + poem_text
    tpl = cf.author_filename_tpl_js
    tpl_full = tpl

  # Get full completion, mesage content, and write to file
  comp, resps, resp_time = get_openai_response(
    oa_client, model, prompt, cf, call_type)
  write_resp_message_to_file(cf, fn, tpl_full, model, resps)
  write_completion_to_file(cf, fn, cf.full_completion_pfx + tpl_full, model, comp)
  log_prompt_to_file(cf, fn, tpl_full, model, prompt)

  # Return updated response time dataframe (to reuse it in the main loop)
  return log_response_time(
    resp_time_df, fn, model, resp_time, cf.call_types[call_type])


if __name__ == "__main__":
  for modu in cf, pr, ut:
    reload(modu)

  oa_client = OpenAI()
  active_models = cf.model_list

  #breakpoint()

  # dataframe to store response times
  resp_times = {"poem_id": [],
                "gpt-3.5-turbo": [],
                "gpt-4": [], "gpt-4-turbo": [],
                "gpt-4o": [],
                "gpt-4o-mini": [],
                "call_type": []}

  # main loop
  for model in active_models:
    print(f"# Model: {model}")

    reuse_df = False if active_models.index(model) == 0 else True
    #reuse_df = True
    if reuse_df:
      resp_time_df = pd.read_csv(cf.resp_time_df, sep="\t")
    else:
      resp_time_df = pd.DataFrame(resp_times)
      resp_time_df = resp_time_df.astype(
        {"poem_id": "int64", "gpt-3.5-turbo": "float64", "gpt-4": "float64",
         "gpt-4-turbo": "float64", "gpt-4o": "float64",
         "gpt-4o-mini": "float64",
         "call_type": "category"})

    for fn in sorted(os.listdir(cf.corpus_dir)):
      if fn == "metadata.tsv":
        continue
      print("- Start poem:", fn)
      poem_text = ut.get_poem_text_by_fn(os.path.join(cf.corpus_dir, fn))
      for call_type in cf.call_types:
      #for call_type in ['humor', 'author']:
        resp_time_df = process_openai_response(
          oa_client, model, cf, fn, poem_text, call_type)

        # write out response times after each model
        # writing line-wise so far to avoid losing data in case of an error
        resp_time_df.to_csv(cf.resp_time_df, sep="\t", index=False)
