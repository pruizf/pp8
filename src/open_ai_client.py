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


def get_openai_response(oa_client, model, prompt, cf):
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
    response_format={"type": "json_object"}
  )
  td = 1000 * (time.time() - t1)
  resp = completion.choices[0].message.content
  return resp, td


def log_response_time(resp_time_df, fn, model, td, call_type):
  """
  Log the response time for a poem and model.

  Args:
      resp_time_df (pandas.DataFrame): The response time dataframe.
      fn (str): The filename of the poem.
      model (str): The model used to generate the response.
      td (float): The response time in milliseconds.
      call_type: The type of call (humor, completion, author).
  """
  resp_time_row = pd.DataFrame({"poem_id": fn.replace(".txt", ""),
                                model: td, "call_type": call_type}, index=[0])
  resp_time_df = pd.concat([resp_time_df, resp_time_row],
                           ignore_index=True)
  return resp_time_df


def write_response_to_file(cf, fn, tpl, model, resp):
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
      resp (str): The humor response from the model.

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
  with open(out_fn, mode="w") as f:
    f.write(resp)


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
      call_type (str): The type of call (humor, completion, author).

  Returns:
      pandas.DataFrame: The updated response time dataframe.
  """
  print(f"  - {call_type.capitalize()} response", fn)
  assert call_type in cf.call_types

  # Define prompt and template based on call type
  if call_type == "humor":
    prompt = pr.general_prompt_json + pr.gsep + poem_text
    tpl = cf.response_filename_tpl_js
  elif call_type == "completion":
    prompt = pr.poem_comletion_prompt_json + pr.gsep + poem_text
    tpl = cf.completion_filename_tpl_js
  else:
    prompt = pr.poem_author_prompt_json + pr.gsep + poem_text
    tpl = cf.author_filename_tpl_js

  # Get response and write to file
  resp, resp_time = get_openai_response(
    oa_client, model, prompt, cf)
  write_response_to_file(cf, fn, tpl, model, resp)
  log_prompt_to_file(cf, fn, tpl, model, prompt)

  # Return updated response time dataframe (to reuse it in the main loop)
  return log_response_time(
    resp_time_df, fn, model, resp_time, cf.call_types[call_type])


if __name__ == "__main__":
  for modu in cf, pr, ut:
    reload(modu)

  oa_client = OpenAI()
  active_models = cf.oai_models

  # dataframe to store response times
  resp_times = {"poem_id": [],
                "gpt-3.5-turbo": [],
                "gpt-4": [], "gpt-4-turbo": [],
                "gpt-4o": [],
                "call_type": []}

  # main loop
  for model in active_models:
    print(f"# Model: {model}")

    resp_time_df = pd.DataFrame(resp_times)

    reuse_df = False if active_models.index(model) == 0 else True
    if reuse_df:
      resp_time_df = pd.read_csv(cf.resp_time_df, sep="\t")
    else:
      resp_time_df = resp_time_df.astype(
        {"poem_id": "int64", "gpt-3.5-turbo": "float64", "gpt-4": "float64",
         "gpt-4-turbo": "float64", "gpt-4o": "float64", "call_type": "category"})

    for fn in sorted(os.listdir(cf.corpus_dir)):
      print("- Start poem:", fn)
      poem_text = ut.get_poem_text_by_fn(os.path.join(cf.corpus_dir, fn))
      for call_type in cf.call_types:
        resp_time_df = process_openai_response(
          oa_client, model, cf, fn, poem_text, call_type)

        # write out response times after each model
        # writing line-wise so far to avoid losing data in case of an error
        resp_time_df.to_csv(cf.resp_time_df, sep="\t", index=False)
