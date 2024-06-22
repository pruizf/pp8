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


if __name__ == "__main__":
  for modu in cf, pr, ut:
    reload(modu)

  oa_client = OpenAI()
  active_models = cf.oai_models

  reuse_df = False

  # dataframe to store response times
  resp_times = {"poem_id": [],
                "gpt-3.5-turbo": [],
                "gpt-4": [], "gpt-4-turbo": [],
                "gpt-4o": [],
                "call_type": []}
  resp_time_df = pd.DataFrame(resp_times)
  if reuse_df:
    resp_time_df = pd.read_csv(cf.resp_time_df, sep="\t")
  else:
    resp_time_df = resp_time_df.astype(
      {"poem_id": "int64", "gpt-3.5-turbo": "float64", "gpt-4": "float64",
       "gpt-4-turbo": "float64", "gpt-4o": "float64", "call_type": "category"})

  # main loop
  for model in active_models:
    for fn in sorted(os.listdir(cf.corpus_dir))[0:5]:
      print("- Start poem:", fn)
      poem_text = ut.get_poem_text_by_fn(os.path.join(cf.corpus_dir, fn))

      # humor response ------------------------------------
      print("  - Humor response", fn)
      # humor_prompt = pr.general_prompt + pr.gsep + poem_text
      humor_prompt = pr.general_prompt_json + pr.gsep + poem_text
      humor_resp, h_time = get_openai_response(
        oa_client, model, humor_prompt, cf)
      write_response_to_file(cf, fn, cf.response_filename_tpl_js, model, humor_resp)
      #   log response time
      resp_time_df = log_response_time(
        resp_time_df, fn, model, h_time, cf.call_types["humor"])

      # poem knowledge response ---------------------------
      print("  - Poem knowledge response", fn)
      known_text_prompt = pr.poem_comletion_prompt_json + pr.gsep + poem_text
      known_text_resp, kt_time = get_openai_response(
        oa_client, model, known_text_prompt, cf)
      write_response_to_file(
        cf, fn, cf.completion_filename_tpl_js, model, known_text_resp)
      #   log response time
      resp_time_df = log_response_time(
        resp_time_df, fn, model, kt_time, cf.call_types["completion"])

      # author knowledge response -------------------------
      print("  - Author knowledge response", fn)
      author_prompt = pr.poem_author_prompt_json + pr.gsep + poem_text
      author_resp, a_time = get_openai_response(
        oa_client, model, author_prompt, cf)
      write_response_to_file(
        cf, fn, cf.author_filename_tpl_js, model, author_resp)
      #   log response time
      resp_time_df = log_response_time(
        resp_time_df, fn, model, a_time, cf.call_types["author"])

    # write out response times after each model
    resp_time_df.to_csv(cf.resp_time_df, sep="\t", index=False)
