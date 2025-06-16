from importlib import reload

import config as cf
import utils as ut


if __name__ == '__main__':
  reload(cf)
  reload(ut)
  False and ut.postprocess_full_into_individual_responses(
    cf,"../outputs/model_responses/claude/claude-3-5-haiku-latest", "claude-3-5-haiku-latest",
  model_type="claude")
  False and ut.postprocess_full_into_individual_responses(
    cf,"../outputs/model_responses/claude/claude-3-5-sonnet-latest", "claude-3-5-sonnet-latest",
  model_type="claude")
  False and ut.postprocess_full_into_individual_responses(
    cf,"../outputs/model_responses/mistral/mistral-small", "mistral-small",
  model_type="mistral")
  False and ut.postprocess_full_into_individual_responses(
    cf,"../outputs/model_responses/mistral/mistral-large-latest", "mistral-large-latest",
  model_type="deepseek")
  False and ut.postprocess_full_into_individual_responses(
    cf,"../outputs/model_responses/deepseek/deepseek-chat", "deepseek-chat",
  model_type="deepseek")
  False and ut.postprocess_full_into_individual_responses(
    cf,"../outputs/model_responses/gpt/gpt-41", "gpt-41",
  model_type="gpt")
  False and ut.postprocess_full_into_individual_responses(
    cf,"../outputs/model_responses/gemini/gemini-15-pro", "gemini-15-pro",
  model_type="gemini")
  False and ut.postprocess_full_into_individual_responses(
    cf,"../outputs/model_responses/gemini/gemini-20-flash", "gemini-20-flash",
  model_type="gemini")
  ut.postprocess_full_into_individual_responses(
    cf,"../outputs/model_responses/mistral/mistral-medium", "mistral-medium",
  model_type="mistral")
