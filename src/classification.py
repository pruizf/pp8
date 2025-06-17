import argparse
from functools import partial
from importlib import reload
import json
import os
import re
from types import ModuleType
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk.corpus
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
import spacy
import warnings
warnings.filterwarnings('ignore')

import config as cf
import utils as ut


# Set the random seed for reproducibility
np.random.seed(cf.rdm_seed)

# Build dataframe

def read_metadata(conf: ModuleType) -> pd.DataFrame:
  """
  Read metadata file.
  Args:
    conf (ModuleType): The configuration module.
  Returns:
    pd.DataFrame: The metadata in a dataframe.
  """
  md = pd.read_csv(cf.metadata_file, sep="\t", encoding="utf-8")
  return md


def collect_data(ddir: str, model_name: str, md: pd.DataFrame) -> pd.DataFrame:
  """
  Get data for classification from a directory.
  Args:
    ddir (str): The path to the directory containing the data.
    model_name (str): The name of the model.
    md (pd.DataFrame): The metadata in a dataframe

  Returns:
    dict: Data for classification in a dictionary of lists, dictionary keys are column names.
      for a dataframe that will be created later
  """
  assert model_name in cf.model_list_for_clf, f"Model {model_name} not in the list of models."
  model_name = model_name.replace(".", "")
  print(f"  - Collecting data for {model_name} from {ddir} [{ut.get_current_date_hms()}]")
  data = {"text": [], "model": [], "humorSys": [], "exampleNumber": [],
          "centuryBirth": [], "completionNumber": [], "humorGold": []}
  for fname in sorted(os.listdir(ddir)):
    if fname.startswith("humor") and re.search(rf"_\d\.json", fname):
      example_number = int(fname.split("_")[1])
      completion_number = int(fname.split("_")[3].split(".")[0])
      with open(os.path.join(ddir, fname), "r", encoding="utf-8") as f:
        jso = json.load(f)
      text = jso["reason"]
      data["text"].append(text)
      data["exampleNumber"].append(example_number)
      data["centuryBirth"].append(md.loc[md["id"] == example_number, "centuryBirth"].values[0])
      data["completionNumber"].append(completion_number)
      data["humorGold"].append(md.loc[md["id"] == example_number, "comic"].values[0])
      data["humorSys"] = jso["judgement"]
      data["model"].append(model_name)
  return data


def split_into_tokens_spacy(text: str, no_punct=False) -> list:
  """
    Split text into tokens using spacy.
  Args:
    text: Text to tokenize

  Returns:
    list: List of tokens
  """
  doc = spacy_pipeline(text)
  if no_punct:
    doc = [w.text for w in doc if not w.is_punct and not w.is_space]
  else:
    doc = [w.text for w in doc]
  return doc


def split_into_tokens_spacy_no_punct(text: str) -> list:
  """
    Split text into tokens using spacy.
  Args:
    text: Text to tokenize

  Returns:
    list: List of tokens
  """
  doc = spacy_pipeline(text)
  doc = [w.text for w in doc if not w.is_punct and not w.is_space]
  return doc


def pos_spacy(text: str, no_punct=False) -> list:
  """
    Split text into tokens using spacy.
  Args:
    text: Text to tokenize

  Returns:
    list: List of tokens
  """
  doc = spacy_pipeline(text)
  if no_punct:
    doc = [w.pos_ for w in doc if not w.is_punct and not w.is_space]
  else:
    doc = [w.pos_ for w in doc]
  return doc


def pos_spacy_no_punct(text: str) -> list:
  """
    Split text into tokens using spacy.
  Args:
    text: Text to tokenize

  Returns:
    list: List of tokens
  """
  doc = spacy_pipeline(text)
  doc = [w.pos_ for w in doc if not w.is_punct and not w.is_space]
  return doc


def plot_confusion_matrix(cm, classes, out_fn, hl_low=False, cmap='Greens'):
  title = 'Confusion matrix'
  fig, ax = plt.subplots(figsize=(11, 8))
  if hl_low:
    # Apply gamma correction with gamma < 1 to highlight low values
    gamma = 0.5
    cm_hl = np.power(cm, gamma)    
    #sns.heatmap(cm_gamma, annot=cm, cmap=cmap, xticklabels=classes, yticklabels=classes)
  else:
    cm_hl = cm
    sns.heatmap(cm_hl, annot=True, cmap=cmap, xticklabels=classes, yticklabels=classes)
  plt.title(title)
  plt.xlabel('pred')
  plt.ylabel('gold')
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  plt.savefig(out_fn, dpi=300)
  plt.show()


def plot_top_features(coef_df, out_fn, categ, n=20, save_format='png'):
  """Bar plot for the top features for a given category."""
  assert save_format in ('png', 'pdf')
  top_words = coef_df.loc[categ].sort_values(ascending=False).head(n)
  plt.figure(figsize=(11, 8))
  plt.barh(top_words.index, top_words.values, color='green')
  plt.xlabel("Coefficient")
  plt.title(f"Top {n} features for {categ}")
  ax = plt.gca()
  ax.invert_yaxis()
  ax.spines['top'].set_visible(False) 
  ax.spines['right'].set_visible(False) 
  if save_format == 'png':
    plt.savefig(out_fn, format='png', dpi=300)
  else:
    plt.savefig(out_fn)
  plt.clf()
  #plt.show()

def parse_args():
  """
  Parse command line arguments.
  """
  parser = argparse.ArgumentParser(description="Classification experiments.")
  parser.add_argument("batch_name", type=str)
  parser.add_argument("--remove_punct", "-r", action="store_true", help="Remove punctuation based on spacy token objects")
  parser.add_argument("--hl_low", "-r", action="store_true", help="Make low values darker in confusion matrix")
  return parser.parse_args()


if __name__ == "__main__":
  for modu in cf, ut:
    reload(modu)
    
  args = parse_args()

  batch_name = args.batch_name
  
  mpl.rcParams['font.family'] = cf.plot_font_family

  print(f"# Start [{ut.get_current_date_hms()}]")
  if args.remove_punct:
    print("  - Removing punctuation from tokens")
  mddf = read_metadata(cf)
  data_35 = collect_data(os.path.join(cf.response_dir, "gpt" + os.sep + "gpt-35-turbo") , "gpt-3.5-turbo", mddf)
  data_4o = collect_data(os.path.join(cf.response_dir, "gpt" + os.sep + "gpt-4o") , "gpt-4o", mddf)
  data_4o_mini = collect_data(os.path.join(cf.response_dir, "gpt" + os.sep + "gpt-4o-mini") , "gpt-4o-mini", mddf)
  data_41 = collect_data(os.path.join(cf.response_dir, "gpt" + os.sep + "gpt-41") , "gpt-41", mddf)
  data_haiku = collect_data(os.path.join(cf.response_dir, "claude" + os.sep + "claude-3-5-haiku-latest") , "claude-3-5-haiku-latest", mddf)
  data_sonnet = collect_data(os.path.join(cf.response_dir, "claude" + os.sep + "claude-3-5-sonnet-latest") , "claude-3-5-sonnet-latest", mddf)
  data_mistral_s = collect_data(os.path.join(cf.response_dir, "mistral" + os.sep + "mistral-small") , "mistral-small", mddf)
  data_mistral_l = collect_data(os.path.join(cf.response_dir, "mistral" + os.sep + "mistral-large-latest") , "mistral-large-latest", mddf)
  data_mistral_m = collect_data(os.path.join(cf.response_dir, "mistral" + os.sep + "mistral-medium") , "mistral-medium", mddf)
  data_deepseek = collect_data(os.path.join(cf.response_dir, "deepseek" + os.sep + "deepseek-chat") , "deepseek-chat", mddf)
  data_gemini_15 = collect_data(os.path.join(cf.response_dir, "gemini" + os.sep + "gemini-15-pro") , "gemini-15-pro", mddf)
  data_gemini_20 = collect_data(os.path.join(cf.response_dir, "gemini" + os.sep + "gemini-20-flash") , "gemini-20-flash", mddf)
  #data_35 = collect_data(os.path.join(cf.response_dir, "gpt" + os.sep + "gpt-35-turbo") , "gpt-3.5", mddf)
  data_all = {k: data_35[k] + data_4o[k] + data_4o_mini[k] + data_haiku[k] + data_sonnet[k] +
              data_mistral_s[k] + data_mistral_l[k] + data_mistral_m[k] +
              data_deepseek[k] + data_41[k] +
              data_gemini_15[k] + data_gemini_20[k] 
              for k in data_35.keys()}
  #data_all = {k: data_35[k] + data_4o[k] for k in data_35.keys()}
  df = pd.DataFrame(data_all)

  #labelCol = "humorLabel"
  labelCol = "model"
  class_names = sorted(df[labelCol].unique())
  label2id = {class_names[i]: i for i in range(len(class_names))}
  id2label = {i: class_names[i] for i in range(len(class_names))}

  print(f"  - Loading spacy model [{ut.get_current_date_hms()}]")
  spacy_pipeline = spacy.load("es_core_news_sm", disable=["parser", "ner"])
  print("    - Done")

  # will use NLTK stopwords instead of spacy's since latter seem too restrictive (too many words removed)
  stopwords = nltk.corpus.stopwords.words("spanish")

  # corpus vectorizers
  #chosen_tokenizer = split_into_tokens_spacy if not remove_punct else split_into_tokens_spacy_no_punct
  tok_vectorizer = TfidfVectorizer(lowercase=True,
                                   #tokenizer=chosen_tokenizer,
                                   tokenizer=partial(split_into_tokens_spacy, no_punct=args.remove_punct),
                                   stop_words=stopwords,
                                   min_df=0.01)
  #chosen_pos_tagging = pos_spacy if not remove_punct else pos_spacy_no_punct
  pos_vectorizer = TfidfVectorizer(lowercase=True,
                                   tokenizer=partial(pos_spacy, no_punct=args.remove_punct),
                                   stop_words=stopwords,
                                   min_df=0.01)

  # column transformer
  column_trans_wf_pos = ColumnTransformer(
    [
      ('reason', tok_vectorizer, 'text'),
      ('pos', pos_vectorizer, 'text'),
    ],
    verbose=True,
  )

  df_train, df_test = train_test_split(df, test_size=0.2, random_state=cf.rdm_seed, stratify=df["model"])

  X_train = df_train.drop(columns=[labelCol])
  y_train = df_train[labelCol]
  X_test = df_test.drop(columns=[labelCol])
  y_test = df_test[labelCol]

  scores = []
  model_names = ["LR", "SVC", "NB"]
  scoring = "Macro F1"

  models = [('LR', LogisticRegression()),
            ('SVC', LinearSVC()),
            ('NB', MultinomialNB()),]

  for model_name, model in models:
    print(f"  - Training {model_name} [{ut.get_current_date_hms()}]")
    clf_pipeline = make_pipeline(column_trans_wf_pos, model)
    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)
    print("Classification report:\n\n{}".format(classification_report(y_test, y_pred, digits=4)))
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    scores.append(f1)
    
    # feature importances
    tok_feature_names = clf_pipeline.named_steps["columntransformer"].named_transformers_['reason'].get_feature_names_out()
    pos_feature_names = clf_pipeline.named_steps["columntransformer"].named_transformers_['pos'].get_feature_names_out()
    
    feature_names = tok_feature_names.tolist() + pos_feature_names.tolist()
    if model_name == "LR":
      coeffs = clf_pipeline.named_steps['logisticregression'].coef_
    elif model_name == "SVC":
      coeffs = clf_pipeline.named_steps['linearsvc'].coef_
    coeffs_df = pd.DataFrame(coeffs, columns=feature_names, index=model.classes_)
    
    n_out_feats = 30
    n_plot_feats = 20
    out_fn = os.path.join(cf.clf_plot_dir, f"top_features_{model_name}_{str.zfill(str(batch_name), 3)}.txt")
    if os.path.exists(out_fn):
      os.remove(out_fn)
    for categ in model.classes_:
      print(f"\nTop features for {categ}:")
      print(coeffs_df.loc[categ].sort_values(ascending=False).head(n_out_feats))
      out_plot_fn = os.path.join(cf.clf_plot_dir, f"top_features_{model_name}_{categ}_{str.zfill(str(batch_name), 3)}.png")
      plot_top_features(coeffs_df, out_plot_fn, categ=categ, n=n_plot_feats)
      with open(out_fn, "a") as f:
        coeffs_df.loc[categ].sort_values(ascending=False).head(n_out_feats).to_csv(f, header=True, sep="\t")
        f.write("\n")
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_, normalize="true")
    out_cm_text = os.path.join(cf.clf_plot_dir,
      f"cm_{model_name}_{str.zfill(str(batch_name), 3)}.txt")
    out_cr_text = os.path.join(cf.clf_plot_dir,
      f"cr_{model_name}_{str.zfill(str(batch_name), 3)}.txt")
    with open(out_cm_text, "w") as cmf:
      cmf.write(str(cm))
    with open(out_cr_text, "w") as crf:
      crf.write(str(classification_report(y_test, y_pred, digits=4)))
    out_cm_fn = os.path.join(cf.clf_plot_dir,
      f"cm_{model_name}_{str.zfill(str(batch_name), 3)}.png")
    plot_confusion_matrix(cm, model.classes_, out_cm_fn, hl_low=args.hl_low)
