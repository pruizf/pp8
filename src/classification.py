from importlib import reload
import json
import os
import string
from datetime import datetime
from types import ModuleType

# Suppression de l'affichage des messages d'avertissement
import warnings
warnings.filterwarnings('ignore')

# import matplotlib.pyplot as plt
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
# from sklearn import metrics, set_config
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDClassifier, RidgeClassifier
# from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
# from sklearn.pipeline import Pipeline, make_pipeline
# import spacy

import config as cf
import utils as ut


# Set the random seed for reproducibility
np.random.seed(42)

# Build dataframe

def read_metadata(conf: ModuleType) -> pd.DataFrame:
  """
  Read metadata file.
  Args:
    conf (ModuleType): The configuration module.
  Returns:
    pd.DataFrame: The metadata in a dataframe.
  """
  md = pd.read_csv(cf.metadata_file, sep="\t")
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
  assert model_name in cf.oai_models, f"Model {model_name} not in the list of models."
  model_name = model_name.replace(".", "")
  print(f"  - Collecting data for {model_name} from {ddir} [{ut.get_current_date_hms()}]")
  data = {"text": [], "model": [], "exampleNumber": [],
          "centuryBirth": [], "completionNumber": [], "humorLabel": []}
  for fname in sorted(os.listdir(ddir)):
    if fname.startswith("humor"):
      example_number = int(fname.split("_")[1])
      completion_number = int(fname.split("_")[3].split(".")[0])
      jso = json.load(open(os.path.join(ddir, fname), "r"))
      text = jso["reason"]
      data["text"].append(text)
      data["exampleNumber"].append(example_number)
      data["centuryBirth"].append(md.loc[md["id"] == example_number, "centuryBirth"].values[0])
      data["completionNumber"].append(completion_number)
      data["humorLabel"].append(md.loc[md["id"] == example_number, "comic"].values[0])
      data["model"].append(model_name)
  return data


if __name__ == "__main__":
  for modu in cf, ut:
    reload(modu)
  print(f"# Start [{ut.get_current_date_hms()}]")
  mddf = read_metadata(cf)
  data_35 = collect_data(os.path.join(cf.response_dir, "gpt" + os.sep + "gpt-35-turbo") , "gpt-3.5-turbo", mddf)
  data_4o = collect_data(os.path.join(cf.response_dir, "gpt" + os.sep + "gpt-4o") , "gpt-4o", mddf)
  data_all = {k: data_35[k] + data_4o[k] for k in data_35.keys()}
  df = pd.DataFrame(data_all)
