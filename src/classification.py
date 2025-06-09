from importlib import reload
import json
import os
import string
from datetime import datetime
from types import ModuleType

# Suppression de l'affichage des messages d'avertissement
import warnings

import nltk.corpus

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
import spacy

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
  assert model_name in cf.model_list, f"Model {model_name} not in the list of models."
  model_name = model_name.replace(".", "")
  print(f"  - Collecting data for {model_name} from {ddir} [{ut.get_current_date_hms()}]")
  data = {"text": [], "model": [], "exampleNumber": [],
          "centuryBirth": [], "completionNumber": [], "humorLabel": []}
  for fname in sorted(os.listdir(ddir)):
    if fname.startswith("humor"):
      example_number = int(fname.split("_")[1])
      completion_number = int(fname.split("_")[3].split(".")[0])
      #jso = json.load(open(os.path.join(ddir, fname), "r"))
      with open(os.path.join(ddir, fname), "r", encoding="utf-8") as f:
        jso = json.load(f)
      text = jso["reason"]
      data["text"].append(text)
      data["exampleNumber"].append(example_number)
      data["centuryBirth"].append(md.loc[md["id"] == example_number, "centuryBirth"].values[0])
      data["completionNumber"].append(completion_number)
      data["humorLabel"].append(md.loc[md["id"] == example_number, "comic"].values[0])
      data["model"].append(model_name)
  return data

def split_into_tokens_spacy(text: str) -> list:
  """
    Split text into tokens using spacy.
  Args:
    text: Text to tokenize

  Returns:
    list: List of tokens
  """
  doc = spacy_pipeline(text)
  return [w.text for w in doc]


def pos_spacy(text: str) -> list:
  """
    Split text into tokens using spacy.
  Args:
    text: Text to tokenize

  Returns:
    list: List of tokens
  """
  doc = spacy_pipeline(text)
  return [w.pos_ for w in doc]

def plot_confusion_matrix(cm, classes, normalize=False, cmap='Blues'):
  title = 'Matrice de Confusion'
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, cmap=cmap, xticklabels=classes, yticklabels=classes)
  plt.title(title)
  plt.xlabel('Prédits')
  plt.ylabel('Réels')
  plt.show()


if __name__ == "__main__":
  for modu in cf, ut:
    reload(modu)
  print(f"# Start [{ut.get_current_date_hms()}]")
  mddf = read_metadata(cf)
  data_35 = collect_data(os.path.join(cf.response_dir, "gpt" + os.sep + "gpt-35-turbo") , "gpt-3.5-turbo", mddf)
  data_4o = collect_data(os.path.join(cf.response_dir, "gpt" + os.sep + "gpt-4o") , "gpt-4o", mddf)
  data_all = {k: data_35[k] + data_4o[k] for k in data_35.keys()}
  df = pd.DataFrame(data_all)

  #labelCol = "humorLabel"
  labelCol = "model"
  class_names = sorted(df[labelCol].unique())
  label2id = {class_names[i]: i for i in range(len(class_names))}
  id2label = {i: class_names[i] for i in range(len(class_names))}

  #es_model = spacy.load("es_core_news_sm")
  print(f"  - Loading spacy model [{ut.get_current_date_hms()}]")
  spacy_pipeline = spacy.load("es_core_news_sm", disable=["parser", "ner"])
  print("    - Done")

  # will use NLTK stopwords instead of spacy's since latter seem too restrictive (too many words removed)
  stopwords = nltk.corpus.stopwords.words("spanish")

  # corpus vectorizers

  tok_vectorizer = TfidfVectorizer(lowercase=True,
                                   tokenizer=split_into_tokens_spacy,
                                   stop_words=stopwords,
                                   min_df=0.01)
  pos_vectorizer = TfidfVectorizer(lowercase=True,
                                   tokenizer=pos_spacy,
                                   stop_words=stopwords,
                                   min_df=0.01)

  # column transformer
  column_trans_wf_pos = ColumnTransformer(
    [
      # Colonne 'description' : tf-idf
      ('judgement', tok_vectorizer, 'text'),
      ('pos', pos_vectorizer, 'text'),
      #('ngrams', ngram_vectorizer, 'description'),
      #('char_ngrams', descr_vectorizer_char_ng, 'description'),
      # (
      #   'description_stats',
      #   Pipeline(
      #     [
      #       ('text_stats', text_stats_transformer),
      #       ('vect', text_stats_vectorizer),
      #       ('scaling', min_max_scaler)
      #     ]
      #   ),
      #   'description'
      # )
    ],
    verbose=True,
  )

  df_train, df_test = train_test_split(df, test_size=0.2, random_state=cf.rdm_seed)

  X_train = df_train.drop(columns=[labelCol])
  y_train = df_train[labelCol]
  X_test = df_test.drop(columns=[labelCol])
  y_test = df_test[labelCol]

  scores = []
  model_names = ["LR"]
  scoring = "Macro F1"

  models = [('LR', LogisticRegression())]

  for model_name, model in models:
    print(f"  - Training {model_name} [{ut.get_current_date_hms()}]")
    clf_pipeline = make_pipeline(column_trans_wf_pos, model)
    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)
    print("Classification report:\n\n{}".format(classification_report(y_test, y_pred, digits=4)))
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    scores.append(f1)

  # model_names = ["LR", "Extra Trees", "SVC"]

  # Initialiser un dictionnaire qui contiendra toutes les caractéristiques les plus influentes (classes négative et positive confondues)
  all_features = dict()

  # Dictionnaires pour stocker les résultats
  accuracy_test_results_opti = {}
  f1_score_results_opti = {}

  # Fonction pour afficher les caractéristiques
  def plot_feature_importances(importances, feature_names, top_indices, title, color):
      plt.figure(figsize=(10, 6))
      # Barres pour les caractéristiques positives
      plt.barh(range(len(top_indices)), importances[top_indices], align='center', color=color)
      plt.yticks(range(len(top_indices)), feature_names[top_indices])
      plt.gca().invert_yaxis()  # Inverser l'ordre des barres
      plt.xlabel('Importance')
      plt.ylabel('Caractéristiques')
      plt.title(title)
      plt.tight_layout()
      plt.show()

  best_estimators = [('LR', clf_pipeline)]
  for name, model in best_estimators:
      if name not in model_names:
          continue  # Ignorer les modèles qui ne sont pas dans la liste

      # Prédire les classes sur l'ensemble de test
      y_test_pred = model.predict(X_test)

      # Calculer et afficher le rapport de classification
      print(f"Model: {name}")
      print(classification_report(y_test, y_test_pred))

      # Calculer et afficher le score d'exactitude
      accuracy_test_opti = accuracy_score(y_test, y_test_pred)
      accuracy_test_results_opti[name] = accuracy_test_opti

      # Calcul et stockage des F1-scores par classe
      f1_scores_per_class_opti = f1_score(y_test, y_test_pred, average=None)
      f1_score_results_opti[name] = {'Amateur': f1_scores_per_class_opti[0], 'Professionnel': f1_scores_per_class_opti[1]}

      print(f"Accuracy (test set): {accuracy_test_opti:.2f}")
      print(f"F1 Score (Amateur): {f1_scores_per_class_opti[0]:.2f}")
      print(f"F1 Score (Professionnel): {f1_scores_per_class_opti[1]:.2f}")

      # Calcul de la matrice de confusion
      conf_matrix = confusion_matrix(y_test, y_test_pred)
      # Afficher la matrice de confusion
      plot_confusion_matrix(conf_matrix, classes=['Amateur', 'Professionnel'])

      if name == "LR":
          lr_model = model.named_steps['logisticregression']

          # Récupérer les coefficients (importances) des caractéristiques
          importances = lr_model.coef_[0]
          top_indices = np.argsort(importances)[::-1][:30]  # Les 30 indices des caractéristiques les plus importantes
          #top_features_names = X_train.columns[top_indices]
          top_features_names = model.named_steps["columntransformer"].get_feature_names_out()[top_indices]

          top_indices_neg = np.argsort(importances)[:30]  # Coefficients négatifs aussi
          #top_features_names_neg = X_train.columns[top_indices_neg]
          top_features_names_neg = model.named_steps["columntransformer"].get_feature_names_out()[top_indices_neg]

          # Afficher les caractéristiques positives
          #plot_feature_importances(importances, X_train.columns, top_indices, f'Top 30 des caractéristiques pour {name} (Professionnel)', color='mediumaquamarine')
          plot_feature_importances(importances, model.named_steps["columntransformer"].get_feature_names_out(), top_indices, f'Top 30 des caractéristiques pour {name} (Professionnel)', color='mediumaquamarine')

          # Afficher les caractéristiques négatives
          #plot_feature_importances(importances, X_train.columns, top_indices_neg, f'Top 30 des caractéristiques pour {name} (Amateur)', color='mediumturquoise')
          plot_feature_importances(importances, model.named_steps["columntransformer"].get_feature_names_out(), top_indices_neg, f'Top 30 des caractéristiques pour {name} (Amateur)', color='mediumturquoise')

          # Créer un dataframe pour afficher les caractéristiques et leurs coefficients
          importance_df = pd.DataFrame({
              'Feature': model.named_steps["columntransformer"].get_feature_names_out(),
              'Coefficient': importances
          })

          # Séparer les caractéristiques positives et négatives
          positive_features = importance_df[importance_df['Coefficient'] > 0].sort_values(by='Coefficient', ascending=False)
          negative_features = importance_df[importance_df['Coefficient'] < 0].sort_values(by='Coefficient')

          print("Positive class discriminant features:")
          print(positive_features.head(30))

          print("\nNegative class discriminant features:")
          print(negative_features.head(30))

          all_features[name] = list(top_features_names) + list(top_features_names_neg)