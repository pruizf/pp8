# 4o vs 35, 0.8 / 0.2

In [4]: %run classification.py
# Start [13:40:57]
  - Collecting data for gpt-35-turbo from ../outputs/model_responses/gpt/gpt-35-turbo [13:40:57]
  - Collecting data for gpt-4o from ../outputs/model_responses/gpt/gpt-4o [13:40:57]
  - Loading spacy model [13:40:57]
    - Done
  - Training LR [13:40:57]
[ColumnTransformer] ..... (1 of 2) Processing judgement, total=   2.8s
[ColumnTransformer] ........... (2 of 2) Processing pos, total=   2.1s
Classification report:

              precision    recall  f1-score   support

gpt-35-turbo     0.9091    0.9375    0.9231        96
      gpt-4o     0.9406    0.9135    0.9268       104

    accuracy                         0.9250       200
   macro avg     0.9248    0.9255    0.9250       200
weighted avg     0.9255    0.9250    0.9250       200

