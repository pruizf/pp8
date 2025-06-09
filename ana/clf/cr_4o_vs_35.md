# 4o vs 35, 0.8 / 0.2

In [4]: %run classification.py
## Start [13:40:57]
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

# 4o vs 4omini vs 35

In [3]: %run classification_new.py
## Start [17:04:29]
  - Collecting data for gpt-35-turbo from ../outputs/model_responses/gpt/gpt-35-turbo [17:04:29]
  - Collecting data for gpt-4o from ../outputs/model_responses/gpt/gpt-4o [17:04:29]
  - Collecting data for gpt-4o-mini from ../outputs/model_responses/gpt/gpt-4o-mini [17:04:29]
  - Loading spacy model [17:04:29]
    - Done
  - Training LR [17:04:29]
[ColumnTransformer] ..... (1 of 2) Processing judgement, total=   4.7s
[ColumnTransformer] ........... (2 of 2) Processing pos, total=   4.0s
Classification report:

              precision    recall  f1-score   support

gpt-35-turbo     0.9697    0.9231    0.9458       104
      gpt-4o     0.9302    0.9302    0.9302        86
 gpt-4o-mini     0.9391    0.9818    0.9600       110

    accuracy                         0.9467       300
   macro avg     0.9464    0.9450    0.9453       300
weighted avg     0.9472    0.9467    0.9465       300

# 4o vs 4omini vs 35 stratified on model

In [5]: %run classification_new.py
## Start [17:09:09]
  - Collecting data for gpt-35-turbo from ../outputs/model_responses/gpt/gpt-35-turbo [17:09:09]
  - Collecting data for gpt-4o from ../outputs/model_responses/gpt/gpt-4o [17:09:09]
  - Collecting data for gpt-4o-mini from ../outputs/model_responses/gpt/gpt-4o-mini [17:09:09]
  - Loading spacy model [17:09:09]
    - Done
  - Training LR [17:09:09]
[ColumnTransformer] ..... (1 of 2) Processing judgement, total=   4.2s
[ColumnTransformer] ........... (2 of 2) Processing pos, total=   3.6s
Classification report:

              precision    recall  f1-score   support

gpt-35-turbo     0.9787    0.9200    0.9485       100
      gpt-4o     0.9010    0.9100    0.9055       100
 gpt-4o-mini     0.9333    0.9800    0.9561       100

    accuracy                         0.9367       300
   macro avg     0.9377    0.9367    0.9367       300
weighted avg     0.9377    0.9367    0.9367       300

