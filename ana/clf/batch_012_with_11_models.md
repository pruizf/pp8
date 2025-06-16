In [10]: %run classification_new.py 012
# Start [20:24:44]
  - Collecting data for gpt-35-turbo from ../outputs/model_responses/gpt/gpt-35-turbo [20:24:44]
  - Collecting data for gpt-4o from ../outputs/model_responses/gpt/gpt-4o [20:24:44]
  - Collecting data for gpt-4o-mini from ../outputs/model_responses/gpt/gpt-4o-mini [20:24:45]
  - Collecting data for gpt-41 from ../outputs/model_responses/gpt/gpt-41 [20:24:45]
  - Collecting data for claude-3-5-haiku-latest from ../outputs/model_responses/claude/claude-3-5-haiku-latest [20:24:45]
  - Collecting data for claude-3-5-sonnet-latest from ../outputs/model_responses/claude/claude-3-5-sonnet-latest [20:24:45]
  - Collecting data for mistral-small from ../outputs/model_responses/mistral/mistral-small [20:24:45]
  - Collecting data for mistral-large-latest from ../outputs/model_responses/mistral/mistral-large-latest [20:24:45]
  - Collecting data for deepseek-chat from ../outputs/model_responses/deepseek/deepseek-chat [20:24:45]
  - Collecting data for gemini-15-pro from ../outputs/model_responses/gemini/gemini-15-pro [20:24:45]
  - Collecting data for gemini-20-flash from ../outputs/model_responses/gemini/gemini-20-flash [20:24:45]
  - Loading spacy model [20:24:45]
    - Done
  - Training LR [20:24:46]
[ColumnTransformer] ........ (1 of 2) Processing reason, total=  16.0s
[ColumnTransformer] ........... (2 of 2) Processing pos, total=  14.8s
Classification report:

                          precision    recall  f1-score   support

 claude-3-5-haiku-latest     0.8909    0.9800    0.9333       100
claude-3-5-sonnet-latest     0.9091    0.9000    0.9045       100
           deepseek-chat     0.8667    0.9100    0.8878       100
           gemini-15-pro     0.8835    0.9100    0.8966       100
         gemini-20-flash     0.9176    0.7800    0.8432       100
            gpt-35-turbo     0.8421    0.8000    0.8205       100
                  gpt-41     0.9574    0.9000    0.9278       100
                  gpt-4o     0.7975    0.6300    0.7039       100
             gpt-4o-mini     0.9065    0.9700    0.9372       100
    mistral-large-latest     0.7373    0.8700    0.7982       100
           mistral-small     0.8476    0.8900    0.8683       100

                accuracy                         0.8673      1100
               macro avg     0.8688    0.8673    0.8656      1100
            weighted avg     0.8688    0.8673    0.8656      1100


Top features for claude-3-5-haiku-latest:
completamente    4.460734
soneto           3.751943
ADJ              3.607613
estructura       2.697810
genera           2.367641
profundamente    2.249982
culto            2.080844
poeta            1.952906
expresiones      1.933566
usa              1.928116
alejándose       1.914492
elemento         1.798457
existencial      1.769946
casi             1.754893
poético          1.697754
transmite        1.695536
lenguaje         1.688702
comicidad        1.683764
referencias      1.662382
juega            1.647541
:                1.549366
mediante         1.506530
dramático        1.480759
generan          1.476535
paródico         1.458700
estilo           1.456520
ingenioso        1.401718
tradicional      1.308005
elegante         1.251141
mordaz           1.238737
Name: claude-3-5-haiku-latest, dtype: float64

Top features for claude-3-5-sonnet-latest:
)               5.181952
(               4.496158
ADJ             3.631258
soneto          2.417706
culmina         1.890070
sino            1.870509
solemne         1.787791
SYM             1.643166
construye       1.599269
emplea          1.522639
utiliza         1.489158
creando         1.475756
final           1.443688
NUM             1.410590
metáforas       1.394654
empleando       1.346669
formal          1.199486
irónicamente    1.181514
oro             1.161718
satíricos       1.150689
elevados        1.144727
elevado         1.144426
primero         1.123817
PUNCT           1.094775
/               1.086749
culminando      1.078129
cualquier       1.046225
amoroso         1.007971
siglo           0.964722
revela          0.961750
Name: claude-3-5-sonnet-latest, dtype: float64

Top features for deepseek-chat:
centrándose      4.157087
presentado       3.122807
comedia          2.692172
temas            2.451467
lenguaje         2.361440
intención        2.357901
junto            2.324668
juicio           2.200174
tener            2.091626
ser              1.855567
considera        1.849177
subjetiva        1.821390
,                1.806262
típicos          1.791319
lejos            1.734622
SYM              1.711635
percibido        1.710232
capa             1.647135
ADP              1.623290
especialmente    1.426204
comicidad        1.374138
juguetón         1.364359
presenta         1.364314
satírico         1.358051
haciendo         1.312946
'                1.294195
utilizado        1.288142
refuerza         1.284626
lírica           1.279615
descripción      1.267747
Name: deepseek-chat, dtype: float64

Top features for gemini-15-pro:
                 5.258064
"                4.769723
SPACE            3.322800
PUNCT            2.665649
describe         2.417395
hablante         2.093993
sugieran         2.030732
yuxtaposición    1.806467
expresa          1.741161
crea             1.360704
centra           1.302519
.                1.231160
elevado          1.209567
provoquen        1.184342
propio           1.137239
poeta            0.982612
reverente        0.961450
humorísticos     0.960952
imagen           0.957714
DET              0.914955
sino             0.815124
sátira           0.811748
admiración       0.775621
expresando       0.743725
tema             0.711525
naturaleza       0.707256
figura           0.702943
incluso          0.685181
divina           0.679918
literarios       0.679634
Name: gemini-15-pro, dtype: float64

Top features for gemini-20-flash:
"                  4.797903
comicidad          3.352106
.                  2.522158
si                 2.435619
incongruencia      2.315155
podría             2.259125
depende            2.157006
interpretarse      2.105844
exageración        2.067105
reside             1.982408
rima               1.778612
interpretación     1.766155
PUNCT              1.716854
cómicos            1.655528
hipérbole          1.649095
idealización       1.625166
elevado            1.621654
,                  1.612665
debido             1.538862
aunque             1.485405
inherentemente     1.473523
determinar         1.450668
resumen            1.404221
grandiosidad       1.404105
consonante         1.389938
grandilocuencia    1.375956
formal             1.362542
seriedad           1.342509
sorpresa           1.312221
inicial            1.275125
Name: gemini-20-flash, dtype: float64

Top features for gpt-35-turbo:
exageración     3.021515
además          2.956217
humorístico     2.742966
cómico          2.338779
trata           2.187189
hacen           2.044784
toque           1.889120
resultar        1.818207
descripción     1.803669
exagerada       1.775614
debido          1.712591
exagerado       1.663028
forma           1.541432
combinación     1.390294
sarcástico      1.358727
considera       1.333225
ADP             1.321704
resulta         1.253191
SCONJ           1.231270
poema           1.226745
sentimientos    1.204532
comparación     1.192045
situación       1.182232
manera          1.169346
tono            1.120129
lector          1.115980
personas        1.099756
lectores        1.061717
historia        1.061032
palabras        1.051362
Name: gpt-35-turbo, dtype: float64

Top features for gpt-41:
,                 6.116865
recursos          3.412492
”                 3.036669
“                 3.022339
considerarse      2.265414
(                 2.012991
juegos            2.001463
emplea            1.835672
burla             1.818885
literarios        1.815312
ADJ               1.743288
remate            1.736343
ironía            1.734349
comicidad         1.731700
:                 1.723768
principalmente    1.663112
solemne           1.655927
absurdas          1.639660
PUNCT             1.560718
palabras          1.539228
texto             1.499978
)                 1.496893
lírico            1.491062
poesía            1.485469
parodia           1.470185
tampoco           1.424629
presentado        1.421656
propio            1.375473
irónico           1.339846
ello              1.311790
Name: gpt-41, dtype: float64

Top features for gpt-4o:
.                  2.605414
contenido          2.015542
humor              1.894983
NOUN               1.768551
cómica             1.754917
CCONJ              1.679579
formal             1.671856
manera             1.616936
elección           1.550410
debido             1.500282
serio              1.448323
situaciones        1.363790
cómicas            1.330526
reír               1.312616
sorpresa           1.305531
lugar              1.283753
hacer              1.264211
destacando         1.258219
parece             1.253305
VERB               1.251029
solemnes           1.246498
ADV                1.241221
tono               1.197534
solemne            1.188841
palabras           1.178104
entretenimiento    1.172502
características    1.140648
temas              1.136629
ridículas          1.127687
razones            1.115792
Name: gpt-4o, dtype: float64

Top features for gpt-4o-mini:
PRON         4.476401
presenta     3.642428
,            3.390035
enfoque      2.912870
través       2.589258
aunque       2.334840
DET          2.229067
uso          2.223628
evoca        2.006277
'            1.998038
risa         1.997707
provoca      1.913464
lucha        1.913351
resumen      1.886730
sugiere      1.859478
así          1.848990
aleja        1.830997
ADP          1.787759
invita       1.741837
.            1.707141
obra         1.637743
imágenes     1.586325
crítica      1.573673
vida         1.555124
puede        1.547612
belleza      1.543177
emocional    1.509176
largo        1.478607
contexto     1.432873
pesar        1.408534
Name: gpt-4o-mini, dtype: float64

Top features for mistral-large-latest:
debido           4.261492
mención          2.806693
razón            2.705564
AUX              2.646801
toque            2.477703
temática         2.229884
profundidad      1.812112
interpretados    1.797851
principal        1.778387
resumen          1.703561
SYM              1.668561
embargo          1.665232
primer           1.559641
ligero           1.558216
añaden           1.485751
poema            1.472686
'                1.458778
CCONJ            1.448820
añade            1.448471
subraya          1.446468
centra           1.412725
profunda         1.390207
serios           1.379337
metafórico       1.364027
solemne          1.327149
idea             1.320590
humorísticos     1.315864
atmósfera        1.293554
.                1.288104
hace             1.286521
Name: mistral-large-latest, dtype: float64

Top features for mistral-small:
contiene            4.274468
diseñado            3.504592
ADV                 2.569848
contrario           2.383091
ser                 2.170004
titulado            2.157627
sátira              2.058722
palabra             1.995480
aunque              1.983192
AUX                 1.894332
intencionalmente    1.894034
poema               1.838939
diversión           1.830438
métrica             1.824816
rima                1.762265
lugar               1.728027
CCONJ               1.664760
cómicos             1.659930
destinado           1.652296
considerar          1.589512
chistes             1.498864
tradicional         1.420341
puede               1.276955
gracioso            1.262501
utilizados          1.254996
poético             1.247694
comedia             1.233540
ningún              1.232627
escrito             1.173541
hablante            1.125476
Name: mistral-small, dtype: float64


