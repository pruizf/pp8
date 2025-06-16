In [16]: %run classification_new.py 013
# Start [20:57:09]
  - Collecting data for gpt-35-turbo from ../outputs/model_responses/gpt/gpt-35-turbo [20:57:09]
  - Collecting data for gpt-4o from ../outputs/model_responses/gpt/gpt-4o [20:57:09]
  - Collecting data for gpt-4o-mini from ../outputs/model_responses/gpt/gpt-4o-mini [20:57:09]
  - Collecting data for gpt-41 from ../outputs/model_responses/gpt/gpt-41 [20:57:09]
  - Collecting data for claude-3-5-haiku-latest from ../outputs/model_responses/claude/claude-3-5-haiku-latest [20:57:09]
  - Collecting data for claude-3-5-sonnet-latest from ../outputs/model_responses/claude/claude-3-5-sonnet-latest [20:57:09]
  - Collecting data for mistral-small from ../outputs/model_responses/mistral/mistral-small [20:57:10]
  - Collecting data for mistral-large-latest from ../outputs/model_responses/mistral/mistral-large-latest [20:57:10]
  - Collecting data for mistral-medium from ../outputs/model_responses/mistral/mistral-medium [20:57:10]
  - Collecting data for deepseek-chat from ../outputs/model_responses/deepseek/deepseek-chat [20:57:10]
  - Collecting data for gemini-15-pro from ../outputs/model_responses/gemini/gemini-15-pro [20:57:10]
  - Collecting data for gemini-20-flash from ../outputs/model_responses/gemini/gemini-20-flash [20:57:10]
  - Loading spacy model [20:57:11]
    - Done
  - Training LR [20:57:11]
[ColumnTransformer] ........ (1 of 2) Processing reason, total=  30.5s
[ColumnTransformer] ........... (2 of 2) Processing pos, total=  28.4s
Classification report:

                          precision    recall  f1-score   support

 claude-3-5-haiku-latest     0.9074    0.9800    0.9423       100
claude-3-5-sonnet-latest     0.9388    0.9200    0.9293       100
           deepseek-chat     0.8318    0.8900    0.8599       100
           gemini-15-pro     0.8900    0.8900    0.8900       100
         gemini-20-flash     0.8864    0.7800    0.8298       100
            gpt-35-turbo     0.8632    0.8200    0.8410       100
                  gpt-41     0.9474    0.9000    0.9231       100
                  gpt-4o     0.7722    0.6100    0.6816       100
             gpt-4o-mini     0.9065    0.9700    0.9372       100
    mistral-large-latest     0.7411    0.8300    0.7830       100
          mistral-medium     0.9057    0.9600    0.9320       100
           mistral-small     0.8667    0.9100    0.8878       100

                accuracy                         0.8717      1200
               macro avg     0.8714    0.8717    0.8698      1200
            weighted avg     0.8714    0.8717    0.8698      1200


Top features for claude-3-5-haiku-latest:
completamente    4.747257
soneto           4.064573
ADJ              3.447490
profundamente    2.392730
estructura       2.377662
genera           2.213742
expresiones      2.111452
usa              2.079356
poeta            2.027149
elemento         2.007635
poético          1.876911
alejándose       1.852573
culto            1.808153
casi             1.783301
juega            1.772498
transmite        1.719433
existencial      1.718192
referencias      1.707042
:                1.643294
paródico         1.581036
comicidad        1.576179
mediante         1.575001
lenguaje         1.551856
estilo           1.538027
ingenioso        1.460745
generan          1.440620
dramático        1.410814
elegante         1.406484
tradicional      1.373235
barroco          1.263513
Name: claude-3-5-haiku-latest, dtype: float64

Top features for claude-3-5-sonnet-latest:
)               5.108905
(               4.349454
ADJ             3.518916
soneto          2.659838
culmina         1.985870
solemne         1.851563
utiliza         1.831661
sino            1.781378
construye       1.716155
emplea          1.666422
NUM             1.492064
empleando       1.482513
SYM             1.478621
creando         1.420607
metáforas       1.405301
final           1.376394
irónicamente    1.328874
primero         1.253294
oro             1.203333
varios          1.135940
/               1.106074
elevados        1.099108
satíricos       1.093475
cualquier       1.085628
amoroso         1.036183
revela          1.008055
muerte          0.999094
:               0.984642
siglo           0.969613
elevado         0.965131
Name: claude-3-5-sonnet-latest, dtype: float64

Top features for deepseek-chat:
centrándose      4.200180
presentado       3.378633
comedia          2.791608
temas            2.541140
intención        2.433589
tener            2.232538
junto            2.222303
juicio           2.201362
lenguaje         2.129229
ser              2.069661
ADP              2.035119
considera        1.953885
lejos            1.865643
percibido        1.839263
capa             1.814310
subjetiva        1.812525
SYM              1.679162
típicos          1.660400
,                1.639565
utilizado        1.449775
presenta         1.386542
juguetón         1.375540
haciendo         1.362872
bien             1.345516
hacia            1.316090
especialmente    1.303229
través           1.300978
comicidad        1.253774
parece           1.246839
descripción      1.223124
Name: deepseek-chat, dtype: float64

Top features for gemini-15-pro:
                 5.285566
"                4.882538
SPACE            3.350841
describe         2.550450
PUNCT            2.510636
hablante         2.224744
sugieran         2.100764
yuxtaposición    1.877415
expresa          1.785245
centra           1.373089
crea             1.367496
.                1.229025
propio           1.202968
provoquen        1.124706
elevado          1.102791
poeta            1.001764
reverente        1.000833
imagen           0.954684
humorísticos     0.943939
DET              0.891356
admiración       0.803883
sino             0.758303
sátira           0.755694
naturaleza       0.706451
trata            0.694354
incluso          0.693853
figura           0.684538
intención        0.682725
VERB             0.681392
literarios       0.680382
Name: gemini-15-pro, dtype: float64

Top features for gemini-20-flash:
"                  4.955459
comicidad          3.307911
.                  2.510662
incongruencia      2.489627
si                 2.346940
podría             2.288298
reside             2.143598
depende            1.925549
exageración        1.900031
interpretarse      1.866934
hipérbole          1.777966
rima               1.689920
interpretación     1.662284
idealización       1.642628
resumen            1.600585
inherentemente     1.576455
aunque             1.560979
determinar         1.524243
grandiosidad       1.470999
cómicos            1.468469
grandilocuencia    1.452010
,                  1.449213
cómica             1.424340
sorpresa           1.387182
PUNCT              1.386762
consonante         1.383037
elevado            1.375435
subjetivo          1.374482
debido             1.323402
primer             1.311814
Name: gemini-20-flash, dtype: float64

Top features for gpt-35-turbo:
exageración     2.899647
además          2.898918
humorístico     2.790009
cómico          2.403708
trata           2.318339
hacen           2.075901
toque           1.904215
resultar        1.878356
descripción     1.858974
exagerada       1.790516
forma           1.645026
debido          1.558311
ADP             1.541836
exagerado       1.515327
poema           1.465565
sarcástico      1.429634
considera       1.391639
resulta         1.325122
combinación     1.310633
sentimientos    1.284473
situación       1.261698
SCONJ           1.260043
manera          1.244233
lector          1.214725
palabras        1.152603
lectores        1.149523
tono            1.143105
comparación     1.136281
personas        1.134165
percepción      1.054976
Name: gpt-35-turbo, dtype: float64

Top features for gpt-41:
,                 6.063819
recursos          3.314381
”                 3.105925
“                 3.092628
considerarse      2.466646
juegos            2.083285
emplea            1.967352
(                 1.958270
burla             1.927731
literarios        1.919627
:                 1.860879
remate            1.853700
principalmente    1.744351
absurdas          1.737476
comicidad         1.716760
presentado        1.708807
palabras          1.708626
solemne           1.626874
ironía            1.557258
propio            1.548182
tampoco           1.530913
texto             1.529897
parodia           1.482743
ADJ               1.473153
)                 1.446485
forma             1.444519
ello              1.373664
razones           1.373508
poesía            1.359134
provocar          1.309195
Name: gpt-41, dtype: float64

Top features for gpt-4o:
.                  2.561368
contenido          2.041513
humor              1.945689
cómica             1.934861
NOUN               1.833286
manera             1.711701
elección           1.617731
CCONJ              1.608924
serio              1.474241
cómicas            1.443008
lugar              1.437013
situaciones        1.429081
formal             1.415815
poema              1.412124
reír               1.398444
sorpresa           1.376124
hacer              1.376010
ADV                1.344101
parece             1.313816
palabras           1.288312
VERB               1.282469
solemnes           1.274184
destacando         1.234346
entretenimiento    1.211889
debido             1.210594
temas              1.195203
descripciones      1.192845
características    1.183341
razones            1.182131
solemne            1.178949
Name: gpt-4o, dtype: float64

Top features for gpt-4o-mini:
PRON         4.609040
presenta     3.700442
,            3.288614
enfoque      2.863401
través       2.748924
aunque       2.334798
uso          2.249537
DET          2.189047
evoca        2.057019
ADP          2.045215
risa         2.035339
resumen      1.999551
provoca      1.956351
sugiere      1.940799
lucha        1.885811
'            1.868303
así          1.822284
aleja        1.807908
invita       1.757064
puede        1.713657
obra         1.709868
.            1.676579
imágenes     1.658182
vida         1.616471
belleza      1.597651
largo        1.544249
emocional    1.542729
hablante     1.518932
crítica      1.512134
pesar        1.499097
Name: gpt-4o-mini, dtype: float64

Top features for mistral-large-latest:
debido         3.979064
AUX            2.811132
razón          2.807098
mención        2.654995
toque          2.558109
temática       2.271273
poema          2.084884
resumen        1.898765
profundidad    1.822502
principal      1.801271
primer         1.737959
ligero         1.735197
embargo        1.670630
SYM            1.547612
profunda       1.530992
centra         1.522309
'              1.464312
serios         1.436709
añade          1.433321
CCONJ          1.432945
añaden         1.428525
capa           1.406863
subraya        1.384283
idea           1.379095
hace           1.367814
atmósfera      1.366213
sugieren       1.332891
sugiere        1.321604
solemne        1.315359
estrofa        1.297132
Name: mistral-large-latest, dtype: float64

Top features for mistral-medium:
refuerzan      3.271431
PUNCT          2.908341
ADJ            2.553345
estructura     2.499717
irónico        2.464125
debido         2.461930
refuerza       2.446829
exageración    2.339915
grotesca       2.287831
lírico         2.206749
,              2.120177
cotidiano      2.045501
culto          1.987271
vocabulario    1.972540
elevado        1.895642
ritmo          1.891001
confirma       1.865576
formal         1.825044
ausencia       1.674366
ambigüedad     1.618244
grotesco       1.589058
barroca        1.561409
genera         1.540258
'              1.514015
ironía         1.499964
además         1.485828
podrían        1.434159
grotescas      1.417158
satírico       1.409900
típico         1.391383
Name: mistral-medium, dtype: float64

Top features for mistral-small:
contiene            4.541545
diseñado            3.663126
ADV                 2.785026
contrario           2.315862
intencionalmente    2.262268
sátira              2.258448
titulado            2.233988
ser                 2.137492
aunque              1.936233
destinado           1.894576
lugar               1.861659
poema               1.850731
palabra             1.846285
diversión           1.797235
métrica             1.760618
AUX                 1.730374
rima                1.684422
CCONJ               1.669053
cómicos             1.563218
considerar          1.538011
gracioso            1.510599
chistes             1.473587
comedia             1.393028
tradicional         1.385893
puede               1.356030
poético             1.317367
suficiente          1.195443
crear               1.181957
poética             1.118187
mismo               1.103316
Name: mistral-small, dtype: float64

