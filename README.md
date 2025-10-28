# pp8

Work for the following [presentation](./pp8_pres.pdf):

> Pablo Ruiz Fabo (2025). Aspects of comic verse salient to non-specialized large language models: Says who? Presented at Plotting Poetry 8. Prague, June 16-18, 2025. https://www.plottingpoetry.org/conference/2025prague/programme

## Repository structure

- The corpus is at [`corpus`](./corpus); metadata are at [`corpus/metadata.tsv`](./corpus/metadata.tsv)
- Scripts are at [`src`](./src)
  - The LLM clients are there, besides evaluation scripts for the binary classification task humorous/not
  - The model identification task based on model outputs is run with  `src/classsification.py`
- LLM outputs are at [`outputs`](./outputs), together with some analysis results:
  - `model_responses`: responses from the LLMs
  - `plots`: confusion matrices, feature importances and classification reports for the different runs of the model identification task
- Textometric analyses are at [`ana`](./ana)
- The [wk](./wk) directory contains corpus creation stuff
