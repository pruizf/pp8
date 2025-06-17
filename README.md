# pp7_llm

Work for the following presentation:

> Pablo Ruiz Fabo (2025). Aspects of comic verse salient to non-specialized large language models. Presented at Plotting Poetry 8. Prague, June 16-18, 2025. https://www.plottingpoetry.org/conference/2025prague/programme

## Repository structure

- The corpus is at [`corpus`](./corpus); metadata are at [`corpus/metadata.tsv`](./corpus/metadata.tsv)
- Scripts are at [`src`](./src)
  - The LLM clients are there and evaluation scripts for the binary classification task humorous/not are also there
  - The model classification task based on model outputs is run with  `src/classsification.py`
- LLM outputs are at [`outputs`](./outputs), in turn this is divided into:
  - `model_responses`: responses from the LLMs
  - `plots`: for the different runs of the model classification task (confusion matrices, feature importances, etc.)
- Textometric analyses are at [`ana`](./ana)
- The [wk](./wk) directory contains corpus creation stuff
