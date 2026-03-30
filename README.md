# zs-3dvg-vlm

We investigate the effects of multi-view perception, query reconstruction, and input ablations for zero-shot 3D visual grounding, and observe consistent performance variations across different settings.

---

## File Structure

- `inference/` : Inference pipeline (multi-view projection, ablations, variants)
- `parse_query/` : Query parsing and relation extraction
- `eval/` : Evaluation scripts (including McNemar test)
- `prompts/` : Prompt templates for relation extraction

---

## Implementation
To reproduce our results, please first follow the original SeeGround setup.

Then replace or update the following components with the implementations provided in this repository:

- Multi-view projection module
- Query reconstruction and parsing pipeline
- Ablation configurations in inference scripts

---

## Acknowledgement

This codebase is built upon [SeeGround (Apache-2.0)](https://github.com/iris0329/SeeGround). 

We sincerely thank the authors for releasing their code.
