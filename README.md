# zs-3dvg-vlm
**Paper**
*An Empirical Study of Zero-Shot 3D Visual Grounding with Vision-Language Models*

---

## Abstract
3D Visual Grounding (3DVG) aims to localize objects in a 3D scene that correspond to a given natural language query and plays a critical role in applications such as robotics and autonomous systems. With recent advances in Vision-Language Models (VLMs), zero-shot approaches to 3DVG that leverage pre-trained VLMs without task-specific 3D supervision have gained increasingly attention.

However, such approaches heavily rely on pre-trained knowledge and are sensitive to the configuration of textual and visual inputs. While fully supervised 3DVG methods have been extensively studied, a systematic analysis of zero-shot VLM-based 3DVG remains limited.

In this work, we conduct a comprehensive analysis of VLM-based zero-shot 3D Visual Grounding by varying natural language query formulations and visual input configurations, with a particular focus on modality contribution. Our analysis reveals that current VLM-based zero-shot approaches exhibit limited capability in relational reasoning and tend to rely on textual cues rather than visual evidence.

These findings highlight inherent structural limitations of existing zero-shot VLM-based 3DVG pipelines. Based on our observations, we further discuss the necessity of incorporating structured 3D representations or explicit mechanisms for modeling spatial relationships to enable more reliable reasoning in future zero-shot 3D Visual Grounding systems.

---

## Overview
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
