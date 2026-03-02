# From Controlled Scenarios to the Real-World: Cross-Domain Degradation Pattern Matching for All-in-One Image Restoration

[![Paper](https://img.shields.io/badge/Paper-Research_(In_Press)-red.svg)](https://spj.science.org/doi/10.34133/research.1191)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🔥 News
- **[2026.02]** 🎉 Our paper is accepted by *Research* and available as an **Article in Press**.
- Code release coming soon.

> 🚧 **Code Status**
> The implementation is currently being refactored and documented.
> We will release the full training and inference code soon.


## Abstract
As a fundamental imaging task, All-in-One Image Restoration (AiOIR) aims to achieve image restoration caused by multiple degradation patterns via a single model with unified parameters. However, existing methods typically rely on sample-wise supervision, which tends to entangle degradation features with image content. Furthermore, the inevitable distribution shift between training data (source domain) and real-world samples (target domain) weakens degradation awareness, severely limiting the generalization capability of model in real-world scenarios. To address this, a Unified Domain-Adaptive Image Restoration (UDAIR) computer vision model is proposed by achieving the transition from learning unstable local features to learning robust universal prototypes. To decouple degradation from content, a Cross-Sample Contrastive Learning (CSCL) mechanism is implemented by a Codebook-based module. By contrasting samples with shared degradations but diverse content, the proposed model learns discrete embeddings as degradation prototypes. Furthermore, to actively bridge the distribution gap during inference, a correlation alignment-based test-time adaptation mechanism is designed to dynamically pull drifting target features towards their corresponding degradation cluster centers to effectively eliminate residual alignment discrepancies. Experimental results on 10 open-source datasets demonstrate that UDAIR achieves new state-of-the-art performance for the AiOIR task, in which each technical module contributes to desired performance improvement. Most importantly, the feature cluster validates the degradation identification under multiple degradation patterns, and qualitative comparisons showcase robust generalization to real-world scenarios.

## Quick Start
Coming soon.

## Citation
If you find our framework or concept useful for your research, please consider citing our paper:

```bibtex
@article{
doi:10.34133/research.1191,
author = {Junyu Fan  and Chuanlin Liao  and Endi Xie  and Dongyue Guo  and Xiaolin Gou  and Duan Wei  and Junyang Hu  and Yi Lin },
title = {From Controlled Scenarios to the Real-World: Cross-Domain Degradation Pattern Matching for All-in-One Image Restoration},
journal = {Research},
volume = {0},
number = {ja},
pages = {},
year = {},
doi = {10.34133/research.1191},
URL = {https://spj.science.org/doi/abs/10.34133/research.1191},
eprint = {https://spj.science.org/doi/pdf/10.34133/research.1191}}
