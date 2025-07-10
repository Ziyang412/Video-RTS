# VIDEO-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning

This is the official implementation for Video-RTS, still in-progress. 

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://sites.google.com/cs.unc.edu/videorts2025/)  [![arXiv](https://img.shields.io/badge/arXiv-2405.19209-b31b1b.svg)](https://arxiv.org/abs/2507.06485)

### Authors: [Ziyang Wang*](https://ziyangw2000.github.io/),  [Jaehong Yoon*](https://jaehong31.github.io/), [Shoubin Yu](https://yui010206.github.io/), [Md Mohaiminul Islam](https://md-mohaiminul.github.io/), [Gedas Bertasius](https://www.gedasbertasius.com/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

### University of North Carolina at Chapel Hill


We introduce Video-RTS, a new approach to improve video reasoning capability with drastically improved data efficiency by combining data-efficient RL with a video-adaptive test-time scaling (TTS) strategy.

<img src="./assets/fig1.png" alt="teaser image" width="800"/>

<img src="./assets/fig2.png" alt="vis image" width="600"/>


## **Installation**




## Acknowledgments
We thank the developers of [LLoVi](https://github.com/CeeZh/LLoVi), [LifelongMemory](https://github.com/Agentic-Learning-AI-Lab/lifelong-memory), [EVA-CLIP](https://huggingface.co/BAAI/EVA-CLIP-18B#eva-clip-8b), [Kmeans-pytorch](https://github.com/subhadarship/kmeans_pytorch) and [SKlearn Clustering](https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html) for their public code release. We also thank the authors of [VideoAgent](https://arxiv.org/pdf/2403.10517) for the helpful discussion. 

# Reference
Please cite our paper if you use our models in your works:

```bibtex
@InProceedings{wang2025video,
    author    = {Wang, Ziyang and Yoon, Jaehong and Yu, Shoubin and Islam, Md Mohaiminul  and Bertasius, Gedas and Bansal, Mohit},
    title     = {VIDEO-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning},
    booktitle = {Arxiv},
}
