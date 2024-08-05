---
title: "FLOPS-efficient DETR Training"
excerpt: "A recipe to train Object Detection Transformers with high MFU.<br/><img src='https://raw.githubusercontent.com/MasterSkepticista/detr/main/.github/detr.png' width='640'>"
collection: portfolio
---

Object Detection Transformers, are known for their long, inefficient training schedules. While there have been many successors to the original [DETR](https://arxiv.org/abs/2005.12872) that improve algorithmic convergence rates dramatically, like [Deformable-DETR](https://arxiv.org/abs/2010.04159), or [Conditional-DETR](https://arxiv.org/abs/2108.06152), these implementations do not achieve a high MFU (arithmetic intensity) on the GPU.


The bottleneck
---
<img src='https://raw.githubusercontent.com/MasterSkepticista/detr/main/.github/detr.png'>
DETR has three main components: a ResNet backbone, a stack of encoder-decoder transformer blocks, and a bipartite matcher. Of the three, no popular GPU-optimized bipartite matching (hungarian) algorithms exist. In fact, the original DETR implementation calls [`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) sequentially, for each input-target pair, on the CPU.


Baseline
---
Let's measure what we are starting with. FAIR's DETR implementation would take roughly 10 days to train a 300-epoch baseline on our 4x-3090 GPU cluster (I actually had to employ a few [tricks](url) to get it to scale to 94% efficiency!). Welcome to the GPU-poor equivalent of 2023.

We can now start with a few broad goals:
* Perform bipartite matching on GPU
* Vectorize matching across samples (process all in parallel).
* Optimize via XLA JIT.

So, I chose JAX to implement the training pipeline. It is easier to work with TFRecord-based `tf.data` pipelines, and XLA almost always outperforms even `torch.compile` optimized PyTorch in my experience. On the flip side, writing XLA-philosophy code comes with slightly sub-par pythonic experience. But I see this getting [better](https://news.ycombinator.com/item?id=29130582) with JAX.

~Build~ Refactor
---
I got a head start. Scenic's [re-implementation](https://github.com/google-research/scenic/tree/main/scenic/projects/baselines/detr) of DETR was one (and only one) written in JAX. But if you were to try to run it today, as I did in 2023, it would probably fail (unless this refactor [PR](https://github.com/google-research/scenic/pull/1062) I later created got merged).


Lucky for me, `scenic` already [had](https://github.com/google-research/scenic/blob/main/scenic/model_lib/matchers/hungarian_jax.py) a GPU and TPU implementation of Hungarian matching.
