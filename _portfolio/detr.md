---
title: "PyTorch: I’m Fast, JAX: You Call That Fast?"
excerpt: "A recipe to train Object Detection Transformers (really) fast.<br/><img src='https://raw.githubusercontent.com/MasterSkepticista/detr/main/.github/detr.png' width='640'>"
collection: portfolio
---

Ok, I did not mean PyTorch is slow. But it is always a fun (and worthy) exercise to flex how fast you can _really_ go with compilers when you lay out computation in the right way.

This is my work log of building a Detection Transformer ([DETR](https://arxiv.org/abs/2005.12872)) training pipeline in JAX. DETR architecture is special for many reasons - one of which is that it predicts bounding boxes and class labels directly, instead of relying on region-proposals and custom post-processing techniques. Second, I personally love the elegance of an end-to-end differentiable method like DETR. It fits the _spirit of deep learning_ and borrows wisdom from Rich Sutton's [Bitter Lesson](http://incompleteideas.net/IncIdeas/BitterLesson.html) of AI research.

DETR is slow to train, though. While there have been many successors to the original DETR that improve algorithmic convergence rates, like [Deformable-DETR](https://arxiv.org/abs/2010.04159), or [Conditional-DETR](https://arxiv.org/abs/2108.06152), none of these implementations focus on running 'efficiently' on the GPU. I will walk through certain techniques that can provide up to $$30\%$$ higher GPU utilization against a best-effort optimized [PyTorch implementation of DETR](https://github.com/facebookresearch/detr).

I highly recommend you read the original DETR [paper](https://arxiv.org/abs/2005.12872).


The Bottleneck
===
<img src='https://raw.githubusercontent.com/MasterSkepticista/detr/main/.github/detr.png'>
<p align='center'> Figure: DETR Architecture </p>
DETR has three main components: a CNN backbone (typically a ResNet), a stack of encoder-decoder transformer blocks, and a bipartite matcher. Of the three, bipartite matching (hungarian) algorithm runs on the CPU. In fact, the original DETR implementation calls [`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) sequentially, for each input-target pair. This leaves the GPU idle. Part of the gains we will see later, are by reducing this idle time.

> Idle GPU is wasted GPU.

Baseline
===
Let us measure what we are starting with. Our bench this time is an 8-A6000 cluster without NVLink. I made a couple of changes to ensure PyTorch version was 'as fast as possible'. Here is a summary of digressions:
* Use `torch>=2.3` to ensure [Flash Attention](https://arxiv.org/abs/2205.14135) is used in `F.scaled_dot_product_attention`.
* Set `torch.set_float32_matmul_precision(...)` to "medium".

With these changes, it took 3 days to train a 300-epoch baseline on our 8-GPU cluster. I will skip the napkin math, but this is already faster than authors' numbers when normalized for per GPU FLOP throughput - notably from use of the new flash attention kernel that Ampere GPUs support.

> N.B.: While I did try `torch.compile` with different options on sub-parts of the model/training step, it either ended up giving the same throughput, or failed to compile. 

Refactor
===
I decided to implement DETR in JAX. You can think of JAX as a native front-end language to write [XLA](https://openxla.org/xla) optimized programs. XLA generally outperforms the superset of all PyTorch optimizations _when done right, by a large margin_. One downside of working with XLA/JAX is that it is harder to debug `jit` compiled programs. PyTorch, on the other hand, dispatches CUDA kernels eagerly (except when wrapped with `torch.compile`), which makes it easiest to debug and work with. But when you consider the cost of few compile minutes over how long production training runs like these typically are, it is worth the tradeoff.

Luckily a dusty [re-implementation](https://github.com/google-research/scenic/tree/main/scenic/projects/baselines/detr) of DETR in JAX made for a good head-start. But it did not work out-of-the-box due to deprecated JAX and Flax APIs. To get the ball rolling, I made a minimal set of [changes](https://github.com/google-research/scenic/pull/1062), without any optimizations.

Scenic also [provides](https://github.com/google-research/scenic/blob/main/scenic/model_lib/matchers/hungarian_jax.py) GPU and TPU implementations of Hungarian matching. This is already significant work off-the-table.

This implementation takes 6.5 days to replicate the PyTorch baseline. How fast can we go?

<img style="display: block; margin: auto;" src="/images/detr_opts/pt_baseline.png" width="512">

Optimize!
===
Disable Matching for padded objects
---
This is actually a bug-fix rather than an optimization. COCO dataset does not guarantee a fixed number of objects for each image. This means the bipartite matcher would have to map a fixed set of object queries (say 100) to a randomly varying number of target objects for each image, triggering an expensive retrace of the graph.
> N.B.: XLA compiler can generate optimized graphs in part because memory allocation/deallocation is predictable, and constant-folding/fusion of operators is simpler when the entire computational graph layout is static. This is the price you pay for performance. You can read more [here](https://www.tensorflow.org/guide/function).

To prevent retracing, we add 'padding' objects and a boolean mask that allows us to filter dummy objects when computing loss.

```python
# Adding padded dimensions
# https://github.com/MasterSkepticista/detr/blob/main/input_pipeline.py#L145
padded_shapes = {
    'inputs': [max_size, max_size, 3],
    'padding_mask': [max_size, max_size],
    'label': {
        'boxes': [max_boxes, 4],
        'area': [max_boxes,],
        'objects/id': [max_boxes,],
        'is_crowd': [max_boxes,],
        'labels': [max_boxes,],
        'image/id': [],
        'size': [2,],
        'orig_size': [2,],
    },
}
```

But this still computes bipartite matching on `padded` objects. We can remove
constants from the `cost` matrix as they do not affect the final matching.

```patch

-- cost = cost * mask + (1.0 - mask) * cost_upper_bound
++ cost = cost * mask
```

With this bug-fix, we are now 40% faster, i.e. $$1.4$$ steps/s. It now takes 4.7 days to train the baseline.

<img style="display: block; margin: auto;" src="/images/detr_opts/disable_padded.png">

Mixed Precision MatMuls
---
Yes, there are no 'free-lunches', but I think we can make a strong case for the invention of `bfloat16` data type.
We migrate `float32` matmuls to `bfloat16`, without any loss in final AP scores.
For `flax`, it is the same as supplying `dtype=jnp.bfloat16` on supported modules.

```python
# Example conversion.
conv = nn.Conv(..., dtype=jnp.bfloat16)
dense = nn.Dense(..., dtype=jnp.bfloat16)
...
```

This gets us above $$2.1$$ steps/s. We now have performance parity with PyTorch, with 3.1 days taken to train the baseline!

<img style="display: block; margin: auto;" src="/images/detr_opts/mixed_prec.png">

Huh! We should've called it a day... but let's keep going.

Parallel Bipartite Matching on Decoders
---
To achieve a high overall $$\text{mAP}$$ score, DETR authors propose computing loss over each decoder output. DETR uses a sequential stack of 6 decoders, each emitting bounding-box and classifier predictions for a given number of queries.

```python
# models/detr_base_model.py#L377
# Computing matchings for each decoder head (auxiliary predictions)
# outputs = {
#   "pred_logits": ndarray, 
#   "pred_boxes": ndarray,
#   "aux_outputs": [
#     {"pred_logits": ndarray, "pred_boxes": ndarray},
#     {"pred_logits": ndarray, "pred_boxes": ndarray},
#     ...
#   ]
# }
if matches is None:
  cost, n_cols = self.compute_cost_matrix(outputs, batch['label'])
  matches = self.matcher(cost, n_cols)
  if 'aux_outputs' in outputs:
    matches = [matches]
    for aux_pred in outputs['aux_outputs']:
      cost, n_cols = self.compute_cost_matrix(aux_pred, batch['label'])
      matches.append(self.matcher(cost, n_cols))
```

Computing optimal matchings on these decoder outputs can actually be done in parallel using `vmap`.
```python
# models/detr_base_model.py#L377
# After vectorization
if matches is None:
  predictions = [{
    "pred_logits": outputs["pred_logits"],
    "pred_boxes": outputs["pred_boxes"]
  }]
  if 'aux_outputs' in outputs:
    predictions.extend(outputs["aux_outputs"])

  def _compute_matches(predictions, targets):
    cost, n_cols = self.compute_cost_matrix(predictions, targets)
    return self.matcher(cost, n_cols)

  # Stack list of pytrees.
  predictions = jax.tree.map(lambda *args: jnp.stack(args), *predictions)

  # Compute matches in parallel for all outputs.
  matches = jax.vmap(_compute_matches, (0, None))(predictions, batch["label"])
  matches = list(matches)
```

With this change, we are now stepping **10% faster** than PyTorch, at $$2.4$$ steps/s, i.e. 2.7 days to train.

<img style="display: block; margin: auto;" src="/images/detr_opts/parallel_match.png">

Use Flash Attention
---
XLA did not use flash attention kernel all along. It was added only recently through [`jax.nn.dot_product_attention`](https://github.com/google/jax/pull/21371) for Ampere and later architectures. Perhaps future XLA versions might automatically recognize a dot-product attention signature during `jit`, without us having to explicitly call via SDPA API. But that is not the case today, so we will make-do with this custom function call.

```python
# models/detr.py#L261
if True:
  x = jax.nn.dot_product_attention(query, key, value, mask=mask, implementation="cudnn")
else:
  x = attention_layers.dot_product_attention(
      query,
      key,
      value,
      mask=mask,
      dropout_rate=self.dropout_rate,
      broadcast_dropout=self.broadcast_dropout,
      dropout_rng=self.make_rng('dropout') if train else None,
      deterministic=not train,
      capture_attention_weights=False)
```
> Note: As of writing SDPA API does not support attention dropout. This is because JAX and cuDNN use different PRNG implementations. Once `dropout` makes its way to SDPA API, we can set flash attention to be our default.

For now, let us be content with the _potential_ speedup. We are now at $$3.0$$ steps/s, **33% faster** than PyTorch, taking 2 days to train.

<img style="display: block; margin: auto;" src="/images/detr_opts/flash_attn.png">

Summary
---
Further gains are possible by replacing exact bipartite matching with an approximate matcher. In fact, it may be a good reason to do so - just like minibatch SGD does not give an accurate gradient estimate at each step. It is arguably its strong suit on why it converges so well. 

Why should a matching algorithm be exact, if we are spending ~0.5M steps to converge anyway? Are there gains to be had by having an 'approximate' matching? Yes, and one way to go about it is using a regularized optimal transport solver. But that is for another day.

Training code for DETR with all above optimizations is available [here](https://github.com/masterskepticista/detr).