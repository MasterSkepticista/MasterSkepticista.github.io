---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* B.Tech. in Electronics and Communications Engineering, Nirma University, 2020 \
  Capstone Project: Mechanistic Interpretability of Structural MRI Segmentation Models.

Work experience
======
* Research Engineer, Intel
  * Building [OpenFL](https://github.com/securefederatedai/openfl) (a [**Linux Foundation** Project](https://openfl.io))
  * Deployed India's **first** SGX-based Secure Federated Learning Stack for Chest X-Ray Anomaly Detection ([Link](https://health.economictimes.indiatimes.com/news/health-it/aster-dm-intel-carpl-collaborate-for-secure-federated-learning-platform/92599071)).
  
* Intern, Intel
  * Built a mechanistic interpretability toolbox - to measure uncertainty attention and weight patterns for image activations.
  * Developed a hardened C++ application to run a tumor segmentation model within SGX Trusted Execution Environment.

Skills
======
* **Programming:** Python, C++
* **Frameworks:** JAX, PyTorch
* **Tools:** OpenMPI, Docker, Gramine

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
  
<!-- Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul> -->

Open Source
===========
[securefederatedai/openfl](https://github.com/securefederatedai/openfl), 
[google/scenic](https://github.com/google-research/scenic/tree/main),
[google/jax](https://github.com/google/jax),
[google/flax](https://github.com/google/flax)
  
<!-- Service and leadership
======
* Currently signed in to 43 different slack teams -->

<!-- * Developed a tool to capture membership inference attacks on synthetic MRIs generated from Diffusion models. -->
<!-- * Slashed training time of Object Detection Transformer from 10 days to 3.5 days (details [here](https://github.com/masterskepticista/detr)).
* Cut annual cloud spends by $112k - profiling and tuning repurposed training nodes for up to 94% scaling with 10GbE.
* Optimized ETL, deduplication, cleanup of 1.6M FHD images by 4.5x using TensorFlow and TFRecords. -->
<!-- * Deployed a ViT-based classification model across at KPN Retail, Coimbatore that brings INR 1.5cr of annualized revenue. -->
<!-- * Delivered multimodal RAG-based summarizer on Intel's internal GenAI LLM platform with >4,100 monthly avg users.
* Containerized Movidius Inference validation stack, reducing infra costs by **60%** and **12x** faster setup time. -->
