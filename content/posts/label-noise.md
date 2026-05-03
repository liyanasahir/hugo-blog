---
title: "A Field Guide to Learning with Noisy Labels"
date: "2025-12-01"
slug: "label-noise"
draft: true
tags: ["notes", "label-noise", "deep-learning", "memorization"]
math: true
---

Annotation is expensive. And when it's expensive, it gets noisy.

Labels can become unreliable for a number of reasons: non-expert annotators who disagree on ambiguous cases, labels derived from weak signals like search queries or hashtags, or predictions from an earlier, less robust model used to bootstrap a larger dataset. In real-world settings (fraud detection, medical imaging, content moderation), you rarely have the luxury of a perfectly clean training set. You train on what you have, and what you have is messy.

This post is a structured overview of the label noise literature, from reading conducted around my interest in learning with label noise and presented in a Paper Reading Group session conducted at work in 2024 (that is to say, I might be missing some important work that has come out since). It covers the problem formulation, four foundational empirical observations that I took away that I think motivates most of the field, a taxonomy of approaches, and a deeper look at one recent paper that I think handles the problem particularly cleanly. It ends with some open questions that I don't think the literature has settled yet.

---

## Problem Formulation

We have a dataset of $n$ examples $\{(x_1, \tilde{y}_1), \ldots, (x_n, \tilde{y}_n)\} \sim \tilde{D}^n$, where $\tilde{y}_i$ is the *observed* (possibly noisy) label, not the true label $y_i$. The goal is to learn a classifier $f_\rho$ from the noisy distribution such that:

$$f_\rho(x) = \arg\max_y P(Y = y \mid X = x)$$

We want to recover the Bayes-optimal classifier despite the label corruption.

### Types of label noise

The literature standardizes around two noise models, both characterized by a **noise transition matrix** $Q$ where $Q_{ij} = P(\tilde{Y} = j \mid Y = i)$:

**Symmetric (uniform) noise**: each label has a probability $\epsilon / (n-1)$ of being flipped uniformly to any other class:

$$Q = \begin{bmatrix} 1-\epsilon & \frac{\epsilon}{n-1} & \cdots & \frac{\epsilon}{n-1} \\ \frac{\epsilon}{n-1} & 1-\epsilon & \cdots & \frac{\epsilon}{n-1} \\ \vdots & & \ddots & \vdots \\ \frac{\epsilon}{n-1} & \cdots & \frac{\epsilon}{n-1} & 1-\epsilon \end{bmatrix}$$

Concretely: for a 3-class problem with 30% symmetric noise, a dog has a 15% chance of being labeled as cat and a 15% chance of being labeled as horse, regardless of how visually similar those classes are.

**Asymmetric (pair) noise**: labels flip only to semantically similar classes with probability $\epsilon$. This is more realistic: a "cat" is more likely to be mislabeled as "dog" than as "airplane."

$$Q = \begin{bmatrix} 1-\epsilon & \epsilon & 0 & \cdots & 0 \\ 0 & 1-\epsilon & \epsilon & & 0 \\ \vdots & & \ddots & \ddots & \vdots \\ 0 & & & 1-\epsilon & \epsilon \\ \epsilon & 0 & \cdots & 0 & 1-\epsilon \end{bmatrix}$$

A third type, **instance-dependent noise**, is harder to model but more faithful to reality: the probability of mislabeling depends on the specific input, not just the class. An ambiguous image at a class boundary is more likely to be mislabeled than a prototypical one. Recent work (*Part-dependent Label Noise*, Xia et al., NeurIPS 2020; *Instance-dependent Label-noise Learning*, Wang et al., NeurIPS 2021) has made progress here, but most of the established literature focuses on symmetric and asymmetric noise because they yield tractable theoretical analysis.

**What this tractability assumption costs you**: the transition matrix framework assumes noise is class-conditional, meaning knowing the true class is sufficient to characterize how it gets mislabeled. In practice, annotator behavior, ambiguous instances, and domain-specific label conventions all introduce structure that a single matrix cannot capture. Methods developed and evaluated under the symmetric noise assumption often degrade faster than expected when deployed on real-world noisy datasets like CIFAR-10N or WebVision, where the noise is genuinely instance-dependent.

<figure style="margin: 2rem 0; text-align: center;">
<svg viewBox="0 0 560 215" xmlns="http://www.w3.org/2000/svg" style="max-width: 560px; width: 100%; height: auto;">
  <text x="140" y="18" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" fill="#4a9e9e">Symmetric noise (ε = 30%)</text>
  <text x="420" y="18" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" fill="#4a9e9e">Asymmetric noise (ε = 30%)</text>
  <text x="44" y="66" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10" fill="#888">dog</text>
  <text x="44" y="106" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10" fill="#888">cat</text>
  <text x="44" y="146" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10" fill="#888">horse</text>
  <text x="80" y="34" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#888">dog</text>
  <text x="130" y="34" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#888">cat</text>
  <text x="180" y="34" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#888">horse</text>
  <rect x="55" y="44" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.75"/><text x="80" y="69" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" fill="white">0.70</text>
  <rect x="105" y="44" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.15"/><text x="130" y="69" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#666">0.15</text>
  <rect x="155" y="44" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.15"/><text x="180" y="69" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#666">0.15</text>
  <rect x="55" y="84" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.15"/><text x="80" y="109" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#666">0.15</text>
  <rect x="105" y="84" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.75"/><text x="130" y="109" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" fill="white">0.70</text>
  <rect x="155" y="84" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.15"/><text x="180" y="109" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#666">0.15</text>
  <rect x="55" y="124" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.15"/><text x="80" y="149" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#666">0.15</text>
  <rect x="105" y="124" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.15"/><text x="130" y="149" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#666">0.15</text>
  <rect x="155" y="124" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.75"/><text x="180" y="149" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" fill="white">0.70</text>
  <line x1="280" y1="28" x2="280" y2="178" stroke="#ddd" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="324" y="66" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10" fill="#888">dog</text>
  <text x="324" y="106" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10" fill="#888">cat</text>
  <text x="324" y="146" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10" fill="#888">horse</text>
  <text x="360" y="34" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#888">dog</text>
  <text x="410" y="34" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#888">cat</text>
  <text x="460" y="34" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#888">horse</text>
  <rect x="335" y="44" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.75"/><text x="360" y="69" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" fill="white">0.70</text>
  <rect x="385" y="44" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.45"/><text x="410" y="69" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" fill="white">0.30</text>
  <rect x="435" y="44" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.04"/><text x="460" y="69" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#ccc">0.00</text>
  <rect x="335" y="84" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.04"/><text x="360" y="109" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#ccc">0.00</text>
  <rect x="385" y="84" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.75"/><text x="410" y="109" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" fill="white">0.70</text>
  <rect x="435" y="84" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.45"/><text x="460" y="109" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" fill="white">0.30</text>
  <rect x="335" y="124" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.45"/><text x="360" y="149" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" fill="white">0.30</text>
  <rect x="385" y="124" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.04"/><text x="410" y="149" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#ccc">0.00</text>
  <rect x="435" y="124" width="50" height="40" rx="4" fill="#4a9e9e" fill-opacity="0.75"/><text x="460" y="149" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="600" fill="white">0.70</text>
  <text x="130" y="195" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#aaa">← observed label (ỹ) →</text>
  <text x="410" y="195" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#aaa">← observed label (ỹ) →</text>
  <text x="27" y="105" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#aaa" transform="rotate(-90,27,105)">true label (y)</text>
  <text x="307" y="105" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#aaa" transform="rotate(-90,307,105)">true label (y)</text>
</svg>
<figcaption style="font-size: 13px; color: #999; margin-top: 0.5rem;">Figure 1. Noise transition matrix Q for symmetric vs. asymmetric noise (3-class, ε = 0.30). Cell opacity encodes flip probability. Symmetric noise spreads evenly; asymmetric noise flips only to the next class (dog→cat, cat→horse, horse→dog).</figcaption>
</figure>

---

## Four Foundational Observations

Before getting into methods, it's worth building up an intuition for *why* label noise is hard. The following four observations, drawn from my reading of the literature from 2017 to 2021, together explain most of the challenge and motivate most of the solutions. Each comes with an important scope condition: the regime in which it holds, and where it tends to break down.

### #1: Neural networks can memorize random labels

**Paper**: *Understanding Deep Learning Requires Rethinking Generalization* (Zhang et al., ICLR 2017, Best Paper)

The uncomfortable baseline: if you train a neural network on CIFAR-10 with randomly shuffled labels, it will eventually achieve near-zero training loss. Networks have enough capacity to memorize the entire training set, noise included. Regularization (dropout, weight decay, data augmentation) improves generalization performance in the clean setting, but is neither necessary nor sufficient to prevent memorization of noisy labels.

**Scope**: this is well-established across architectures and dataset sizes. The rate of memorization varies (wider, deeper networks memorize faster) but the eventual outcome does not. The implication is sharp: given enough training time, your model *will* overfit to noisy labels. The question is not whether, but when.

### #2: Neural networks learn simple patterns before memorizing

**Paper**: *A Closer Look at Memorization in Deep Networks* (Arpit et al., ICML 2017)

There is a qualitative difference in how networks learn real data versus random noise. For real data, networks first discover generalizable patterns. For noisy data, they memorize instance by instance. Crucially, regularizers slow down memorization of noisy labels much more than they slow down learning on clean data, suggesting a temporal window early in training where the model is learning signal rather than noise.

Imagine plotting clean-label accuracy and noisy-label accuracy on the same axes over training epochs: they rise together in the early phase, then diverge as the network starts fitting noisy examples. The peak of clean-label accuracy (before the divergence) is the target you're trying to preserve.

**Scope**: this temporal structure is most visible at moderate noise rates (roughly 20-60% symmetric noise). At very high noise rates, the clean signal may be too weak to produce a meaningful early learning phase. In long-tailed settings, tail classes may show almost no clean learning phase at all, since there are simply too few clean examples to establish a pattern before memorization begins.

<figure style="margin: 2rem 0; text-align: center;">
<svg viewBox="0 0 500 210" xmlns="http://www.w3.org/2000/svg" style="max-width: 500px; width: 100%; height: auto;">
  <line x1="50" y1="160" x2="460" y2="160" stroke="#ddd" stroke-width="1"/>
  <line x1="50" y1="20" x2="50" y2="160" stroke="#ddd" stroke-width="1"/>
  <text x="255" y="180" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#aaa">training epochs →</text>
  <text x="16" y="90" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#aaa" transform="rotate(-90,16,90)">accuracy</text>
  <path d="M50,148 C100,122 160,90 230,70 290,56 360,50 460,48 L460,62 C360,64 290,72 230,88 160,108 100,140 50,162 Z" fill="#4a9e9e" fill-opacity="0.08"/>
  <path d="M50,148 C100,122 160,90 230,70 290,56 330,56 360,64 400,78 440,96 460,108" fill="none" stroke="#4a9e9e" stroke-width="2.2" stroke-dasharray="6,3" stroke-linecap="round"/>
  <path d="M50,148 C100,120 160,88 230,66 290,50 360,44 460,42" fill="none" stroke="#333" stroke-width="2.2" stroke-linecap="round"/>
  <line x1="308" y1="58" x2="308" y2="162" stroke="#e0e0e0" stroke-width="1" stroke-dasharray="3,3"/>
  <rect x="248" y="165" width="130" height="22" rx="6" fill="#f0fafa" stroke="#4a9e9e" stroke-width="0.8"/>
  <text x="313" y="180" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#4a9e9e">memorization begins ↑</text>
  <line x1="280" y1="28" x2="306" y2="28" stroke="#333" stroke-width="2.2" stroke-linecap="round"/>
  <text x="312" y="32" font-family="-apple-system, sans-serif" font-size="11" fill="#555">clean labels</text>
  <line x1="280" y1="46" x2="306" y2="46" stroke="#4a9e9e" stroke-width="2.2" stroke-dasharray="6,3" stroke-linecap="round"/>
  <text x="312" y="50" font-family="-apple-system, sans-serif" font-size="11" fill="#555">noisy labels</text>
  <text x="44" y="50" text-anchor="end" font-family="-apple-system, sans-serif" font-size="9" fill="#ccc">high</text>
  <text x="44" y="164" text-anchor="end" font-family="-apple-system, sans-serif" font-size="9" fill="#ccc">low</text>
</svg>
<figcaption style="font-size: 13px; color: #999; margin-top: 0.5rem;">Figure 2. Memorization dynamics over training. Both curves rise together in the early phase as the network learns generalizable patterns, then diverge as it starts fitting noisy examples. Methods that intervene before the divergence point preserve clean accuracy.</figcaption>
</figure>

### #3: Small-loss instances are more likely to have correct labels

This follows directly from observations #1 and #2: if networks learn patterns before memorizing noise, then early in training, instances with small loss are more likely to carry genuine signal. This is the key insight behind a large family of **sample selection** methods.

The canonical operationalization is **Co-teaching** (Han et al., NeurIPS 2018): train two networks simultaneously, have each select its small-loss instances, and cross-update: each network trains on the other's selected examples. The cross-update prevents both networks from collapsing to the same memorized patterns; different random initializations produce different decision boundaries and thus different failure modes.

Prior work explored related ideas:
- **MentorNet**: a separate mentor network selects clean instances, but requires clean validation data or a predefined curriculum when none is available.
- **Decoupling**: trains two networks and updates only on instances where they disagree - simpler than loss thresholding, but less principled about the selection criterion.
- **Co-teaching+**: extends Co-teaching by also requiring disagreement between networks, combining both signals.

What does Co-teaching buy over Decoupling? The loss-based selection is more directly motivated by the memorization dynamics: it selects examples the network hasn't yet memorized. Disagreement-based selection (Decoupling) is more conservative and can fail when both networks agree on a wrong answer early in training.

**Scope**: the small-loss heuristic works well at moderate noise rates but has two known failure modes. First, at very high symmetric noise (above ~80%), the proportion of truly clean small-loss instances drops sharply, and the selection becomes unreliable. Second, in long-tailed distributions, tail class examples are inherently harder and tend to have higher loss even when correctly labeled, making them indistinguishable from noisy examples by loss alone. This is a fundamental limitation that purely loss-based methods cannot address without additional structure.

### #4: Self-supervised representations are robust to label noise

**Paper**: *Contrastive Learning Improves Model Robustness Under Label Noise* (Ghosh & Lan, CVPR 2021)

Representations learned through contrastive objectives (MoCo, SimCLR, PCL) are significantly more robust to label noise than representations learned from scratch with supervised loss. The reason is structural: contrastive learning is label-agnostic. It learns to cluster semantically similar instances together based on augmentation invariance, not label agreement. When labels are noisy, the SSL objective still pulls toward the correct semantic structure.

The effect is visible in the geometry of the learned representations: supervised representations under noise collapse semantically similar classes together while mixing in noisy examples; SSL representations maintain cleaner cluster separation. Starting from a good SSL initialization before supervised fine-tuning dramatically reduces the damage noisy labels can do.

**Scope**: this observation assumes that the SSL objective produces features that are genuinely discriminative for your downstream task, which holds for image classification but is less obvious for tabular or graph-structured data, where meaningful augmentations are harder to define. It also assumes a reasonably balanced class distribution: contrastive learning on severely imbalanced data tends to produce representations dominated by majority class geometry, which may not protect minority class examples from label noise.

---

## A Taxonomy of Approaches

The literature organizes into four broad families. A useful survey is *Learning from Noisy Labels with Deep Neural Networks: A Survey* (Song et al., TNNLS 2022).

<figure style="margin: 2rem 0; text-align: center;">
<svg viewBox="0 0 580 210" xmlns="http://www.w3.org/2000/svg" style="max-width: 580px; width: 100%; height: auto;">
  <rect x="190" y="8" width="200" height="36" rx="18" fill="#4a9e9e"/>
  <text x="290" y="31" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" fill="white">Noisy Label Methods</text>
  <path d="M290 44 C290 62 75 62 75 80" fill="none" stroke="#4a9e9e" stroke-width="1.5" stroke-opacity="0.5"/>
  <path d="M290 44 C290 62 195 62 195 80" fill="none" stroke="#4a9e9e" stroke-width="1.5" stroke-opacity="0.5"/>
  <path d="M290 44 C290 62 385 62 385 80" fill="none" stroke="#4a9e9e" stroke-width="1.5" stroke-opacity="0.5"/>
  <path d="M290 44 C290 62 505 62 505 80" fill="none" stroke="#4a9e9e" stroke-width="1.5" stroke-opacity="0.5"/>
  <rect x="25" y="80" width="100" height="42" rx="8" fill="#f0fafa" stroke="#4a9e9e" stroke-width="1.2"/>
  <text x="75" y="99" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">Robust</text>
  <text x="75" y="113" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">Architecture</text>
  <rect x="145" y="80" width="100" height="42" rx="8" fill="#f0fafa" stroke="#4a9e9e" stroke-width="1.2"/>
  <text x="195" y="99" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">Robust</text>
  <text x="195" y="113" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">Regularization</text>
  <rect x="335" y="80" width="100" height="42" rx="8" fill="#f0fafa" stroke="#4a9e9e" stroke-width="1.2"/>
  <text x="385" y="99" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">Robust</text>
  <text x="385" y="113" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">Loss Design</text>
  <rect x="455" y="80" width="100" height="42" rx="8" fill="#f0fafa" stroke="#4a9e9e" stroke-width="1.2"/>
  <text x="505" y="99" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">Sample</text>
  <text x="505" y="113" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">Selection</text>
  <line x1="75" y1="122" x2="75" y2="138" stroke="#ccc" stroke-width="1"/>
  <line x1="195" y1="122" x2="195" y2="138" stroke="#ccc" stroke-width="1"/>
  <line x1="385" y1="122" x2="385" y2="138" stroke="#ccc" stroke-width="1"/>
  <line x1="505" y1="122" x2="505" y2="138" stroke="#ccc" stroke-width="1"/>
  <text x="75" y="150" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#999">noise adapt. layer</text>
  <text x="195" y="150" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#999">early stopping,</text>
  <text x="195" y="162" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#999">consistency reg.</text>
  <text x="385" y="150" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#999">MAE, GCE,</text>
  <text x="385" y="162" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#999">Peer Loss</text>
  <text x="505" y="150" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#999">Co-teaching,</text>
  <text x="505" y="162" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#999">DivideMix</text>
  <rect x="455" y="173" width="100" height="20" rx="10" fill="#4a9e9e" fill-opacity="0.12" stroke="#4a9e9e" stroke-width="0.8"/>
  <text x="505" y="187" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#4a9e9e">+ RRL (deep dive ↓)</text>
</svg>
<figcaption style="font-size: 13px; color: #999; margin-top: 0.5rem;">Figure 3. Four families of noisy-label methods. Sample selection currently achieves strongest benchmark results; RRL extends this family with SSL-based representation regularization.</figcaption>
</figure>

**Robust architecture**: modify the network to be inherently noise-resistant, typically by adding a noise adaptation layer that models the transition matrix explicitly, learning to correct predictions before computing loss. The limitation is fundamental: these methods treat noise as a fixed property of the label space, and are agnostic to the memorization dynamics of the network itself.

**Robust regularization**: add explicit or implicit regularization to slow memorization. Early stopping is surprisingly effective and underrated. More principled variants include consistency regularization across augmented views, and representation-level regularization (discussed below). The limitation: regularization slows memorization; it doesn't prevent it given sufficient training time.

**Robust loss design**: replace cross-entropy with a theoretically noise-robust alternative:
- *Mean Absolute Error (MAE)*: provably robust to symmetric noise, but severe underfitting in practice: it treats all errors equally, losing the gradient signal from easy examples.
- *Generalized Cross Entropy (GCE)* (Zhang & Sabuncu, NeurIPS 2018): interpolates between CCE and MAE via a parameter $q$. Empirically better than both extremes.
- *Peer Loss* (Liu & Guo, ICML 2020): statistically robust without requiring knowledge of noise rates, by comparing the loss of the true example against a randomly drawn peer example. Extensions handle instance-dependent noise.

On a standard benchmark (CIFAR-10, 60% symmetric noise), GCE sits around 72-73% test accuracy, Peer Loss around 77-78%, while vanilla CE drops to around 61%. The gains are real but not decisive, and they come at the cost of training stability. The shared limitation: robust losses reduce the *pressure* to memorize noisy labels but don't eliminate the tendency, because they don't directly engage with the memorization dynamics.

**Sample selection**: identify and down-weight or exclude noisy examples during training, using the small-loss heuristic (Co-teaching variants), GMM-based separation of clean and noisy loss distributions (DivideMix), or hybrid approaches that combine multiple signals. This family currently achieves the strongest results on standard benchmarks, but inherits the scope limitations of observation #3 at high noise rates and under class imbalance.

---

## A Closer Look: RRL (ICLR 2023)

*Mitigating Memorization of Noisy Labels via Regularization Between Representations*

This paper sits at the intersection of observations #2 and #4. It's worth spending time on because it's an example of the field doing something right: instead of proposing a new loss function or selection heuristic, it asks a more fundamental question about *what kind of representations* make a network robust to noisy labels.

RRL is essentially a principled operationalization of observation #4, designed to respect the learning dynamics from observations #2 and #3. The core idea: use self-supervised features as a regularizer on supervised features, so that when the supervised signal is corrupted by noise, the SSL geometry pulls the representations back toward correct structure.

### Decoupling encoder from classifier

A neural classifier can be written as $C(X) = g(f(X))$, an encoder $f$ followed by a linear classifier $g$. RRL considers three learning paths:

- **Path 1**: learn $f$ and $g$ jointly (standard training)
- **Path 2**: initialize $f$ from SSL pre-training, fine-tune both
- **Path 3**: initialize from SSL, freeze $f$, only update $g$

The finding: under high noise, a properly chosen fixed encoder outperforms an unfixed one. But freezing entirely is suboptimal at low noise. RRL finds a principled compromise by adding a regularization term that allows the encoder to move, but penalizes it for moving in directions that disagree with the SSL feature structure.

### The training framework

<figure style="margin: 2rem 0; text-align: center;">
<svg viewBox="0 0 580 200" xmlns="http://www.w3.org/2000/svg" style="max-width: 580px; width: 100%; height: auto;">
  <defs>
    <marker id="arr-rrl" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
      <path d="M0,0 L0,7 L7,3.5 z" fill="#4a9e9e" opacity="0.65"/>
    </marker>
  </defs>
  <rect x="8" y="82" width="68" height="36" rx="8" fill="#f0fafa" stroke="#4a9e9e" stroke-width="1.2"/>
  <text x="42" y="101" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">image xₙ</text>
  <path d="M76 90 C100 90 108 48 132 48" fill="none" stroke="#4a9e9e" stroke-width="1.4" stroke-opacity="0.6" marker-end="url(#arr-rrl)"/>
  <path d="M76 100 C100 100 108 152 132 152" fill="none" stroke="#4a9e9e" stroke-width="1.4" stroke-opacity="0.6" marker-end="url(#arr-rrl)"/>
  <rect x="132" y="28" width="105" height="38" rx="8" fill="#f0fafa" stroke="#4a9e9e" stroke-width="1.2"/>
  <text x="184" y="47" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">SSL encoder  f→h</text>
  <text x="184" y="59" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#4a9e9e">label-agnostic</text>
  <line x1="237" y1="47" x2="268" y2="47" stroke="#4a9e9e" stroke-width="1.4" stroke-opacity="0.6" marker-end="url(#arr-rrl)"/>
  <rect x="268" y="28" width="88" height="38" rx="8" fill="#4a9e9e" fill-opacity="0.12" stroke="#4a9e9e" stroke-width="1"/>
  <text x="312" y="47" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">tₙ = h(f(xₙ))</text>
  <text x="312" y="59" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#4a9e9e">SSL features</text>
  <rect x="132" y="132" width="105" height="38" rx="8" fill="#f0fafa" stroke="#4a9e9e" stroke-width="1.2"/>
  <text x="184" y="151" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">SL encoder  f→g</text>
  <text x="184" y="163" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#4a9e9e">noisy labels</text>
  <line x1="237" y1="151" x2="268" y2="151" stroke="#4a9e9e" stroke-width="1.4" stroke-opacity="0.6" marker-end="url(#arr-rrl)"/>
  <rect x="268" y="132" width="88" height="38" rx="8" fill="#4a9e9e" fill-opacity="0.12" stroke="#4a9e9e" stroke-width="1"/>
  <text x="312" y="151" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">sₙ = g(f(xₙ))</text>
  <text x="312" y="163" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#4a9e9e">SL features</text>
  <path d="M312 66 L312 90" fill="none" stroke="#4a9e9e" stroke-width="1.4" stroke-dasharray="4,3"/>
  <path d="M312 112 L312 132" fill="none" stroke="#4a9e9e" stroke-width="1.4" stroke-dasharray="4,3"/>
  <rect x="244" y="90" width="136" height="22" rx="11" fill="#e8f4f4" stroke="#4a9e9e" stroke-width="1"/>
  <text x="312" y="105" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="600" fill="#4a9e9e">ℓ_Reg (repr. regularizer)</text>
  <line x1="356" y1="151" x2="388" y2="151" stroke="#4a9e9e" stroke-width="1.4" stroke-opacity="0.6" marker-end="url(#arr-rrl)"/>
  <rect x="388" y="132" width="72" height="38" rx="8" fill="#f0fafa" stroke="#4a9e9e" stroke-width="1.2"/>
  <text x="424" y="151" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="#2d7a7a">classifier g</text>
  <text x="424" y="163" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#2d7a7a">→ ŷ</text>
  <rect x="478" y="28" width="90" height="24" rx="6" fill="#4a9e9e" fill-opacity="0.1" stroke="#4a9e9e" stroke-width="0.8"/>
  <text x="523" y="44" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#4a9e9e">ℓ_Info (SSL loss)</text>
  <rect x="478" y="145" width="90" height="24" rx="6" fill="#4a9e9e" fill-opacity="0.1" stroke="#4a9e9e" stroke-width="0.8"/>
  <text x="523" y="161" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#4a9e9e">ℓ_CE (SL loss)</text>
  <line x1="356" y1="47" x2="478" y2="47" stroke="#ccc" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="460" y1="160" x2="478" y2="160" stroke="#ccc" stroke-width="1" stroke-dasharray="3,3"/>
</svg>
<figcaption style="font-size: 13px; color: #999; margin-top: 0.5rem;">Figure 4. RRL dual-path training. Both paths share encoder f. The representation regularizer (dashed bridge) penalizes divergence between SSL feature geometry (tₙ) and SL features (sₙ), preventing noisy supervision from corrupting the encoder.</figcaption>
</figure>

RRL adds a self-supervised path $f \to h$ alongside the supervised path $f \to g$. The full loss is:

$$L((x_n, \tilde{y}_n); f, g, h) = \underbrace{\ell(g(f(x_n)), \tilde{y}_n)}_{\text{SL}} + \underbrace{\ell_{\text{Info}}(h(f(x_n)), \mathcal{B})}_{\text{SSL}} + \lambda \underbrace{\ell_{\text{Reg}}(h(f(x_n)), g(f(x_n)), \mathcal{B})}_{\text{Representation Regularizer}}$$

The $\lambda$ parameter controls how strongly the SSL geometry constrains the supervised update. Ablations show performance is relatively stable across a range of $\lambda$ values, with degradation at extremes: very small $\lambda$ reduces to standard co-training, very large $\lambda$ over-constrains the encoder.

The representation regularizer penalizes disagreement between the pairwise distance structure of SSL and SL features:

$$\ell_{\text{Reg}} = \frac{1}{|\mathcal{B}|-1} \sum_{x_{n'} \in \mathcal{B}, n \neq n'} d(\phi^w(t_n, t_{n'}), \phi^w(s_n, s_{n'}))$$

where $t_n = h(f(x_n))$ are SSL features and $s_n = g(f(x_n))$ are SL features. The regularizer activates primarily when SSL and SL features *disagree*, specifically when noisy supervision is pulling the representation away from the SSL-derived semantic structure. When they agree, the regularizer contributes minimally.

### Theoretical grounding

Under three simplifying assumptions, (1) the network memorizes clean instances, (2) variance of predictions on clean instances goes to zero, (3) SSL features follow Gaussian distributions, the paper derives a clean bound relating SSL feature quality to network robustness:

$$\mathbb{E}_\mathcal{D}[\mathbf{1}(g^*(f^*(X), Y))] = e \cdot \left(\frac{1}{2} - \frac{1}{2 + \Delta(\Sigma, \mu_1, \mu_2)}\right)$$

where $\Delta(\Sigma, \mu_1, \mu_2) = 8 \cdot \text{tr}(\Sigma) / ||\mu_1 - \mu_2||^2$ measures class separability of the SSL features. Better SSL representations produce more separated distributions and lower expected error.

The Gaussian assumption (3) is the most load-bearing and the least obviously satisfied. SSL features in practice are not Gaussian - they tend to be concentrated on manifolds, often with heavy tails. For vision features this may be a reasonable approximation; for graph or tabular data, it is much less clear. The theorem is useful for building intuition but should be treated cautiously as a practical guarantee.

### What RRL gets right, and what it doesn't address

The regularizer is empirically effective and theoretically motivated. It complements any base loss function: adding it to CE, GCE, or Peer Loss consistently improves performance, especially at high noise rates. The choice of distance measure for $\phi^w$ makes little difference, which is a good sign for robustness.

What it doesn't address: the method assumes SSL features are genuinely useful as a reference signal, which requires that the SSL objective produces discriminative representations for your downstream classes. Under severe class imbalance, contrastive learning tends to be dominated by majority class geometry. A noisy minority class may not cluster meaningfully under any standard SSL objective, making the regularizer's reference signal unreliable precisely where the noise is most damaging.

---

## Supplementary: Where the Standard Assumptions Break Down

### Graphs with noisy labels

*Robust Training of GNNs via Noise Governance* (RTGNN, Qian et al., WSDM 2023) extends the sample selection approach to graphs, using graph augmentation and pseudo-labels alongside consistency regularization. The core challenge is that graphs introduce a new source of noise beyond label corruption: structural noise in the edges themselves. A noisy edge aggregates information from the wrong neighborhood, compounding the label noise problem.

More problematically for applied settings: anomalous nodes also tend to have high loss, making them indistinguishable from noisy nodes by loss alone. RTGNN conflates noise and anomalies under class imbalance. In fraud detection, where fraud nodes are simultaneously rare, anomalous, and subject to label noise from disputes and chargebacks, this is not a minor edge case: it's the default condition.

### Noisy labels in long-tailed distributions

*Identifying Hard Noise in Long-Tailed Sample Distribution* (Yi et al., ECCV 2022) addresses a compounded challenge: datasets that are simultaneously long-tailed and noisy. In long-tailed settings, tail-class examples are rare enough that the loss distributions for clean and noisy examples overlap significantly, and the small-loss heuristic breaks down. A noisy head-class example may have lower loss than a clean tail-class example simply because the head class is better represented in training.

The practical implication: most of the sample selection literature has been developed and evaluated on balanced datasets with synthetic noise. The benchmark performance numbers do not transfer cleanly to real-world settings where imbalance and noise co-occur. This gap between benchmark evaluation and deployment reality is, I think, the most underacknowledged problem in this literature.

---

## Recent Developments (2022–2025)

The post above reflects the literature as of early 2024. A few papers since then are worth noting, particularly those that address the scope limitations discussed throughout.

**UNICON** (Karim et al., CVPR 2022) extends DivideMix by combining contrastive learning with a uniform selection procedure that avoids the class-imbalance failures of purely GMM-based separation. It treats the GMM as a soft prior rather than a hard gate, which helps in regimes where clean and noisy loss distributions overlap significantly.

**SED** (Li et al., NeurIPS 2023) takes a semantic embedding distillation approach: use a teacher network trained on pseudo-clean subsets to distill noise-robust representations into the student. The key contribution is a per-sample distillation weight based on semantic consistency - examples whose semantics disagree between teacher and student are down-weighted. This partially addresses the problem of noisy minority classes, since semantic consistency is a less biased signal than loss magnitude.

**DISC** (Li et al., ICLR 2024) is notable for explicitly targeting the distribution mismatch between noisy training sets and clean test distributions. It uses a distribution-aware sample selection criterion that accounts for class frequency, making the selection less biased toward majority classes under imbalance. This is one of the first papers to directly attack the imbalance-noise interaction that I flag as an open question below.

**PNP** (Zhang et al., ICLR 2024) - *Progressive Noise Purification* - frames noise correction as a progressive refinement problem: start with a rough label correction and iteratively improve it using the model's own evolving representations. The progressive framing helps avoid the circularity problem with the small-loss heuristic by anchoring early-round corrections to a cleaner signal.

**Benchmark re-evaluation** (2024): Several teams have re-evaluated standard methods on CIFAR-10N, CIFAR-100N, and WebVision with controlled compute budgets, finding that the benchmark rankings shift considerably from prior work. DivideMix and its variants remain competitive, but simpler methods (early stopping + temperature scaling) close the gap significantly when hyperparameters are properly tuned - a reminder that the reported improvements in the literature are often against undertuned baselines.

---

## Open Questions and Deliberations

The field has made real progress. Sample selection methods reliably outperform robust losses on standard benchmarks; SSL-based regularization adds a principled handle on representation quality; the memorization dynamics of deep networks are now reasonably well characterized for the image classification setting.

But several things remain unsettled, and I think they're worth sitting with.

**The small-loss heuristic has a circularity problem.** We use training loss to identify which examples have correct labels. But training loss reflects the current model's beliefs, which were themselves shaped by noisy labels in earlier iterations. At high noise rates, the model's loss surface may be so corrupted that small-loss examples are simply the ones the model has already memorized, not the ones with correct labels. The heuristic works empirically at moderate noise, but the theoretical justification for why it should work at all is weaker than it appears at first.

**Most methods treat noise as static.** The transition matrix framework assumes noise rates are fixed properties of the labeling process. In reality, noise can be dynamic: in online labeling systems, annotator behavior shifts over time; in fraud detection, labels can be revised months after the fact as chargebacks resolve; in medical imaging, the same image may be labeled differently as clinical understanding evolves. Methods that estimate and correct for a static noise matrix are doing something useful, but they're solving a simplified version of the problem.

**The SSL robustness assumption may not transfer across modalities.** Observation #4 is grounded in image classification experiments where contrastive augmentations are well-motivated (random crops, color jitter, flips). For tabular data, the right augmentation strategy is genuinely unclear: corrupting feature values, dropping columns, or adding Gaussian noise all have different semantic implications depending on the domain. For financial transaction data in particular, the "similar instance" concept that contrastive learning relies on may not have a natural definition. If the SSL features aren't meaningfully clusterable for your domain, the entire family of SSL-based regularization methods loses its foundation.

**Class imbalance and label noise interact in ways the literature has barely begun to address.** The CIKM 2025 work on equitable coreset selection that I've been involved in is one attempt to take imbalance seriously in a data selection context. But the deeper question (how do you identify and correct noisy labels in a class where you have only a few hundred examples, each of which is critical?) doesn't have a satisfying answer yet. Small-loss selection actively harms minority classes in this regime. Robust losses apply uniform pressure regardless of class frequency. SSL representations are biased toward majority geometry. Every standard approach has a structural disadvantage in exactly the setting where getting it right matters most.

**We don't have good evaluation frameworks for the realistic case.** Benchmarks like CIFAR-10N (real human annotation noise on CIFAR-10) are a step forward from synthetic noise, but they're still balanced classification tasks with moderate noise rates. There is no widely-used benchmark that combines instance-dependent noise, severe class imbalance, and graph-structured data, which is the setting many real applied problems actually live in. Until that benchmark exists, it will be difficult to know whether the methods that win on current evaluations are solving the right problem.

The honest summary: the label noise literature has converged on solutions that work well in the settings it has chosen to study. The settings it has chosen to study are not the hardest ones. The gap between benchmark conditions and deployment reality, in terms of noise type, class distribution, data structure, and label dynamics, remains wide enough that translating research progress into production systems requires significant additional work that rarely gets published. That translation problem is, in my view, where the most interesting open questions live.

---

*This post is based on a Paper Reading Group session at Mastercard AI Garage (February 2024), co-presented with Kamna Meena. The primary paper covered in depth was RRL (ICLR 2023); the literature overview draws from Song et al. (TNNLS 2022) and my own reading of the field.*

---

## Citation

If you find this useful, please cite this post as:

Sahir, Liyana. (Feb 2024). Learning with Label Noise: A Literature Overview. *liyanasahir.in*. https://liyanasahir.in/posts/label-noise/.

```bibtex
@article{sahir2024labelnoise,
  title   = "A Field Guide to Learning with Noisy Labels",
  author  = "Sahir, Liyana",
  journal = "liyanasahir.in",
  year    = "2024",
  month   = "Feb",
  url     = "https://liyanasahir.in/posts/label-noise"
}
```
