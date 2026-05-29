---
title: "Stochastic parrots"
date: 2026-05-28
lastTended: 2026-05-29
stage: growing
topics:
  - Ethical AI
description: "On the 'stochastic parrots' paper, coherence, and what we project onto language models."
---

The phrase comes from the 2021 paper by Bender, Gebru, McMillan-Major, and Shmitchell: *On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?* The core claim: language models produce text by stitching together statistical patterns from training data, without any understanding of meaning. They are "stochastic parrots."

A phrase from the paper I have stuck on my office desk:

> "Coherence is in the eye of the beholder."

We read coherence *into* the output because we are meaning-making creatures. The model doesn't understand; *we* understand, and then attribute that understanding back to the model. The illusion is on our side, not theirs.

This framing is useful as a corrective, it stops us from anthropomorphising too quickly. But is it the full picture? If a system produces outputs that are indistinguishable from understanding in every measurable context, what's the status of the claim that it "doesn't really understand"? At what point does the distinction stop being useful?

The paper's publication led to the firing of Timnit Gebru and later Margaret Mitchell from Google, making it as much a story about institutional power and who gets to ask critical questions as it is about language models. Margaret Mitchell, being asked to not put her name on the paper, decided to sign herself as Shmargaret Shmitchell.

The paper argues that making large language models bigger is expensive in many ways: training them emits huge amounts of carbon, a tiny BLEU improvement can cost hundreds of thousands of dollars in compute, models like GPT-3 use millions of litres of water, and the environmental burden falls unevenly on vulnerable communities, while the rush to gather ever-larger datasets creates “documentation debt” because we don’t know what is in the data, what biases it carries, or whose voices are missing.


{{< todo >}}How much worse has it gotten since 2021, along all the stated axes? The paper was written before GPT-4, before the current scaling race. Integrate learnings from Empire of AI and other publications.{{< /todo >}}


