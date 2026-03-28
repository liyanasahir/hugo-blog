---
title: "Research"
---

## Research interests

My research sits at the intersection of representation learning and data-centric ML, with a thread running through vision, graphs, and tabular data. I care about what happens when the data itself is imperfect: noisy labels, severe class imbalance, scarce supervision, or distributions that shift in ways that are hard to anticipate.

Some of this is theoretical curiosity: what a model memorizes versus generalizes, how neighborhood structure shapes representations, how learned embeddings behave when the underlying distribution is adversarial or long-tailed. But much of it is grounded in working directly with large-scale financial transaction data, where class imbalance is not a benchmark setting but a structural reality, where fraud patterns evolve adversarially, and where a model's failure modes have real consequences. That operational experience shapes how I think about research problems. I'm drawn to methods that are principled enough to publish and robust enough to actually deploy.

Lately I've been thinking about tabular representation learning: how SSL objectives developed for vision and language translate (or don't) to structured, heterogeneous tabular data, and what the right inductive biases look like for this setting.


## Publications

> **Towards Equitable Coreset Selection: Addressing Challenges Under Class Imbalance**
> L Sahir Kallooriyakath, AN Reddy, BS Achary, A Sharma, K Shah, S Gupta, S Asthana
> *CIKM 2025* · [PDF](https://dl.acm.org/doi/pdf/10.1145/3746252.3760971)

Selecting which data to train on is as important as selecting how to train. This work looks at when and why standard coreset selection methods fail under class imbalance, and proposes a more equitable approach without sacrificing efficiency.

> **AMEND: Adaptive Margin and Expanded Neighborhood for Efficient Generalized Category Discovery**
> A Banerjee, LS Kallooriyakath, S Biswas
> *WACV 2024* · [PDF](https://openaccess.thecvf.com/content/WACV2024/papers/Banerjee_AMEND_Adaptive_Margin_and_Expanded_Neighborhood_for_Efficient_Generalized_Category_WACV_2024_paper.pdf)

> **AdaPrompt: Prompt Tuning with Adaptive Neighbours for Generalized Category Discovery**
> LS Kallooriyakath, A Banerjee, S Biswas
> *ICIP 2024* · [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10648034)

Both from my masters at IISc, advised by [Prof. Soma Biswas](https://scholar.google.com/citations?user=...). These tackle generalized category discovery (GCD): the problem of learning representations that simultaneously recognize known classes and discover novel ones from unlabeled data. The open-world setting forces you to think carefully about what good representations actually mean when the label space itself is incomplete.

> **Study of Topology Bias in GNN-based Knowledge Graph Algorithms**
> A Surisetty, A Malhotra, D Chaurasiya, S Modak, S Yerramsetty, A Singh, L Sahir, E Abdel-Raheem
> *ICDMW 2023* · [PDF](https://www.researchgate.net/profile/Aakarsh-Malhotra/publication/377661475_Study_of_Topology_Bias_in_GNN-based_Knowledge_Graphs_Algorithms/links/65b1d5646c7ad06ab427a713/Study-of-Topology-Bias-in-GNN-based-Knowledge-Graphs-Algorithms.pdf)

See [Google Scholar](https://scholar.google.com/citations?user=6SdmbwQAAAAJ) for a full list of publications.


## Talks

**The Trustworthiness of AI**
GDG DevFest Kozhikode, December 2024
On what trustworthy AI means in practice: reliability, interpretability, and the gap between benchmark performance and real-world deployment.

**Women in AI: Initiatives and Impact**
AI Impact Summit, 2026
Representing Mastercard AI Garage's work on building more inclusive AI research communities.

**Career Journeys in AI** *(panel moderator)*
WiDS India, 2024
Moderated a panel on career paths in AI, with a focus on non-linear journeys and underrepresented voices in the field.


## Community

**Mastercard AI Garage** — I co-organize the Paper Reading Group which hosts weekly sessions to discuss the latest as well as the seminal works in AI/ML. I've also co-organized the ML Symposium 2025, and the Product Innovation Championship 2026. Active member of Women@AIG.

**Older work** — Founding member of SHE GCEK; previously with WITI India.
