# penhackit
Trainable Intelligent Agent for Automated Penetration Testing

**Master’s Thesis (TFM)**

## Overview

This repository contains the codebase and experimental framework developed for the **Master’s Thesis (TFM)** focused on the design and implementation of a **trainable intelligent agent for cybersecurity tasks**, with a particular emphasis on **penetration testing (pentesting) automation**.

The core objective of the project is to explore how an intelligent agent can **learn operational behavior from human experts** and progressively evolve towards **autonomous decision-making** in controlled and ethical cybersecurity environments.

Rather than relying solely on static scripts or fully generative models, this work investigates a **modular, interpretable, and extensible agent architecture** that separates perception, decision, and execution.

---

## Research Motivation

Pentesting workflows are highly procedural but context-dependent. While experienced analysts follow recognizable patterns, these are rarely captured in a structured or reusable form.

This thesis addresses the following research questions:

- Can an agent **learn pentesting workflows from human behavior** instead of handcrafted rules?
- How should **environment state** be represented to enable reliable decision-making?
- Can **Imitation Learning** serve as a solid foundation before introducing Reinforcement Learning?
- How can modern representation models (e.g., LLM-based encoders) be integrated **without turning the agent into a black box**?

---

## High-Level Architecture

The system is designed as a **trainable decision loop** composed of independent but interoperable modules:

1. **Environment Interface**  
   Controlled CLI-based operating systems and security tools.

2. **State Representation Layer**  
   Normalization of raw terminal output and encoding into structured or semantic representations.

3. **Decision Model**  
   Lightweight policy models trained to predict the next action.

4. **Learning Framework**  
   Behavioral Cloning as the baseline, with Reinforcement Learning extensions planned.

5. **Agent Runtime**  
   Observe → encode → decide → act execution loop.

---

## Learning Paradigm

The thesis follows a **progressive learning strategy**:

### Phase 1 – Behavioral Cloning
- Offline supervised learning from expert demonstrations.
- Focus on reproducing valid operational flows.
- Emphasis on safety, stability, and reproducibility.

### Phase 2 – Hybrid Extensions (Planned)
- Behavioral Cloning with Reset (BCR).
- Reinforcement Learning for exploration and optimization.
- Feedback-driven policy refinement.

---

## Scope and Ethical Considerations

This project is developed strictly within **authorized, controlled, and ethical environments**.

- No real-world unauthorized attacks are performed.
- All data originates from **synthetic or permitted lab sessions**.
- The goal is defensive improvement, automation research, and analyst training support.

---

## Repository Structure (Conceptual)

```text
├── data/
├── preprocessing/
├── training/
├── models/
├── agent/
├── cli/
├── config/
├── logs/
└── docs/
```
The structure may evolve as the thesis progresses.

## Current Status

This repository is actively developed as part of the Master’s Thesis.
- Foundational architecture implemented.
- Initial learning pipelines validated.
- Ongoing work towards real-world pentesting environments.

## Technologies and Concepts

- Python
- Machine Learning & Imitation Learning
- Behavioral Cloning
- Reinforcement Learning (planned)
- Large Language Models (state encoders)
- Cybersecurity & Pentesting methodologies

## Author

Miguel Ángel Fortes Santiago
Master’s Degree in Computer Science and Engineering
University of Málaga

📧 mafortes.it@uma.es

 # Disclaimer

This repository is part of an academic research project.
All experiments are conducted in safe, controlled, and ethical environments.
