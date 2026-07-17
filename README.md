# penhackit
Trainable Intelligent Agent for Automated Penetration Testing

**Master’s Thesis (TFM)**

## Overview

This repository contains the codebase and experimental framework developed for the **Master’s Thesis (TFM)** focused on the design and implementation of a **trainable intelligent agent for cybersecurity tasks**, with a particular emphasis on **penetration testing (pentesting) automation**.

PenHackIt is a research prototype for the assistance and partial automation of penetration-testing sessions.

The core objective of the project is to explore how an intelligent agent can **learn operational behavior from human experts** and progressively evolve towards **autonomous decision-making** in controlled and ethical cybersecurity environments.

The system registers the evolution of a pentesting session, structures the knowledge obtained from security tools, represents the current decision context, and uses supervised machine-learning models to select the next action from a controlled action catalogue.

The project focuses on traceability, reproducibility and controlled execution. It does not generate arbitrary terminal commands directly from a language model. Instead, decisions, command construction, execution and result interpretation are handled by separate components.

The prototype also includes an offline training pipeline and an automatic reporting module based on the information stored during each session.

---

## Research Motivation

Pentesting workflows are highly procedural but context-dependent. While experienced analysts follow recognizable patterns, these are rarely captured in a structured or reusable form.

This thesis addresses the following research questions:

- Can an agent **learn pentesting workflows from human behavior** instead of handcrafted rules?
- How should **environment state** be represented to enable reliable decision-making?
- Can **Imitation Learning** serve as a solid foundation before introducing Reinforcement Learning?
- How can modern representation models (e.g., LLM-based encoders) be integrated **without turning the agent into a black box**?

## Objectives

The main objective of this project is to design and implement a trainable intelligent agent capable of:

- Recording the complete evolution of a pentesting session
- Structuring the knowledge discovered during execution
- Representing the current decision context as a machine-learning state
- Learning state-action patterns from previously recorded sessions
- Selecting actions during controlled autonomous executions
- Generating technical reports from the final session knowledge

The project investigates whether pentesting sessions can be represented as reusable sequences of states and actions, and whether supervised learning can model part of the operational decision process.

---

## System Architecture

PenHackIt is organised around three main modules:

### 1. Session Core

The session core implements the operational loop of the agent.

Its main responsibilities include:

- Maintaining a persistent Knowledge Base
- Selecting the current operational focus
- Building the decision state
- Selecting an action
- Constructing and executing commands
- Parsing heterogeneous tool output
- Generating normalised events
- Updating the Knowledge Base
- Evaluating progress and stopping conditions

The main execution cycle is:

```text
Knowledge Base
      ↓
Focus selection
      ↓
State construction
      ↓
Action selection
      ↓
Command building
      ↓
Tool execution
      ↓
Output parsing
      ↓
Event generation
      ↓
Knowledge Base update
```

### 2. Training Pipeline

Recorded sessions are processed offline to generate supervised learning datasets.

For each valid decision step, the system extracts:

- The state available before the decision
- The selected action identifier
- Session and transition metadata
- Execution results for traceability and evaluation

The resulting state-action samples are stored in JSONL format.

Semantic states are converted into fixed-length numerical vectors before training. The same feature ordering and mappings are used during training and inference.

The implemented classifiers are:

- Decision Tree
- Random Forest
- CatBoost

The models solve a multiclass classification problem in which the target is the identifier of the next action.


### 3. Reporting Pipeline

After a session finishes, PenHackIt can generate a structured technical report from its final Knowledge Base.

```text
Final Knowledge Base
        ↓
Compact Knowledge Base
        ↓
Report template
        ↓
Section-specific prompts
        ↓
Generation backend
        ↓
Markdown / PDF report
```

Supported reporting backends include:

- Deterministic baseline
- Ollama
- Transformers

Language models are used only for report writing. They do not select actions, execute tools or modify the session Knowledge Base.

---

## Knowledge Representation

### Knowledge Base

The Knowledge Base stores the complete accumulated knowledge of a session, including:

- Networks
- Hosts
- Ports
- Services
- Vulnerabilities
- Findings and evidence
- Credentials
- Opened sessions
- Attempts and execution history
- Coverage and progress information

### Focus

The focus identifies the entity currently being analysed, such as a network, host or service.

It reduces the decision context and allows the agent to operate over one relevant part of the Knowledge Base at a time.

### State

The state is a compact representation of the information required to select the next action.

It may include:

- Session objective
- Current focus
- Known hosts and services
- Coverage indicators
- Previous action and result
- Generated event types
- Progress and control flags

The Knowledge Base represents everything known by the agent, whereas the state contains only the information required for the current decision.

---

## Actions, Builders, Parsers and Events

The agent operates through a discrete catalogue of actions organised by pentesting phase.

Examples include:

- Local inspection
- Network reconnaissance
- Host and service enumeration
- Vulnerability analysis
- Credential testing
- Controlled exploitation
- Post-exploitation actions

The decision model predicts an action identifier rather than an arbitrary shell command.

Each selected action is processed through the following chain:

Action
  ↓
Builder
  ↓
Command
  ↓
Execution
  ↓
Parser
  ↓
Event
  ↓
Knowledge Base update

Builders validate preconditions and resolve command parameters from the Knowledge Base.

Parsers transform heterogeneous tool output into normalised events such as:

HOST_DISCOVERED
PORT_OPEN
SERVICE_DETECTED
CREDENTIAL_VALID
SESSION_OPENED

This separation improves control, traceability and reproducibility.

---

## Operating Modes

The architecture defines three operating modes:

### Observation

The user controls the session while PenHackIt records and structures the execution.

### Suggestion

The system proposes the next action, while the user decides whether to accept it or select an alternative.

### Autonomous

A previously trained model selects actions from the controlled catalogue and the agent executes the operational loop automatically.

The experimental validation of the thesis focuses primarily on autonomous execution.

---

## Experimental Evaluation

The system was evaluated in authorised and controlled laboratory environments.

Two complementary evaluation approaches were used:

### Offline Evaluation

The trained classifiers were evaluated using held-out state-action samples.

The analysis included:

- Accuracy
- Macro and weighted precision
- Macro and weighted recall
- Macro and weighted F1-score
- Confusion matrices
- Training time
- Inference time

### Online Evaluation

The selected model was integrated into the complete agent loop and evaluated in reproducible vulnerable laboratory scenarios.

The online evaluation measured:

- Goal completion
- Number of execution steps
- Progress during the session
- Repeated actions
- Tool errors and timeouts
- Execution time
- Knowledge Base growth
- Opened sessions and generated findings

The experiments validate the feasibility of the architecture in controlled scenarios, but they do not imply systematic success or direct generalisation to arbitrary real-world environments.



---

## Repository Structure

```text
penhackit/
├── config/
│   └── settings.json
├── data/
│   ├── datasets/
│   ├── evaluations/
│   └── sessions/
├── models/
│   ├── decision_tree/
│   ├── random_forest/
│   └── catboost/
├── logs/
├── llm_models/
├── src/
└── README.md
```
The workspace directory contains the artefacts generated during execution, including sessions, datasets, trained models, evaluations, reports and logs.

## Current Status

The Master's Thesis prototype has been implemented and evaluated.

The current version includes:

Persistent session management
Structured Knowledge Base
Focus and state construction
Discrete action catalogue
Command builders
Tool execution and parsing
Normalised event generation
Offline dataset extraction
State vectorisation
Decision Tree, Random Forest and CatBoost training
Offline and online evaluation
Automatic Markdown and PDF reporting
Local reporting backends

Future work includes extending the action catalogue, parsers, scenarios and state representation, as well as studying active learning, reinforcement learning and hybrid decision systems.

## Technologies and Concepts

- Python
- Machine Learning & Imitation Learning
- Behavioral Cloning
- Decision Trees
- Random Forest
- CatBoost
- JSON / JSONL
- Ollama
- Hugging Face Transformers
- Docker
- Kali Linux
- Pentesting tools and methodologies
- Cybersecurity & Pentesting methodologies

## Ethical Use

PenHackIt has been developed exclusively for academic research in authorised and controlled environments.

Do not use this software against systems without explicit permission
The included experiments target intentionally vulnerable laboratory environments
The project is intended for defensive research, reproducible experimentation and security training
The author assumes no responsibility for unauthorised or illegal use

## Author

Miguel Ángel Fortes Santiago
Master’s Degree in Computer Science and Engineering
University of Málaga

📧 mafortes.it@uma.es

 # Disclaimer

This repository is part of an academic research project.
All experiments are conducted in safe, controlled, and ethical environments.
