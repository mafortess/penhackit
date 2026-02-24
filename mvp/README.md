# MVP 
0) Propósito del MVP
- Validar un flujo end-to-end de “agente”:
- CLI → sesión → ejecución de comandos → parse → eventos → KB → logs/dataset → entrenamiento → reporte (MD/PDF)
- Soportar 3 modos de sesión y 3 backends de reporte (baseline/ollama/transformers).

## 1) CLI (punto de entrada)
- Banner aleatorio al inicio de cada loop.
- Menú principal:
    Run session
    Train models
    Generate report
    Exit

## 2) Run session (3 modos)
### 2.1 Session bootstrap (común)
Crea:
- mvp/sessions/session_<session_id>/
- session_config.json (metadatos mínimos)
- session_context.json (modo, goal_type, target, max_steps)
- kb.json (KB inicial con estructura fija)

KB inicial incluye:
- Memoria “pentest”: hosts/services/findings/notes
- Memoria “entorno”: net{interfaces, ipv4, default_gw, arp_neighbors, routes}
- Runtime mínimo: step_idx, last_action_*, last_rc, last_event_type

### 2.2 Loop de sesión (pasos)
En cada t:
- state = build_state(kb, session_context) (vector tabular fijo por contadores/flags)
- Se decide una acción / comando según modo
- Se ejecuta comando (o se omite)
- Se parsea stdout/stderr → lista de events
- Se actualiza KB con eventos (merge + dedup)
- Se guardan artefactos y logs

### 2.3 Modo Autonomous
- El sistema decide la acción sin intervención del usuario.
- Decisor configurable (en el código actual):
    - rules (heurística determinista)
    - model (sklearn cargado de mvp/models/...)
    - scripted (placeholder, usa t)
- Acción → command_builder(action_id, kb) (rellena placeholders como {ip})
- Ejecuta comando automáticamente.

### 2.4 Modo Suggestion
- Igual que Autonomous para inferencia (rules/model/scripted)
Diferencia:
- El sistema sugiere (acción + comando)
- Usuario:
    - Enter → acepta sugerencia
    - escribe comando → override
    - 0 → stop
- Registra en KB:
    - última sugerencia, comando sugerido, si se aceptó

### 2.5 Modo Observation
- El humano conduce.
- Entrada por paso:
    - action_id numérico (predefinido)
    - o comando libre
    - o 0 stop

- Si comando libre:
    - extract_action_id_from_cmd(cmd) intenta mapear a acción conocida
        - si match → se loggea dataset BC
        - si no match → se loggea como FREEFORM (no entra al dataset BC)

## 3) Catálogo de acciones y ejecución real
- ACTIONS: mapping action_id -> (name, cmd_template)
    - Ejecuta comandos reales vía subprocess.run(shell=True, timeout=30)
    - En MVP actual el foco es “recon local” (Windows-centric)

## 4) Parsing determinista → eventos
Convierte outputs en eventos normalizados:
- INSPECT_IPCONFIG → NET_INFO
    - ipv4, gateway, interfaces (heurístico)
- INSPECT_ARP → ARP_TABLE
- vecinos ARP (ip/mac/type) + fallback por IP
- errores → COMMAND_ERROR
- default → NO_EVENT

## 5) KB (memoria) y merge
update_kb(kb, events):
- Dedup de IPv4/GW
- Interfaces dedup por (name, ipv4)
- ARP dedup por IP
- Refleja ARP como hosts (source=arp)
- Errores y eventos desconocidos → notes

## 6) Trazabilidad / logging por sesión
Archivos por sesión:
- kb.json (estado persistente de la sesión)
- steps.jsonl
    1 línea META + líneas por paso (state, action_id, command, t)
- command_outputs.jsonl
    1 línea META + líneas CMD (cmd, rc, stdout, stderr)
- dataset.jsonl (solo observation cuando hay acción identificable)
    rows: {t, state, action_id}
- dataset_freeform.jsonl
    comandos libres sin mapping (para análisis/expansión futura)

## 7) Train models (BC offline)
Flujo interactivo:
- Selección de dataset dir (root o subcarpetas con dataset.jsonl)
- Selección de modelo:
    Logistic Regression / Decision Tree / Random Forest / MLP
- Vectorización:
    Keys union de state → feature_names
    X numérico + y=action_id
- Split train/test (stratify si aplica)

- Output:
    mvp/models/<dataset_name>/<model_key>[_i]/model.joblib
    metrics.json con:
    feature_names, accuracy, confusion_matrix, classification_report

## 8) Generate report (MD + PDF)
### 8.1 Inputs
- Carga kb.json de una sesión seleccionada (ahora hardcodeada en el código)

### 8.2 Report MD
- Cabecera fija + metadatos
- Figures:
    figures/counts.png
    figures/hosts.png

- Secciones fijas (REPORT_SECTIONS)

- 3 backends:
    - Baseline: texto determinista desde kb_compact (no inventa)
    - Ollama:
        - lista modelos (/api/tags o ollama list)
        - genera sección a sección con streaming
    - Transformers local:
        - selecciona modelo local bajo mvp/llm_models/<id>/config.json
        - selecciona device (cpu/cuda)
        - carga con cache (_HF_CACHE) + streaming con TextIteratorStreamer

- Sanitización:
    - quita fences ``` y headings repetidos en la respuesta LLM

- Evita sobrescribir: next_available_path(report.md -> report_1.md ...)

### 8.3 Export PDF
- md_to_pdf_simple:
    Render muy básico (títulos/bullets por heurística)
    ReportLab, recorte lineal (MVP)

## 9) Qué valida “de verdad” el MVP
- Ejecución real de comandos + captura de outputs
- Parsing determinista → KB actualizable
- Dataset BC real (observation) → entrenamiento sklearn → inferencia (autonomous/model)
- Pipeline de reporte reproducible con:
    - baseline
    - LLM local (Ollama/HF) sección a sección + figuras