## Session transition / dataset fields

| Field | Meaning | Use |
|---|---|---|
| `session_id` | Unique session identifier. | Traceability. |
| `step_id` | Step number inside the session. | Temporal ordering. |
| `state_t` | Structured state before the action. | Conceptual model input. |
| `x_t` | Numeric vector derived from `state_t`. | Actual model input. |
| `action_t` | Executed or labelled semantic action. | Training label / decision output. |
| `command_t` | Concrete command generated from the action. | Reproducibility and debugging. |
| `result_t` | Raw execution result. | Debugging and parser input. |
| `events_t` | Events generated from the result. | KB update. |
| `state_t+1` | Structured state after applying events. | Progress evaluation. |