## Knowledge Base structure

| KB section | Stored information | Use inside the agent |
|---|---|---|
| `session` | Session metadata such as identifier, name, mode and timestamps. | Traceability and session management. |
| `scope` | Target, goal and authorized analysis boundaries. | Restricts actions to the allowed scope. |
| `net` | Local attacker-side network information. | Separates local environment data from target data. |
| `networks` | Known networks and discovered network ranges. | Provides network-level context. |
| `hosts` | Discovered target hosts and their attributes. | Main source for host selection and enumeration. |
| `services` | Detected services, ports, protocols, products and versions. | Drives enumeration, vulnerability checks and exploitation attempts. |
| `findings` | Relevant technical findings discovered during the session. | Used for progress tracking and report generation. |
| `credentials` | Valid or candidate credentials found during the assessment. | Supports login attempts and access validation. |
| `history_refs` | References to previous actions, commands and outputs. | Provides traceability without overloading the KB. |
| `focus` | Current active host, service or target entity. | Guides the next decision of the agent. |