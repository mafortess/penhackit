## State variables

| Variable | Type | Meaning | Source |
|---|---|---|---|
| `hosts_count` | int | Number of discovered hosts. | `kb.hosts` |
| `alive_hosts_count` | int | Number of hosts marked as alive. | `kb.hosts[*].status` |
| `open_ports_count` | int | Total number of open ports. | `kb.hosts[*].services` / `kb.services` |
| `services_count` | int | Number of detected services. | `kb.services` |
| `http_services_count` | int | Number of HTTP/HTTPS services. | `kb.services` |
| `smb_services_count` | int | Number of SMB services. | `kb.services` |
| `ftp_services_count` | int | Number of FTP services. | `kb.services` |
| `findings_count` | int | Number of relevant findings. | `kb.findings` |
| `credentials_count` | int | Number of valid or candidate credentials. | `kb.credentials` |
| `has_focus_host` | bool | Indicates whether a host is currently selected. | `kb.focus` |
| `has_focus_service` | bool | Indicates whether a service is currently selected. | `kb.focus` |
| `last_action_id` | int | Last executed action identifier. | Session transition log |
| `last_action_success` | bool | Indicates whether the last action succeeded. | Session transition log |
| `progress_score` | float | Simple score representing recent KB progress. | KB diff before/after action |
| `stagnation_score` | float | Simple score representing lack of recent progress. | KB diff before/after action |