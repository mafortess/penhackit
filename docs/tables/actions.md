## Semantic actions

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 0 | `STOP` | control | Ends the session. |
| 100 | `SELECT_NEXT_HOST` | recon/enum | Selects the next relevant host. |
| 110 | `SELECT_NEXT_SERVICE` | enum/vuln | Selects the next relevant service. |
| 200 | `DISCOVER_HOSTS` | recon | Discovers active hosts inside the authorized scope. |
| 210 | `PORTSCAN_TOP_TCP` | recon | Scans common TCP ports. |
| 220 | `PORTSCAN_FULL_TCP` | recon | Scans all TCP ports. |
| 230 | `SERVICE_DETECT` | enum | Detects services, products and versions. |
| 300 | `ENUM_HTTP` | enum | Enumerates HTTP services. |
| 310 | `ENUM_SMB` | enum | Enumerates SMB services. |
| 320 | `ENUM_FTP` | enum | Enumerates FTP services. |
| 400 | `CHECK_VERSION_VULNS` | vuln | Searches candidate vulnerabilities based on service versions. |
| 500 | `TRY_DEFAULT_CREDS` | exploit | Tests default or weak credentials. |
| 600 | `OPEN_SESSION_ATTEMPT` | exploit | Attempts to obtain a session through a vulnerable service. |