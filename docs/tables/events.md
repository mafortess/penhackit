## Normalized events

| Event | Typical source | Effect on KB |
|---|---|---|
| `HOST_DISCOVERED` | `nmap -sn`, `arp-scan` | Adds or updates a host. |
| `PORT_OPEN` | `nmap` | Adds an open port to a host. |
| `SERVICE_DETECTED` | `nmap -sV` | Adds a detected service. |
| `SERVICE_VERSION_DETECTED` | `nmap -sV`, banners | Adds product/version evidence. |
| `WEB_PATH_FOUND` | `gobuster`, `dirb` | Adds a web-related finding. |
| `SMB_SHARE_FOUND` | `enum4linux`, `smbclient` | Adds an SMB-related finding. |
| `CANDIDATE_VULN_FOUND` | `searchsploit`, NSE scripts | Adds a candidate vulnerability. |
| `VALID_CREDENTIAL_FOUND` | `hydra`, manual/scripted login | Adds a valid credential. |
| `SESSION_OPENED` | exploit, Metasploit, manual/scripted access | Marks the session objective as reached. |
| `TOOL_ERROR` | Any external tool | Records an execution error. |
| `TIMEOUT` | Executor | Records a timeout event. |
| `ACTION_COMPLETED` | Session loop | Records the completion of a step. |