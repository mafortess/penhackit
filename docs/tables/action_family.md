## Semantic action families

| ID range | Family | Phase | Description |
|---:|---|---|---|
| 0-99 | Control | control | Session control actions such as stop or no-op. |
| 100-199 | Local attacker context | attacker_context | Collects local network and host information from the attacker environment. |
| 200-299 | Reconnaissance | recon | Discovers hosts, scans ports and detects exposed services. |
| 300-399 | Service enumeration | enum | Performs protocol-specific enumeration for HTTP, SMB, FTP, SSH, DNS, NFS, databases and remote access services. |
| 400-499 | Vulnerability discovery | vuln_lookup | Identifies candidate vulnerabilities using version matching, NSE scripts and service-specific checks. |
| 500-599 | Credential validation | credential_attack | Tests known, weak or default credentials in the authorized lab environment. |
| 600-699 | Exploitation | exploit | Attempts to obtain sessions through vulnerable services or exposed shells. |
| 700-759 | Post-exploitation | post_exploit | Collects evidence from established sessions and searches for privilege escalation vectors. |
| 760-779 | Pivoting | pivoting | Enables access to internal networks through compromised hosts. |
| 780-799 | Evidence collection | credential_access/exfiltration | Searches, archives or downloads relevant evidence for reporting. |