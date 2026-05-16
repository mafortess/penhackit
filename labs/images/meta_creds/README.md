# Name:
meta_creds

# Description:
Metasploitable2-derived image focused on remote access services.

# Purpose:
Test detection of login services and controlled credential-based attacks.

# Main services:

ssh
proftpd
openbsd-inetd
xinetd
rmnologin

# Expected ports:

21/tcp
22/tcp
23/tcp, 512/tcp, 513/tcp or 514/tcp if enabled through inetd/xinetd

# Validation:
docker exec -it <container> bash -lc "cat /tmp/services.log"
docker exec -it <container> bash -lc "netstat -tulpn"
nmap -sV -p- <ip>

# Notes:
For the MVP, use small controlled credential sets instead of large wordlists.
