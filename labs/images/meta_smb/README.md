# Name:
meta_smb

# Description:
Metasploitable2-derived image focused on SMB/Samba services.

# Purpose:
Test SMB service detection, enumeration and attack paths.

# Main services:

samba
openbsd-inetd
xinetd
portmap
ssh

# Expected ports:

139/tcp
445/tcp
22/tcp

# Validation:
docker exec -it <container> bash -lc "cat /tmp/services.log"
docker exec -it <container> bash -lc "netstat -tulpn"
nmap -sV -p- <ip>

# Notes:
The agent must not know this profile during execution. Services must be discovered through scanning and enumeration.
