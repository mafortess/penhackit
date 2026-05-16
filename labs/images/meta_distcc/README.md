# Name:
meta_distcc

# Description:
Metasploitable2-derived image focused on the distcc service.

# Purpose:
Test detection and exploitation of distcc.

# Main services:

distcc
xinetd
openbsd-inetd
ssh

# Expected ports:

3632/tcp
22/tcp

# Validation:
docker exec -it <container> bash -lc "cat /tmp/services.log"
docker exec -it <container> bash -lc "netstat -tulpn"
nmap -sV -p- <ip>

# Notes:
This profile is useful for testing a direct path: scan, service detection, exploitation and session acquisition.
