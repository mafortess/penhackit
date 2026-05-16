# Name:
meta_full

# Description:
Default Metasploitable2 profile using the original image and its original /bin/services.sh.

# Purpose:
Provide a full vulnerable host with the widest available attack surface.

# Base image:
tleemcjr/metasploitable2

# Service strategy:
This profile should use the original /bin/services.sh from the base image.

# Recommended compose command:
docker compose service should run:

command: /bin/bash -lc "/bin/services.sh >/tmp/services.log 2>&1 || true; tail -f /dev/null"

# Expected services:

ftp
ssh
telnet
smtp
http
samba / smb
rexec / rlogin / remote shell
java-rmi
mysql
postgresql
vnc
x11
irc
ajp13
tomcat

# Expected ports:

21/tcp
22/tcp
23/tcp
25/tcp
80/tcp
139/tcp
445/tcp
512/tcp
513/tcp
514/tcp
1099/tcp
1524/tcp
2121/tcp
3306/tcp
5432/tcp
5900/tcp
6000/tcp
6667/tcp
8009/tcp
8180/tcp

# Validation:
docker exec -it <container> bash -lc "cat /tmp/services.log"
docker exec -it <container> bash -lc "netstat -tulpn"
nmap -sV -p- <ip>

# Notes:
This profile may take time to stabilize after deployment. Wait before scanning.

The agent must not know the exposed services beforehand. It should discover them through scanning and enumeration.
