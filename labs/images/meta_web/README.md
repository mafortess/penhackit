Name:
meta_web

Description:
Metasploitable2-derived image focused on web and database services.

Purpose:
Test web enumeration, technology detection and attack paths involving Apache/Tomcat.

Main services:

apache2
tomcat5.5
mysql
postgresql-8.3
ssh

Expected ports:

80/tcp
8180/tcp
3306/tcp
5432/tcp
22/tcp

Validation:
docker exec -it <container> bash -lc "cat /tmp/services.log"
docker exec -it <container> bash -lc "netstat -tulpn"
nmap -sV -p- <ip>
curl -I http://<ip>:80
curl -I http://<ip>:8180

Notes:
This profile may take longer to stabilize because it starts web and database services.
