# Name:
S01 - Single Host

# Description:
Scenario with a single Metasploitable2 instance in one network.

# Topology:

Network: 10.6.6.0/24
Host: 10.6.6.10
Profile: original meta_full

# Recommended target:
10.6.6.10

# Goal:
obtain_session

# Deployment:
docker compose up -d

# Validation:
docker ps
docker exec -it s01_meta_full bash -lc "cat /tmp/services.log"
docker exec -it s01_meta_full bash -lc "netstat -tulpn"
nmap -sV -p- 10.6.6.10

# Use:
Validate the basic agent loop against one target host.
