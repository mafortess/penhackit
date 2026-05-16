# Name:
S04 - Pivoting

# Description:
Scenario with a public network and an internal network. One Metasploitable2 instance acts as the pivot host.

# Topology:

kali_attacker: 172.19.0.5
public_net: 172.19.0.0/24
m2_pivot: 172.19.0.10 and 172.18.0.2
internal_net: 172.18.0.0/24
172.18.0.20: meta_smb
172.18.0.30: meta_distcc
172.18.0.40: meta_web

# Initial target:

172.19.0.10

# Internal targets:

172.18.0.20
172.18.0.30
172.18.0.40

# Goal:
obtain_session

# Deployment:
docker compose up -d --build

# Initial validation:
docker ps
docker exec -it s04_kali_attacker bash
ip a
ip r
nmap -sV -p- 172.19.0.10

# Use:
Advanced scenario for studying pivoting. In the MVP, it can be used only to obtain a session on the pivot host.
