# Name:
S02 - Multi Host Same Network

# Description:
Scenario with multiple Metasploitable2 instances in the same network.

# Topology:

Network: 10.6.6.0/24
10.6.6.10: meta_full
10.6.6.20: meta_smb
10.6.6.30: meta_distcc
10.6.6.40: meta_web
10.6.6.50: meta_creds

# Possible targets:

10.6.6.10
10.6.6.20
10.6.6.30
10.6.6.40
10.6.6.50
10.6.6.0/24

# Goal:
obtain_session

# Deployment:
docker compose up -d --build

# Host discovery:
sudo arp-scan --interface=<bridge_interface> 10.6.6.0/24

# Validation:
docker ps
nmap -sV -p- 10.6.6.10
nmap -sV -p- 10.6.6.20
nmap -sV -p- 10.6.6.30
nmap -sV -p- 10.6.6.40
nmap -sV -p- 10.6.6.50

# Use:
Test multiple hosts in one network and different service profiles.
