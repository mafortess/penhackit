# Name:
PenHackIt Labs

# Description:
Docker-based vulnerable lab environments used to test and evaluate the PenHackIt agent.

# Purpose:
Provide controlled pentesting scenarios based on Metasploitable2 instances with different topologies and service profiles.

# Directory structure:
labs/images
Contains custom Metasploitable2-derived images.

# labs/scenarios
Contains Docker Compose scenarios/topologies.

# Image profiles:
meta_full
Uses the original tleemcjr/metasploitable2 image and its original /bin/services.sh.

meta_smb
Metasploitable2-derived image focused on SMB/Samba services.

meta_distcc
Metasploitable2-derived image focused on distcc.

meta_web
Metasploitable2-derived image focused on web and database services.

meta_creds
Metasploitable2-derived image focused on remote login and credential-based access.

# Scenarios:
s01_single_host
One network and one Metasploitable2 host.

s02_multi_host_same_network
One network with multiple Metasploitable2 hosts using different service profiles.

s03_multi_network_multi_host
Multiple networks with multiple Metasploitable2 hosts.

s04_pivoting
Public/internal network topology with a Metasploitable2 pivot host.

# Common deployment:
Run from the scenario directory:

docker compose up -d --build

For scenarios using only original images and no custom builds:

# docker compose up -d

# Stop scenario:
docker compose down

# Clean scenario volumes:
docker compose down -v

# Common validation:
docker ps
docker exec -it <container_name> bash -lc "cat /tmp/services.log"
docker exec -it <container_name> bash -lc "netstat -tulpn"
nmap -sV -p- <target_ip>

Notes:
Some Metasploitable2 services may take time to become fully available after container startup.

The agent should not receive information about the internal service profile of a host during execution. It should discover services through scanning, parsing and KB updates.

# The MVP goal for these labs is:
obtain_session

Other goals and more advanced flows can be added later.