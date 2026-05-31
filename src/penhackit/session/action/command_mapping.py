import re

def extract_action_id_from_cmd(cmd: str) -> int:
    """
    Extracts the closest semantic action_id from a free-form command.
    Used mainly in observation mode to map human commands to action labels.
    """
    if not cmd:
        return None

    s = cmd.strip().lower()
    s = re.sub(r"\s+", " ", s)

    # Windows (acepta "ipconfig" y "ipconfig /all")
    if s == "ipconfig" or s.startswith("ipconfig "):
        return 1
    if s == "arp" or s.startswith("arp "):
        return 2
    if s == "route" or s.startswith("route "):
        return 3
    # ping -n 1 <ipv4>
    if re.fullmatch(r"ping -n 1 (?:\d{1,3}\.){3}\d{1,3}", s):
        return 4
    # ping -n 1 <ipv4> (y también ping <ipv4> como fallback MVP)
    if re.fullmatch(r"ping (?:\d{1,3}\.){3}\d{1,3}", s):
        return 4
    
    # ============================================================
    # Local attacker context
    # ============================================================

    if s == "hostname":
        return 100
    if s == "ip a":
        return 101
    if s == "ip r":
        return 102
    if s == "ip neigh":
        return 103
    if s == "ss -tulpn":
        return 104
    if re.fullmatch(r"ping -c 1(?: -w \d+)?(?: -w \d+)? (?:\d{1,3}\.){3}\d{1,3}", s):
        return 105
    if s.startswith("traceroute "):
        return 106
    
     # ============================================================
    # Recon / discovery
    # ============================================================

    if s.startswith("nmap -sn "):
        return 200
    if s == "arp-scan --localnet":
        return 201
    if s.startswith("arp-scan "):
        return 202
    if s.startswith("netdiscover -r "):
        return 203
    if s.startswith("fping ") and " -g " in s:
        return 204

    # ============================================================
    # Port scanning
    # ============================================================

    if s.startswith("nmap ") and "--top-ports" in s and "--open" in s and "-su" not in s:
        return 210
    if s.startswith("nmap ") and "-p-" in s and "--open" in s:
        return 211
    if s.startswith("nmap ") and " -f " in s:
        return 212
    if s.startswith("nmap ") and "-su" in s:
        return 213

    # ============================================================
    # Service detection
    # ============================================================

    if s.startswith("nmap ") and "-sv" in s and "-sc" in s:
        return 220
    if s.startswith("nmap ") and "-sv" in s and "--version-light" in s:
        return 221
    if s.startswith("nmap ") and " -a " in s:
        return 222
    if s.startswith("nmap ") and "-sc" in s:
        return 230

    # ============================================================
    # HTTP / HTTPS
    # ============================================================

    if s.startswith("curl -i ") or s.startswith("curl -i --") or s.startswith("curl -k -i "):
        return 300
    if s.startswith("curl -l ") or s.startswith("curl -l --") or "curl -l" in s:
        return 301
    if s.startswith("curl -k -i ") and "https://" in s:
        return 302
    if "robots.txt" in s and s.startswith("curl "):
        return 303
    if s.startswith("gobuster dir "):
        return 310
    if s.startswith("feroxbuster "):
        return 311
    if s.startswith("nikto "):
        return 312
    if s.startswith("whatweb "):
        return 313
    if s.startswith("wafw00f "):
        return 314

    # ============================================================
    # SMB
    # ============================================================

    if s.startswith("smbclient -l ") or s.startswith("smbclient -l//") or s.startswith("smbclient -l //"):
        return 320
    if s.startswith("enum4linux "):
        return 321
    if s.startswith("rpcclient ") and "enumdomusers" in s:
        return 322
    if s.startswith("nmap ") and "smb-os-discovery" in s:
        return 323
    if s.startswith("nmap ") and "smb-protocols" in s:
        return 324

    # ============================================================
    # FTP
    # ============================================================

    if s.startswith("nc ") or s.startswith("netcat "):
        return 330
    if s.startswith("nmap ") and "ftp-anon" in s:
        return 331
    if s.startswith("nmap ") and "ftp-" in s:
        return 332

    # ============================================================
    # SSH
    # ============================================================

    if s.startswith("nmap ") and ("ssh2-enum-algos" in s or "ssh-hostkey" in s):
        return 341

    # ============================================================
    # DNS
    # ============================================================

    if s.startswith("dig ") and "version.bind" in s:
        return 350
    if s.startswith("dig ") and " any" in s:
        return 351
    if s.startswith("dig axfr "):
        return 352

    # ============================================================
    # NFS / RPC
    # ============================================================

    if s.startswith("showmount -e "):
        return 360
    if s.startswith("rpcinfo -p "):
        return 361
    
    # ============================================================
    # Databases / RDP / VNC
    # ============================================================

    if s.startswith("nmap ") and ("nfs-showmount" in s or "nfs-ls" in s or "nfs-statfs" in s):
        return 362
    if s.startswith("nmap ") and "mysql-info" in s:
        return 370
    if s.startswith("nmap ") and "pgsql" in s:
        return 371
    if s.startswith("nmap ") and ("rdp-enum-encryption" in s or "rdp-ntlm-info" in s):
        return 380
    if s.startswith("nmap ") and "vnc-info" in s:
        return 381

    # ============================================================
    # Vulnerability discovery
    # ============================================================

    if s.startswith("searchsploit "):
        return 400
    if s.startswith("nmap ") and "--script vuln" in s:
        return 401
    if s.startswith("nmap ") and "smb-vuln" in s:
        return 410
    if s.startswith("nmap ") and "ssl-enum-ciphers" in s:
        return 412
    if s.startswith("nmap ") and "ftp-vsftpd-backdoor" in s:
        return 413

    # ============================================================
    # Credential attacks / lab auth checks
    # ============================================================

    if s.startswith("hydra ") and "ssh://" in s:
        return 500
    if s.startswith("hydra ") and "ftp://" in s:
        return 501
    if s.startswith("hydra ") and "http-post-form" in s:
        return 502
    
    if s.startswith("sshpass ") and " ssh " in s:
        return 520
    if s.startswith("hydra ") and "telnet://" in s:
        return 521
    if s.startswith("mysql ") and "select version()" in s:
        return 522
    if s.startswith("pgpassword=") and "psql " in s:
        return 523
    if s.startswith("curl ") and "/manager/html" in s and "-u " in s:
        return 524

    # ============================================================
    # Metasploit lab exploitation
    # ============================================================

    if s.startswith("msfconsole ") and "usermap_script" in s:
        return 600
    if s.startswith("msfconsole ") and "vsftpd_234_backdoor" in s:
        return 601
    if s.startswith("msfconsole ") and "distcc_exec" in s:
        return 602
    if s.startswith("msfconsole ") and "tomcat_mgr_upload" in s:
        return 603
    if s.startswith("msfconsole ") and "postgres_payload" in s:
        return 604
    if s.startswith("msfconsole ") and "unreal_ircd_3281_backdoor" in s:
        return 605
    if re.fullmatch(r"nc -nv(?: -w \d+)? (?:\d{1,3}\.){3}\d{1,3} 1524", s):
        return 606
    if s.startswith("msfconsole ") and "rlogin_login" in s:
        return 607
    if s.startswith("msfconsole ") and "java_rmi_server" in s:
        return 608
    # ============================================================
    # Post-exploitation
    # ============================================================

    if s == "whoami":
        return 700
    if s == "uname -a":
        return 701
    if s == "id":
        return 702
    if s == "ip route":
        return 705
    if s == "ss -tulnp":
        return 706
    if s == "netstat -tulnp":
        return 707
    if s == "cat /etc/passwd":
        return 708
    if s == "cat /etc/passwd | grep home":
        return 709
    if s == "ps aux":
        return 710
    if s == "env":
        return 711
    if s == "sudo -l":
        return 712
    if s == "ls -l /etc/sudoers":
        return 713
    if s.startswith("find / -perm -4000"):
        return 714
    if s == "nmap --interactive":
        return 715
    if s.startswith("find . -exec /bin/sh -p"):
        return 716
    if s == "cat /etc/shadow":
        return 717
    if s == "ls -la /etc/ssh":
        return 718
    if s == "cat /etc/crontab":
        return 719
    if s.startswith("msfconsole ") and "post/linux/gather/enum_system" in s:
        return 730
    if s.startswith("msfconsole ") and "post/linux/gather/enum_configs" in s:
        return 731
    if s.startswith("msfconsole ") and "post/linux/gather/enum_network" in s:
        return 732
    if s.startswith("msfconsole ") and "local_exploit_suggester" in s:
        return 733
    if s.startswith("msfconsole ") and "post/multi/manage/autoroute" in s:
        return 760
    if s.startswith("msfconsole ") and "auxiliary/server/socks_proxy" in s:
        return 761
    if s.startswith("proxychains nmap ") and "-st" in s and "-pn" in s:
        return 762
    if s.startswith("msfconsole ") and "portfwd add" in s:
        return 763
    if s.startswith("mysql ") and "-h 127.0.0.1" in s:
        return 764
    if s.startswith("mysqldump ") and "-h 127.0.0.1" in s:
        return 765
    if s.startswith("grep -r \"pass\" -ni /var/www"):
        return 780
    if s.startswith("grep -r \"db\" -ni /var/www"):
        return 781
    if s == "cat /var/www/dvwa/config/config.inc.php":
        return 782
    if s == "cat /etc/phpmyadmin/config-db.php":
        return 783
    if s == "cat /var/www/tikiwiki/db/local.php":
        return 784
    if "tar -cvf /tmp/mysql.tar" in s:
        return 790
    if "tar -cvf /tmp/pgsql.tar" in s:
        return 791
    if s.startswith("download "):
        return 792
    return None