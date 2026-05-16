#!/bin/sh

LOG_FILE="/tmp/services.log"

start_service() {
    SERVICE_NAME="$1"

    echo ""
    echo "[+] Starting ${SERVICE_NAME}..." | tee -a "$LOG_FILE"
    
    service "$SERVICE_NAME" start >> "$LOG_FILE" 2>&1
    RC=$?    
    
    if [ "$RC" -eq 0 ]; then
        echo "[OK] ${SERVICE_NAME} started" | tee -a "$LOG_FILE"
    else
        echo "[WARN] ${SERVICE_NAME} failed with rc=${RC}" | tee -a "$LOG_FILE"
    fi
}

echo "[+] Starting Metasploitable2 full service profile..." | tee "$LOG_FILE"

start_service apache2
start_service atd
start_service cron
start_service distcc
start_service mysql
start_service mysql-ndb
start_service mysql-ndb-mgm
start_service networking
start_service openbsd-inetd
start_service portmap
start_service postfix
start_service postgresql-8.3
start_service proftpd
start_service rmnologin
start_service rsync
start_service samba
start_service ssh
start_service sysklogd
start_service tomcat5.5
start_service xinetd
start_service x11-common
start_service xserver-xorg-input-wacom
start_service snmpd

echo "[+] Running rc.local..." | tee -a "$LOG_FILE"
/etc/init.d/rc.local start >> "$LOG_FILE" 2>&1 || true

echo "[+] Services startup finished" | tee -a "$LOG_FILE"
echo "[+] Current listening services:" | tee -a "$LOG_FILE"
netstat -tulpn >> "$LOG_FILE" 2>&1 || true