#!/bin/sh

LOG_FILE="/tmp/services.log"

start_service() {
    SERVICE_NAME="$1"

    echo "[+] Starting ${SERVICE_NAME}..." | tee -a "$LOG_FILE"

    service "$SERVICE_NAME" start >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "[OK] ${SERVICE_NAME} started" | tee -a "$LOG_FILE"
    else
        echo "[WARN] ${SERVICE_NAME} failed to start" | tee -a "$LOG_FILE"
    fi
}

echo "[+] Starting Metasploitable2 DISTCC profile..." | tee "$LOG_FILE"

start_service networking
start_service openbsd-inetd
start_service xinetd

start_service distcc
start_service ssh
start_service sysklogd

echo "[+] Running rc.local..." | tee -a "$LOG_FILE"
/etc/init.d/rc.local start >> "$LOG_FILE" 2>&1 || true

echo "[+] DISTCC profile startup finished." | tee -a "$LOG_FILE"
echo "[+] Keeping container alive." | tee -a "$LOG_FILE"