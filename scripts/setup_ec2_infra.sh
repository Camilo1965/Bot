#!/bin/bash
# ClawdBot EC2 infrastructure setup script
# Run as root or with sudo on the target Ubuntu/Debian EC2 instance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="clawdbot"

# ── 1. Backup script permissions ──────────────────────────────────────────────
if [ -f "${PROJECT_ROOT}/scripts/backup_clawdbot.sh" ]; then
    chmod +x "${PROJECT_ROOT}/scripts/backup_clawdbot.sh"
    echo "✅ backup_clawdbot.sh made executable"
else
    echo "⚠️  backup_clawdbot.sh not found — skipping"
fi

# ── 2. Crontab for daily backup at 3 AM (avoid duplicates) ────────────────────
CRON_CMD="0 3 * * * ${PROJECT_ROOT}/scripts/backup_clawdbot.sh"
if crontab -l 2>/dev/null | grep -Fq "backup_clawdbot.sh"; then
    echo "ℹ️  Cron entry for backup already exists — skipping"
else
    (crontab -l 2>/dev/null || true; echo "$CRON_CMD") | crontab -
    echo "✅ Cron entry added: daily backup at 3 AM"
fi

# ── 3. Systemd service ────────────────────────────────────────────────────────
SERVICE_SRC="${PROJECT_ROOT}/infra/clawdbot.service"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"

if [ -f "$SERVICE_SRC" ]; then
    cp "$SERVICE_SRC" "$SERVICE_DST"
    # Adjust WorkingDirectory and ExecStart paths if needed
    sed -i "s|/home/ubuntu/Bot-master|${PROJECT_ROOT}|g" "$SERVICE_DST"
    sed -i "s|/home/ubuntu/Bot-master/.venv312/bin/python|${PROJECT_ROOT}/.venv/Scripts/python|g" "$SERVICE_DST"
    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
    echo "✅ systemd service installed and enabled: ${SERVICE_NAME}"
    echo "   Start manually with: sudo systemctl start ${SERVICE_NAME}"
else
    echo "⚠️  ${SERVICE_SRC} not found — skipping systemd setup"
fi

# ── 4. Quick status ───────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────────────────────────"
echo "  ClawdBot EC2 infra setup complete"
echo "────────────────────────────────────────────────────────────"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Service:      ${SERVICE_NAME}"
echo "  Backup cron:  0 3 * * *"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl start   ${SERVICE_NAME}"
echo "    sudo systemctl stop    ${SERVICE_NAME}"
echo "    sudo systemctl status  ${SERVICE_NAME}"
echo "    sudo journalctl -u     ${SERVICE_NAME} -f"
echo "────────────────────────────────────────────────────────────"
