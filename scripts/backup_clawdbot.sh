#!/bin/bash
# ClawdBot PostgreSQL daily backup script
DATE=$(date +%Y-%m-%d_%H%M)
DIR="/home/ubuntu/backups/clawdbot"
mkdir -p "$DIR"

# pg_dump requires PGPASSWORD in env
export PGPASSWORD=${DB_PASSWORD:-clawdbot_secret}
pg_dump -h ${DB_HOST:-localhost} -p ${DB_PORT:-5432} -U ${DB_USER:-clawdbot} ${DB_NAME:-clawdbot} | gzip > "$DIR/backup_$DATE.sql.gz"

# Keep only last 30 days
find "$DIR" -name "*.sql.gz" -mtime +30 -delete

echo "$(date): backup completado → backup_$DATE.sql.gz" >> /home/ubuntu/backup.log
