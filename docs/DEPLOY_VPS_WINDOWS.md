# Deploy ClawdBot en Windows VPS (sin Docker)

Guía express para correr el bot 24/7 en un Windows VPS con MT5, SQLite (sin Docker) y URL pública del dashboard enviada por Telegram.

## Requisitos del VPS

- Windows Server 2019/2022 o Windows 10/11 Pro.
- 4 GB RAM mínimo, 2 vCPU, 40 GB disco.
- Acceso RDP.
- Internet saliente sin restricciones (para CCXT, Cloudflare, Telegram).

## 1. Preparar el VPS

Conecta por RDP. Instala:

1. **Python 3.12** desde python.org → marcar "Add Python to PATH".
2. **Git for Windows** desde git-scm.com.
3. **MetaTrader 5** desde tu broker → loguéate con tu cuenta real.

## 2. Clonar el repo

```powershell
cd C:\
git clone https://github.com/<tu-usuario>/Bot.git
cd Bot
py -3.12 -m venv .venv312
.venv312\Scripts\activate
pip install -r requirements.txt
pip install -r dashboard\api\requirements.txt
```

## 3. Configurar `.env`

```powershell
copy .env.example .env
notepad .env
```

Descomenta el bloque "Modo VPS zero-dep":

```
DB_BACKEND=sqlite
REDIS_BACKEND=memory
SQLITE_DB_PATH=data/clawdbot.sqlite
```

Configura tus credenciales MT5 (`MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`) y de Telegram:

```
TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=987654321
```

Para conseguir el token y el chat id:
1. Abre Telegram → busca `@BotFather` → `/newbot` → te da el token.
2. Habla con tu bot (manda cualquier mensaje).
3. Visita `https://api.telegram.org/bot<TU_TOKEN>/getUpdates` → copia el `chat.id`.

## 4. Build del dashboard (una vez)

```powershell
cd dashboard\web
npm install
npm run build
cd ..\..
```

## 5. Arrancar el bot

Doble clic en `START_BOT.bat` (o ejecuta desde PowerShell).

Verás:
- `DB: SQLite zero-dep (sin Docker)` → confirma que no usa Docker.
- `Obteniendo URL publica  https://xxxxx.trycloudflare.com` → URL pública generada.
- `Telegram: URL enviada.` → mensaje enviado a tu chat.

Abre Telegram en el móvil → tendrás un mensaje con la URL del dashboard. Click → carga desde cualquier red.

## 6. Que sobreviva al cierre de RDP

Por defecto el bot se mata cuando cierras la sesión RDP. Soluciones:

**Opción A (recomendada): Task Scheduler**

1. Abre Task Scheduler → Create Task.
2. General → "Run whether user is logged on or not" + "Run with highest privileges".
3. Triggers → At startup.
4. Actions → Start a program → `C:\Bot\START_BOT.bat`.
5. Settings → unchecks "Stop the task if it runs longer than...".

**Opción B: NSSM (servicio Windows)**

```powershell
choco install nssm
nssm install ClawdBot C:\Bot\START_BOT.bat
nssm start ClawdBot
```

## 7. Operación diaria

- **URL del dashboard**: el bot manda una nueva por Telegram cada arranque. La URL cambia tras reinicio (Cloudflare Quick Tunnel), pero mientras el bot esté arriba la URL se mantiene.
- **Parar**: doble clic en `STOP_BOT.bat`.
- **Logs**: `logs\bot_stdout.log`, `logs\bot_stderr.log`, `logs\cloudflared.log`.
- **DB**: archivo `data\clawdbot.sqlite`. Backup = copiar el archivo. Cero servicios externos.

## Limitaciones del modo SQLite

- Sin hypertables Timescale → consultas históricas grandes (>30 días, alta resolución) más lentas. Para volumen actual del bot, irrelevante.
- Fakeredis solo vive en proceso → no compartes cache entre múltiples instancias. En un VPS de un solo bot, no importa.
- Si más adelante necesitas Postgres, instala PostgreSQL nativo en el VPS y cambia `DB_BACKEND=postgres` — no hace falta Docker.
