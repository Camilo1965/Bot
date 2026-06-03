# ClawdBot Dashboard — UX/UI Design Spec

Spec completo del dashboard web de operación. Apunta a una experiencia de "command center" de trading: dark theme, denso pero respirado, números monoespacio, gráficos con interacciones, telemetría en tiempo real.

> Audiencia: operador único (tú). Single-user. No multi-tenant. No login social. Acceso por API key local o sesión en localhost.

---

## 1. Stack técnico recomendado

| Capa | Elección | Por qué |
|------|----------|---------|
| Frontend | **Next.js 15 (App Router)** + React 19 + TypeScript | SSR/streaming, ecosystem, performance |
| Estilos | **Tailwind CSS v4** + **shadcn/ui** | Tokens diseño consistentes, sin reinventar |
| Charts | **TradingView Lightweight Charts** (candles) + **Recharts** (resto) | Lightweight Charts es el estándar de oro para velas; Recharts para el resto (simple, declarativo) |
| Estado server | **TanStack Query** (React Query) | Cache + revalidation + WebSocket sync |
| WebSocket | **Socket.IO client** o native ws hook | Push real-time del backend |
| Backend API | **FastAPI** + Uvicorn | Async nativo, pydantic schemas, OpenAPI auto |
| State store bot | **Redis** (live) + **DuckDB** (analytics) | Redis para hot reads (positions, last prob); DuckDB para historical queries rápidas |
| Auth | **Local API key** (single header `X-Api-Key`) en `.env` | Simple, suficiente para localhost |
| Deploy | **Docker Compose** (bot + redis + fastapi + nextjs) | Local-first, reproducible |
| Forms | **React Hook Form** + **Zod** | Validation server-aligned |
| Icons | **Lucide React** | Set consistente con shadcn |
| Fuentes | **JetBrains Mono** (números) + **Inter** (UI) | Monospace para precios/PnL; sans para chrome |

Alternativa rápida si quieres ahorro: **Streamlit** (1-2 días para v0, sacrificando finesse). Recomendación: arrancar con Streamlit v0 para validar features, luego invertir en Next.js v1 para look profesional.

---

## 2. Design system

### 2.1 Color tokens (dark-first)

| Token | Hex | Uso |
|-------|-----|-----|
| `bg-base` | `#0A0E14` | Fondo principal |
| `bg-surface` | `#10151D` | Cards, panels |
| `bg-surface-2` | `#161C26` | Modals, sidebar |
| `bg-elevated` | `#1E2530` | Hover, dropdown |
| `border-subtle` | `#1F2937` | Bordes invisibles |
| `border-default` | `#374151` | Bordes de cards |
| `border-strong` | `#4B5563` | Inputs focus |
| `text-primary` | `#F3F4F6` | Texto principal |
| `text-secondary` | `#9CA3AF` | Labels, captions |
| `text-muted` | `#6B7280` | Disabled, placeholders |
| `accent-blue` | `#3B82F6` | Acciones primarias, links |
| `accent-blue-glow` | `#1E40AF` | Sombras de botones primarios |
| `pnl-positive` | `#10B981` | PnL +, win, BUY |
| `pnl-negative` | `#EF4444` | PnL -, loss, SELL |
| `pnl-neutral` | `#9CA3AF` | Hold, BE |
| `warning` | `#F59E0B` | Drift, near kill-switch |
| `critical` | `#DC2626` | Kill-switch active, position margin |
| `success-glow` | `#059669` | Confirmaciones |
| `purple-accent` | `#8B5CF6` | VIBE features, ML probs |
| `gradient-pnl-pos` | `linear-gradient(135deg, #10B981 0%, #059669 100%)` | Hero PnL positive |
| `gradient-pnl-neg` | `linear-gradient(135deg, #EF4444 0%, #B91C1C 100%)` | Hero PnL negative |

### 2.2 Tipografía

- Display (hero numbers): **JetBrains Mono Bold 48-72px tabular-nums**
- Heading 1: **Inter SemiBold 24px**
- Heading 2: **Inter SemiBold 18px**
- Body: **Inter Regular 14px**
- Caption: **Inter Regular 12px** `text-secondary`
- Code/Symbols/Prices: **JetBrains Mono Regular 14px tabular-nums**

### 2.3 Spacing & radius

- Scale Tailwind default (4px base).
- Card padding: `p-6` (24px).
- Section gap: `gap-6`.
- Radius: cards `rounded-xl` (12px), buttons `rounded-lg` (8px), badges `rounded-full`.

### 2.4 Sombra y elevación

- Cards reposo: `shadow-none` con `border-subtle`.
- Cards hover: `shadow-lg shadow-black/40`.
- Modals: `shadow-2xl shadow-black/60`.
- Glow accents (kill-switch crítico): `shadow-[0_0_24px_rgba(220,38,38,0.4)]`.

### 2.5 Estados animados

- Posición abierta → pulsa suave `animate-pulse` en el badge.
- PnL update → transition `transition-colors duration-300`.
- Loading → skeleton con `bg-surface animate-pulse`.
- Drift alert → `animate-bounce` en el ícono por 3s.

---

## 3. Layout global

```
┌──────────────────────────────────────────────────────────────────────────┐
│  TopBar (h=56px)                                                          │
│  [ClawdBot logo] [Live ●]  [Equity $X.XX (+Y%)]  [Bell] [Kill] [Settings]│
├──────────┬───────────────────────────────────────────────────────────────┤
│          │                                                                │
│  Side    │                                                                │
│  Nav     │     Main Content Area                                          │
│  (224px) │     (responsive grid, gap-6, p-6)                              │
│          │                                                                │
│  • Overview                                                               │
│  • Positions                                                              │
│  • Signals                                                                │
│  • Performance                                                            │
│  • Risk                                                                   │
│  • Models                                                                 │
│  • Backtest                                                               │
│  • Journal                                                                │
│  • Settings                                                               │
│  • Alerts                                                                 │
│          │                                                                │
└──────────┴───────────────────────────────────────────────────────────────┘
```

### 3.1 TopBar (sticky)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ [🐺 ClawdBot]  ● Live MT5  │  Equity $142.18 ▲ +6.24% (1d)              │
│                                       (gradient pill, color por signo)  │
│                                                                          │
│                            [🔔 3]  [⛔ Kill]  [⚙️]                       │
└─────────────────────────────────────────────────────────────────────────┘
```

- Estado del bot: dot verde (live) / amarillo (paused) / rojo (killed).
- Equity hero number con micro-sparkline 24h (alt) + delta %.
- `🔔` count = alerts no leídas.
- `⛔ Kill` botón rojo con confirmación modal.

### 3.2 Sidebar nav

- Iconos lucide + label.
- Indicador activo: barra izquierda `accent-blue` 3px + bg `bg-elevated`.
- Badge en items con conteo (Positions: número de abiertas; Alerts: no leídas).
- Collapse a 56px en mobile (solo icons).

---

## 4. Páginas detalladas

### 4.1 Overview (`/`) — Command center

**Objetivo:** vista panorámica para decidir en 5 segundos si todo está bien.

**Layout grid:**

```
┌──────────────────────────────┬──────────────────────────────┐
│  Hero: Equity $XXX           │  Hero: Today's PnL +$X.YZ    │
│  Spark 24h + delta % bg gr   │  +A.BC%   X trades   Y WR%   │
│  size col-span-1             │  size col-span-1             │
├──────────────────────────────┼──────────────────────────────┤
│  Equity Curve 30d            │  Open Positions (live grid)  │
│  Multi-line: portfolio +     │  3 cards: pos1, pos2, ...    │
│  per-symbol toggleable       │                              │
│  col-span-2                  │  col-span-1                  │
├──────────────────────────────┴──────────────────────────────┤
│  Recent Signals (last 20)                                    │
│  Table: timestamp, symbol, side, raw_prob, cal_prob,         │
│         threshold, decision, regime, htf_filter              │
├──────────────────────────────────────────────────────────────┤
│  Per-Symbol Health Strip (mini-cards horizontal scroll)      │
│  BTC | ETH | SOL | DOGE | NEAR | ATOM | LINK                 │
│  c/u: 7d PnL spark + WR + last trade rel time               │
└──────────────────────────────────────────────────────────────┘
```

**Componentes específicos:**

1. **EquityHeroCard**
   ```
   ┌─────────────────────────────────────────┐
   │  EQUITY                  ●●●●●●●●●●●●●● │  ← sparkline 24h derecha
   │  $142.18                                 │  ← font 48px tabular
   │  ▲ +6.24% │ +$8.34 today                 │  ← chip verde
   │                                          │
   │  Initial $100  ·  +42.18% all-time      │  ← caption
   └─────────────────────────────────────────┘
   ```
   - Fondo: gradient sutil dependiente de signo (verde-tinted o rojo-tinted, 5% opacity).
   - Animación: número cuenta hacia arriba (CountUp 800ms ease-out) al cambiar.

2. **TodayPnLCard**
   - Mismo layout que equity, pero focus en delta del día.
   - Sub-stats: trades hoy, WR rolling 24h, % capital usado.

3. **EquityCurveChart**
   - Recharts AreaChart.
   - X: timestamp. Y: equity USD.
   - Líneas: portfolio (bold), per-symbol (toggle on click en chip leyenda).
   - Tooltip: fecha + equity + trades hasta ese punto.
   - Annotations: marcadores rojos en kill-switch activations.

4. **OpenPositionCard** (1 por posición abierta)
   ```
   ┌───────────────────────────────────────────┐
   │  NEAR/USDT          LONG · 8m ago          │  ← badge LONG verde
   │  Entry $4.823   SL $4.679 (-3%)            │
   │  Current $4.901 ▲ +1.62%                   │
   │  ━━━━━━━━━━━━━━░░░░░░ TP $5.064            │  ← progress bar entry→TP
   │  size $14.20  ·  pnl +$0.23  ·  prob 0.71  │
   └───────────────────────────────────────────┘
   ```
   - Progress bar visual entry→TP con SL como marca roja inversa.
   - PnL color en tiempo real (verde/rojo).
   - Click → modal con detalle full (charts, history, manual close).

5. **RecentSignalsTable**
   - Columns: time (rel), symbol chip, side (BUY/HOLD/SELL), raw_prob (sparkline bar), cal_prob, threshold (gris), decision (chip), regime (chip if used), filters passed/failed.
   - Filtros: por símbolo, por decision.
   - Click row → expande con feature breakdown.

6. **SymbolHealthStrip**
   ```
   ┌─ BTC ──────┐ ┌─ ETH ──────┐ ┌─ SOL ──────┐ ...
   │ ●●●●●●●●●● │ │ ●●●●●●●●●● │ │ ●●●●●●●●●● │
   │ +3.2% 7d   │ │ +5.5% 7d   │ │ +10.7% 7d  │
   │ WR 60%     │ │ WR 78%     │ │ WR 91%     │
   │ 8h ago     │ │ 2h ago     │ │ 1d ago     │
   └────────────┘ └────────────┘ └────────────┘
   ```
   - Sparkline mini = equity 7d.
   - Status dot: verde (last trade win), rojo (last trade loss), gris (no recent).
   - Click → naviga a Performance > {symbol}.

### 4.2 Positions (`/positions`)

**Vista de posiciones (abiertas + historial).**

Tabs: **Open** | **Closed (today)** | **All history**.

#### Open tab

Cards expandidas full-width:

```
┌────────────────────────────────────────────────────────────────┐
│ NEAR/USDT  LONG  ●●●●  $14.20 notional                 [✕ Close]│
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Mini candle chart 4h ──────────────────┐  Entry  $4.823    │
│  │                                          │  Now    $4.901   │
│  │      ▲ entry                             │  SL     $4.679   │
│  │                                          │  TP     $5.064   │
│  │  ───────────────                          │  Stop    $4.679 │
│  │                                          │  (trailing inactive)│
│  └──────────────────────────────────────────┘                  │
│                                                                 │
│  PnL  +$0.23 (+1.62%)   ·   8 min held   ·   TTL 4h remaining  │
│  ML prob 0.71  ·  Threshold 0.55  ·  Regime SKIP  ·  HTF ✓     │
│                                                                 │
│  [📋 Copy trade ID]  [📊 View signal context]  [✕ Manual Close]│
└────────────────────────────────────────────────────────────────┘
```

- Candle chart embebido: TradingView Lightweight Charts, 4h, marker en entry.
- Trailing stop visualizado si activo: línea punteada que sube con peak.
- Close manual: modal de confirmación con razón opcional.

#### Closed tab

- Tabla densa con todos los campos del journal.
- Filtros: símbolo, lado, close_reason (SL/TP/TTL/manual), rango fecha.
- Export CSV.
- Cada fila clickeable → modal full detail.

### 4.3 Signals (`/signals`)

**Stream live de señales evaluadas por símbolo.**

Layout: 7 columns (uno por símbolo activo) horizontal scroll.

```
┌─ BTC/USDT ───────────────┐ ┌─ ETH/USDT ───────────────┐
│                          │ │                          │
│  Now: prob 0.34          │ │  Now: prob 0.62 ✓        │
│  Threshold: 0.70 ✗       │ │  Threshold: 0.50 ✓       │
│                          │ │                          │
│  ┌─ Last 50 probs ──────┐│ │  ┌─ Last 50 probs ──────┐│
│  │     ▁▂▁▃▂▁▂▄▂▃▁▂▃▅▂  ││ │  │  ▅▆▅▇▆▆▇▆▇▆▅▇▆▆▇▆▇▆  ││
│  │  ─── threshold ─────  ││ │  │  ─── threshold ─────  ││  ← linea horizontal pt
│  └────────────────────────┘│ │  └────────────────────────┘│
│                          │ │                          │
│  Raw: 0.073  →  Cal: 0.34│ │  Raw: 0.110 → Cal: 0.62  │  ← arrow visual
│                          │ │                          │
│  Last signal: HOLD 2m    │ │  Last signal: BUY 1h ago │
│  HTF: ✓   Vol: ✓   Reg:- │ │  HTF: ✓   Vol: ✓   Reg:- │
│                          │ │                          │
└──────────────────────────┘ └──────────────────────────┘
```

- Cada columna actualiza cada N seconds via WebSocket.
- Sparkline interactivo: hover muestra el prob exacto.
- Cuando cruza threshold → flash verde/rojo + emoji indicador.
- Filtros pasados/fallados como chips verdes/rojos.
- Click columna → vista expandida en modal con histograma de probs últimos 7d.

### 4.4 Performance (`/performance`)

**Analytics de PnL y métricas.**

Top: selector de rango (`1d / 7d / 30d / 90d / all`) + selector símbolo (`all` o específico).

**Layout:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Big metric cards row (4 cards)                                 │
│  [PnL $X.XX] [Trades N] [Win Rate Y%] [Profit Factor Z.ZZ]      │
├─────────────────────────────────────────────────────────────────┤
│  Equity curve (full width)                                       │
│  + drawdown subplot                                              │
├──────────────────────────────────┬──────────────────────────────┤
│  Per-symbol breakdown table       │  Distribution charts          │
│  - Sym, Trades, WR, PnL%, PF, DD  │  - PnL histogram             │
│  - sortable                       │  - Holding time histogram     │
│                                   │  - Win/loss size box plot     │
├──────────────────────────────────┴──────────────────────────────┤
│  Trade heat map (calendar)                                       │
│  GitHub-style: 7w × 7d grid, color por PnL diario                │
├─────────────────────────────────────────────────────────────────┤
│  Hour-of-day & Day-of-week performance grids                     │
│  Heatmap WR%/PnL por hora UTC × símbolo                          │
└─────────────────────────────────────────────────────────────────┘
```

- Heat map calendario tipo GitHub contribution graph.
- Heat map hour-of-day reveal patrones temporales (NEAR opera Asia hours, etc).
- Export PDF report.

### 4.5 Risk (`/risk`)

**Vista de exposición y kill-switch.**

```
┌─────────────────────────────────────────────────────────────────┐
│  Risk Gauges (3 medidores arco)                                  │
│  [Portfolio Risk 4.2% / 10%]  [Daily PnL -1.1% / -5%]            │
│  [Drawdown 2.3% / 15%]                                           │
├─────────────────────────────────────────────────────────────────┤
│  Current Exposure breakdown (donut chart)                        │
│  - Por símbolo + cash idle                                       │
│  - Total risk_usd sumando posiciones abiertas                    │
├─────────────────────────────────────────────────────────────────┤
│  Kill-Switch Status panel                                        │
│  ┌─ Triggers ──────────────────────────────────────────────┐    │
│  │ ✓ Daily PnL >= -5%        Current -1.1%        SAFE     │    │
│  │ ✓ Consec losses <3        Current 1            SAFE     │    │
│  │ ⚠ 7d DD <15%              Current 2.3%         SAFE     │    │
│  │ ⚠ Per-sym WR >30%         All OK                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  [🔧 Configure thresholds]  [📜 View activation history]         │
├─────────────────────────────────────────────────────────────────┤
│  Recent kill-switch events log                                   │
│  Table: timestamp, trigger, action_taken, recovery_time          │
└─────────────────────────────────────────────────────────────────┘
```

- Medidores: arc charts (D3 o react-gauge), verde→amarillo→rojo según %.
- Donut con tooltip por slice.
- Triggers UI con check/warning/critical icons live.

### 4.6 Models (`/models`)

**Health por modelo: AUC trend, calibration, drift.**

Lista de símbolos como filas de cards (1 por modelo activo).

```
┌─────────────────────────────────────────────────────────────────┐
│  BTC/USDT   v2 (long)   Trained 2026-05-27   180d   26 features │
├─────────────────────────────────────────────────────────────────┤
│  AUC: 0.6248 ┃ F1: 0.1364 ┃ Last refit cal: 2026-06-02          │
│                                                                  │
│  ┌─ AUC trend last 6 retrains ──┐  ┌─ Raw prob histogram ──────┐│
│  │      ▂▃▄▄▅▆                  │  │  ▇▆▅▄▃▂▁                  ││
│  │  0.55──────0.65               │  │  0.0       1.0            ││
│  └──────────────────────────────┘  └────────────────────────────┘│
│                                                                  │
│  ┌─ Calibration curve ──────────┐  ┌─ Drift KS-test 7d ────────┐│
│  │  diagonal vs fitted iso      │  │  p-value: 0.84  STABLE     ││
│  │      ╱╱╱                     │  │  KS-statistic: 0.07        ││
│  │  ╱╱╱                          │  │                            ││
│  └──────────────────────────────┘  └────────────────────────────┘│
│                                                                  │
│  Feature importance (top 8 bars)                                 │
│  atr ████████████████████   0.074                                │
│  dow ██████████████████     0.064                                │
│  ...                                                             │
│                                                                  │
│  [🔄 Trigger retrain]  [📊 Run backtest]  [⬇️ Download model]    │
└─────────────────────────────────────────────────────────────────┘
```

- Calibration curve plot (perfectly calibrated = diagonal, actual = curve).
- Drift KS-test rolling 7d resultado live.
- Botón "Retrain now" hace POST a `/api/models/{symbol}/retrain` (async; toast progress).

### 4.7 Backtest (`/backtest`)

**Lanzar backtests desde UI.**

Form:
- Símbolo (multi-select).
- Modo: `disk` / `inline` / `portfolio`.
- Días: slider 7-180.
- Override params (opcional, expand panel): pt, sl, tp, ttl, risk.

Run → progress bar live (WebSocket de logs) → resultados al terminar.

Result view: misma sección que Performance pero etiquetada "BACKTEST RESULT".

History list: backtests pasados con timestamp + params + click para re-ver.

### 4.8 Journal (`/journal`)

**Trade journal + VIBE insights.**

Tabs: **Trades** | **VIBE Insights** | **Notes**.

#### Trades tab
- Tabla completa del journal con todos los campos.
- Búsqueda fulltext (por símbolo, razón, etc).
- Notas manuales por trade (campo editable).

#### VIBE Insights tab
- Output del `vibe.journal_analyzer`:
  - Disposition effect score
  - Overtrading periods detected
  - Anchoring bias evidence
  - Suggested behavioral corrections
- Cada insight con explicación + ejemplo de trade afectado.

#### Notes tab
- Markdown editor libre para journaling personal.
- Auto-save cada cambio.

### 4.9 Settings (`/settings`)

**Editor de configuración.**

Tabs: **Symbols** | **Risk** | **Execution** | **Notifications** | **API Keys**.

#### Symbols tab
- Tabla con todas las entradas de `SYMBOL_CONFIG`.
- Click "Edit" → drawer con form de TODOS los campos.
- Validación Zod: rangos sanos (pt ∈ [0,1], sl ∈ [0.005, 0.10]).
- Botón "Run quick backtest with new params" antes de save.
- Save → POST a `/api/symbols/{symbol}/config`; bot reload config sin restart.
- Botón "Add new symbol" → wizard (fetch test, train, calibrate, backtest, promote).

#### Risk tab
- Sliders: MAX_POSITIONS, RISK_PER_TRADE, MAX_PORTFOLIO_RISK_PCT, leverage, DAILY_LOSS_LIMIT.
- Kill-switch triggers thresholds.

#### Notifications tab
- Telegram/Discord webhook URL.
- Per-event toggle: trade_open, trade_close, daily_report, kill_switch, drift.

#### API Keys tab
- MT5 credentials (masked, edit con confirmación).
- Binance API key (futuro live).

### 4.10 Alerts (`/alerts`)

**Inbox de notificaciones.**

- Lista cronológica reverse.
- Filtros: severity (info/warn/critical), source (kill_switch/drift/trade/system).
- Cada alert: timestamp, severity badge, source chip, mensaje, acciones (acknowledge / link a contexto).
- Search bar.
- Marcar como read individualmente o "marcar todos".

---

## 5. Componentes reutilizables (design system extras)

| Componente | Propósito | Notas |
|------------|-----------|-------|
| `SymbolChip` | Display BTC/USDT con icono crypto | Color tinted por símbolo |
| `PnLValue` | Wrapper formato número con color | `+$X.YZ (+A.BC%)`, tabular-nums |
| `SideBadge` | LONG/SHORT/HOLD | Verde/rojo/gris |
| `ProbBar` | Barra horizontal raw → cal con threshold marker | Reusable en Signals + Models |
| `Sparkline` | Mini chart sin ejes | Para health strips |
| `MetricCard` | Hero number + delta + sparkline | Reusable cards de KPI |
| `RiskGauge` | Arco con valor + threshold | Para Risk page |
| `Skeleton` | Loading state grid-aware | shadcn default |
| `CountUp` | Animación de número incrementando | Para hero numbers en updates |
| `RelativeTime` | "2m ago" auto-update cada 30s | Wrapper de date-fns |
| `Toast` | Notificación efímera | shadcn sonner |
| `ConfirmDialog` | Modal con confirm + razón opcional | Para kills, manual close |

---

## 6. WebSocket events (backend → frontend)

| Event | Payload | Frecuencia |
|-------|---------|------------|
| `equity.tick` | `{balance, equity, dd, ts}` | 1Hz |
| `position.opened` | `{trade_id, symbol, side, entry, sl, tp, size}` | Event |
| `position.updated` | `{trade_id, current_price, pnl_usd, pnl_pct, current_stop}` | 1Hz por posición |
| `position.closed` | `{trade_id, exit, pnl, reason}` | Event |
| `signal.evaluated` | `{symbol, raw_prob, cal_prob, threshold, decision, filters}` | Por evaluación (~varies por TF) |
| `kill_switch.triggered` | `{trigger, msg, action_taken}` | Event raro |
| `drift.detected` | `{symbol, ks_stat, p_value}` | Cron horario |
| `model.retrained` | `{symbol, auc_old, auc_new, brier_delta}` | Event raro |
| `alert.created` | `{severity, source, message}` | Event |

Frontend subscribe selectivamente según ruta activa para minimizar bandwidth.

---

## 7. API endpoints (FastAPI)

```
GET  /api/state              # snapshot completo (balance, positions, last signals)
GET  /api/equity?days=30     # serie temporal equity
GET  /api/positions/open
GET  /api/positions/history?limit=100&symbol=...
GET  /api/signals/recent?limit=50
GET  /api/signals/by-symbol/{symbol}?limit=200

GET  /api/performance?range=30d&symbol=all
GET  /api/performance/heatmap?metric=pnl
GET  /api/performance/by-hour
GET  /api/performance/by-dow

GET  /api/risk/state         # gauges + exposure
GET  /api/risk/killswitch/state
POST /api/risk/killswitch/trigger
POST /api/risk/killswitch/reset
GET  /api/risk/killswitch/history

GET  /api/models             # lista todos
GET  /api/models/{symbol}
GET  /api/models/{symbol}/feature-importance
GET  /api/models/{symbol}/calibration-curve
GET  /api/models/{symbol}/drift-test
POST /api/models/{symbol}/retrain
POST /api/models/{symbol}/refit-calibration

POST /api/backtest/run       # body: {symbols, mode, days, overrides}
GET  /api/backtest/list      # history
GET  /api/backtest/{run_id}

GET  /api/journal/trades?from=&to=&symbol=
POST /api/journal/notes/{trade_id}
GET  /api/journal/vibe-insights

GET  /api/symbols/config
PATCH /api/symbols/{symbol}/config
POST /api/symbols/add        # wizard endpoint

GET  /api/settings/risk
PATCH /api/settings/risk
GET  /api/settings/notifications
PATCH /api/settings/notifications

GET  /api/alerts?status=unread
POST /api/alerts/{id}/ack
POST /api/alerts/ack-all

WS   /ws/stream              # WebSocket subscribe events
```

Auth: header `X-Api-Key: <local-key>` en todos. Key generada al primer arranque y guardada en `.env`.

---

## 8. Bot ↔ Dashboard data flow

```
                    ┌─────────────┐
                    │  Bot loop   │ (bot/main.py)
                    └──────┬──────┘
                           │
              writes state │
                           ▼
                    ┌─────────────┐
                    │   Redis     │ (hot live state)
                    │  - balance  │
                    │  - positions[]
                    │  - last_signals[]
                    └──────┬──────┘
                           │
                  reads/subs│
                           ▼
                    ┌─────────────┐
       writes hist  │  FastAPI    │  serves API + WS
       to DuckDB  ◄─┤  uvicorn    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Next.js    │ (browser)
                    │  React+ws   │
                    └─────────────┘

         ┌───────────────────────────┐
         │  DuckDB (analytics)       │
         │  - trades                 │
         │  - signals_log            │
         │  - equity_snapshots       │
         │  - backtest_runs          │
         │  - kill_switch_events     │
         └───────────────────────────┘
```

Bot empuja a Redis (hot) + DuckDB (cold). FastAPI lee de ambos. WebSocket transmite eventos en tiempo real.

---

## 9. Plan de implementación por sprints

### Sprint 1 (semana 1) — Foundation + Overview MVP

- [ ] Setup monorepo: `/dashboard/api` (FastAPI) + `/dashboard/web` (Next.js) + Docker Compose.
- [ ] Redis schema + DuckDB schema diseñado y migrado.
- [ ] Bot escribe `equity_snapshots`, `positions`, `signals_log` a Redis/DuckDB.
- [ ] FastAPI endpoints `/state`, `/equity`, `/positions/open`, `/signals/recent`.
- [ ] WebSocket básico (`equity.tick`, `position.opened/closed`, `signal.evaluated`).
- [ ] Next.js setup con Tailwind + shadcn + design tokens.
- [ ] Página **Overview** funcional: EquityHero, TodayPnL, equity curve, open positions, recent signals.
- [ ] TopBar con live status + kill button (modal no funcional aún).
- [ ] Sidebar nav.

**Deliverable:** dashboard locales con Overview live conectado al bot. Refrescos en tiempo real.

### Sprint 2 (semana 2) — Positions + Signals + Performance

- [ ] Página **Positions** con tabs Open/Closed/All.
- [ ] Embed candle chart con Lightweight Charts en position card.
- [ ] Manual close endpoint + UI.
- [ ] Página **Signals**: 7-col live stream + histogram modal.
- [ ] Página **Performance**: KPI cards, equity curve con drawdown subplot.
- [ ] Per-symbol breakdown table sortable.
- [ ] Distribution charts (PnL hist, holding time, win/loss box).

**Deliverable:** vista operacional completa de positions y rendimiento.

### Sprint 3 (semana 3) — Risk + Models

- [ ] Página **Risk**: gauges, exposure donut, kill-switch panel.
- [ ] Kill-switch trigger config UI.
- [ ] Página **Models**: per-symbol cards con AUC trend + calibration curve + drift test + feature importance.
- [ ] Retrain trigger endpoint + toast progress.
- [ ] Refit calibration endpoint + UI.

**Deliverable:** observabilidad y control de riesgo + modelos.

### Sprint 4 (semana 4) — Backtest + Journal + Settings

- [ ] Página **Backtest**: form de configuración + run + result view.
- [ ] WebSocket progress live de logs durante backtest.
- [ ] History list con click-to-view.
- [ ] Página **Journal**: tabla trades + búsqueda + notas.
- [ ] VIBE insights tab.
- [ ] Notes markdown editor.
- [ ] Página **Settings**: symbols editor con validación + quick-backtest preview.
- [ ] Risk settings sliders.
- [ ] Notifications config (Telegram webhook test).
- [ ] API keys management con masking.

**Deliverable:** workflow completo de operación + tuning desde UI.

### Sprint 5 (semana 5) — Polish + Alerts + Deploy

- [ ] Página **Alerts** inbox.
- [ ] Heat map calendar en Performance.
- [ ] Hour-of-day + day-of-week heat maps.
- [ ] Animaciones refinadas (CountUp, transitions).
- [ ] Loading skeletons consistentes.
- [ ] Error boundaries.
- [ ] Empty states con ilustraciones.
- [ ] Responsive mobile (sidebar collapse, cards stack).
- [ ] Tests E2E con Playwright (smoke flow).
- [ ] Docker Compose production build.
- [ ] README de deployment.

**Deliverable:** dashboard production-ready con UX pulido.

### Sprint 6 (opcional) — Avanzado

- [ ] Multi-monitor support (dashboard fullscreen mode).
- [ ] Voice alerts (Web Speech API) en kill-switch.
- [ ] PWA install para móvil con push notifications.
- [ ] Export reportes PDF.
- [ ] Modo "presentación" sin secrets visibles (para screenshots).
- [ ] Theme switcher (dark default, light alternativo).
- [ ] i18n ES/EN.

---

## 10. Estructura de carpetas propuesta

```
Bot/
├─ dashboard/
│  ├─ api/                         # FastAPI
│  │  ├─ main.py
│  │  ├─ routers/
│  │  │  ├─ state.py
│  │  │  ├─ positions.py
│  │  │  ├─ signals.py
│  │  │  ├─ performance.py
│  │  │  ├─ risk.py
│  │  │  ├─ models.py
│  │  │  ├─ backtest.py
│  │  │  ├─ journal.py
│  │  │  ├─ settings.py
│  │  │  └─ alerts.py
│  │  ├─ ws/
│  │  │  └─ stream.py
│  │  ├─ db/
│  │  │  ├─ redis_client.py
│  │  │  └─ duckdb_client.py
│  │  ├─ schemas/                   # Pydantic
│  │  └─ requirements.txt
│  │
│  ├─ web/                          # Next.js
│  │  ├─ app/
│  │  │  ├─ layout.tsx
│  │  │  ├─ page.tsx                # Overview
│  │  │  ├─ positions/page.tsx
│  │  │  ├─ signals/page.tsx
│  │  │  ├─ performance/page.tsx
│  │  │  ├─ risk/page.tsx
│  │  │  ├─ models/page.tsx
│  │  │  ├─ backtest/page.tsx
│  │  │  ├─ journal/page.tsx
│  │  │  ├─ settings/page.tsx
│  │  │  └─ alerts/page.tsx
│  │  ├─ components/
│  │  │  ├─ ui/                     # shadcn
│  │  │  ├─ design/                 # custom design system
│  │  │  │  ├─ EquityHeroCard.tsx
│  │  │  │  ├─ TodayPnLCard.tsx
│  │  │  │  ├─ SymbolChip.tsx
│  │  │  │  ├─ PnLValue.tsx
│  │  │  │  ├─ SideBadge.tsx
│  │  │  │  ├─ ProbBar.tsx
│  │  │  │  ├─ Sparkline.tsx
│  │  │  │  ├─ RiskGauge.tsx
│  │  │  │  ├─ CountUp.tsx
│  │  │  │  └─ RelativeTime.tsx
│  │  │  └─ charts/
│  │  │     ├─ EquityCurve.tsx
│  │  │     ├─ CandleChart.tsx
│  │  │     ├─ CalibrationCurve.tsx
│  │  │     ├─ HeatmapCalendar.tsx
│  │  │     └─ HeatmapHourly.tsx
│  │  ├─ hooks/
│  │  │  ├─ useWebSocket.ts
│  │  │  ├─ useEquity.ts
│  │  │  ├─ usePositions.ts
│  │  │  └─ useSignals.ts
│  │  ├─ lib/
│  │  │  ├─ api.ts                  # fetcher con X-Api-Key
│  │  │  ├─ formatters.ts           # currency, pct, time
│  │  │  └─ design-tokens.ts
│  │  ├─ public/
│  │  ├─ tailwind.config.ts
│  │  ├─ tsconfig.json
│  │  └─ package.json
│  │
│  └─ docker-compose.yml
```

---

## 11. Criterios de aceptación finales del dashboard

1. **Loading time**: Overview interactive en <1.5s en localhost.
2. **WebSocket reconnect**: si bot se reinicia, dashboard reconecta auto en <3s.
3. **No data loss**: si dashboard cae, bot sigue persistiendo a Redis+DuckDB; al volver, frontend repuebla state.
4. **Responsive**: usable en pantalla 1920×1080, 1366×768, y tablet 1024×768.
5. **Performance**: 60fps en navegación, sin jank en updates de equity tick.
6. **Accesibilidad**: contraste AAA en números críticos (PnL, alerts).
7. **Tests**: cobertura E2E del happy path (login → Overview → open position → close trade).
8. **Documentación**: README con setup local + deploy + troubleshooting.

---

## 12. Mockups ASCII de referencia rápida

### 12.1 Hero del Overview

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    EQUITY                                          ▁▂▃▄▄▅▆▇          ║
║    $142.18                                                           ║
║    ▲ +6.24%  ·  +$8.34 today  ·  +42.18% all-time                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 12.2 Posición abierta card

```
╔══════════════════════════════════════════════════════════════════════╗
║  ⬢ NEAR/USDT      LONG  ●   8 min held                       [✕]    ║
║                                                                      ║
║  Entry  $4.823       SL ─── $4.679   ──┐                             ║
║  Now    $4.901 ▲     TP ─── $5.064 ────│                             ║
║                                                                      ║
║  ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━○                       ║
║  SL                              [▶]                              TP ║
║                                                                      ║
║  PnL  +$0.23 (+1.62%)   ·   size $14.20   ·   prob 0.71              ║
║  HTF ✓   Vol ✓   Regime SKIP   TTL 4h remaining                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 12.3 Signals stream columna

```
┌────────────────────────┐
│  BTC/USDT     ●        │
│  pt=0.70               │
│                        │
│      ▁▂▃▂▃▁▂▄▃         │
│  ━━━━━━━━━━━━━━ pt     │
│                        │
│  Raw 0.073→Cal 0.34    │
│  HOLD                  │
│                        │
│  Last: HOLD  2m        │
│  HTF ✓  Vol ✓          │
└────────────────────────┘
```

---

## 13. Próximos pasos concretos para empezar

1. Crear branch `feat/dashboard-mvp-sprint1`.
2. `mkdir dashboard/{api,web}`.
3. Bootstrap FastAPI con `uv` o `poetry`: añadir routers vacíos + endpoint `/state` que lea de Redis (mock inicial).
4. Bootstrap Next.js: `npx create-next-app@latest dashboard/web --typescript --tailwind --app`.
5. Instalar shadcn: `npx shadcn-ui@latest init` + componentes base (card, button, badge, table, dropdown, dialog, toast).
6. Setup design tokens en `tailwind.config.ts` con la paleta del §2.1.
7. Implementar Overview con datos mock primero, luego conectar al bot.
8. Modificar `bot/main.py` para escribir snapshot a Redis cada 1s.
9. WebSocket inicial: emitir `equity.tick` desde bot al frontend.
10. Demo end-to-end en sprint review.
