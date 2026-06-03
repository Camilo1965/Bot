"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Tab = "symbols" | "risk" | "notifications" | "apikeys";

const TABS: { key: Tab; label: string }[] = [
  { key: "symbols", label: "Symbols" },
  { key: "risk", label: "Risk" },
  { key: "notifications", label: "Notifications" },
  { key: "apikeys", label: "API Keys" },
];

interface SymbolConfig {
  symbol: string;
  timeframe: string;
  prob_threshold: number;
  fixed_sl_pct: number;
  fixed_tp_pct: number;
  enabled: boolean;
  max_position_pct?: number;
}

function SymbolsTab() {
  const [configs, setConfigs] = useState<SymbolConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/api/symbols/config`)
      .then((r) => r.json())
      .then((data) => {
        setConfigs(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="text-[#9CA3AF] text-sm text-center py-6">Loading…</div>;
  if (error) return <div className="text-[#EF4444] text-sm text-center py-4">Failed to load: {error}</div>;
  if (configs.length === 0) return <div className="text-[#9CA3AF] text-sm text-center py-6">No symbol configs found.</div>;

  return (
    <div className="overflow-x-auto">
      <div className="text-xs text-[#9CA3AF] mb-3">
        Read-only view. Edit SYMBOL_CONFIG in the bot source to change these values.
      </div>
      <table className="w-full text-sm text-left">
        <thead>
          <tr className="text-[10px] uppercase tracking-widest text-[#9CA3AF] border-b border-[#374151]">
            {["Symbol", "Timeframe", "Threshold", "SL%", "TP%", "Max Pos%", "Enabled"].map((h) => (
              <th key={h} className="pb-2 pr-4 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {configs.map((c) => (
            <tr
              key={c.symbol}
              className="border-b border-[#374151]/40 hover:bg-white/5 transition-colors"
            >
              <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{c.symbol}</td>
              <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{c.timeframe}</td>
              <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{c.prob_threshold.toFixed(3)}</td>
              <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{(c.fixed_sl_pct * 100).toFixed(2)}%</td>
              <td className="py-2 pr-4 font-mono text-[#F3F4F6]">{(c.fixed_tp_pct * 100).toFixed(2)}%</td>
              <td className="py-2 pr-4 font-mono text-[#9CA3AF]">
                {c.max_position_pct !== undefined ? `${(c.max_position_pct * 100).toFixed(1)}%` : "—"}
              </td>
              <td className="py-2 pr-4">
                <span
                  className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                    c.enabled
                      ? "bg-[#10B981]/20 text-[#10B981]"
                      : "bg-[#374151] text-[#9CA3AF]"
                  }`}
                >
                  {c.enabled ? "ON" : "OFF"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RiskTab() {
  return (
    <div className="space-y-4">
      <div className="text-xs text-[#9CA3AF] mb-2">
        Risk parameters are set via environment variables. Restart the bot after changing them.
      </div>
      <div className="space-y-2">
        {[
          { label: "MAX_DAILY_LOSS_PCT", desc: "Kill switch triggers when daily PnL falls below this percentage" },
          { label: "MAX_CONSECUTIVE_LOSSES", desc: "Kill switch triggers after this many consecutive losses" },
          { label: "MAX_DRAWDOWN_7D_PCT", desc: "Kill switch triggers when 7d drawdown exceeds this" },
          { label: "MAX_OPEN_POSITIONS", desc: "Maximum number of simultaneously open positions" },
          { label: "POSITION_SIZE_PCT", desc: "Default position size as % of available balance" },
        ].map((item) => (
          <div
            key={item.label}
            className="flex items-start gap-3 border border-[#374151] rounded-lg px-4 py-3"
          >
            <code className="text-[#3B82F6] text-xs font-mono shrink-0 mt-0.5">{item.label}</code>
            <span className="text-[#9CA3AF] text-xs">{item.desc}</span>
          </div>
        ))}
      </div>
      <p className="text-xs text-[#9CA3AF] mt-2">
        Current values are read from <code className="text-[#F3F4F6]">.env</code> at startup.
        Live editing via dashboard is not yet supported.
      </p>
    </div>
  );
}

function NotificationsTab() {
  return (
    <div className="space-y-4">
      <div className="text-xs text-[#9CA3AF] mb-2">
        Notification settings are configured via environment variables.
      </div>
      <div className="border border-[#374151] rounded-lg px-4 py-4 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-[#F3F4F6] font-medium">Telegram</div>
            <div className="text-xs text-[#9CA3AF] mt-0.5">
              Set <code>TELEGRAM_ENABLED=true</code>, <code>TELEGRAM_BOT_TOKEN</code>, and{" "}
              <code>TELEGRAM_CHAT_ID</code> in your <code>.env</code> file.
            </div>
          </div>
          <div className="text-xs text-[#9CA3AF] font-mono bg-[#374151] px-2 py-1 rounded">
            TELEGRAM_ENABLED
          </div>
        </div>
        <div className="border-t border-[#374151] pt-3 flex items-center justify-between">
          <div>
            <div className="text-sm text-[#F3F4F6] font-medium">Alert Levels</div>
            <div className="text-xs text-[#9CA3AF] mt-0.5">
              Set <code>NOTIFY_ON_SIGNAL</code>, <code>NOTIFY_ON_TRADE</code>, and{" "}
              <code>NOTIFY_ON_KILL_SWITCH</code>.
            </div>
          </div>
        </div>
      </div>
      <p className="text-xs text-[#9CA3AF]">
        Toggle controls coming in a future release once the settings API is wired up.
      </p>
    </div>
  );
}

function ApiKeysTab() {
  function masked(key: string | undefined) {
    if (!key) return "not set";
    if (key.length <= 8) return "••••••••";
    return key.slice(0, 4) + "••••••••" + key.slice(-4);
  }

  const exchangeKey = process.env.NEXT_PUBLIC_EXCHANGE_API_KEY;

  return (
    <div className="space-y-4">
      <div className="text-xs text-[#9CA3AF] mb-2">
        API keys are loaded from <code className="text-[#F3F4F6]">.env</code> and never exposed in
        the browser. Only masked previews are shown here.
      </div>
      <div className="space-y-2">
        {[
          { label: "Exchange API Key", env: "EXCHANGE_API_KEY", value: exchangeKey },
          { label: "Exchange API Secret", env: "EXCHANGE_API_SECRET", value: undefined },
          { label: "Telegram Bot Token", env: "TELEGRAM_BOT_TOKEN", value: undefined },
          { label: "Database URL", env: "DATABASE_URL", value: undefined },
        ].map((item) => (
          <div
            key={item.env}
            className="flex items-center justify-between border border-[#374151] rounded-lg px-4 py-3"
          >
            <div>
              <div className="text-sm text-[#F3F4F6]">{item.label}</div>
              <code className="text-[10px] text-[#9CA3AF]">{item.env}</code>
            </div>
            <code className="text-xs font-mono text-[#9CA3AF] bg-[#374151] px-2 py-1 rounded">
              {masked(item.value)}
            </code>
          </div>
        ))}
      </div>
      <p className="text-xs text-[#9CA3AF]">
        To rotate keys, update <code className="text-[#F3F4F6]">.env</code> and restart the bot.
      </p>
    </div>
  );
}

export default function SettingsPage() {
  const [tab, setTab] = useState<Tab>("symbols");

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold text-[#F3F4F6]">Settings</h1>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-[#374151] overflow-x-auto">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px whitespace-nowrap ${
              tab === t.key
                ? "border-[#3B82F6] text-[#F3F4F6]"
                : "border-transparent text-[#9CA3AF] hover:text-[#F3F4F6]"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="bg-[#10151D] border border-[#374151] rounded-lg p-5">
        {tab === "symbols" && <SymbolsTab />}
        {tab === "risk" && <RiskTab />}
        {tab === "notifications" && <NotificationsTab />}
        {tab === "apikeys" && <ApiKeysTab />}
      </div>
    </div>
  );
}
