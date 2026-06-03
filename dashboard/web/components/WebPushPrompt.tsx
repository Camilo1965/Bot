"use client";

import { useEffect, useState } from "react";

const STORAGE_KEY = "clawdbot:webpush:enabled";

export function WebPushPrompt() {
  const [perm, setPerm] = useState<NotificationPermission>("default");
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined" || !("Notification" in window)) return;
    setPerm(Notification.permission);
    setEnabled(localStorage.getItem(STORAGE_KEY) === "1");
  }, []);

  async function enable() {
    if (typeof window === "undefined" || !("Notification" in window)) return;
    const result = await Notification.requestPermission();
    setPerm(result);
    if (result === "granted") {
      localStorage.setItem(STORAGE_KEY, "1");
      setEnabled(true);
      new Notification("ClawdBot", { body: "Browser alerts enabled." });
    }
  }

  function disable() {
    localStorage.setItem(STORAGE_KEY, "0");
    setEnabled(false);
  }

  if (typeof window !== "undefined" && !("Notification" in window)) {
    return <div className="text-xs text-[#9CA3AF]">Browser does not support notifications.</div>;
  }
  if (perm === "denied") {
    return <div className="text-xs text-[#EF4444]">Notifications blocked in browser settings.</div>;
  }
  if (perm === "granted" && enabled) {
    return (
      <button onClick={disable}
        className="px-3 py-1.5 rounded text-xs font-medium bg-[#10B981]/20 text-[#10B981] hover:bg-[#374151]">
        Browser Alerts: ON · Click to disable
      </button>
    );
  }
  return (
    <button onClick={enable}
      className="px-3 py-1.5 rounded text-xs font-medium bg-[#374151] text-[#9CA3AF] hover:bg-[#3B82F6]/20 hover:text-[#3B82F6]">
      Enable Browser Alerts
    </button>
  );
}

export function pushNotify(title: string, body: string, tag?: string) {
  if (typeof window === "undefined" || !("Notification" in window)) return;
  if (localStorage.getItem(STORAGE_KEY) !== "1") return;
  if (Notification.permission !== "granted") return;
  try {
    new Notification(title, { body, tag, icon: "/favicon.ico" });
  } catch {}
}
