import type { Metadata, Viewport } from "next";
import "./globals.css";
import { AppShell } from "@/components/AppShell";

export const metadata: Metadata = {
  title: "ClawdBot — Trading Dashboard",
  description: "ML-powered crypto trading bot dashboard",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    title: "ClawdBot",
    statusBarStyle: "black-translucent",
  },
};

export const viewport: Viewport = {
  themeColor: "#10151D",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0A0E14] text-[#F3F4F6] font-sans antialiased flex h-screen overflow-hidden">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
