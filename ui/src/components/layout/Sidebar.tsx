"use client";

import React from "react";

export type NavSection =
  | "overview"
  | "liverace"
  | "strategy"
  | "decision"
  | "opponents"
  | "simulation"
  | "analytics"
  | "modelinsights";

interface SidebarProps {
  activeSection: NavSection;
  onSelectSection: (section: NavSection) => void;
}

interface NavItemDef {
  id: NavSection;
  label: string;
  icon: string;
  badge?: string;
}

const NAV_ITEMS: NavItemDef[] = [
  { id: "overview", label: "Overview", icon: "⊞" },
  { id: "liverace", label: "Live Race", icon: "🏎" },
  { id: "strategy", label: "Strategy", icon: "🗺" },
  { id: "decision", label: "Decision Engine", icon: "⚡" },
  { id: "opponents", label: "Opponents", icon: "🎯" },
  { id: "simulation", label: "Simulation", icon: "🎲" },
  { id: "analytics", label: "Analytics", icon: "📈" },
  { id: "modelinsights", label: "Model Insights", icon: "🔬" },
];

export function Sidebar({ activeSection, onSelectSection }: SidebarProps) {
  return (
    <aside className="app-sidebar">
      <div
        style={{
          fontSize: "0.68rem",
          fontWeight: 700,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "var(--text-muted)",
          padding: "4px 12px 8px 12px",
        }}
      >
        Strategy Control Deck
      </div>

      <nav style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
        {NAV_ITEMS.map((item) => {
          const isActive = activeSection === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectSection(item.id)}
              className={`nav-item ${isActive ? "active" : ""}`}
            >
              <span style={{ fontSize: "1rem", lineHeight: 1, width: "18px", textAlign: "center" }}>
                {item.icon}
              </span>
              <span>{item.label}</span>
              {item.badge && (
                <span
                  className="font-mono"
                  style={{
                    marginLeft: "auto",
                    fontSize: "0.65rem",
                    padding: "1px 5px",
                    borderRadius: "var(--radius-pill)",
                    background: "var(--border-subtle)",
                    color: "var(--text-muted)",
                  }}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Subtle Bottom Footer */}
      <div style={{ marginTop: "auto", padding: "12px 10px 4px 10px", borderTop: "1px solid var(--border-subtle)" }}>
        <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", lineHeight: 1.3 }}>
          Formula 1 SIS v1.0
        </div>
        <div style={{ fontSize: "0.65rem", color: "var(--text-dim)", marginTop: "2px" }}>
          Light Precision Analytics
        </div>
      </div>
    </aside>
  );
}
