"use client";

import React from "react";

export type AlertSeverity = "CRITICAL" | "WARNING" | "INFORMATION";

interface AlertBannerProps {
  severity: AlertSeverity;
  title: string;
  message: string;
  actionText?: string;
  onAction?: () => void;
  onDismiss?: () => void;
}

export function AlertBanner({
  severity,
  title,
  message,
  actionText,
  onAction,
  onDismiss,
}: AlertBannerProps) {
  let color = "var(--color-info)";
  let bg = "var(--color-info-bg)";
  let border = "var(--color-info-border)";
  let icon = "ℹ";

  switch (severity) {
    case "CRITICAL":
      color = "var(--color-danger)";
      bg = "var(--color-danger-bg)";
      border = "var(--color-danger-border)";
      icon = "!";
      break;
    case "WARNING":
      color = "var(--color-warning)";
      bg = "var(--color-warning-bg)";
      border = "var(--color-warning-border)";
      icon = "⚠";
      break;
    case "INFORMATION":
      color = "var(--color-info)";
      bg = "var(--color-info-bg)";
      border = "var(--color-info-border)";
      icon = "ℹ";
      break;
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: "12px",
        padding: "10px 14px",
        background: bg,
        border: `1px solid ${border}`,
        borderRadius: "var(--radius-md)",
        position: "relative",
      }}
    >
      <div
        className="font-mono"
        style={{
          width: "20px",
          height: "20px",
          borderRadius: "50%",
          background: color,
          color: "#fff",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "0.75rem",
          fontWeight: 800,
          flexShrink: 0,
          marginTop: "1px",
        }}
      >
        {icon}
      </div>

      <div style={{ flex: 1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span
            className="font-mono"
            style={{
              fontSize: "0.72rem",
              fontWeight: 800,
              letterSpacing: "0.05em",
              textTransform: "uppercase",
              color: color,
            }}
          >
            {severity} · {title}
          </span>
        </div>
        <p style={{ fontSize: "0.82rem", color: "var(--text-primary)", marginTop: "2px", lineHeight: 1.4 }}>
          {message}
        </p>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
        {actionText && onAction && (
          <button
            type="button"
            onClick={onAction}
            style={{
              background: color,
              color: "#fff",
              border: "none",
              borderRadius: "var(--radius-sm)",
              padding: "4px 8px",
              fontSize: "0.75rem",
              fontWeight: 700,
              cursor: "pointer",
            }}
          >
            {actionText}
          </button>
        )}
        {onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-muted)",
              cursor: "pointer",
              fontSize: "0.9rem",
              padding: "2px 4px",
            }}
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}
