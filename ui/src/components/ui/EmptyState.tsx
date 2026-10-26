"use client";

import React from "react";

interface EmptyStateProps {
  title: string;
  description: string;
  icon?: string;
  actionText?: string;
  onAction?: () => void;
  compact?: boolean;
}

export function EmptyState({
  title,
  description,
  icon = "⏱",
  actionText,
  onAction,
  compact = false,
}: EmptyStateProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: compact ? "24px 16px" : "48px 24px",
        textAlign: "center",
        background: "var(--surface-secondary)",
        border: "1px dashed var(--border)",
        borderRadius: "var(--radius-lg)",
      }}
    >
      <div
        style={{
          width: compact ? "32px" : "42px",
          height: compact ? "32px" : "42px",
          borderRadius: "50%",
          background: "var(--surface)",
          border: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: compact ? "16px" : "20px",
          marginBottom: "12px",
          boxShadow: "var(--shadow-xs)",
        }}
      >
        {icon}
      </div>

      <h4
        style={{
          fontSize: compact ? "0.85rem" : "0.95rem",
          fontWeight: 700,
          color: "var(--text-primary)",
          marginBottom: "4px",
        }}
      >
        {title}
      </h4>

      <p
        style={{
          fontSize: "0.8rem",
          color: "var(--text-secondary)",
          maxWidth: "360px",
          lineHeight: 1.45,
          marginBottom: actionText ? "16px" : "0",
        }}
      >
        {description}
      </p>

      {actionText && onAction && (
        <button type="button" onClick={onAction} className="btn-secondary" style={{ marginTop: "12px" }}>
          {actionText}
        </button>
      )}
    </div>
  );
}
