"use client";

import React from "react";

interface LoadingStateProps {
  title?: string;
  message?: string;
  currentStep?: number;
  totalSteps?: number;
  progressPct?: number;
}

export function LoadingState({
  title = "Evaluating Race Scenarios",
  message = "Running Monte Carlo rollouts across prospective tyre lifecycles...",
  currentStep,
  totalSteps,
  progressPct,
}: LoadingStateProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "36px 20px",
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-lg)",
        textAlign: "center",
      }}
    >
      <div
        style={{
          width: "36px",
          height: "36px",
          borderRadius: "50%",
          border: "2px solid var(--border)",
          borderTopColor: "var(--f1-red)",
          animation: "spin 0.8s linear infinite",
          marginBottom: "14px",
        }}
      />

      <div
        style={{
          fontSize: "0.75rem",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          color: "var(--f1-red)",
          marginBottom: "4px",
        }}
      >
        {title}
      </div>

      <p style={{ fontSize: "0.82rem", color: "var(--text-secondary)", maxWidth: "420px", lineHeight: 1.4 }}>
        {message}
      </p>

      {progressPct !== undefined && (
        <div style={{ width: "240px", marginTop: "14px" }}>
          <div
            style={{
              height: "5px",
              background: "var(--surface-secondary)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-pill)",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: `${Math.min(100, Math.max(0, progressPct))}%`,
                height: "100%",
                background: "var(--f1-red)",
                borderRadius: "var(--radius-pill)",
                transition: "width 0.2s ease",
              }}
            />
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: "0.7rem",
              color: "var(--text-muted)",
              marginTop: "4px",
            }}
          >
            <span>{currentStep && totalSteps ? `${currentStep} / ${totalSteps}` : "Progress"}</span>
            <span className="font-mono">{progressPct}%</span>
          </div>
        </div>
      )}
    </div>
  );
}
