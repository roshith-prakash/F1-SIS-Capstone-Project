"use client";

import React, { useState } from "react";
import { Header } from "./Header";
import { Sidebar, NavSection } from "./Sidebar";
import { TimingStrip } from "@/components/dashboard/TimingStrip";
import { CalculatingOverlay } from "@/components/overlay/CalculatingOverlay";

interface AppShellProps {
  children: (activeSection: NavSection) => React.ReactNode;
  defaultSection?: NavSection;
}

export function AppShell({ children, defaultSection = "overview" }: AppShellProps) {
  const [activeSection, setActiveSection] = useState<NavSection>(defaultSection);

  return (
    <div className="app-shell">
      {/* Calculating Overlay Modal */}
      <CalculatingOverlay />

      {/* Main Header */}
      <Header />

      {/* App Body with Sidebar + Content */}
      <div className="app-body">
        <Sidebar activeSection={activeSection} onSelectSection={setActiveSection} />
        <main className="app-content">
          {children(activeSection)}
        </main>
      </div>

      {/* Bottom Timing Strip */}
      <TimingStrip />
    </div>
  );
}
