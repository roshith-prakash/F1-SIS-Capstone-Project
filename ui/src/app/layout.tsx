import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "F1-SIS | Pit Wall Strategy Intelligence",
  description: "Real-time Formula 1 Monte Carlo race strategy and predictive intelligence dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
