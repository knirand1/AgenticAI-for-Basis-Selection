import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Basis-Selection Agent",
  description: "Submit (x, y) observations and view the selected basis fit.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}
