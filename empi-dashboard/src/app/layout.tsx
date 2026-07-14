import type { Metadata } from "next";
import { Montserrat, Raleway } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { TopNav } from "@/components/TopNav";

const montserrat = Montserrat({
  variable: "--font-montserrat",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800", "900"],
});

const raleway = Raleway({
  variable: "--font-raleway",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Entity Matching Dashboard — AllianceChicago",
  description: "eMPI reviewer console for the entity-resolution pipeline.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${montserrat.variable} ${raleway.variable}`}>
      <body className="min-h-screen bg-bg text-ink antialiased">
        <Providers>
          <TopNav />
          <main className="mx-auto max-w-[1180px] px-6 pt-6 pb-16">
            {children}
          </main>
        </Providers>
      </body>
    </html>
  );
}
