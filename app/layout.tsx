import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ChatApp",
  description: "Chat app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-gray-100 antialiased">
        <main className="min-h-screen min-w-screen">
          <section className="h-full w-full">{children}</section>
        </main>
      </body>
    </html>
  );
}
