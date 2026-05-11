import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Context-Aware Retrieval Engine",
  description:
    "RAG system with dual retrieval strategies, hybrid BM25+dense search, and cross-encoder reranking.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full bg-gray-950 text-gray-100 antialiased">
        <div className="min-h-full flex flex-col">
          <header className="border-b border-gray-800 bg-gray-900 px-6 py-3 flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="h-6 w-6 rounded bg-brand-600 flex items-center justify-center">
                <span className="text-xs font-bold text-white">R</span>
              </div>
              <span className="font-semibold text-gray-100 text-sm tracking-wide">
                Context-Aware Retrieval Engine
              </span>
            </div>
            <span className="text-xs text-gray-500 ml-auto">
              AirAsia GenAI Assessment
            </span>
          </header>
          <main className="flex-1">{children}</main>
        </div>
      </body>
    </html>
  );
}
