"use client";

export function Toast({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div className="fixed bottom-6 left-1/2 z-[200] -translate-x-1/2 rounded-lg bg-ink px-4 py-2.5 text-sm font-semibold text-white shadow-lg">
      {message}
    </div>
  );
}
