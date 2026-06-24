import { useState, useRef, useEffect } from 'react';

export default function StatusLog({ entries }) {
  const [open, setOpen] = useState(false);
  const panelRef = useRef(null);
  const scrollRef = useRef(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handleClick = (e) => {
      if (panelRef.current && !panelRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [open]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (open && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [open, entries.length]);

  const lastMsg = entries.length > 0 ? entries[entries.length - 1].text : '';

  return (
    <div className="relative" ref={panelRef}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded px-2.5 py-1 text-xs transition-colors hover:bg-white/5"
        style={{ color: 'var(--text-secondary)' }}
      >
        <span className="truncate" style={{ maxWidth: 120 }}>{lastMsg || 'Status'}</span>
        <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor"
             style={{ transform: open ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.15s' }}>
          <path d="M2 4l3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {open && (
        <div
          className="absolute right-0 top-full mt-1 z-50 rounded-lg border shadow-xl overflow-hidden"
          style={{
            width: 360,
            maxHeight: 300,
            background: 'var(--bg-panel)',
            borderColor: 'var(--border-color)',
          }}
        >
          <div className="flex items-center justify-between px-3 py-2 border-b"
               style={{ borderColor: 'var(--border-color)' }}>
            <span className="text-[10px] font-semibold uppercase tracking-wider"
                  style={{ color: 'var(--text-secondary)' }}>
              Activity Log
            </span>
            <span className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>
              {entries.length} entries
            </span>
          </div>
          <div ref={scrollRef} className="overflow-y-auto p-2" style={{ maxHeight: 250 }}>
            {entries.length === 0 ? (
              <div className="text-xs text-center py-4" style={{ color: 'var(--text-secondary)' }}>
                No activity yet
              </div>
            ) : (
              entries.map((entry, i) => (
                <div key={i} className="flex gap-2 py-1 text-[11px]" style={{ color: 'var(--text-primary)' }}>
                  <span className="shrink-0 font-mono" style={{ color: 'var(--text-secondary)' }}>
                    {entry.time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                  <span className="break-all">{entry.text}</span>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
