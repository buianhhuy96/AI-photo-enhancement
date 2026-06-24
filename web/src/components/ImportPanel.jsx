import { useState, useEffect, useCallback } from 'react';
import { listDir, browseFolder } from '../api';

function FolderTree({ onSelectFolder, selectedPath }) {
  const [currentPath, setCurrentPath] = useState(null);
  const [parentPath, setParentPath] = useState(null);
  const [dirs, setDirs] = useState([]);
  const [pathInput, setPathInput] = useState('');

  const loadDir = useCallback(async (path) => {
    try {
      const data = await listDir(path);
      setCurrentPath(data.current);
      setParentPath(data.parent);
      setDirs(data.dirs);
      setPathInput(data.current);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => { loadDir('~'); }, [loadDir]);

  const handlePathSubmit = (e) => {
    if (e.key === 'Enter') loadDir(pathInput.trim());
  };

  return (
    <div className="flex flex-col h-full">
      {/* Path input */}
      <div className="p-2 border-b" style={{ borderColor: 'var(--border-color)' }}>
        <input
          type="text"
          value={pathInput}
          onChange={(e) => setPathInput(e.target.value)}
          onKeyDown={handlePathSubmit}
          className="w-full rounded px-2 py-1.5 text-xs outline-none"
          style={{
            background: 'var(--bg-card)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-color)',
          }}
          placeholder="/path/to/folder"
        />
      </div>

      {/* Folder list */}
      <div className="flex-1 overflow-y-auto">
        {/* Parent directory */}
        {parentPath && (
          <button
            onClick={() => loadDir(parentPath)}
            className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs hover:bg-white/5"
            style={{ color: 'var(--text-secondary)' }}
          >
            📁 ..
          </button>
        )}
        {/* Current directory (clickable to load images) */}
        {currentPath && (
          <button
            onClick={() => onSelectFolder(currentPath)}
            className="flex w-full items-center justify-between gap-2 px-3 py-1.5 text-left text-xs hover:bg-white/5"
            style={{
              color: selectedPath === currentPath ? '#fff' : 'var(--text-primary)',
              background: selectedPath === currentPath ? 'var(--accent)' : 'transparent',
            }}
          >
            <span className="truncate">📂 {currentPath.split('/').pop() || currentPath}</span>
            <span style={{ color: selectedPath === currentPath ? '#ddd' : 'var(--text-secondary)' }}>
              (this folder)
            </span>
          </button>
        )}
        {/* Subdirectories */}
        {dirs.map((dir) => (
          <div key={dir.path} className="flex items-center">
            <button
              onClick={() => onSelectFolder(dir.path)}
              className="flex flex-1 items-center justify-between gap-1 px-3 py-1.5 text-left text-xs hover:bg-white/5 truncate"
              style={{
                color: selectedPath === dir.path ? '#fff' : 'var(--text-primary)',
                background: selectedPath === dir.path ? 'var(--accent)' : 'transparent',
              }}
            >
              <span className="truncate">📁 {dir.name}</span>
              {dir.imageCount > 0 && (
                <span className="shrink-0 text-[10px]" style={{ color: selectedPath === dir.path ? '#ddd' : 'var(--text-secondary)' }}>
                  {dir.imageCount}
                </span>
              )}
            </button>
            <button
              onClick={() => loadDir(dir.path)}
              className="shrink-0 px-2 py-1.5 text-[10px] hover:bg-white/10"
              style={{ color: 'var(--text-secondary)' }}
              title="Open folder"
            >
              ▶
            </button>
          </div>
        ))}
        {dirs.length === 0 && (
          <div className="px-3 py-4 text-xs text-center" style={{ color: 'var(--text-secondary)' }}>
            No subfolders
          </div>
        )}
      </div>
    </div>
  );
}

function ImageGrid({ images, checked, onToggle }) {
  if (images.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center" style={{ background: '#111' }}>
        <div className="text-center">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
               style={{ color: 'var(--text-secondary)', margin: '0 auto 12px' }}>
            <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
          </svg>
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Select a folder to view images
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-3" style={{ background: '#111' }}>
      <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))' }}>
        {images.map((img, i) => {
          const isChecked = checked.has(i);
          return (
            <div
              key={i}
              onClick={() => onToggle(i)}
              className="relative cursor-pointer rounded overflow-hidden"
              style={{
                aspectRatio: '4/3',
                border: isChecked ? '2px solid var(--accent)' : '2px solid transparent',
                opacity: isChecked ? 1 : 0.45,
                transition: 'all 0.15s',
              }}
            >
              <img
                src={img.url}
                alt={img.name}
                className="w-full h-full"
                style={{ objectFit: 'cover', display: 'block' }}
                draggable={false}
              />
              <div
                className="absolute top-1.5 left-1.5 flex items-center justify-center rounded"
                style={{
                  width: 18, height: 18,
                  background: isChecked ? 'var(--accent)' : 'rgba(0,0,0,0.5)',
                  border: isChecked ? 'none' : '1.5px solid rgba(255,255,255,0.5)',
                }}
              >
                {isChecked && (
                  <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
                    <path d="M2 6l3 3 5-6" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
              </div>
              <div
                className="absolute bottom-0 left-0 right-0 px-1 py-0.5 text-center truncate text-[10px]"
                style={{ background: 'rgba(0,0,0,0.6)', color: '#ccc' }}
              >
                {img.name}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function ImportPanel({ onImportFiles, processing }) {
  const [images, setImages] = useState([]);
  const [checked, setChecked] = useState(new Set());
  const [browsing, setBrowsing] = useState(false);
  const [selectedFolder, setSelectedFolder] = useState('');

  const handleSelectFolder = useCallback(async (folderPath) => {
    setSelectedFolder(folderPath);
    setBrowsing(true);
    try {
      const data = await browseFolder(folderPath);
      setImages(data.images);
      setChecked(new Set(data.images.map((_, i) => i)));
    } catch {
      setImages([]);
      setChecked(new Set());
    } finally {
      setBrowsing(false);
    }
  }, []);

  const toggleCheck = (idx) => {
    setChecked((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const selectAll = () => setChecked(new Set(images.map((_, i) => i)));
  const selectNone = () => setChecked(new Set());

  const handleImport = () => {
    const paths = images.filter((_, i) => checked.has(i)).map((img) => img.path);
    if (paths.length) onImportFiles(paths);
  };

  return (
    <div className="flex flex-1 min-h-0">
      {/* Left sidebar — Folder tree */}
      <div
        className="w-56 shrink-0 overflow-hidden flex flex-col border-r"
        style={{ background: 'var(--bg-sidebar)', borderColor: 'var(--border-color)' }}
      >
        <div className="px-3 py-2 text-xs font-semibold uppercase tracking-wider border-b"
             style={{ color: 'var(--text-secondary)', borderColor: 'var(--border-color)' }}>
          Folders
        </div>
        <FolderTree onSelectFolder={handleSelectFolder} selectedPath={selectedFolder} />
      </div>

      {/* Center — Image grid */}
      <div className="flex flex-1 flex-col min-w-0">
        {browsing ? (
          <div className="flex flex-1 items-center justify-center" style={{ background: '#111' }}>
            <div className="flex flex-col items-center gap-2">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-t-transparent"
                   style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Loading...</span>
            </div>
          </div>
        ) : (
          <ImageGrid images={images} checked={checked} onToggle={toggleCheck} />
        )}
      </div>

      {/* Right sidebar — Import controls */}
      <div
        className="w-56 shrink-0 flex flex-col border-l"
        style={{ background: 'var(--bg-sidebar)', borderColor: 'var(--border-color)' }}
      >
        <div className="px-3 py-2 text-xs font-semibold uppercase tracking-wider border-b"
             style={{ color: 'var(--text-secondary)', borderColor: 'var(--border-color)' }}>
          Import
        </div>
        <div className="flex flex-col gap-2 p-3">
          {/* Import button */}
          <button
            onClick={handleImport}
            disabled={processing || checked.size === 0}
            className="w-full rounded-lg py-2 text-sm font-semibold text-white transition-colors disabled:opacity-50"
            style={{ background: checked.size > 0 ? 'var(--accent)' : '#475569' }}
          >
            {processing
              ? 'Importing...'
              : checked.size > 0
                ? `Import ${checked.size} Photo${checked.size !== 1 ? 's' : ''}`
                : 'Import'
            }
          </button>

          {/* Selection controls */}
          {images.length > 0 && (
            <>
              <div className="flex gap-1">
                <button onClick={selectAll}
                  className="flex-1 rounded py-1.5 text-xs font-medium transition-colors hover:bg-white/5"
                  style={{ color: 'var(--text-primary)', background: 'var(--bg-card)' }}>
                  Select All
                </button>
                <button onClick={selectNone}
                  className="flex-1 rounded py-1.5 text-xs font-medium transition-colors hover:bg-white/5"
                  style={{ color: 'var(--text-primary)', background: 'var(--bg-card)' }}>
                  Select None
                </button>
              </div>

              {/* Info */}
              <div className="mt-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
                <div className="flex justify-between mb-1">
                  <span>Total:</span>
                  <span>{images.length} photos</span>
                </div>
                <div className="flex justify-between">
                  <span>Selected:</span>
                  <span>{checked.size} photos</span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Current folder info */}
        {selectedFolder && (
          <div className="mt-auto border-t p-3" style={{ borderColor: 'var(--border-color)' }}>
            <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-secondary)' }}>
              Current Folder
            </div>
            <div className="text-xs truncate" style={{ color: 'var(--text-primary)' }} title={selectedFolder}>
              {selectedFolder}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
