import { useState, useCallback, useRef, useEffect } from 'react';
import Viewer from './components/Viewer';
import Filmstrip from './components/Filmstrip';
import ControlPanel from './components/ControlPanel';
import ImportPanel from './components/ImportPanel';
import StatusLog from './components/StatusLog';
import SettingsPanel from './components/SettingsPanel';
import { restoreSession, importFiles, uploadFiles, getImage, processImage, blendImage, exportAll, removeImages, deleteImages, runPipeline } from './api';

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [thumbnails, setThumbnails] = useState([]);
  const [imageCount, setImageCount] = useState(0);

  const [selectedIndex, setSelectedIndex] = useState(0);
  const [currentImage, setCurrentImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [isProcessed, setIsProcessed] = useState(false);
  const [originalSize, setOriginalSize] = useState({ width: 0, height: 0 });

  const [activeTab, setActiveTab] = useState('import');
  const [processing, setProcessing] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');
  const [logEntries, setLogEntries] = useState([]);
  const [showOriginal, setShowOriginal] = useState(false);

  // Multi-select for filmstrip
  const [selectedSet, setSelectedSet] = useState(new Set());

  const [quality, setQuality] = useState(0);
  const [strength, setStrength] = useState(0.5);
  const [use4bit, setUse4bit] = useState(true);
  const [outputFormat, setOutputFormat] = useState('png');
  const [jpgQuality, setJpgQuality] = useState(95);
  const [exportProgress, setExportProgress] = useState(null); // null = not exporting, 0-1 = progress

  const EMPTY_SET = useRef(new Set()).current;
  const blendTimer = useRef(null);
  const imageCache = useRef(new Map()); // index → full-res URL

  const log = useCallback((msg) => {
    setStatusMsg(msg);
    if (msg) setLogEntries((prev) => [...prev, { time: new Date(), text: msg }]);
  }, []);

  // Restore previous session from .cache on mount
  useEffect(() => {
    (async () => {
      try {
        const data = await restoreSession();
        if (data) {
          setSessionId(data.session_id);
          setThumbnails(data.thumbnails);
          setImageCount(data.count);
          setActiveTab('develop');
          // Load first image
          const img = await getImage(data.session_id, 0);
          imageCache.current.set(0, img);
          setCurrentImage(img.url);
          if (img.width) setOriginalSize({ width: img.width, height: img.height });
          setLogEntries((prev) => [...prev, { time: new Date(), text: `Restored ${data.count} image(s) from cache` }]);
        }
      } catch {
        // no cache or server down — ignore
      }
    })();
  }, []);

  const handleImportFiles = useCallback(async (filePaths) => {
    setProcessing(true);
    log('Importing...');
    try {
      const data = await importFiles(filePaths, sessionId);
      setSessionId(data.session_id);
      setThumbnails(data.thumbnails);
      setImageCount(data.count);
      setSelectedSet(EMPTY_SET);
      imageCache.current.clear();

      if (data.added === 0) {
        log(`All images already in library (${data.count} total)`);
        setActiveTab('develop');
        return;
      }

      log(`Added ${data.added} image(s) — ${data.count} total in library`);
      const newIndex = data.count - data.added;
      setSelectedIndex(newIndex);
      const img = await getImage(data.session_id, newIndex);
      imageCache.current.set(newIndex, img);
      setCurrentImage(img.url);
      if (img.width) setOriginalSize({ width: img.width, height: img.height });
      setResultImage(null);
      setIsProcessed(false);
      setActiveTab('develop');
    } catch (e) {
      log(`Import error: ${e.message}`);
    } finally {
      setProcessing(false);
    }
  }, [sessionId]);

  const handleUploadFiles = useCallback(async (fileList) => {
    setProcessing(true);
    log('Uploading...');
    try {
      const data = await uploadFiles(Array.from(fileList), sessionId);
      setSessionId(data.session_id);
      setThumbnails(data.thumbnails);
      setImageCount(data.count);
      setSelectedSet(EMPTY_SET);
      imageCache.current.clear();

      if (data.added === 0) {
        log(`All images already in library (${data.count} total)`);
        setActiveTab('develop');
        return;
      }

      log(`Uploaded ${data.added} image(s) — ${data.count} total in library`);
      const newIndex = data.count - data.added;
      setSelectedIndex(newIndex);
      const img = await getImage(data.session_id, newIndex);
      imageCache.current.set(newIndex, img);
      setCurrentImage(img.url);
      if (img.width) setOriginalSize({ width: img.width, height: img.height });
      setResultImage(null);
      setIsProcessed(false);
      setActiveTab('develop');
    } catch (e) {
      log(`Upload error: ${e.message}`);
    } finally {
      setProcessing(false);
    }
  }, [sessionId]);

  const handleSelectImage = useCallback(async (index) => {
    if (!sessionId) return;
    setSelectedIndex(index);
    setSelectedSet(EMPTY_SET);
    setResultImage(null);
    setIsProcessed(false);
    log(`Image ${index + 1} of ${imageCount}`);

    // Show cached full-res immediately, or thumbnail as placeholder
    const cached = imageCache.current.get(index);
    if (cached) {
      setCurrentImage(cached.url);
      if (cached.width) setOriginalSize({ width: cached.width, height: cached.height });
      return;
    }
    // Instant: show thumbnail while loading full-res
    if (thumbnails[index]) setCurrentImage(thumbnails[index].url);

    try {
      const img = await getImage(sessionId, index);
      imageCache.current.set(index, img);
      if (img.width) setOriginalSize({ width: img.width, height: img.height });
      // Only update if still on same image (user may have clicked elsewhere)
      setSelectedIndex((cur) => {
        if (cur === index) setCurrentImage(img.url);
        return cur;
      });
    } catch (e) {
      log(`Error: ${e.message}`);
    }
  }, [sessionId, imageCount, thumbnails]);

  const handleMultiSelect = useCallback((index, mode) => {
    setSelectedSet((prev) => {
      const next = new Set(prev);
      if (mode === 'toggle') {
        if (next.has(index)) next.delete(index);
        else next.add(index);
      } else if (mode === 'range') {
        const start = Math.min(selectedIndex, index);
        const end = Math.max(selectedIndex, index);
        for (let i = start; i <= end; i++) next.add(i);
      }
      return next;
    });
  }, [selectedIndex]);

  const handleRemoveFromLibrary = useCallback(async () => {
    if (!sessionId || selectedSet.size === 0) return;
    setProcessing(true);
    try {
      const indices = Array.from(selectedSet).sort((a, b) => a - b);
      const data = await removeImages(sessionId, indices);
      setThumbnails(data.thumbnails);
      setImageCount(data.count);
      setSelectedSet(EMPTY_SET);
      imageCache.current.clear();
      // Reset viewer
      if (data.count > 0) {
        const newIdx = Math.min(selectedIndex, data.count - 1);
        setSelectedIndex(newIdx);
        const img = await getImage(sessionId, newIdx);
        setCurrentImage(img.url);
      } else {
        setSelectedIndex(0);
        setCurrentImage(null);
      }
      setResultImage(null);
      setIsProcessed(false);
      log(`Removed ${indices.length} image(s). ${data.count} remaining.`);
    } catch (e) {
      log(`Error: ${e.message}`);
    } finally {
      setProcessing(false);
    }
  }, [sessionId, selectedSet, selectedIndex]);

  const handleDeleteFromDisk = useCallback(async () => {
    if (!sessionId || selectedSet.size === 0) return;
    if (!confirm(`Delete ${selectedSet.size} image(s) from disk? This cannot be undone.`)) return;
    setProcessing(true);
    try {
      const indices = Array.from(selectedSet).sort((a, b) => a - b);
      const data = await deleteImages(sessionId, indices);
      setThumbnails(data.thumbnails);
      setImageCount(data.count);
      setSelectedSet(EMPTY_SET);
      imageCache.current.clear();
      if (data.count > 0) {
        const newIdx = Math.min(selectedIndex, data.count - 1);
        setSelectedIndex(newIdx);
        const img = await getImage(sessionId, newIdx);
        setCurrentImage(img.url);
      } else {
        setSelectedIndex(0);
        setCurrentImage(null);
      }
      setResultImage(null);
      setIsProcessed(false);
      log(`Deleted ${indices.length} image(s). ${data.count} remaining.`);
    } catch (e) {
      log(`Error: ${e.message}`);
    } finally {
      setProcessing(false);
    }
  }, [sessionId, selectedSet, selectedIndex]);

  // ── Pipeline handler ──
  const pipelineTimer = useRef(null);
  const handlePipeline = useCallback((steps) => {
    if (!sessionId) return;

    // If no steps are active, revert to original
    if (!steps || steps.length === 0) {
      const cached = imageCache.current.get(selectedIndex);
      if (cached) setCurrentImage(cached.url);
      setResultImage(null);
      log('Reverted to original');
      return;
    }

    // Debounce pipeline calls (slider dragging)
    if (pipelineTimer.current) clearTimeout(pipelineTimer.current);
    pipelineTimer.current = setTimeout(async () => {
      setProcessing(true);
      log('Processing pipeline...');
      try {
        const data = await runPipeline(sessionId, selectedIndex, steps);
        if (data.url) {
          setResultImage(data.url);
        }
        log('Pipeline complete');
      } catch (e) {
        log(`Pipeline error: ${e.message}`);
      } finally {
        setProcessing(false);
      }
    }, 150);
  }, [sessionId, selectedIndex]);

  const handleExport = useCallback(async () => {
    if (!sessionId) return;
    setProcessing(true);
    setExportProgress(0);
    log('Exporting...');
    try {
      const data = await exportAll(sessionId, {
        quality, strength, use4bit, outputFormat, jpgQuality,
      }, (progress, message) => {
        setExportProgress(progress);
        log(message);
      });
      log(data.status);
    } catch (e) {
      log(`Export error: ${e.message}`);
    } finally {
      setProcessing(false);
      setExportProgress(null);
    }
  }, [sessionId, quality, strength, use4bit, outputFormat, jpgQuality]);

  const displayImage = showOriginal ? currentImage : (resultImage || currentImage);

  return (
    <div className="flex h-screen w-screen flex-col">
      {/* ─── Top bar ─── */}
      <header
        className="flex h-10 shrink-0 items-center justify-between border-b px-4"
        style={{ background: 'var(--bg-sidebar)', borderColor: 'var(--border-color)' }}
      >
        <div className="flex items-center gap-4">
          <span className="text-sm font-bold tracking-wide" style={{ color: 'var(--accent)' }}>
            ✦ WindowSeat
          </span>
          <nav className="flex gap-1">
            {[
              { id: 'import', label: 'Library' },
              { id: 'develop', label: 'Develop' },
              { id: 'settings', label: 'Settings' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className="rounded px-3 py-1 text-xs font-medium transition-colors"
                style={{
                  background: activeTab === tab.id ? 'var(--accent)' : 'transparent',
                  color: activeTab === tab.id ? '#fff' : 'var(--text-secondary)',
                }}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
        <StatusLog entries={logEntries} />
      </header>

      {/* ─── Main area ─── */}
      <div className="flex flex-1 min-h-0">
        {/* Right sidebar — only in Develop */}
        {activeTab === 'develop' && (
          <div
            className="order-2 w-72 shrink-0 overflow-y-auto border-l flex flex-col"
            style={{ background: 'var(--bg-sidebar)', borderColor: 'var(--border-color)' }}
          >
            <ControlPanel
              quality={quality} onQualityChange={setQuality}
              strength={strength} onStrengthChange={setStrength}
              use4bit={use4bit} onUse4bitChange={setUse4bit}
              outputFormat={outputFormat} onOutputFormatChange={setOutputFormat}
              jpgQuality={jpgQuality} onJpgQualityChange={setJpgQuality}
              onPipeline={handlePipeline}
              onExport={handleExport}
              processing={processing}
              exportProgress={exportProgress}
              originalSize={originalSize}
            />
          </div>
        )}

        {/* Center content */}
        <div className="order-1 flex flex-1 flex-col min-w-0">
          {activeTab === 'import' ? (
            <ImportPanel onImportFiles={handleImportFiles} onUploadFiles={handleUploadFiles} processing={processing} />
          ) : activeTab === 'settings' ? (
            <SettingsPanel />
          ) : (
            <Viewer image={displayImage} processing={processing} />
          )}
        </div>
      </div>

      {/* ─── Filmstrip ─── */}
      {thumbnails.length > 0 && (
        <div className="shrink-0" style={{ background: 'var(--bg-sidebar)' }}>
          <Filmstrip
            thumbnails={thumbnails}
            selectedIndex={selectedIndex}
            selectedSet={selectedSet}
            onSelect={handleSelectImage}
            onMultiSelect={handleMultiSelect}
          />
          {/* Action bar below filmstrip when multi-selected */}
          {selectedSet.size > 0 && (
            <div className="flex items-center justify-center gap-3 border-t px-4 py-2"
                 style={{ borderColor: 'var(--border-color)' }}>
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                {selectedSet.size} selected
              </span>
              <button
                onClick={handleRemoveFromLibrary}
                disabled={processing}
                className="rounded px-3 py-1 text-xs font-medium transition-colors disabled:opacity-50"
                style={{ background: 'var(--bg-card)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
              >
                Remove from Library
              </button>
              <button
                onClick={handleDeleteFromDisk}
                disabled={processing}
                className="rounded px-3 py-1 text-xs font-medium transition-colors disabled:opacity-50"
                style={{ background: '#7f1d1d', color: '#fca5a5', border: '1px solid #991b1b' }}
              >
                Delete from Disk
              </button>
              <button
                onClick={() => setSelectedSet(new Set())}
                className="rounded px-3 py-1 text-xs font-medium transition-colors"
                style={{ color: 'var(--text-secondary)' }}
              >
                Clear Selection
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
