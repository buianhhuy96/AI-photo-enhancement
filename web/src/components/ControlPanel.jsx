import { useState, useRef, useCallback, memo, useEffect } from 'react';

const QUALITY_LABELS = { '-1': 'Fast', '0': 'Balanced', '1': 'High', '2': 'Very High', '3': 'Ultra' };

/**
 * Scrubable value: click-and-drag on the number to change it.
 * horizontal = true: drag left/right. false: drag up/down (inverted: up = increase).
 */
function DragValue({ value, min, max, step, onChange, displayValue, horizontal = true, className = '', style = {} }) {
  const dragRef = useRef(null);

  const handlePointerDown = useCallback((e) => {
    e.preventDefault();
    const startPos = horizontal ? e.clientX : e.clientY;
    const startVal = value;
    const range = max - min;
    // Pixels per full range: more pixels = more precise
    const pxPerRange = horizontal ? 600 : 400;

    const handleMove = (ev) => {
      const delta = horizontal
        ? (ev.clientX - startPos)
        : -(ev.clientY - startPos); // up = increase
      const newVal = Math.min(max, Math.max(min,
        startVal + (delta / pxPerRange) * range
      ));
      // Snap to step
      const snapped = Math.round(newVal / step) * step;
      onChange(parseFloat(snapped.toFixed(10)));
    };

    const handleUp = () => {
      document.removeEventListener('pointermove', handleMove);
      document.removeEventListener('pointerup', handleUp);
      document.body.style.cursor = '';
    };

    document.addEventListener('pointermove', handleMove);
    document.addEventListener('pointerup', handleUp);
    document.body.style.cursor = horizontal ? 'ew-resize' : 'ns-resize';
  }, [value, min, max, step, onChange, horizontal]);

  return (
    <span
      ref={dragRef}
      onPointerDown={handlePointerDown}
      className={`select-none ${className}`}
      style={{ cursor: horizontal ? 'ew-resize' : 'ns-resize', ...style }}
    >
      {displayValue ?? value}
    </span>
  );
}

function ToolSection({ title, children, defaultOpen = true, enabled, onToggle }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b" style={{ borderColor: 'var(--border-color)' }}>
      <div className="flex items-center px-1 py-px" style={{ background: 'var(--bg-section-header)' }}>
        {/* Left: toggle switch (if provided) */}
        {onToggle && (
          <button
            onClick={onToggle}
            className="shrink-0 w-5 h-2.5 mr-2 relative transition-colors"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '2px' }}
          >
            <span
              className="absolute w-2 h-1.5 transition-all"
              style={{
                background: enabled ? '#fff' : 'var(--text-secondary)',
                borderRadius: '1px',
                top: '50%',
                transform: 'translateY(-50%)',
                left: enabled ? '10px' : '2px',
              }}
            />
          </button>
        )}
        {/* Right-aligned title + collapse toggle */}
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-1 ml-auto text-right transition-colors hover:bg-white/5 px-1"
        >
          <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-primary)' }}>
            {title}
          </span>
          <span
            className="text-[8px] transition-transform duration-150"
            style={{ color: 'var(--text-secondary)', transform: open ? 'rotate(0deg)' : 'rotate(-90deg)' }}
          >
            ▼
          </span>
        </button>
      </div>
      {open && <div className="px-2 pt-1.5 pb-2">{children}</div>}
    </div>
  );
}

function SliderRow({ label, value, min, max, step, onChange, displayValue }) {
  return (
    <div className="flex items-center gap-1 mb-1">
      <span className="text-[10px] shrink-0 w-12 text-right" style={{ color: 'var(--text-primary)' }}>{label}</span>
      <input
        type="range"
        className="flex-1 h-1 min-w-0 slider-sm"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <DragValue
        value={value} min={min} max={max} step={step} onChange={onChange}
        displayValue={displayValue}
        className="text-[10px] shrink-0 w-14 text-center"
        style={{ color: 'var(--text-secondary)' }}
      />
    </div>
  );
}

export default memo(function ControlPanel({
  quality, onQualityChange,
  strength, onStrengthChange,
  use4bit, onUse4bitChange,
  outputFormat, onOutputFormatChange,
  jpgQuality, onJpgQualityChange,
  onPipeline,
  onExport,
  processing, exportProgress,
  originalSize,
}) {
  const [resizeEnabled, setResizeEnabled] = useState(false);
  const [resizeW, setResizeW] = useState(originalSize?.width || 0);
  const [resizeH, setResizeH] = useState(originalSize?.height || 0);
  const [lockRatio, setLockRatio] = useState(true);
  const [denoiseStrength, setDenoiseStrength] = useState(0.5);
  const [denoiseLevel, setDenoiseLevel] = useState(0);
  const [deblurLevel, setDeblurLevel] = useState(0);
  const [restoreEnabled, setRestoreEnabled] = useState(false);
  const [reflectionEnabled, setReflectionEnabled] = useState(false);
  const [skinRetouchEnabled, setSkinRetouchEnabled] = useState(false);
  const [skinRetouchStrength, setSkinRetouchStrength] = useState(0.5);
  const [skinDetailSize, setSkinDetailSize] = useState(0.05);
  const [skinToneEnabled, setSkinToneEnabled] = useState(false);
  const [skinToneStrength, setSkinToneStrength] = useState(0.5);
  const [pipelineOrder, setPipelineOrder] = useState([]);

  const activateStep = (key) => {
    setPipelineOrder((prev) => prev.includes(key) ? prev : [...prev, key]);
  };
  const deactivateStep = (key) => {
    setPipelineOrder((prev) => prev.filter((k) => k !== key));
  };
  const aspectRatio = originalSize?.width && originalSize?.height
    ? originalSize.width / originalSize.height : 1;

  // Build and run pipeline from current state
  const runPipeline = useCallback((overrides = {}) => {
    const order = overrides.order ?? pipelineOrder;
    const rEnabled = overrides.resizeEnabled ?? resizeEnabled;
    const refEnabled = overrides.reflectionEnabled ?? reflectionEnabled;
    const rstEnabled = overrides.restoreEnabled ?? restoreEnabled;
    const skinEnabled = overrides.skinRetouchEnabled ?? skinRetouchEnabled;
    const toneEnabled = overrides.skinToneEnabled ?? skinToneEnabled;
    const w = overrides.resizeW ?? resizeW;
    const h = overrides.resizeH ?? resizeH;
    const dStr = overrides.denoiseStrength ?? denoiseStrength;
    const q = overrides.quality ?? quality;
    const s = overrides.strength ?? strength;
    const dn = overrides.denoiseLevel ?? denoiseLevel;
    const db = overrides.deblurLevel ?? deblurLevel;
    const skinStr = overrides.skinRetouchStrength ?? skinRetouchStrength;
    const skinDetail = overrides.skinDetailSize ?? skinDetailSize;
    const toneStr = overrides.skinToneStrength ?? skinToneStrength;

    const steps = [];
    for (const key of order) {
      if (key === 'resize' && rEnabled && w > 0 && h > 0) {
        steps.push({ name: 'resize', params: { width: w, height: h, denoise_strength: dStr } });
      } else if (key === 'reflection' && refEnabled) {
        steps.push({ name: 'reflection', params: { quality: q, strength: s, use_4bit: use4bit } });
      } else if (key === 'restore' && rstEnabled) {
        steps.push({ name: 'restore', params: { denoise: dn, deblur: db } });
      } else if (key === 'skin_retouch' && skinEnabled) {
        steps.push({ name: 'skin_retouch', params: { strength: skinStr, detail_size: skinDetail } });
      } else if (key === 'skin_tone' && toneEnabled) {
        steps.push({ name: 'skin_tone', params: { strength: toneStr } });
      }
    }
    onPipeline(steps);
  }, [pipelineOrder, resizeEnabled, reflectionEnabled, restoreEnabled, skinRetouchEnabled, skinToneEnabled,
      resizeW, resizeH, denoiseStrength, quality, strength, use4bit,
      denoiseLevel, deblurLevel, skinRetouchStrength, skinDetailSize, skinToneStrength, onPipeline]);

  // Reset dimensions when original size changes (image switch)
  useEffect(() => {
    if (originalSize?.width) {
      setResizeW(originalSize.width);
      setResizeH(originalSize.height);
      setResizeEnabled(false);
      setReflectionEnabled(false);
      setRestoreEnabled(false);
      setSkinRetouchEnabled(false);
      setSkinToneEnabled(false);
      setPipelineOrder([]);
    }
  }, [originalSize?.width, originalSize?.height]);

  const handleWidthChange = (val) => {
    const w = parseInt(val) || 0;
    setResizeW(w);
    const h = lockRatio && w > 0 ? Math.round(w / aspectRatio) : resizeH;
    if (lockRatio && w > 0) setResizeH(h);
    if (resizeEnabled && w > 0 && h > 0) {
      runPipeline({ resizeW: w, resizeH: h });
    }
  };

  const handleHeightChange = (val) => {
    const h = parseInt(val) || 0;
    setResizeH(h);
    const w = lockRatio && h > 0 ? Math.round(h * aspectRatio) : resizeW;
    if (lockRatio && h > 0) setResizeW(w);
    if (resizeEnabled && w > 0 && h > 0) {
      runPipeline({ resizeW: w, resizeH: h });
    }
  };

  return (
    <div className="flex flex-col h-full text-[11px]">
      {/* ═══ Resize ═══ */}
      <ToolSection title="Resize" defaultOpen={true}
        enabled={resizeEnabled}
        onToggle={() => {
          const next = !resizeEnabled;
          setResizeEnabled(next);
          const newOrder = next
            ? (pipelineOrder.includes('resize') ? pipelineOrder : [...pipelineOrder, 'resize'])
            : pipelineOrder.filter(k => k !== 'resize');
          setPipelineOrder(newOrder);
          runPipeline({ resizeEnabled: next, order: newOrder });
        }}
      >
        <div className="mt-0.5">
          <div className="flex items-center gap-1.5">
            <div className="flex-1">
              <label className="text-[9px] uppercase tracking-wider block mb-0.5"
                     style={{ color: 'var(--text-secondary)' }}>W</label>
              <DragValue
                value={resizeW} min={1} max={9999} step={1}
                onChange={(v) => handleWidthChange(String(Math.round(v)))}
                horizontal={false}
                displayValue={resizeW}
                className="block w-full px-1.5 py-1 text-[11px] outline-none text-center"
                style={{
                  background: 'var(--bg-card)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                }}
              />
            </div>
            <button
              onClick={() => setLockRatio((v) => !v)}
              className="mt-3 shrink-0 p-1 transition-colors"
              style={{
                background: lockRatio ? '#b0b0b0' : 'var(--bg-card)',
                color: lockRatio ? '#fff' : 'var(--text-secondary)',
              }}
              title={lockRatio ? 'Locked' : 'Unlocked'}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                   strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                {lockRatio ? (
                  <><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                  <path d="M7 11V7a5 5 0 0110 0v4"/></>
                ) : (
                  <><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                  <path d="M7 11V7a5 5 0 019.9-1"/></>
                )}
              </svg>
            </button>
            <div className="flex-1">
              <label className="text-[9px] uppercase tracking-wider block mb-0.5"
                     style={{ color: 'var(--text-secondary)' }}>H</label>
              <DragValue
                value={resizeH} min={1} max={9999} step={1}
                onChange={(v) => handleHeightChange(String(Math.round(v)))}
                horizontal={false}
                displayValue={resizeH}
                className="block w-full px-1.5 py-1 text-[11px] outline-none text-center"
                style={{
                  background: 'var(--bg-card)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                }}
              />
            </div>
          </div>
          <div className="mt-1 text-[9px]" style={{ color: 'var(--text-secondary)' }}>
            Original: {originalSize?.width || '?'} × {originalSize?.height || '?'}
          </div>
          <div className="mt-1.5">
            <SliderRow
              label="Denoise"
              value={denoiseStrength}
              min={0} max={1} step={0.05}
              onChange={(v) => {
                setDenoiseStrength(v);
                if (resizeEnabled) runPipeline({ denoiseStrength: v });
              }}
              displayValue={`${Math.round(denoiseStrength * 100)}%`}
            />
          </div>
        </div>
      </ToolSection>

      {/* ═══ Reflection Removal ═══ */}
      <ToolSection title="Reflection Removal" defaultOpen={true}
        enabled={reflectionEnabled}
        onToggle={() => {
          const next = !reflectionEnabled;
          setReflectionEnabled(next);
          const newOrder = next
            ? (pipelineOrder.includes('reflection') ? pipelineOrder : [...pipelineOrder, 'reflection'])
            : pipelineOrder.filter(k => k !== 'reflection');
          setPipelineOrder(newOrder);
          runPipeline({ reflectionEnabled: next, order: newOrder });
        }}
      >
        <SliderRow
          label="Quality"
          value={quality}
          min={-1} max={3} step={1}
          onChange={(v) => {
            onQualityChange(Math.round(v));
            if (reflectionEnabled) runPipeline({ quality: Math.round(v) });
          }}
          displayValue={QUALITY_LABELS[String(quality)]}
        />
        <SliderRow
          label="Strength"
          value={strength}
          min={0} max={1} step={0.01}
          onChange={(v) => {
            onStrengthChange(v);
            if (reflectionEnabled) runPipeline({ strength: v });
          }}
          displayValue={`${Math.round(strength * 100)}%`}
        />
      </ToolSection>

      {/* ═══ Skin Retouch ═══ */}
      <ToolSection title="Skin Retouch" defaultOpen={true}
        enabled={skinRetouchEnabled}
        onToggle={() => {
          const next = !skinRetouchEnabled;
          setSkinRetouchEnabled(next);
          const newOrder = next
            ? (pipelineOrder.includes('skin_retouch') ? pipelineOrder : [...pipelineOrder, 'skin_retouch'])
            : pipelineOrder.filter(k => k !== 'skin_retouch');
          setPipelineOrder(newOrder);
          runPipeline({ skinRetouchEnabled: next, order: newOrder });
        }}
      >
        <SliderRow
          label="Blemish"
          value={skinRetouchStrength}
          min={0} max={1} step={0.05}
          onChange={(v) => {
            setSkinRetouchStrength(v);
            if (skinRetouchEnabled) runPipeline({ skinRetouchStrength: v });
          }}
          displayValue={`${Math.round(skinRetouchStrength * 100)}%`}
        />
        <SliderRow
          label="Detail Size"
          value={skinDetailSize}
          min={0.02} max={0.15} step={0.01}
          onChange={(v) => {
            setSkinDetailSize(v);
            if (skinRetouchEnabled) runPipeline({ skinDetailSize: v });
          }}
          displayValue={`${(skinDetailSize * 100).toFixed(1)}%`}
        />
      </ToolSection>

      {/* ═══ Skin Tone ═══ */}
      <ToolSection title="Skin Tone" defaultOpen={true}
        enabled={skinToneEnabled}
        onToggle={() => {
          const next = !skinToneEnabled;
          setSkinToneEnabled(next);
          const newOrder = next
            ? (pipelineOrder.includes('skin_tone') ? pipelineOrder : [...pipelineOrder, 'skin_tone'])
            : pipelineOrder.filter(k => k !== 'skin_tone');
          setPipelineOrder(newOrder);
          runPipeline({ skinToneEnabled: next, order: newOrder });
        }}
      >
        <SliderRow
          label="Evenness"
          value={skinToneStrength}
          min={0} max={1} step={0.05}
          onChange={(v) => {
            setSkinToneStrength(v);
            if (skinToneEnabled) runPipeline({ skinToneStrength: v });
          }}
          displayValue={`${Math.round(skinToneStrength * 100)}%`}
        />
      </ToolSection>

      {/* ═══ Denoise & Deblur ═══ */}
      <ToolSection title="Denoise / Deblur" defaultOpen={true}
        enabled={restoreEnabled}
        onToggle={() => {
          const next = !restoreEnabled;
          setRestoreEnabled(next);
          const newOrder = next
            ? (pipelineOrder.includes('restore') ? pipelineOrder : [...pipelineOrder, 'restore'])
            : pipelineOrder.filter(k => k !== 'restore');
          setPipelineOrder(newOrder);
          runPipeline({ restoreEnabled: next, order: newOrder });
        }}
      >
        <SliderRow
          label="Denoise"
          value={denoiseLevel}
          min={0} max={1} step={0.05}
          onChange={(v) => {
            setDenoiseLevel(v);
            if (restoreEnabled) runPipeline({ denoiseLevel: v });
          }}
          displayValue={`${Math.round(denoiseLevel * 100)}%`}
        />
        <SliderRow
          label="Deblur"
          value={deblurLevel}
          min={0} max={1} step={0.05}
          onChange={(v) => {
            setDeblurLevel(v);
            if (restoreEnabled) runPipeline({ deblurLevel: v });
          }}
          displayValue={`${Math.round(deblurLevel * 100)}%`}
        />
      </ToolSection>

      {/* Export & Output settings — always visible at bottom */}
      <div className="mt-auto px-3 py-2 border-t" style={{ borderColor: 'var(--border-color)' }}>

        {/* Pipeline order cards — reorderable */}
        {pipelineOrder.length > 0 && (
          <div className="mb-2">
            <div className="text-[9px] font-medium uppercase tracking-wider mb-1" style={{ color: 'var(--text-secondary)' }}>
              Processing Order
            </div>
            <div className="flex flex-col gap-0.5">
              {(() => {
                const stepMap = {
                  resize: { label: 'Resize', color: '#6366f1' },
                  restore: { label: 'Denoise / Deblur', color: '#10b981' },
                  reflection: { label: 'Reflection', color: '#f59e0b' },
                  skin_retouch: { label: 'Skin Retouch', color: '#ec4899' },
                  skin_tone: { label: 'Skin Tone', color: '#a855f7' },
                };
                const activeSteps = pipelineOrder
                  .map((key) => stepMap[key] ? { ...stepMap[key], key } : null)
                  .filter(Boolean);
                const moveStep = (key, dir) => {
                  setPipelineOrder((prev) => {
                    const arr = [...prev];
                    const idx = arr.indexOf(key);
                    const target = idx + dir;
                    if (target < 0 || target >= arr.length) return prev;
                    [arr[idx], arr[target]] = [arr[target], arr[idx]];
                    setTimeout(() => runPipeline({ order: arr }), 0);
                    return arr;
                  });
                };
                return activeSteps.map((step, i) => (
                  <div
                    key={step.key}
                    className="flex items-center gap-1.5 px-1.5 py-0.5 select-none"
                    style={{
                      background: step.color + '10',
                      border: `1px solid ${step.color}30`,
                    }}
                  >
                    <div className="flex flex-col leading-none">
                      <button
                        onClick={() => moveStep(step.key, -1)}
                        disabled={i === 0}
                        className="text-[8px] px-0.5 hover:bg-white/10 disabled:opacity-20"
                        style={{ color: 'var(--text-secondary)' }}
                      >▲</button>
                      <button
                        onClick={() => moveStep(step.key, 1)}
                        disabled={i === activeSteps.length - 1}
                        className="text-[8px] px-0.5 hover:bg-white/10 disabled:opacity-20"
                        style={{ color: 'var(--text-secondary)' }}
                      >▼</button>
                    </div>
                    <span className="text-[9px] font-mono" style={{ color: 'var(--text-secondary)' }}>{i + 1}.</span>
                    <span className="text-[10px] font-medium" style={{ color: step.color }}>
                      {step.label}
                    </span>
                  </div>
                ));
              })()}
            </div>
          </div>
        )}

        <label className="flex items-center gap-1.5 cursor-pointer mb-2">
          <input
            type="checkbox"
            checked={use4bit}
            onChange={(e) => onUse4bitChange(e.target.checked)}
            className="w-3 h-3"
          />
          <span style={{ color: 'var(--text-primary)' }}>
            4-bit quantization (less VRAM)
          </span>
        </label>

        <div className="text-[9px] font-medium uppercase tracking-wider mb-1" style={{ color: 'var(--text-secondary)' }}>
          Output Format
        </div>
        <div className="flex gap-0.5 mb-1.5">
          {['png', 'jpg', 'webp'].map((fmt) => (
            <button
              key={fmt}
              onClick={() => onOutputFormatChange(fmt)}
              className="flex-1 py-0.5 text-[10px] font-medium uppercase transition-colors"
              style={{
                background: outputFormat === fmt ? '#b0b0b0' : 'var(--bg-card)',
                color: outputFormat === fmt ? '#fff' : 'var(--text-secondary)',
              }}
            >
              {fmt}
            </button>
          ))}
        </div>
        {outputFormat === 'jpg' && (
          <SliderRow
            label="JPEG Quality"
            value={jpgQuality}
            min={50} max={100} step={1}
            onChange={(v) => onJpgQualityChange(Math.round(v))}
            displayValue={`${jpgQuality}%`}
          />
        )}

        {/* Progress bar */}
        {exportProgress != null && (
          <div className="mt-1.5 mb-1">
            <div className="flex items-center justify-between">
              <span className="text-[9px]" style={{ color: 'var(--text-secondary)' }}>Exporting...</span>
              <span className="text-[9px] font-mono" style={{ color: 'var(--text-secondary)' }}>
                {Math.round(exportProgress * 100)}%
              </span>
            </div>
            <div className="w-full h-1 overflow-hidden" style={{ background: 'var(--bg-card)' }}>
              <div
                className="h-full transition-all duration-200"
                style={{ width: `${exportProgress * 100}%`, background: 'var(--accent)' }}
              />
            </div>
          </div>
        )}

        <button
          onClick={onExport}
          disabled={processing}
          className="mt-1.5 w-full py-1.5 text-[11px] font-medium transition-colors disabled:opacity-50"
          style={{
            background: 'transparent',
            border: '1px solid var(--border-color)',
            color: 'var(--text-primary)',
          }}
        >
          {exportProgress != null ? `Exporting... ${Math.round(exportProgress * 100)}%` : '📁 Export All'}
        </button>
      </div>
    </div>
  );
});
