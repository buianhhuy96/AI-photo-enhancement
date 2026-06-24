import { memo } from 'react';

export default memo(function Filmstrip({ thumbnails, selectedIndex, selectedSet = new Set(), onSelect, onMultiSelect }) {
  const handleClick = (i, e) => {
    if (e.ctrlKey || e.metaKey) {
      // Toggle individual selection
      onMultiSelect(i, 'toggle');
    } else if (e.shiftKey && selectedIndex != null) {
      // Range selection
      onMultiSelect(i, 'range');
    } else {
      // Single select
      onSelect(i);
    }
  };

  return (
    <div
      className="shrink-0 border-t flex items-center overflow-x-auto"
      style={{
        height: 'var(--filmstrip-height)',
        background: 'var(--bg-sidebar)',
        borderColor: 'var(--border-color)',
      }}
    >
      {/* Centered filmstrip container */}
      <div className="flex items-center gap-0.5 mx-auto px-2 py-1">
        {thumbnails.map((thumb, i) => {
          const isActive = i === selectedIndex;
          const isInSet = selectedSet.has(i);
          let borderColor = 'transparent';
          if (isActive) borderColor = '#b0b0b0';
          else if (isInSet) borderColor = '#f59e0b';
          const h = 44;
          const aspect = thumb.aspect || 1.333;
          const w = Math.round(h * aspect);

          return (
            <button
              key={i}
              onClick={(e) => handleClick(i, e)}
              className="shrink-0 overflow-hidden transition-all duration-150"
              style={{
                width: w,
                height: h,
                border: `2px solid ${borderColor}`,
                opacity: isActive || isInSet ? 1 : 0.6,
                outline: 'none',
                padding: 0,
                background: '#000',
              }}
              title={`${thumb.name}${isInSet ? ' (selected)' : ''}`}
            >
              <img
                src={thumb.url}
                alt={thumb.name}
                className="w-full h-full"
                style={{ objectFit: 'cover', display: 'block' }}
                draggable={false}
              />
            </button>
          );
        })}
      </div>
    </div>
  );
});
