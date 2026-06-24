import { memo } from 'react';

export default memo(function Viewer({ image, processing }) {
  return (
    <div className="relative flex-1 overflow-hidden" style={{ background: '#111' }}>
      {processing && !image && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-t-transparent"
                 style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Loading...</span>
          </div>
        </div>
      )}
      {image && (
        <img
          src={image}
          alt="Preview"
          className="absolute inset-0 w-full h-full object-contain select-none"
          draggable={false}
        />
      )}
      {!image && !processing && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Import photos to get started
          </span>
        </div>
      )}
    </div>
  );
});
