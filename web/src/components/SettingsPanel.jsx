import { useState, useEffect, useCallback } from 'react';
import { getSettingsStatus, downloadModel, installPackages, setHfToken } from '../api';

export default function SettingsPanel() {
  const [env, setEnv] = useState(null);
  const [models, setModels] = useState({});
  const [packages, setPackages] = useState({});
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(null);
  const [installing, setInstalling] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');
  const [hfToken, setHfTokenInput] = useState('');
  const [connected, setConnected] = useState(false);

  const loadStatus = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getSettingsStatus();
      setEnv(data.env);
      setModels(data.models || {});
      setPackages(data.packages || {});
      setConnected(true);
    } catch {
      setConnected(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadStatus(); }, [loadStatus]);

  const handleDownload = async (modelId) => {
    setDownloading(modelId);
    setStatusMsg('');
    try {
      const msg = await downloadModel(modelId, (_, message) => setStatusMsg(message));
      setStatusMsg(msg || 'Done');
      await loadStatus();
    } catch (e) {
      setStatusMsg(`Error: ${e.message}`);
    } finally {
      setDownloading(null);
    }
  };

  const handleDownloadAll = async () => {
    const missing = Object.entries(models).filter(([, m]) => !m.downloaded).map(([id]) => id);
    for (const id of missing) {
      await handleDownload(id);
    }
  };

  const handleInstallPackages = async () => {
    setInstalling(true);
    setStatusMsg('');
    try {
      const msg = await installPackages((_, message) => setStatusMsg(message));
      setStatusMsg(msg || 'Done');
      await loadStatus();
    } catch (e) {
      setStatusMsg(`Error: ${e.message}`);
    } finally {
      setInstalling(false);
    }
  };

  const handleSetToken = async () => {
    if (!hfToken.trim()) return;
    try {
      await setHfToken(hfToken.trim());
      setStatusMsg('Token saved');
      setHfTokenInput('');
      await loadStatus();
    } catch (e) {
      setStatusMsg(`Token error: ${e.message}`);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-1 items-center justify-center" style={{ background: '#111' }}>
        <div className="flex flex-col items-center gap-2">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-t-transparent"
               style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
          <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Loading settings...</span>
        </div>
      </div>
    );
  }

  const hasModels = Object.keys(models).length > 0;
  const hasPackages = Object.keys(packages).length > 0;
  const allModelsReady = hasModels && Object.values(models).every(m => m.downloaded);
  const allPackagesReady = hasPackages && Object.values(packages).every(p => p.installed);
  const missingPackages = Object.entries(packages).filter(([, p]) => !p.installed);
  const missingModels = Object.entries(models).filter(([, m]) => !m.downloaded);

  return (
    <div className="flex flex-1 overflow-y-auto p-6" style={{ background: '#111' }}>
      <div className="w-full max-w-2xl mx-auto space-y-5">

        {/* Connection warning */}
        {!connected && (
          <div className="p-3 text-xs" style={{ background: '#7f1d1d20', border: '1px solid #991b1b', color: '#fca5a5' }}>
            ⚠ Backend not connected. Start with: <code className="font-mono">python3 serve.py --mock</code>
          </div>
        )}

        {/* ═══ Setup Progress ═══ */}
        <section>
          <h2 className="text-sm font-semibold uppercase tracking-wider mb-3"
              style={{ color: 'var(--text-primary)' }}>
            Setup
          </h2>
          <div className="p-4 space-y-3" style={{ background: 'var(--bg-card)' }}>
            <StepRow
              num={1}
              label="Python Environment"
              status={env?.python_version ? 'done' : 'pending'}
              detail={env?.python_version ? `Python ${env.python_version}` : 'Not detected'}
            />
            <StepRow
              num={2}
              label="Python Packages"
              status={allPackagesReady ? 'done' : 'pending'}
              detail={allPackagesReady ? 'All installed' : !hasPackages ? 'Unknown — connect backend' : `${missingPackages.length} missing`}
              action={hasPackages && !allPackagesReady && !installing && (
                <button onClick={handleInstallPackages}
                  className="text-[10px] font-medium px-2.5 py-1 transition-colors"
                  style={{ background: '#b0b0b0', color: '#1a1a1a' }}>
                  Install All
                </button>
              )}
            />
            <StepRow
              num={3}
              label="GPU / CUDA"
              status={env?.cuda_available ? 'done' : 'warn'}
              detail={env?.cuda_available ? `${env.gpu_name} (${env.gpu_vram})` : 'No GPU — will use CPU (slow)'}
            />
            <StepRow
              num={4}
              label="HuggingFace Access"
              status={env?.hf_token_set ? 'done' : 'pending'}
              detail={env?.hf_token_set ? 'Token configured' : 'Required for model downloads'}
              action={!env?.hf_token_set && (
                <div className="flex items-center gap-1">
                  <input
                    type="password"
                    placeholder="hf_..."
                    value={hfToken}
                    onChange={(e) => setHfTokenInput(e.target.value)}
                    className="text-[10px] px-2 py-1 w-32 outline-none"
                    style={{ background: 'var(--bg-sidebar)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
                  />
                  <button onClick={handleSetToken}
                    className="text-[10px] font-medium px-2 py-1"
                    style={{ background: '#b0b0b0', color: '#1a1a1a' }}>
                    Save
                  </button>
                </div>
              )}
            />
            <StepRow
              num={5}
              label="Model Weights"
              status={allModelsReady ? 'done' : 'pending'}
              detail={allModelsReady ? 'All downloaded' : !hasModels ? 'Unknown — connect backend' : `${missingModels.length} remaining`}
              action={hasModels && !allModelsReady && !downloading && (
                <button onClick={handleDownloadAll}
                  className="text-[10px] font-medium px-2.5 py-1 transition-colors"
                  style={{ background: '#b0b0b0', color: '#1a1a1a' }}>
                  Download All
                </button>
              )}
            />
          </div>
        </section>

        {/* ═══ Packages ═══ */}
        <section>
          <h2 className="text-sm font-semibold uppercase tracking-wider mb-3"
              style={{ color: 'var(--text-primary)' }}>
            Packages
          </h2>
          <div className="overflow-hidden" style={{ background: 'var(--bg-card)' }}>
            {!hasPackages && (
              <div className="px-4 py-3 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                Connect to backend to see package status
              </div>
            )}
            {Object.entries(packages).map(([id, pkg]) => (
              <div key={id}
                className="flex items-center justify-between px-4 py-2 border-b last:border-b-0"
                style={{ borderColor: 'var(--border-color)' }}>
                <div className="flex-1 min-w-0">
                  <span className="text-[11px]" style={{ color: 'var(--text-primary)' }}>{pkg.name}</span>
                  <span className="text-[9px] ml-2 font-mono" style={{ color: 'var(--text-secondary)' }}>{id}</span>
                </div>
                <span className="text-[10px] font-mono shrink-0"
                      style={{ color: pkg.installed ? '#10b981' : '#f59e0b' }}>
                  {pkg.installed ? pkg.version : '✗ missing'}
                </span>
              </div>
            ))}
          </div>
          {!allPackagesReady && (
            <button onClick={handleInstallPackages}
              disabled={installing}
              className="mt-2 text-[10px] font-medium px-3 py-1.5 transition-colors disabled:opacity-50"
              style={{ background: '#b0b0b0', color: '#1a1a1a' }}>
              {installing ? 'Installing...' : 'Install Missing Packages'}
            </button>
          )}
        </section>

        {/* ═══ Model Weights ═══ */}
        <section>
          <h2 className="text-sm font-semibold uppercase tracking-wider mb-3"
              style={{ color: 'var(--text-primary)' }}>
            Model Weights
          </h2>
          <div className="overflow-hidden" style={{ background: 'var(--bg-card)' }}>
            {!hasModels && (
              <div className="px-4 py-3 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                Connect to backend to see model status
              </div>
            )}
            {Object.entries(models).map(([id, model]) => (
              <div key={id}
                className="flex items-center justify-between px-4 py-3 border-b last:border-b-0"
                style={{ borderColor: 'var(--border-color)' }}>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
                    {model.name}
                  </div>
                  <div className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>
                    {model.size}
                  </div>
                </div>
                <div className="shrink-0 ml-3">
                  {model.downloaded ? (
                    <span className="text-[10px] font-medium px-2 py-1"
                          style={{ color: '#10b981' }}>
                      ✓ Ready
                    </span>
                  ) : (
                    <button
                      onClick={() => handleDownload(id)}
                      disabled={downloading !== null}
                      className="text-[10px] font-medium px-2.5 py-1 transition-colors disabled:opacity-50"
                      style={{ background: '#b0b0b0', color: '#1a1a1a' }}>
                      {downloading === id ? 'Downloading...' : 'Download'}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ═══ Environment ═══ */}
        <section>
          <h2 className="text-sm font-semibold uppercase tracking-wider mb-3"
              style={{ color: 'var(--text-primary)' }}>
            Environment
          </h2>
          <div className="p-4 space-y-2" style={{ background: 'var(--bg-card)' }}>
            {env && (
              <>
                <Row label="Python" value={env.python_version} />
                <Row label="PyTorch" value={env.torch_version} />
                <Row label="CUDA" value={env.cuda_available ? '✓ Available' : '✗ Not available'}
                     highlight={env.cuda_available} />
                {env.gpu_name && <Row label="GPU" value={env.gpu_name} />}
                {env.gpu_vram && <Row label="VRAM" value={env.gpu_vram} />}
                <Row label="HF Token" value={env.hf_token_set ? '✓ Set' : '✗ Not set'}
                     highlight={env.hf_token_set} />
                <Row label="Disk Free" value={env.disk_free} />
                <Row label="Mode" value={env.mock_mode ? '⚠️ Mock (no AI)' : '✓ Full (AI active)'}
                     highlight={!env.mock_mode} />
              </>
            )}
          </div>
        </section>

        {/* Status message */}
        {statusMsg && (
          <div className="text-xs px-3 py-2" style={{ background: 'var(--bg-sidebar)', color: 'var(--text-secondary)' }}>
            {statusMsg}
          </div>
        )}

        {/* Refresh */}
        <button
          onClick={loadStatus}
          className="text-xs px-3 py-1.5 transition-colors hover:bg-white/5"
          style={{ color: 'var(--text-secondary)', border: '1px solid var(--border-color)' }}>
          ↻ Refresh Status
        </button>
      </div>
    </div>
  );
}

function StepRow({ num, label, status, detail, action }) {
  const colors = { done: '#10b981', pending: '#f59e0b', warn: '#ef4444' };
  const icons = { done: '✓', pending: '○', warn: '⚠' };
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs font-bold w-5 text-center" style={{ color: colors[status] }}>
        {icons[status]}
      </span>
      <div className="flex-1 min-w-0">
        <div className="text-[11px] font-medium" style={{ color: 'var(--text-primary)' }}>
          {num}. {label}
        </div>
        <div className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>{detail}</div>
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

function Row({ label, value, highlight }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span className="text-xs font-mono" style={{ color: highlight ? '#10b981' : 'var(--text-primary)' }}>
        {value}
      </span>
    </div>
  );
}
