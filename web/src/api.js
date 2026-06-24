const API = '';

export async function restoreSession() {
  const res = await fetch(`${API}/api/restore`);
  if (!res.ok) return null;
  const data = await res.json();
  return data.restored ? data : null;
}

export async function getSettingsStatus() {
  const res = await fetch(`${API}/api/settings/status`);
  if (!res.ok) throw new Error(`Failed to get settings: ${res.status}`);
  return res.json();
}

export async function downloadModel(modelId, onProgress) {
  const res = await fetch(`${API}/api/settings/download?model_id=${encodeURIComponent(modelId)}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (onProgress) onProgress(data.progress, data.message);
        if (data.done) return data.message;
      }
    }
  }
}

export async function installPackages(onProgress) {
  const res = await fetch(`${API}/api/settings/install-packages`, { method: 'POST' });
  if (!res.ok) throw new Error(`Install failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (onProgress) onProgress(data.progress, data.message);
        if (data.done) return data.message;
      }
    }
  }
}

export async function setHfToken(token) {
  const res = await fetch(`${API}/api/settings/set-hf-token?token=${encodeURIComponent(token)}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Failed to set token: ${res.status}`);
  return res.json();
}

export async function listDir(path = '~') {
  const res = await fetch(`${API}/api/listdir?path=${encodeURIComponent(path)}`);
  if (!res.ok) throw new Error(`Failed to list directory: ${res.status}`);
  return res.json();
}

export async function browseFolder(path) {
  const res = await fetch(`${API}/api/browse?path=${encodeURIComponent(path)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Browse failed: ${res.status}`);
  }
  return res.json();
}

export async function uploadFiles(files, sessionId = null) {
  const form = new FormData();
  files.forEach((f) => form.append('files', f));
  if (sessionId) form.append('session_id', sessionId);
  const res = await fetch(`${API}/api/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function importByPath(path, sessionId = null) {
  const body = { path };
  if (sessionId) body.session_id = sessionId;
  const res = await fetch(`${API}/api/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Import failed: ${res.status}`);
  }
  return res.json();
}

export async function importFiles(filePaths, sessionId = null) {
  const body = { files: filePaths };
  if (sessionId) body.session_id = sessionId;
  const res = await fetch(`${API}/api/import-files`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Import failed: ${res.status}`);
  }
  return res.json();
}

export async function removeImages(sessionId, indices) {
  const res = await fetch(`${API}/api/remove/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ indices }),
  });
  if (!res.ok) throw new Error(`Remove failed: ${res.status}`);
  return res.json();
}

export async function deleteImages(sessionId, indices) {
  const res = await fetch(`${API}/api/delete/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ indices }),
  });
  if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
  return res.json();
}

export async function getImage(sessionId, index) {
  const res = await fetch(`${API}/api/image/${sessionId}/${index}`);
  if (!res.ok) throw new Error(`Failed to load image: ${res.status}`);
  return res.json();
}

export async function resizeImage(sessionId, index, width, height, denoiseStrength = 0.5) {
  const qs = new URLSearchParams({ width, height, denoise_strength: denoiseStrength });
  const res = await fetch(`${API}/api/resize/${sessionId}/${index}?${qs}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Resize failed: ${res.status}`);
  return res.json();
}

export async function denoiseImage(sessionId, index, strength = 1.0) {
  const qs = new URLSearchParams({ strength });
  const res = await fetch(`${API}/api/denoise/${sessionId}/${index}?${qs}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Denoise failed: ${res.status}`);
  return res.json();
}

export async function deblurImage(sessionId, index, strength = 1.0) {
  const qs = new URLSearchParams({ strength });
  const res = await fetch(`${API}/api/deblur/${sessionId}/${index}?${qs}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Deblur failed: ${res.status}`);
  return res.json();
}

export async function runPipeline(sessionId, index, steps) {
  // Start the job
  const res = await fetch(`${API}/api/pipeline/${sessionId}/${index}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ steps }),
  });
  if (!res.ok) throw new Error(`Pipeline failed: ${res.status}`);
  const data = await res.json();

  if (data.status === 'done') return data; // instant (cached)

  // Poll for result
  const jobId = data.job_id;
  while (true) {
    await new Promise((r) => setTimeout(r, 2000));
    const pollRes = await fetch(`${API}/api/pipeline/status/${jobId}`);
    if (!pollRes.ok) {
      const err = await pollRes.text();
      throw new Error(err || `Pipeline failed: ${pollRes.status}`);
    }
    const status = await pollRes.json();
    if (status.status === 'done') return status;
    if (status.status === 'error') throw new Error(status.detail || 'Pipeline failed');
    // else still processing, continue polling
  }
}

export async function processImage(sessionId, index, params = {}) {
  const qs = new URLSearchParams({
    quality: params.quality ?? 0,
    strength: params.strength ?? 0.5,
    use_4bit: params.use4bit ?? true,
    output_format: params.outputFormat ?? 'png',
    jpg_quality: params.jpgQuality ?? 95,
  });
  const res = await fetch(`${API}/api/process/${sessionId}/${index}?${qs}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Processing failed: ${res.status}`);
  return res.json();
}

export async function blendImage(sessionId, index, strength) {
  const qs = new URLSearchParams({ strength });
  const res = await fetch(`${API}/api/blend/${sessionId}/${index}?${qs}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Blend failed: ${res.status}`);
  return res.json();
}

export async function exportAll(sessionId, params = {}, onProgress) {
  const qs = new URLSearchParams({
    quality: params.quality ?? 0,
    strength: params.strength ?? 0.5,
    use_4bit: params.use4bit ?? true,
    output_format: params.outputFormat ?? 'png',
    jpg_quality: params.jpgQuality ?? 95,
  });
  const res = await fetch(`${API}/api/export/${sessionId}?${qs}`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Export failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let finalStatus = '';
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep incomplete line
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (onProgress) onProgress(data.progress, data.message);
        if (data.done) finalStatus = data.message;
      }
    }
  }
  return { status: finalStatus };
}
