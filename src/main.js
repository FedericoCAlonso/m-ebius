/* ============================================================
   MÖBIUS — main.js
   Punto de entrada y coordinación de la PWA.
   ============================================================ */

import { db } from './storage/db.js';
import {
  detectMarkers,
  clusterIntoSheets,
  SHEET_TEMPLATES
} from './vision/detector.js';
import {
  computeHomographyAndScale,
  conformalFilter,
  computeBBox,
  renderPlane,
  findNearestEdge
} from './vision/math.js';

/* ─────────────────────────────────────────────────────────────
   ESTADO GLOBAL
   ───────────────────────────────────────────────────────────── */
const state = {
  cvReady:       false,
  processing:    false,
  cameraStream:  null,
  
  originalImageMat: null, // Matriz RGBA de la foto original
  activePlaneIndex: -1,   // Índice del plano actualmente seleccionado
  planes:        [],      // [{id, label, H, H_inv}]
  
  measurements:  [],      // [{id, planeLabel, refName, distMm, points: [{x,y}, {x,y}]}]
  measIdCounter: 0,
  snapEnabled: false,
  
  viewer: {
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    isDragging: false,
    startX: 0,
    startY: 0
  }
};

/* ─────────────────────────────────────────────────────────────
   ① CARGA DE OPENCV
   ───────────────────────────────────────────────────────────── */

document.addEventListener('opencvReady', () => {
  state.cvReady = true;
  hideCvErrorPanel();
  setStatus('OpenCV listo ✓', 'ready');
  toast('OpenCV.js cargado. Listo para procesar.', 'success');
});

document.addEventListener('opencvLoadFailed', () => {
  setStatus('OpenCV no disponible', 'error');
  showCvErrorPanel();
});

function showCvErrorPanel() {
  const panel = document.getElementById('cv-error-panel');
  if (panel) panel.classList.remove('hidden');
}

function hideCvErrorPanel() {
  const panel = document.getElementById('cv-error-panel');
  if (panel) panel.classList.add('hidden');
}

/* ─────────────────────────────────────────────────────────────
   UTILIDADES — INTERFAZ
   ───────────────────────────────────────────────────────────── */

function setStatus(text, type = 'loading') {
  const dot  = document.getElementById('status-dot');
  const span = document.getElementById('status-text');
  if (dot) dot.className  = `status-dot ${type}`;
  if (span) span.textContent = text;
}

function enableUI() {
  const dz = document.getElementById('drop-zone');
  if (dz) {
    dz.classList.remove('disabled');
    const sub = dz.querySelector('.drop-sub');
    if (sub && sub.textContent.includes('Cargando')) {
      sub.textContent = 'o usá los botones de abajo · Formatos: JPG, PNG, WEBP';
    }
  }
  document.getElementById('btn-file').disabled   = false;
  document.getElementById('btn-camera').disabled = false;
}

function showProcessing(label = 'Procesando imagen…', sub = '') {
  const ov = document.getElementById('processing-overlay');
  document.getElementById('proc-label').textContent = label;
  document.getElementById('proc-sub').textContent   = sub;
  ov.classList.add('visible');
}

function updateProcessingSub(text) {
  document.getElementById('proc-sub').textContent = text;
}

function hideProcessing() {
  document.getElementById('processing-overlay').classList.remove('visible');
}

function toast(msg, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => {
    el.style.animation = 'toastOut .3s ease forwards';
    setTimeout(() => el.remove(), 320);
  }, duration);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href     = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { a.remove(); URL.revokeObjectURL(url); }, 1000);
}

function yieldToUI() {
  return new Promise(r => setTimeout(r, 0));
}

/* ─────────────────────────────────────────────────────────────
   CARGA DE IMAGEN → cv.Mat
   ───────────────────────────────────────────────────────────── */

function loadSourceAsMat(source) {
  return new Promise((resolve, reject) => {
    if (source instanceof HTMLCanvasElement) {
      if (source.width === 0 || source.height === 0) {
        reject(new Error('El canvas capturado tiene dimensiones cero.'));
        return;
      }
      try {
        const mat = cv.imread(source);
        resolve({ mat, width: source.width, height: source.height });
      } catch(e) {
        reject(new Error(`cv.imread falló en canvas: ${e.message}`));
      }
      return;
    }

    const img = new Image();
    let url;
    try {
      url = URL.createObjectURL(source);
    } catch(e) {
      reject(new Error(`No se pudo crear URL para el archivo: ${e.message}`));
      return;
    }

    img.onload = () => {
      try {
        if (img.naturalWidth === 0 || img.naturalHeight === 0) {
          URL.revokeObjectURL(url);
          reject(new Error('La imagen tiene dimensiones cero (formato no soportado?).'));
          return;
        }
        const tmpCanvas    = document.createElement('canvas');
        tmpCanvas.width    = img.naturalWidth;
        tmpCanvas.height   = img.naturalHeight;
        const ctx = tmpCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        const mat = cv.imread(tmpCanvas);
        resolve({ mat, width: tmpCanvas.width, height: tmpCanvas.height });
      } catch(e) {
        reject(new Error(`Error al procesar imagen: ${e.message}`));
      }
    };

    img.onerror = (e) => {
      URL.revokeObjectURL(url);
      reject(new Error(
        `El navegador no pudo decodificar "${source.name || 'imagen'}". ` +
        'Asegurate de que sea un JPG, PNG o WEBP válido.'
      ));
    };

    img.src = url;
  });
}

/* ─────────────────────────────────────────────────────────────
   VISOR Y MODO PROYECCIÓN
   ───────────────────────────────────────────────────────────── */

function initViewer(width, height) {
  const mainCanvas = document.getElementById('main-image-canvas');
  const overlayCanvas = document.getElementById('overlay-canvas');
  const container = document.getElementById('viewer-container');
  const wrapper = document.getElementById('viewer-wrapper');

  mainCanvas.width = width;
  mainCanvas.height = height;
  overlayCanvas.width = width;
  overlayCanvas.height = height;

  cv.imshow(mainCanvas, state.originalImageMat);

  const rect = wrapper.getBoundingClientRect();
  const scaleX = rect.width / width;
  const scaleY = rect.height / height;
  state.viewer.scale = Math.min(scaleX, scaleY) * 0.9;
  
  state.viewer.offsetX = (rect.width - width * state.viewer.scale) / 2;
  state.viewer.offsetY = (rect.height - height * state.viewer.scale) / 2;

  updateViewTransform();

  // Eventos de Pan & Zoom
  wrapper.addEventListener('wheel', e => {
    e.preventDefault();
    const zoomFactor = 1.1;
    const direction = e.deltaY > 0 ? -1 : 1;
    const newScale = direction > 0 ? state.viewer.scale * zoomFactor : state.viewer.scale / zoomFactor;
    
    const wrapperRect = wrapper.getBoundingClientRect();
    const mouseX = e.clientX - wrapperRect.left;
    const mouseY = e.clientY - wrapperRect.top;
    
    state.viewer.offsetX = mouseX - (mouseX - state.viewer.offsetX) * (newScale / state.viewer.scale);
    state.viewer.offsetY = mouseY - (mouseY - state.viewer.offsetY) * (newScale / state.viewer.scale);
    state.viewer.scale = newScale;
    
    updateViewTransform();
  }, { passive: false });

  // Pan (Arrastrar) - MOUSE
  wrapper.addEventListener('mousedown', e => {
    if (e.button !== 1 && e.button !== 2) return; 
    e.preventDefault();
    state.viewer.isDragging = true;
    state.viewer.startX = e.clientX - state.viewer.offsetX;
    state.viewer.startY = e.clientY - state.viewer.offsetY;
    wrapper.style.cursor = 'grabbing';
  });

  wrapper.addEventListener('contextmenu', e => e.preventDefault());

  window.addEventListener('mousemove', e => {
    if (!state.viewer.isDragging) return;
    state.viewer.offsetX = e.clientX - state.viewer.startX;
    state.viewer.offsetY = e.clientY - state.viewer.startY;
    updateViewTransform();
  });

  window.addEventListener('mouseup', e => {
    if (e.button !== 1 && e.button !== 2) return;
    state.viewer.isDragging = false;
    wrapper.style.cursor = '';
  });

  // Pan y Zoom - TOUCH (2 dedos)
  let initialPinchDist = null;
  let initialScale = null;

  wrapper.addEventListener('touchstart', e => {
    if (e.touches.length === 2) {
      e.preventDefault();
      const t1 = e.touches[0], t2 = e.touches[1];
      initialPinchDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
      initialScale = state.viewer.scale;
      
      const cx = (t1.clientX + t2.clientX) / 2;
      const cy = (t1.clientY + t2.clientY) / 2;
      state.viewer.isDragging = true;
      state.viewer.startX = cx - state.viewer.offsetX;
      state.viewer.startY = cy - state.viewer.offsetY;
    }
  }, { passive: false });

  wrapper.addEventListener('touchmove', e => {
    if (e.touches.length === 2 && initialPinchDist) {
      e.preventDefault();
      const t1 = e.touches[0], t2 = e.touches[1];
      const dist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
      const zoomFactor = dist / initialPinchDist;
      
      const cx = (t1.clientX + t2.clientX) / 2;
      const cy = (t1.clientY + t2.clientY) / 2;
      
      const rect = wrapper.getBoundingClientRect();
      const relativeCx = cx - rect.left;
      const relativeCy = cy - rect.top;

      const newScale = initialScale * zoomFactor;
      
      state.viewer.offsetX = relativeCx - (relativeCx - state.viewer.offsetX) * (newScale / state.viewer.scale);
      state.viewer.offsetY = relativeCy - (relativeCy - state.viewer.offsetY) * (newScale / state.viewer.scale);
      state.viewer.scale = newScale;
      
      state.viewer.offsetX = cx - state.viewer.startX;
      state.viewer.offsetY = cy - state.viewer.startY;
      
      updateViewTransform();
    }
  }, { passive: false });

  wrapper.addEventListener('touchend', e => {
    if (e.touches.length < 2) {
      initialPinchDist = null;
      state.viewer.isDragging = false;
    }
  });

  // Botones de zoom flotantes
  document.getElementById('btn-zoom-in').onclick = () => zoomByCenter(1.2);
  document.getElementById('btn-zoom-out').onclick = () => zoomByCenter(1/1.2);
  document.getElementById('btn-zoom-fit').onclick = () => {
    state.viewer.scale = Math.min(scaleX, scaleY) * 0.9;
    state.viewer.offsetX = (rect.width - width * state.viewer.scale) / 2;
    state.viewer.offsetY = (rect.height - height * state.viewer.scale) / 2;
    updateViewTransform();
  };

  function zoomByCenter(factor) {
    const r = wrapper.getBoundingClientRect();
    const cx = r.width / 2;
    const cy = r.height / 2;
    const newScale = state.viewer.scale * factor;
    state.viewer.offsetX = cx - (cx - state.viewer.offsetX) * (newScale / state.viewer.scale);
    state.viewer.offsetY = cy - (cy - state.viewer.offsetY) * (newScale / state.viewer.scale);
    state.viewer.scale = newScale;
    updateViewTransform();
  }

  // Lógica central de medición
  let pendingPoint = null;
  let currentTouchPoint = null;

  function processClickPoint(clickX, clickY) {
    let imgX = (clickX - state.viewer.offsetX) / state.viewer.scale;
    let imgY = (clickY - state.viewer.offsetY) / state.viewer.scale;

    const plane = state.planes[state.activePlaneIndex];
    if (!plane) return null;

    if (state.snapEnabled && state.originalImageMat) {
      const snapObj = findNearestEdge(state.originalImageMat, imgX, imgY, 40);
      if (snapObj) {
        imgX = snapObj.x;
        imgY = snapObj.y;
      }
    }

    const H = plane.H;
    const H_data = H.data64F;
    const w = H_data[6] * imgX + H_data[7] * imgY + H_data[8];
    const x_mm = (H_data[0] * imgX + H_data[1] * imgY + H_data[2]) / w;
    const y_mm = (H_data[3] * imgX + H_data[4] * imgY + H_data[5]) / w;

    return { imgX, imgY, x_mm, y_mm };
  }

  function commitMeasurementPoint(point) {
    if (!point) return;
    const plane = state.planes[state.activePlaneIndex];
    if (!plane) return;

    if (!pendingPoint) {
      pendingPoint = point;
      drawMeasurements(pendingPoint); 
    } else {
      const distMm = Math.hypot(point.x_mm - pendingPoint.x_mm, point.y_mm - pendingPoint.y_mm);
      addMeasurement(plane.label, distMm, pendingPoint, point, plane.id);
      pendingPoint = null;
      drawMeasurements();
    }
  }

  // Medición - MOUSE
  overlayCanvas.addEventListener('mousedown', e => {
    if (e.button !== 0 || e.shiftKey || e.ctrlKey) return; 
    const rect = wrapper.getBoundingClientRect();
    const point = processClickPoint(e.clientX - rect.left, e.clientY - rect.top);
    commitMeasurementPoint(point);
  });

  // Medición - TOUCH (1 dedo)
  overlayCanvas.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
      e.preventDefault(); 
      const rect = wrapper.getBoundingClientRect();
      const touch = e.touches[0];
      const clickX = touch.clientX - rect.left;
      const clickY = touch.clientY - rect.top - 60; // Offset Y arriba del dedo
      
      currentTouchPoint = processClickPoint(clickX, clickY);
      drawMeasurements(pendingPoint, currentTouchPoint);
    }
  }, { passive: false });

  overlayCanvas.addEventListener('touchmove', e => {
    if (e.touches.length === 1 && currentTouchPoint) {
      e.preventDefault();
      const rect = wrapper.getBoundingClientRect();
      const touch = e.touches[0];
      const clickX = touch.clientX - rect.left;
      const clickY = touch.clientY - rect.top - 60; 
      
      currentTouchPoint = processClickPoint(clickX, clickY);
      drawMeasurements(pendingPoint, currentTouchPoint);
    }
  }, { passive: false });

  overlayCanvas.addEventListener('touchend', e => {
    if (e.changedTouches.length === 1 && currentTouchPoint) {
      e.preventDefault();
      commitMeasurementPoint(currentTouchPoint);
      currentTouchPoint = null;
      drawMeasurements(pendingPoint);
    }
  });
}

function updateViewTransform() {
  const container = document.getElementById('viewer-container');
  if (container) {
    container.style.transform = `translate(${state.viewer.offsetX}px, ${state.viewer.offsetY}px) scale(${state.viewer.scale})`;
  }
  drawMeasurements();
}

function drawGrid() {
  const overlay = document.getElementById('overlay-canvas');
  if (!overlay) return;
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const plane = state.planes[state.activePlaneIndex];
  if (!plane) return;

  const H_inv = plane.H_inv;
  const H_data = H_inv.data64F;

  function project(x_mm, y_mm) {
    const w = H_data[6] * x_mm + H_data[7] * y_mm + H_data[8];
    return {
      x: (H_data[0] * x_mm + H_data[1] * y_mm + H_data[2]) / w,
      y: (H_data[3] * x_mm + H_data[4] * y_mm + H_data[5]) / w
    };
  }

  const range = 800; // mm
  const step = 50;

  ctx.strokeStyle = 'rgba(110, 231, 247, 0.2)';
  ctx.lineWidth = 1 / state.viewer.scale;
  ctx.beginPath();
  for (let x = -range; x <= range; x += step) {
    let p1 = project(x, -range), p2 = project(x, range);
    ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
  }
  for (let y = -range; y <= range; y += step) {
    let p1 = project(-range, y), p2 = project(range, y);
    ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
  }
  ctx.stroke();

  // Ejes X e Y locales (Origen ArUco)
  ctx.lineWidth = 2 / state.viewer.scale;
  ctx.beginPath(); ctx.strokeStyle = 'rgba(248, 113, 113, 0.6)'; // Red = X
  let pX1 = project(0, 0), pX2 = project(100, 0);
  ctx.moveTo(pX1.x, pX1.y); ctx.lineTo(pX2.x, pX2.y); ctx.stroke();

  ctx.beginPath(); ctx.strokeStyle = 'rgba(52, 211, 153, 0.6)'; // Green = Y
  let pY1 = project(0, 0), pY2 = project(0, 100);
  ctx.moveTo(pY1.x, pY1.y); ctx.lineTo(pY2.x, pY2.y); ctx.stroke();
}

function drawMeasurements(pendingPoint = null, currentTouchPoint = null) {
  const overlay = document.getElementById('overlay-canvas');
  if (!overlay) return;
  const ctx = overlay.getContext('2d');
  
  drawGrid();

  const plane = state.planes[state.activePlaneIndex];
  if (!plane) return;

  const planeMeasurements = state.measurements.filter(m => m.planeId === plane.id);
  const strokeW = 2 / state.viewer.scale;
  const r = 4 / state.viewer.scale;

  const drawPt = (p) => {
    ctx.beginPath(); ctx.arc(p.imgX, p.imgY, r, 0, 2*Math.PI);
    ctx.fillStyle = '#6ee7f7'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();
  };

  planeMeasurements.forEach(m => {
    const p1 = m.points[0], p2 = m.points[1];
    drawPt(p1); drawPt(p2);

    ctx.beginPath(); ctx.moveTo(p1.imgX, p1.imgY); ctx.lineTo(p2.imgX, p2.imgY);
    ctx.strokeStyle = '#6ee7f7'; ctx.lineWidth = strokeW;
    ctx.setLineDash([8 / state.viewer.scale, 5 / state.viewer.scale]);
    ctx.stroke(); ctx.setLineDash([]);
    
    const midX = (p1.imgX + p2.imgX) / 2;
    const midY = (p1.imgY + p2.imgY) / 2;
    ctx.font = `600 ${14 / state.viewer.scale}px 'Inter', sans-serif`;
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(`${m.distMm.toFixed(1)} mm`, midX, midY - (10/state.viewer.scale));
  });

  if (pendingPoint) drawPt(pendingPoint);
  if (currentTouchPoint) drawPt(currentTouchPoint);

  if (pendingPoint && currentTouchPoint) {
    ctx.beginPath(); ctx.moveTo(pendingPoint.imgX, pendingPoint.imgY); ctx.lineTo(currentTouchPoint.imgX, currentTouchPoint.imgY);
    ctx.strokeStyle = '#f87171'; ctx.lineWidth = strokeW;
    ctx.stroke();
    
    const distMm = Math.hypot(currentTouchPoint.x_mm - pendingPoint.x_mm, currentTouchPoint.y_mm - pendingPoint.y_mm);
    const midX = (pendingPoint.imgX + currentTouchPoint.imgX) / 2;
    const midY = (pendingPoint.imgY + currentTouchPoint.imgY) / 2;
    ctx.font = `600 ${14 / state.viewer.scale}px 'Inter', sans-serif`;
    ctx.fillStyle = '#f87171';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(`${distMm.toFixed(1)} mm`, midX, midY - (10/state.viewer.scale));
  }
}

/* ─────────────────────────────────────────────────────────────
   GESTIÓN DE MEDICIONES
   ───────────────────────────────────────────────────────────── */

function addMeasurement(planeLabel, distMm, p1, p2, planeId) {
  const entry = {
    id:         ++state.measIdCounter,
    planeId,
    planeLabel,
    refName:    '',
    distMm,
    points:     [p1, p2],
    timestamp:  new Date().toISOString(),
  };
  state.measurements.push(entry);
  renderMeasurementTable();
  document.getElementById('measurements-section').classList.add('visible');
}

function removeMeasurement(id) {
  state.measurements = state.measurements.filter(m => m.id !== id);
  renderMeasurementTable();
  drawMeasurements();
}

function clearAllMeasurements() {
  state.measurements = [];
  state.measIdCounter = 0;
  renderMeasurementTable();
  drawMeasurements();
}

function renderMeasurementTable() {
  const tbody = document.getElementById('meas-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (state.measurements.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="meas-empty">Hacé clic en dos puntos del plano activo para medir.</td></tr>';
    return;
  }

  for (const m of state.measurements) {
    const tr = document.createElement('tr');
    const inputId = `ref-input-${m.id}`;

    tr.innerHTML = `
      <td style="color:var(--text-3);font-family:monospace;">${m.id}</td>
      <td>${escHtml(m.planeLabel)}</td>
      <td>
        <input type="text" class="ref-input" id="${inputId}" placeholder="Ej. Puerta" value="${escHtml(m.refName)}">
      </td>
      <td class="meas-value">${m.distMm.toFixed(2)} mm</td>
      <td>
        <button class="btn btn-danger btn-sm"
                data-meas-id="${m.id}"
                aria-label="Eliminar medición ${m.id}"
                type="button">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
          </svg>
        </button>
      </td>`;
    tbody.appendChild(tr);

    const inputEl = document.getElementById(inputId);
    if (inputEl) {
      inputEl.addEventListener('input', (e) => {
        m.refName = e.target.value;
      });
    }
  }
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* ─────────────────────────────────────────────────────────────
   EXPORTACIÓN
   ───────────────────────────────────────────────────────────── */

function exportAllPNG() {
  if (state.activePlaneIndex < 0) { toast('No hay planos activos para exportar.', 'info'); return; }
  const plane = state.planes[state.activePlaneIndex];
  if (!plane) return;

  showProcessing('Generando recorte plano...', 'Calculando transformación');
  
  setTimeout(() => {
    try {
      const width = state.originalImageMat.cols;
      const height = state.originalImageMat.rows;
      const sheetCenter = { x: plane.template.w_mm / 2, y: plane.template.h_mm / 2 };
      
      const validPoints = conformalFilter(plane.H, width, height, sheetCenter);
      if (validPoints.length < 4) throw new Error("Área válida demasiado pequeña");
      
      const bbox = computeBBox(validPoints);
      if (!bbox) throw new Error("No se pudo calcular bounding box");

      const tmpCanvas = document.createElement('canvas');
      renderPlane(state.originalImageMat, plane.H, plane.scale, bbox, tmpCanvas);

      tmpCanvas.toBlob(blob => {
        if (!blob) throw new Error("Fallo al generar Blob");
        downloadBlob(blob, `mobius_plano_${plane.id + 1}.png`);
        hideProcessing();
        toast('Recorte plano exportado.', 'success');
      }, 'image/png');
    } catch(e) {
      hideProcessing();
      toast('Error generando imagen: ' + e.message, 'error');
    }
  }, 50);
}

function exportMeasurementsCSV() {
  if (state.measurements.length === 0) {
    toast('No hay mediciones para exportar.', 'info');
    return;
  }

  const header = ['ID', 'Plano', 'Referencia', 'Distancia_mm', 'Timestamp'];
  const rows   = state.measurements.map(m => {
    const ref = `"${m.refName.replace(/"/g, '""')}"`;
    return [
      m.id,
      `"${m.planeLabel}"`,
      ref,
      m.distMm.toFixed(4),
      m.timestamp,
    ];
  });

  const csv  = [header, ...rows].map(r => r.join(',')).join('\r\n');
  const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
  downloadBlob(blob, `mediciones_mobius_${new Date().toISOString().slice(0,10)}.csv`);
  toast('Mediciones exportadas como CSV.', 'success');
}

/* ─────────────────────────────────────────────────────────────
   CONSTRUCCIÓN DEL CARD DE RESULTADO
   ───────────────────────────────────────────────────────────── */

function createPlaneSidebarButton(index, scale, templateName) {
  const label = `Plano ${index + 1} (${templateName})`;
  const list  = document.getElementById('planes-list');
  if (!list) return label;

  const btn = document.createElement('button');
  btn.className = 'plane-btn';
  btn.id = `btn-plane-${index}`;
  btn.type = 'button';
  
  btn.innerHTML = `
    <span class="plane-btn-title">
      <span class="planes-badge">${index + 1}</span>
      ${escHtml(label)}
    </span>
    <span class="plane-btn-meta">${scale.toFixed(2)} px/mm</span>
  `;

  btn.addEventListener('click', () => {
    setActivePlane(index);
  });

  list.appendChild(btn);
  return label;
}

function setActivePlane(index) {
  state.activePlaneIndex = index;
  
  document.querySelectorAll('.plane-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`btn-plane-${index}`);
  if (activeBtn) activeBtn.classList.add('active');

  drawGrid();
  drawMeasurements();
}

/* ─────────────────────────────────────────────────────────────
   PIPELINE PRINCIPAL
   ───────────────────────────────────────────────────────────── */

function handleFileSelected(file) {
  const ALLOWED_TYPES = ['image/jpeg','image/png','image/webp','image/bmp','image/tiff'];
  const ALLOWED_EXTS  = ['.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'];

  const typeOk = file.type && ALLOWED_TYPES.includes(file.type);
  const extOk  = ALLOWED_EXTS.some(ext =>
    file.name.toLowerCase().endsWith(ext)
  );

  if (!typeOk && !extOk) {
    toast(
      `Formato no soportado: "${file.name}". Usá JPG, PNG o WEBP.`,
      'error'
    );
    return;
  }

  processImage(file);
}

async function processImage(source) {
  if (state.processing) return;
  if (!state.cvReady) {
    toast(
      'OpenCV.js aún se está cargando (∼8 MB). Esperá unos segundos e intentá de nuevo.',
      'info', 5000
    );
    return;
  }

  state.processing = true;
  clearResults();

  showProcessing('Cargando imagen…', 'Decodificando pixeles');
  await yieldToUI();

  let srcMat = null;

  try {
    const { mat, width, height } = await loadSourceAsMat(source);
    srcMat = mat;

    updateProcessingSub(`Detectando marcadores ArUco en ${width}×${height} px…`);
    await yieldToUI();

    const markers = detectMarkers(srcMat);

    if (markers.length < 4) {
      throw new Error(
        `Se detectaron solo ${markers.length} marcador(es). ` +
        'Se necesitan al menos 4 de un mismo template (ej. IDs 0-3) para rectificar un plano.'
      );
    }

    updateProcessingSub(`${markers.length} marcadores detectados. Agrupando en hojas…`);
    await yieldToUI();

    const sheets = clusterIntoSheets(markers);

    if (sheets.length === 0) {
      throw new Error(
        'No se pudo formar ninguna hoja completa. ' +
        'Verificá que los 4 marcadores de algún template sean visibles (ej. 0,1,2,3 para A5).'
      );
    }

    toast(`${sheets.length} hoja(s) detectada(s). Procesando…`, 'info');

    for (let i = 0; i < sheets.length; i++) {
      updateProcessingSub(`Procesando plano ${i + 1} de ${sheets.length}…`);
      await yieldToUI();

      const sheet = sheets[i];
      const { H, scale } = computeHomographyAndScale(sheet);

      const H_inv = new cv.Mat();
      cv.invert(H, H_inv, cv.DECOMP_LU);

      const label = createPlaneSidebarButton(i, scale, sheet.template.name);
      state.planes.push({ id: i, label, H: H.clone(), H_inv, scale, template: sheet.template });

      H.delete();
      await yieldToUI();
    }

    state.originalImageMat = srcMat;

    const count = state.planes.length;
    if (count > 0) {
      document.getElementById('planes-count').textContent = String(count);
      document.getElementById('upload-view').classList.add('hidden');
      document.getElementById('workspace-view').classList.add('visible');
      document.getElementById('measurements-section').classList.add('visible');
      
      initViewer(width, height);
      setActivePlane(0);
      
      toast(
        `${count} plano(s) rectificado(s) correctamente.`,
        'success'
      );
      renderMeasurementTable();
    } else {
      toast('No se pudo rectificar ningún plano. Revisá la calidad de la imagen.', 'error');
    }

  } catch (err) {
    console.error('[Möbius]', err);
    toast(`Error: ${err.message}`, 'error', 8000);
    if (srcMat && srcMat !== state.originalImageMat) { try { srcMat.delete(); } catch(_) {} }
  } finally {
    hideProcessing();
    state.processing = false;
  }
}

function clearResults() {
  if (state.originalImageMat) {
    try { state.originalImageMat.delete(); } catch(_) {}
    state.originalImageMat = null;
  }
  state.planes.forEach(p => {
    if (p.H) try { p.H.delete(); } catch(_) {}
    if (p.H_inv) try { p.H_inv.delete(); } catch(_) {}
  });
  state.planes = [];
  state.activePlaneIndex = -1;
  state.measurements = [];
  state.measIdCounter = 0;
  
  document.getElementById('planes-list').innerHTML = '';
  document.getElementById('planes-count').textContent = '0';
  document.getElementById('upload-view').classList.remove('hidden');
  document.getElementById('workspace-view').classList.remove('visible');
  document.getElementById('measurements-section').classList.remove('visible');
  
  const mainCtx = document.getElementById('main-image-canvas').getContext('2d');
  mainCtx.clearRect(0, 0, mainCtx.canvas.width, mainCtx.canvas.height);
  const overCtx = document.getElementById('overlay-canvas').getContext('2d');
  overCtx.clearRect(0, 0, overCtx.canvas.width, overCtx.canvas.height);

  renderMeasurementTable();
}

let levelListener = null;

/* ─────────────────────────────────────────────────────────────
   CÁMARA (getUserMedia)
   ───────────────────────────────────────────────────────────── */

async function openCamera() {
  const overlay = document.getElementById('camera-overlay');
  const video   = document.getElementById('camera-video');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width:  { ideal: 4096 },
        height: { ideal: 3072 },
      },
      audio: false,
    });

    state.cameraStream = stream;
    video.srcObject    = stream;
    await video.play();
    overlay.classList.add('visible');

    const levelContainer = document.getElementById('camera-level-container');
    const bubble = document.getElementById('camera-level-bubble');
    const text = document.getElementById('camera-level-text');
    const btnEnable = document.getElementById('btn-enable-sensors');
    
    if(levelContainer) levelContainer.style.opacity = '1';

    function handleOrientation(event) {
      if (event.beta === null || event.gamma === null) return;
      let beta = event.beta; 
      let gamma = event.gamma; 
      
      let pErr = beta - 90;
      let rErr = gamma;
      
      const maxMove = 30; 
      let bx = (rErr / 45) * maxMove; 
      let by = (pErr / 45) * maxMove;
      
      const dist = Math.hypot(bx, by);
      if (dist > maxMove) {
        bx = (bx/dist)*maxMove;
        by = (by/dist)*maxMove;
      }
      
      if(bubble) bubble.style.transform = `translate(calc(-50% + ${bx}px), calc(-50% + ${by}px))`;
      
      const isLevel = Math.abs(pErr) < 3 && Math.abs(rErr) < 3;
      if (isLevel) {
        if(bubble) bubble.style.backgroundColor = '#4ade80';
        if(text) { text.style.color = '#4ade80'; text.innerText = '¡NIVELADO!'; }
      } else {
        if(bubble) bubble.style.backgroundColor = '#fff';
        if(text) { text.style.color = '#fff'; text.innerText = `${Math.abs(pErr).toFixed(0)}° v / ${Math.abs(rErr).toFixed(0)}° h`; }
      }
    }
    
    if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
      if(btnEnable) {
        btnEnable.style.display = 'block';
        if(text) text.innerText = 'Permiso requerido';
        btnEnable.onclick = () => {
          DeviceOrientationEvent.requestPermission()
            .then(res => {
              if (res === 'granted') {
                btnEnable.style.display = 'none';
                window.addEventListener('deviceorientation', handleOrientation);
                levelListener = handleOrientation;
              } else {
                if(text) text.innerText = 'Permiso denegado';
              }
            })
            .catch(console.error);
        };
      }
    } else {
      window.addEventListener('deviceorientation', handleOrientation);
      levelListener = handleOrientation;
    }

  } catch (err) {
    console.error('[Möbius] Camera error:', err);
    const msg = err.name === 'NotAllowedError'
      ? 'Acceso a la cámara denegado. Otorgá permiso en la configuración del navegador.'
      : `No se pudo acceder a la cámara: ${err.message}`;
    toast(msg, 'error', 6000);
  }
}

function closeCamera() {
  const overlay = document.getElementById('camera-overlay');
  const video   = document.getElementById('camera-video');

  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }
  video.srcObject = null;
  overlay.classList.remove('visible');

  const levelContainer = document.getElementById('camera-level-container');
  if(levelContainer) levelContainer.style.opacity = '0';
  if (levelListener) {
    window.removeEventListener('deviceorientation', levelListener);
    levelListener = null;
  }
}

function captureFrame() {
  const video = document.getElementById('camera-video');
  if (!video.videoWidth) { toast('La cámara aún no está lista.', 'info'); return; }

  const tmpCanvas    = document.createElement('canvas');
  tmpCanvas.width    = video.videoWidth;
  tmpCanvas.height   = video.videoHeight;
  tmpCanvas.getContext('2d').drawImage(video, 0, 0);

  closeCamera();
  processImage(tmpCanvas);
}

/* ─────────────────────────────────────────────────────────────
   EVENT LISTENERS DE UI
   ───────────────────────────────────────────────────────────── */

function initUI() {
  const dropZone   = document.getElementById('drop-zone');
  const fileInput  = document.getElementById('file-input');

  enableUI();

  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop',     e => e.preventDefault());

  ['dragenter', 'dragover'].forEach(ev => {
    if (dropZone) {
      dropZone.addEventListener(ev, e => {
        e.preventDefault();
        e.stopPropagation();
        if (!state.processing) dropZone.classList.add('drag-over');
      });
    }
  });
  
  ['dragleave', 'dragend'].forEach(ev => {
    if (dropZone) {
      dropZone.addEventListener(ev, e => {
        dropZone.classList.remove('drag-over');
      });
    }
  });

  if (dropZone) {
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
      const file = e.dataTransfer?.files?.[0];
      if (file) {
        handleFileSelected(file);
      } else {
        toast('No se detectó ninguna imagen en lo que arrastraste.', 'error');
      }
    });

    dropZone.addEventListener('click', e => {
      if (!e.target.closest('.btn')) {
        fileInput.click();
      }
    });

    dropZone.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });
  }

  document.getElementById('btn-file').addEventListener('click', e => {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (file) { handleFileSelected(file); fileInput.value = ''; }
  });

  document.getElementById('btn-camera').addEventListener('click', e => {
    e.stopPropagation();
    openCamera();
  });

  document.getElementById('btn-capture').addEventListener('click', captureFrame);
  document.getElementById('btn-camera-cancel').addEventListener('click', closeCamera);

  document.getElementById('btn-reset').addEventListener('click', clearResults);

  const btnExportPng = document.getElementById('btn-export-png');
  if (btnExportPng) btnExportPng.addEventListener('click', exportAllPNG);
  
  const btnExportCsv = document.getElementById('btn-export-csv');
  if (btnExportCsv) btnExportCsv.addEventListener('click', exportMeasurementsCSV);

  document.getElementById('btn-clear-meas').addEventListener('click', clearAllMeasurements);

  document.getElementById('meas-tbody').addEventListener('click', e => {
    const btn = e.target.closest('[data-meas-id]');
    if (btn) removeMeasurement(Number(btn.dataset.measId));
  });

  const btnRetryCV = document.getElementById('btn-retry-cv');
  if (btnRetryCV) {
    btnRetryCV.addEventListener('click', () => {
      hideCvErrorPanel();
      setStatus('Reintentando carga…', 'loading');
      if (typeof window._tryLoadCV === 'function') {
        window._cvSourceIdx = 0;
        window._tryLoadCV();
      } else {
        window.location.reload();
      }
    });
  }

  document.addEventListener('paste', e => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) { processImage(file); break; }
      }
    }
  });

  // Snap Toggle
  const btnSnap = document.getElementById('btn-snap-toggle');
  if(btnSnap) {
    btnSnap.onclick = () => {
      state.snapEnabled = !state.snapEnabled;
      if (state.snapEnabled) {
        btnSnap.style.background = 'var(--primary)';
        btnSnap.style.color = 'white';
        toast('Imantado a bordes ACTIVADO', 'info');
      } else {
        btnSnap.style.background = 'var(--surface)';
        btnSnap.style.color = 'var(--text-secondary)';
        toast('Imantado a bordes DESACTIVADO', 'info');
      }
    };
  }

  // Guardar Proyecto
  const btnSaveProj = document.getElementById('btn-save-project');
  if(btnSaveProj) {
    btnSaveProj.onclick = () => {
      if (!state.imgBase64) return toast('No hay un proyecto abierto.', 'error');
      showProcessing('Guardando Proyecto...', 'Preparando archivo JSON');
      setTimeout(() => {
        const proj = {
          version: 1,
          imgBase64: state.imgBase64,
          planes: state.planes.map(p => ({
            id: p.id, label: p.label,
            H_data: Array.from(p.H.data64F),
            templateW: p.template.w_mm, templateH: p.template.h_mm
          })),
          measurements: state.measurements,
          measIdCounter: state.measIdCounter
        };
        const blob = new Blob([JSON.stringify(proj)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'proyecto_mobius.json';
        a.click();
        hideProcessing();
        toast('Proyecto guardado con éxito', 'success');
      }, 100);
    };
  }

  // Cargar Proyecto
  const btnLoadProj = document.getElementById('btn-load-project');
  const inputProj = document.getElementById('project-input');
  if(btnLoadProj && inputProj) {
    btnLoadProj.onclick = () => inputProj.click();
    inputProj.onchange = e => {
      const file = e.target.files[0];
      if (!file) return;
      showProcessing('Cargando Proyecto...', 'Leyendo archivo');
      const reader = new FileReader();
      reader.onload = async ev => {
        try {
          const proj = JSON.parse(ev.target.result);
          if (!proj.imgBase64) throw new Error('Formato inválido');
          
          state.processing = true;
          clearResults();
          document.getElementById('upload-view').classList.add('hidden');
          document.getElementById('workspace-view').classList.add('visible');
          document.getElementById('measurements-section').classList.add('visible');
          
          state.imgBase64 = proj.imgBase64;
          const img = new Image();
          img.src = state.imgBase64;
          await new Promise(r => img.onload = r);
          
          state.originalImg = img;
          const mat = cv.imread(img);
          cv.cvtColor(mat, mat, cv.COLOR_RGBA2RGB);
          state.originalImageMat = mat.clone();
          
          state.planes = proj.planes.map(p => {
            const H = new cv.Mat(3, 3, cv.CV_64F);
            for(let i=0; i<9; i++) H.data64F[i] = p.H_data[i];
            const H_inv = new cv.Mat();
            cv.invert(H, H_inv);
            return {
              id: p.id, label: p.label, H, H_inv,
              template: { w_mm: p.templateW, h_mm: p.templateH }
            };
          });
          
          state.measurements = proj.measurements || [];
          state.measIdCounter = proj.measIdCounter || state.measurements.length;
          
          // Reconstruir lista de planos en la barra lateral
          const listEl = document.getElementById('planes-list');
          if (listEl) listEl.innerHTML = '';
          state.planes.forEach((p, idx) => {
            createPlaneSidebarButton(idx, p.scale || 1.0, p.template.name);
          });
          if(state.planes.length > 0) setActivePlane(0);
          renderMeasurementTable();
          
          hideProcessing();
          state.processing = false;
          toast('Proyecto cargado', 'success');
        } catch(err) {
          console.error(err);
          hideProcessing();
          state.processing = false;
          toast('Error al leer proyecto', 'error');
        }
      };
      reader.readAsText(file);
    };
  }

  // Modales de impresión e interacciones de impresión
  document.getElementById('btn-show-print-modal')?.addEventListener('click', () => {
    document.getElementById('modal-print')?.showModal();
  });

  document.getElementById('btn-close-print-modal')?.addEventListener('click', () => {
    document.getElementById('modal-print')?.close();
  });

  document.getElementById('btn-print-a5')?.addEventListener('click', () => {
    printTemplateSheet('A5');
    document.getElementById('modal-print')?.close();
  });

  document.getElementById('btn-print-a4')?.addEventListener('click', () => {
    printTemplateSheet('A4');
    document.getElementById('modal-print')?.close();
  });

  document.getElementById('btn-print-a3')?.addEventListener('click', () => {
    printTemplateSheet('A3');
    document.getElementById('modal-print')?.close();
  });
}

/* ─────────────────────────────────────────────────────────────
   REGISTRO DEL SERVICE WORKER
   ───────────────────────────────────────────────────────────── */

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js', { scope: './' })
      .then(reg => console.log('[Möbius] SW registrado:', reg.scope))
      .catch(err => console.warn('[Möbius] SW no registrado:', err));
  });
}

/* ─────────────────────────────────────────────────────────────
   INICIALIZACIÓN
   ───────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', initUI);

/* ─────────────────────────────────────────────────────────────
   GENERADOR DE HOJAS PARA IMPRIMIR
   ───────────────────────────────────────────────────────────── */

const MARKER_B64 = [
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADaklEQVR4Ad3BsQ3AQBDDMGn/oZ0RVByQ4kl5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxC0ggSxpkESSNIGGcSJI0gYZxJkDSChHEmQdIIEsaZBEkjSBhnEiSNIGGcSZA0goRxJkHSCBLGmQRJI0gYZxIkjSBhnEmQNIKEcSZB0ggSxpkESSNIGGcSJI0gYZxJkDSChHEmQdIIEsaZBEkjSBhnEiSNIGGcSZA0goRxJkHSCBLGmQRJI0gYZxIkjSBhnEmQNIKEcSZB0ggSxpkESSNIGGcSJI0gYZxJkDSChHEmQdIIEsaZBEkjSBhnEiSNIGGcSZA0goRxJkHSCBLGmQS5G0HKCBLkbgQpI0iQuxGkjCBB7kaQMoIEuRtByggS5G4EKSNIkLsRpIwgQe5GkDKCBLkbQcoIEuRuBCkjSJC7EaSMIEHuRpAyggS5G0HKCBLkbgQpI0iQuxGkjCBB7kaQMoIEuRtByggS5G4EKSNIkLsRpIwgQe5GkDKCBLkbQcoIEuRuBCkjSJC7EaSMIEHuRpAyggS5G0HKCBLkbgQpI0iQuxGkjCBB7kaQMoIEuRtByggS5G4EKSNIkLsRpIwgQe5GkDKCBLkbQcoIEuQH40yC/GCcSZAfjDMJ8oNxJkF+MM4kyA/GmQT5wTiTID8YZxLkB+NMgvxgnEmQH4wzCfKDcSZBfjDOJMgPxpkE+cE4kyA/GGcS5AfjTIL8YJxJkB+MMwnyg3EmQX4wziTID8aZBPnBOJMgPxhnEuQH40yC/GCcSZAfjDMJ8oNxJkF+MMwnyg3EmQX4wziTID8aZBPnBOJMgPxhnEuQH4wiV/KDUeRKfjCKXMkPRpEr+cEociU/GEWu5AejyJX8YBS5kh+MIlfyg1HkSn4wilzJD0aRK/nBKHIlPxhFruQHo8iV/GAUuZIfjCJX8oNR5Ep+MIpcyQ9GkSv5wShyJT8YRa7kB6PIlfxgFLmSH4wiV/KDUeRKfjCKXMkPRpEr+cEociU/GEWu5AejyJX8YBS5kh+MIlfyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gj5BHyiA+N8sfJbgus8QAAAABJRU5ErkJggg==",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADBElEQVR4Ad3BMRHAQBDEMJs/6A0EF99kTpIj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOULS+AEJksYPSJA0fkCCpPEDEiSNH5AgafyABEnjByRIGj8gQdL4AQmSxg9IkDR+QIKk8QMSJI0fkCBp/IAESeMHJEgaPyBB0vgBCZLGD0iQNH5AgqTxAxIkjR+QIGn8gARJ4wckSBo/IEHS+AEJksYPSJA0fkCCpPEDEiSNH5AgafyABEnjByRIGj8gQdL4AQmSRpBnI0iQNII8G0GCpBHk2QgSJI0gz0aQIGkEeTaCBEkjyLMRJEgaQZ6NIEHSCPJsBAmSRpBnI0iQNII8G0GCpBHk2QgSJI0gz0aQIGkEeTaCBEkjyLMRJEgaQZ6NIEHSCPJsBAmSRpBnI0iQNII8G0GCpBHk2QgSJI0gz0aQIGkEeTaCBEkjyLMRJEgaQZ6NIEHSCPJsBAmSRpBnI0iQNII8G0GCpBHk2QgSJI0gz0aQIGkEeTaCBEkjyLMRJEgaQZ6NIEHSCPJsBAmSRpBnI0iQNII8G0GCpBEkjCKvJI0gYRR5JWkECaPIK0kjSBhFXkkaQcIo8krSCBJGkVeSRpAwirySNIKEUeSVpBEkjCKvJI0gYRR5JWkECaPIK0kjSBhFXkkaQcIo8krSCBJGkVeSRpAwirySNIKEUeSVpBEkjCKvJI0gYRR5JWkECaPIK0kjSBhFXkkaQcIo8krSCBJGkVeSRpAwirySNIKEUeSVpBEkjCKvJI0gYRR5JWkECaPIK0kjSBhFXkkaQcIo8krSCBJGkVeSRpAwirySNIKEUeSVpBEkjCKv5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkiA9v7KfJsmW8mwAAAABJRU5ErkJggg==",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADJ0lEQVR4Ad3BsQ0AMAzDMOn/o90TPBjIUFI+IZ+QT8gn5BPyCfmEfEI+IZ+QT8gn5BPyCfmEfEI+IZ+QT8gn5BPyCfmEfEI+IZ+QT8gn5BPyCfmEfEI+IZ+QA2EmhRwIMynkQJhJIQfCTAo5EGZSyIEwk0IOhJkUciDMpJADYSaFHAgzKeRAmEkhB8JMCjkQZlLIgTCTQg6EmRRyIMykkANhJoUcCDMp5ECYSSEHwkwKORBmUsiBMJNCDoSZFHIgzKSQA2EmhRwIMynkQJhJIQfCTAo5EGZSyIEwk0IOhJkUciDMpJADYSaFHAgzKeRAmEkhB8JMCjkQZlLIgTCTQg6EmRRyIMykkANhJoUcCDMp5ECYSSEHwkwKORBmUsiBMJNCDoSZFHIgzKSQA2EmhRwIMynkQJhJIQfCTAo5EGZSyIEwk0IOhJkUciDMpJADYSaFHAgzKeRAmEkhB8JMCjkQZlLIgTCTQg6EmRRyIMykkANhJoUcCDMp5EBoZCUHQiMrORAaWcmB0MhKDoRGVnIgNLKSA6GRlRwIjazkQGhkJQdCIys5EBpZyYHQyEoOhEZWciA0spIDoZGVHAiNrORAaGQlB0IjKzkQGlnJgdDISg6ERlZyIDSykgOhkZUcCI2s5EBoZCUHQiMrORAaWcmB0MhKDoRGVnIgNLKSA6GRlRwIjazkQGhkJQdCIyupwkqaUEghVVhJEwoppAoraUIhhVRhJU0opJAqrKQJhRRShZU0oZBCqrCSJhRSSBVW0oRCCqnCSppQSCFVWEkTCimkCitpQiGFVGElTSikkCqspAmFFFKFlTShkEKqsJImFFJIFVbShEIKqcJKmlBIIVVYSRMKKaQKK2lCIYVUYSVNKKSQKqykCYUUUoWVNKGQQqqwkiYUUkgVVtKEQgqpwkqaUEghVVhJEwoppAoraUIhhVRhJU0opJAqrKQJhRRShZU0oZBCqrCSJhRSSBVW0oRCCqnCSppQSCGfkE/IJ+QT8gn5hHxCPiGfkE/IJ+QT8gn5hHxCPiGfkE/IJ+QT8gn5hHxCPiGfkE/IJ+QT8gn5hHxCPiGfeIgSpslm7qh6AAAAAElFTkSuQmCC",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADaUlEQVR4Ad3BsREAMBDCMHv/ockIFFyal+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5QqpQyCwUUkgVCpmFQgqpQiGzUEghVShkFgoppAqFzEIhhVShkFkopJAqFDILhRRShUJmoZBCqlDILBRSSBUKmYVCCqlCIbNQSCFVKGQWCimkCoXMQiGFVKGQWSikkCoUMguFFFKFQmahkEKqUMgsFFJIFQqZhUIKqUIhs1BIIVUoZBYKKaQKhcxCIYVUoZBZKKSQKhQyC4UUUoVCZqGQQqpQyCwUUkgVCpmFQgqpQiGzUEghVShkFgoppAqFzEIhhVShkFkopJAqFDILhRRShUJmoZBCqlDILBRSSBUKmYVCCqlCIbNQSCFVKGQWCimkCoXMQiGFVKGQWSikkCoUMguFFFKFQmahkEKqUMgsFFJIFQqZhUIKqUIhs1BIIVUoZBYKKaQKhcxCIYVUoZBZKKSQKhQyC4UUUoVCZqGQQqpQyCwUUkgVCpmFQgqpQiGzUEghVShkFgoppAqFzEIhhVShkFkopJAqFDILhRRShUJmoZBCqlDILBRSSBUKmYVCCqlCIbNQSCFVKGQWCimkCoXMQiGFVKGQWSikkCoUMguFFFKFQmahkEKqUMgsFFJIFQqZhUI2WQIhs1BIIVUoZBYKKaQKhcxCIYVUZBZKKSQKhQyC4UUUoVCZqGQQqpQyCwUUkgVCpmFQgqpQiGzUEghVShkFgoppAqFzEIhhVShkFkopJAqFDILhRRShUJmoZBCqlDILBRSSBUKmYVCCqlCIbNQSCFVKGQWCimkCoXMQiGFVKGQWSikkCoUMguFFFKFQmahkEKqUMgsFFJIFQqZhUIq2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YWZrGQXZrKSXZjJSnZhJivZhZmsZBdmspJdmMlKdmEmK9mFmaxkF2aykl2YyUp2YSYr2YWZrGQXZrKSXZjJSnZhJivZhZmsZBdmspJdmMlKdmEmK9mFmaxkF2aykl2YyUp2YSYr2YWZrGQXZrKSXZjJSnZhJivZhZmsZBdmspJdmMlKdmEmK9mFmazkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+SIBxUex8mUWmejAAAAAElFTkSuQmCC",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADa0lEQVR4Ad3BMRHAQBDEMJs/6A0EFzdpXpJHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkbriTIFIWQSUsgkpZBJWyCCpkEVTIIqiQRVAhi6BCFkGFLIIKWQQVsggqZBFSyCKokEVQIYugQhZBv0mQu/E/CXI3/idB7sb/JMjd+J8EuRv/kyB3438S5G78T4Lcjf9JkLvxPwlyN/4nQe7G/yTI3fifBLkb/5Mgd+N/EuRu/E+C3I3/SZC78T8Jcjf+J0Huxv8kyN34nwS5G/+TIHfjfxLkbvxPgtyN/0mQu/E/CXI3/idB7sb/JMjd+J8EuRv/kyB3438S5G78T4Lcjf9JkLvxPwlyN/4nQe7G/yTI3fifBLkb/5Mgd+N/EuRu/E+C3I3/SZC78T8Jcjf+J5eSxP/kSpL4n1xJEv+TK0nif3IlSfxPriSJ/8mVJPE/uZIk/idXksT/5EqS+J9cSRL/kypJ4n9yJUn8T64kif/JlSTxP7mSJP4nV5LE/+RKkviP3EmaUHEmaUDFSZoQSUkaN2lCxUkaV3EmaVzFkaRxFUeaMIo80oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oR5JI0o8oSPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGP+ADOlsfJTarkTAAAAABJRU5ErkJggg==",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADaklEQVR4Ad3BMRHAQBDEMJs/6A0EFzdpXpJHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkbvxPgtyN/0mQu/E/CXI3/idB7sb/JMjd+J8EuRv/kyB3438S5G78T4Lcjf9JkLvxPwlyN/4nQe7G/yTI3fifBLkb/5Mgd+N/EuRu/E+C3I3/SZC78T8Jcjf+J0Huxv8kyN34nwS5G/+TIHfjfxLkbvxPgtyN/0mQu/E/CXI3/idB7sb/JMjd+J8EuRv/kyB3438S5G78T4KkEeRsBAmSRpCzESRIGkHORpAgaQQ5G0GCpBHkbAQJkkaQsxEkSBpBzkaQIGkEORtBgqQR5GwECZJGkLMRJEgaQc5GkCBpBDkbQYKkEeRsBAmSRpCzESRIGkHORpAgaQQ5G0GCpBHkbAQJkkaQsxEkSBpBzkaQIGkEORtBgqQR5GwECZJGkLMRJEgaQc5GkCBpBDkbQYKkEeRsBAmSRpCzESRIGkHORpAgaQQ5G0GCpBHkbAQJkkaQsxEkSBpBzkaQIGkEORtBgqQR5GwECZLGlfxP0riS/0kaV/I/SeNK/idpXMn/JI0r+Z+kcSX/kzSu5H+SxpX8T9K4kv9JGlfyP0njSv4naVzJ/ySNK/mfpHEl/5M0ruR/ksaV/E/SuJL/SRpX8j9J40r+J2lcyf8kjSv5n6RxJf+TNK7kf5LGlfxP0riS/0kaV/I/SeNK/idpXMn/JI0r+Z+kcSX/kzSu5H+SxpX8T9K4kv9JGldSRpAgaVxJGUGCpHElZQQJksaVlBEkSBpXUkaQIGlcSRlBgqRxJWUECZLGlZQRJEgaV1JGkCBpXEkZQYKkcSVlBAmSxpWUESRIGldSRpAgaVxJGUGCpHElZQQJksaVlBEkSBpXUkaQIGlcSRlBgqRxJWUECZLGlZQRJEgaV1JGkCBpXEkZQYKkcSVlBAmSxpWUESRIGldSRpAgaVxJGUGCpHElZQQJksaVlBEkSBpXUkaQIGlcSRlBgqRxJWUECZLGlZQRJEgaV1JGkCCPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGP+ADOlsfJTarkTAAAAABJRU5ErkJggg==",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADMElEQVR4Ad3BsRHAQAzDMHL/oZURVOg+hQE5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAqFDILhRRShUJmoZBCqlDILBRSSBUKmYVCCqlCIbNQSCFVKGQWCimkCoXMQiGFVKGQWSikkCoUMguFFFKFQmahkEKqUMgsFFJIFQqZhUIKqUIhs1BIIVUoZBYKKaQKhcxCIYVUoZBZKKSQKhQyC4UUUoVCZqGQQqpQyCwUUkgVCpmFQgqpQiGzUEghVShkFgoppAqFzEIhhVShkFkopJAqFDILhRRShUJmoZBCqlDILBRSSBUKmYVCCqlCIbNQSCFVKGQWCimkCoXMQiGFVKGQWSikkCoUMguFFFKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlPwiNrOQHoZGV/CA0spIfhEZW8oPQyEp+EBpZyQ9CIyv5QWhkJT8IjazkB6GRlfwgNLKSH4RGVvKD0MhKfhAaWckPQiMr+UFoZCU/CI2s5AehkZX8IDSykh+ERlbyg9DISn4QGlnJD0IjK/lBaGQlPwiNrOQHoZGV/CA0spIfhEZW8oPQyEp+EBpZyQ9CIyv5QWhkJT8IjazkB6GRlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVnJEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRHySwpskwQyoaAAAAAElFTkSuQmCC",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAAC7klEQVR4Ad3BwQ3AQAzDMGn/od0R/AhSHELKEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRcoRUYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ9U4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIX8IDQyJT8IjUzJD0IjU/KD0MiU/CA0MiU/CI1MyQ9CI1Pyg9DIlPwgNDIlPwiNTMkPQiNTMkPQiNT8oPQyJT8IDQyJT8IjUzJD0IjU/KD0MiU/CA0MiU/CI1MyQ9CI1Pyg9DIlPwgNDIlPwiNTMkPQiNT8oPQyJT8IDQyJT8IjUzJD0IjU/KD0MiU/CA0MiU/CI1MyQ9CI1Pyg9DIlPwgNDIlR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHfCexhcn27L0vAAAAAElFTkSuQmCC",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADaElEQVR4Ad3BwQkDQRDEQCn/oNsh6DEYjq2SR8gj5BHyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gjJI0PkCBpfIAESeMDJEgaHyBB0vgACZLGB0iQND5AgqTxARIkjQ+QIGl8gARJ4wMkSBofIEHS+AAJksYHSJA0PkCCpPEBEiSND5AgaXyABEnjAyRIGh8gQdL4AAmSxgdIkDQ+QIKk8QESJI0PkCBpfIAESeMDJEgaHyBB0vgACZLGB0iQND5AgqTxARIkjQ+QIGn8n1xJGv8nV5LG/8mVpPF/ciVp/J9cSRr/J1eSxv/JlaTxf3IlafyfXEka/ydXksb/yZWk8X9yJWn8n1xJGv8nV5LG/8mVpPF/ciVp/J9cSRr/J1eSxv/JlaTxf3IlafyfXEka/ydXksb/yZWk8X9yJWn8n1xJGv8nV5LG/8mVpPF/ciVp/J9cSRr/J1eSxv/JlaTxf3IlafyfXEkaV1JGkCBpXEkZQYKkcSVlBAmSxpWUESRIGldSRpAgaVxJGUGCpHElZQQJksaVlBEkSBpXUkaQIGlcSRlBgqRxJWUECZLGlZQRJEgaV1JGkCBpXEkZQYKkcSVlBAmSxpWUESRIGldSRpAgaVxJGUGCpHElZQQJksaVlBEkSBpXUkaQIGlcSRlBgqRxJWUECZLGlZQRJEgaV1JGkCBpXEkZQYKkcSVlBAmSxpWUESRIGldSRpAgaVxJGUGCpHElZQQJksaVlBEkSBpXUkaQIGlcSRlBgqQRJIwiV5JGkDCKXEkaQcIociVpBAmjyJWkESSMIleSRpAwilxJGkHCKHIlaQQJo8iVpBEkjCJXkkaQMIpcSRpBwihyJWkECaPIlaQRJIwiV5JGkDCKXEkaQcIociVpBAmjyJWkESSMIleSRpAwilxJGkHCKHIlaQQJo8iVpBEkjCJXkkaQMIpcSRpBwihyJWkECaPIlaQRJIwiV5JGkDCKXEkaQcIociVpBAmjyJWkESSMIleSRpAwilxJGkHCKHIlaQQJo8iVpBEkjCJX8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8ogfRSvIyUOE+8EAAAAASUVORK5CYII=",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADKklEQVR4Ad3BwQkAQRDDMLv/onMl5BEOlpHkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKqsJL/SRVW8j+pwkr+J1VYyf+kCiv5n1RhJf+TKqzkf1KFlfxPqrCS/0kVVvI/qcJK/idVWMn/pAor+Z9UYSX/kyqs5H9ShZX8T6qwkv9JFVbyP6nCSv4nVVjJ/6QKK/mfVGEl/5MqrOR/UoWV/E+qsJL/SRVW8j+pwkr+J1VYyf+kCiv5n1RhJf+TKqzkf1KFlfxPqrCS/0kVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCFVOEBUkgVHiCF7EIhTSikkF0opAmFFLILhTShkEJ2oZAmFFLILhTShEIK2YVCmlBIIbtQSBMKKWQXCmlCIYXsQiFNKKSQXSikCYUUsguFNKGQQnahkCYUUsguFNKEQgrZhUKaUEghu1BIEwopZBcKaUIhhexCIU0opJBdKKQJhRSyC4U0oZBCdqGQJhRSyC4U0oRCCtmFQppQSCG7UEgTCilkFwppQiGF7EIhTSikkF0opAmFFLILhTShkEJ2oZAmFFLILhTShEIK2YVCmlBIIbtQSBMKKWQXCmlCIYXsQiFNKKSQXSikCYUUsgszWckuzGQluzCTlezCTFayCzNZyS7MZCW7MJOV7MJMVrILM1nJLsxkJbswk5XswkxWsgszWckuzGQluzCTlezCTFayCzNZyS7MZCW7MJOV7MJMVrILM1nJLsxkJbswk5XswkxWsgszWckuzGQluzCTlezCTFayCzNZyS7MZCW7MJOV7MJMVrILM1nJEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRH+I7p8la1PQFAAAAAElFTkSuQmCC",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADWklEQVR4Ad3BMREAIADEsNa/6EdCBxaORD4hn5BPyCfkE/IJ+YR8Qj4hn5BPyCfkE/IJ+YR8Qj4hn5BPyCfkE/IJ+YR8Qj4hn5BPyCfkE/IJ+YR8Qj4hn5A0HiBB0niABEnjARIkjQdIkDQeIEHSeIAESeMBEiSNB0iQNB4gQdJ4gARJ4wESJI0HSJA0HiBB0niABEnjARIkjQdIkDQeIEHSeIAESeMBEiSNB0iQNB4gQdJ4gARJ4wESJI0HSJA0HiBB0niABEnjARIkjQdIkDQeIEHSeIAESeMBEiSNB0iQNB4gQdIIcm0ECZJGkGsjSJA0glwbQYKkEeTaCBIkjSDXRpAgaQS5NoIESSPItREkSBpBro0gQdIIcm0ECZJGkGsjSJA0glwbQYKkEeTaCBIkjSDXRpAgaQS5NoIESSPItREkSBpBro0gQdIIcm0ECZJGkGsjSJA0glwbQYKkEeTaCBIkjSDXRpAgaQS5NoIESSPItREkSBpBro0gQdIIcm0ECZJGkGsjSJA0glwbQYKkEeTaCBIkjSDXRpAgaQS5NoIESSPItREkSBpBro0gQdIIcm0ECZJGkGsjSJA0glwbQYKkEeTaCBIkjSDXRpAgaQS5NoIESSPItREkSBpBro0gQdIIcm0ECZJGkGsjSJA0glwbQYKkEeTaCBIkjSDXRpAgaQS5NoIESSPItREkSBpBro0gQdIIcm0ECZJGkGsjSJA0glwbQYKkEeTaCBIkjSDXRpAgaQS5NoIESSPItREkSBpBro0gQdIIcm0ECZJGkGsjSJA0glwbQYKkEeTaCBIkjSDXRpAgaQS5NoIESSPItREkSBpBro0gQdIIcm0ECZJGkGsjSJA0glwbQYI8YQQJ8oQRJMgTRpAgTxhBgjxhBAnyhBEkyBNGkCBPGEGCPGEECfKEESTIE0aQIE8YQYI8YQQJ8oQRJMgTRpAgTxhBgjxhBAnyhBEkyBNGkCBPGEGCPGEECfKEESTIE0aQIE8YQYI8YQQJ8oQRJMgTRpAgTxhBgjxhBAnyhBEkyBNGkCBPGEGCPGEECfIJ+YR8Qj4hn5BPyCfkE/IJ+YR8Qj4hn5BPyCfkE/IJ+YR8Qj4hn5BPyCfkE/IJ+YR8Qj4hn5BPyCfkE/KJA2wKyMkUkzvgAAAAAElFTkSuQmCC",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADJklEQVR4Ad3BwQnAMADEMHv/oa8j+BEKIZI8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hVxhBglxhBAlyhREkyBVGkCBXGEGCXGEECXKFESTIFUaQIFcYQYJcYQQJcoURJMgVRpAgVxhBglxhBAlyhREkyBVGkCBXGEGCXGEECXKFESTIFUaQIFcYQYJcYQQJcoURJMgVRpAgVxhBglxhBAlyhREkyBVGkCBXGEGCXGEECXKFESTIFUaQIFcYQYJcYQQJcoURJMgVRpAgVxhBglxhBAlyhREkyBVGkCBXGEGCXGEECXKFESTIFUaQIFcYQYJcYQQJcoURJMgVRpAgVxhBglxhBAlyhREkyBVGkCBXGEGCXGEECXKFESTIFUaQIFcYQYJcYQQJcoURJMgVRpAgVxhBglxhBAlyhREkyBVGkCBXGEGCXGEECZJGkDCKnJI0goRR5JSkESSMIqckjSBhFDklaQQJo8gpSSNIGEVOSRpBwihyStIIEkaRU5JGkDCKnJI0goRR5JSkESSMIqckjSBhFDklaQQJo8gpSSNIGEVOSRpBwihyStIIEkaRU5JGkDCKnJI0goRR5JSkESSMIqckjSBhFDklaQQJo8gpSSNIGEVOSRpBwihyStIIEkaRU5JGkDCKnJI0goRR5JSkESSMIqckjSBhFDklaQQJo8gpSSNIGEVOSRpBwihyStIIEkaRU5JGkDCKnJI0goRR5JScG/+TIOfG/yTIufE/CXJu/E+CnBv/kyDnxv8kyLnxPwlybvxPgpwb/5Mg58b/JMi58T8Jcm78T4KcG/+TIOfG/yTIufE/CXJu/E+CnBv/kyDnxv8kyLnxPwlybvxPgpwb/5Mg58b/JMi58T8Jcm78T4KcG/+TIOfG/yTIufE/CXJu/E+CnBv/kyDnxv8kyLnxPwlybvxPgpwb/5Mgj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj/gADA2nyYSQePEAAAAASUVORK5CYII="
];

function printTemplateSheet(templateName) {
  const tpl = SHEET_TEMPLATES.find(t => t.name === templateName);
  if (!tpl) return;

  const title = `Möbius - Hoja ${tpl.name}`;
  
  const PAGE_SIZES = {
    'A5': { w: 148, h: 210 },
    'A4': { w: 210, h: 297 },
    'A3': { w: 297, h: 420 }
  };
  
  const pageSize = PAGE_SIZES[tpl.name] || { w: tpl.w_mm + 40, h: tpl.h_mm + 40 };

  const offsetX = (pageSize.w - tpl.w_mm) / 2;
  const offsetY = (pageSize.h - tpl.h_mm) / 2;
  
  const html = `
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8">
      <title>${title}</title>
      <style>
        body { margin: 0; padding: 0; background: #fff; color: #000; font-family: sans-serif; text-align: center; }
        .page { 
          position: relative; 
          width: ${pageSize.w}mm; 
          height: ${pageSize.h}mm; 
          margin: 0 auto;
          box-sizing: border-box;
          background: #fff;
          overflow: hidden;
          page-break-after: always;
        }
        .safe-zone {
          position: absolute;
          top: 10mm; left: 10mm; right: 10mm; bottom: 10mm;
          border: 2px dashed #bbb;
          border-radius: 8mm;
          pointer-events: none;
        }
        .safe-zone::before {
          content: 'ZONA PARA CINTA';
          position: absolute;
          top: -3mm; left: 50%; transform: translateX(-50%);
          background: #fff;
          padding: 0 10px;
          color: #999;
          font-size: 12px;
          font-weight: bold;
          letter-spacing: 2px;
        }
        .marker {
          position: absolute;
          width: 20mm;
          height: 20mm;
          transform: translate(-50%, -50%);
        }
        .marker img {
          width: 100%;
          height: 100%;
          image-rendering: pixelated;
        }
        .label {
          position: absolute;
          font-size: 10px;
          margin-top: 21mm;
          transform: translate(-50%, 0);
          color: #555;
          white-space: nowrap;
        }
        @media print {
          @page { 
            size: ${pageSize.w}mm ${pageSize.h}mm; 
            margin: 0 !important; 
          }
          body { margin: 0; }
          .page { border: none; }
          .no-print { display: none; }
        }
      </style>
    </head>
    <body>
      <div class="no-print" style="padding: 20px;">
        <h2>${title}</h2>
        <p>Imprimí esta página usando "Guardar como PDF" o enviala a tu impresora. <b>Asegurate de imprimir al 100% de escala (sin ajustar a los márgenes).</b></p>
        <button onclick="window.print()" style="padding: 10px 20px; font-size: 16px; cursor: pointer;">🖨️ Imprimir</button>
      </div>

      <div class="page">
        <div class="safe-zone"></div>
        ${tpl.targets.map((target, idx) => {
          const x = target[0] + offsetX; 
          const y = target[1] + offsetY;
          const id = tpl.ids[idx];
          return `
            <div class="marker" style="left: ${x}mm; top: ${y}mm;">
              <img src="${MARKER_B64[id]}" alt="ArUco ID ${id}">
              <div class="label">ID ${id}</div>
            </div>
          `;
        }).join('')}
      </div>
    </body>
    </html>
  `;

  const printWindow = window.open('', '_blank');
  if (printWindow) {
    printWindow.document.open();
    printWindow.document.write(html);
    printWindow.document.close();
  }
}
