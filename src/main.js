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
  findNearestEdge,
  applyH
} from './vision/math.js';
import {
  calcularEstructuraEscena,
  calcularAlturaLibre,
  getCameraParams,
  project3D,
  intersectRayWithVirtualPlane,
  projectVirtualPlane,
  calibrateVirtualPlaneScale
} from './vision/geometry3d.js';


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
  },
  
  // 3D Measurement Mode
  activeTool: '2d', // '2d' | '3d' | 'virtual_plane'
  planeBaseIndex: -1,
  planeWallIndex: -1,
  
  // Virtual plane state
  virtualPlanePoints: [], // Points clicked on the floor to define the base line
  virtualPlaneMeasurements: [], // Measurements on the virtual plane
  virtualPlanePendingPoint: null,
  isTiltingVirtualPlane: false,
  virtualPlaneTiltPoints: [],
  virtualPlaneNormal: null,
  vanishingPointH: null,
  vanishingPointV: null,        // punto de fuga vertical calibrado en imagen
  vanishingPointVPoints: null,  // [{x,y}, {x,y}] los dos puntos de la recta vertical
  isCalibVertVirtualPlane: false,
  meas3dBasePoint: null, // {x, y} px
  isSelectingExportArea: false,
  exportAreaPoints: null,
  virtualPlaneBaseIntersect: null,
  virtualPlaneTopIntersect: null,
  showFlatView: false,
  flatViewTransform: null,
  flatViewPlaneId: -1,
  showGrid: true,
  dragModeEnabled: false
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

  if (state.viewer.initialized) return;
  state.viewer.initialized = true;

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
    const isDragButton = e.button === 1 || e.button === 2 || (state.dragModeEnabled && e.button === 0);
    if (!isDragButton) return; 
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
    const isDragButton = e.button === 1 || e.button === 2 || (state.dragModeEnabled && e.button === 0);
    if (!isDragButton) return;
    state.viewer.isDragging = false;
    wrapper.style.cursor = state.dragModeEnabled ? 'grab' : '';
  });

  // Pan y Zoom - TOUCH (2 dedos, o 1 dedo en modo arrastre)
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
    } else if (e.touches.length === 1 && state.dragModeEnabled) {
      e.preventDefault();
      const t = e.touches[0];
      state.viewer.isDragging = true;
      state.viewer.startX = t.clientX - state.viewer.offsetX;
      state.viewer.startY = t.clientY - state.viewer.offsetY;
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
    } else if (e.touches.length === 1 && state.dragModeEnabled && state.viewer.isDragging) {
      e.preventDefault();
      const t = e.touches[0];
      state.viewer.offsetX = t.clientX - state.viewer.startX;
      state.viewer.offsetY = t.clientY - state.viewer.startY;
      updateViewTransform();
    }
  }, { passive: false });

  wrapper.addEventListener('touchend', e => {
    if (e.touches.length === 0) {
      initialPinchDist = null;
      state.viewer.isDragging = false;
    } else if (e.touches.length === 1 && state.dragModeEnabled) {
      initialPinchDist = null;
    } else if (e.touches.length < 2) {
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

  const btnDrag = document.getElementById('btn-drag-toggle');
  if (btnDrag) {
    btnDrag.onclick = () => {
      state.dragModeEnabled = !state.dragModeEnabled;
      if (state.dragModeEnabled) {
        btnDrag.style.background = 'var(--primary)';
        btnDrag.style.color = 'white';
        btnDrag.title = 'Modo Navegación/Arrastre (Activado)';
        wrapper.style.cursor = 'grab';
      } else {
        btnDrag.style.background = '';
        btnDrag.style.color = '';
        btnDrag.title = 'Modo Navegación/Arrastre (Desactivado)';
        wrapper.style.cursor = '';
      }
    };
  }

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
    if (!plane) return { imgX, imgY };

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

  function handleVirtualPlaneClick(imgX, imgY) {
    const plane = state.planes[state.activePlaneIndex];
    if (!plane || !plane.isVirtual) return;

    if (plane.points.length < 2) {
      plane.points.push({ x: imgX, y: imgY });
      if (plane.points.length === 2) {
        recomputeVirtualPlane(plane);
        updatePlaneListUI();
        setActivePlane(state.activePlaneIndex);
        toast('Plano virtual definido. Ahora podés medir sobre él o ajustar perspectiva/vertical.', 'success');
        drawGrid();
      } else {
        toast('Hacé clic en otro punto del piso para definir la línea base.', 'info');
      }
      drawMeasurements();
    } else if (state.isCalibVertVirtualPlane) {
      if (!plane.vanishingPointVPoints) plane.vanishingPointVPoints = [];
      plane.vanishingPointVPoints.push({ x: imgX, y: imgY });
      const ptsCount = plane.vanishingPointVPoints.length;
      if (ptsCount === 1) {
        toast('Primer punto de la primera recta vertical marcado. Marcá el segundo punto de la misma recta.', 'info');
      } else if (ptsCount === 2) {
        toast('Primera recta vertical marcada. Marcá el primer punto de la segunda recta vertical.', 'info');
      } else if (ptsCount === 3) {
        toast('Primer punto de la segunda recta vertical marcado. Marcá el segundo punto de la misma recta.', 'info');
      } else if (ptsCount === 4) {
        const vp1 = plane.vanishingPointVPoints[0];
        const vp2 = plane.vanishingPointVPoints[1];
        const vp3 = plane.vanishingPointVPoints[2];
        const vp4 = plane.vanishingPointVPoints[3];

        const L1 = {
          x: vp1.y - vp2.y,
          y: vp2.x - vp1.x,
          z: vp1.x * vp2.y - vp1.y * vp2.x
        };
        const L2 = {
          x: vp3.y - vp4.y,
          y: vp4.x - vp3.x,
          z: vp3.x * vp4.y - vp3.y * vp4.x
        };

        const V_v = {
          x: L1.y * L2.z - L1.z * L2.y,
          y: L1.z * L2.x - L1.x * L2.z,
          z: L1.x * L2.y - L1.y * L2.x
        };

        if (Math.abs(V_v.z) > 1e-7) {
          plane.vanishingPointV = { x: V_v.x / V_v.z, y: V_v.y / V_v.z };
          toast('Fuga vertical calibrada con éxito.', 'success');
        } else {
          const dx1 = vp2.x - vp1.x;
          const dy1 = vp2.y - vp1.y;
          const dx2 = vp4.x - vp3.x;
          const dy2 = vp4.y - vp3.y;
          const dx = (dx1 + dx2) / 2;
          const dy = (dy1 + dy2) / 2;
          const len = Math.hypot(dx, dy) || 1;
          const farScale = 100000;
          plane.vanishingPointV = {
            x: vp1.x + (dx / len) * farScale,
            y: vp1.y + (dy / len) * farScale
          };
          toast('Rectas verticales paralelas detectadas. Fuga vertical en el infinito.', 'info');
        }
        state.isCalibVertVirtualPlane = false;
        recomputeVirtualPlane(plane);
        drawGrid();
        drawMeasurements();
      }
    } else if (state.isTiltingVirtualPlane) {
      if (!plane.virtualPlaneTiltPoints) plane.virtualPlaneTiltPoints = [];
      plane.virtualPlaneTiltPoints.push({ x: imgX, y: imgY });
      const pts = plane.virtualPlaneTiltPoints.length;
      if (pts === 1) {
        toast('Primer punto marcado. Marcá el segundo punto de la recta paralela.', 'info');
      } else if (pts === 2) {
        const bp1 = plane.points[0];
        const bp2 = plane.points[1];
        const tp1 = plane.virtualPlaneTiltPoints[0];
        const tp2 = plane.virtualPlaneTiltPoints[1];
        
        const L_base = {
          x: bp1.y - bp2.y,
          y: bp2.x - bp1.x,
          z: bp1.x * bp2.y - bp1.y * bp2.x
        };
        const L_tilt = {
          x: tp1.y - tp2.y,
          y: tp2.x - tp1.x,
          z: tp1.x * tp2.y - tp1.y * tp2.x
        };
        
        const V_h = {
          x: L_base.y * L_tilt.z - L_base.z * L_tilt.y,
          y: L_base.z * L_tilt.x - L_base.x * L_tilt.z,
          z: L_base.x * L_tilt.y - L_base.y * L_tilt.x
        };
        
        if (Math.abs(V_h.z) > 1e-7) {
          plane.vanishingPointH = { x: V_h.x / V_h.z, y: V_h.y / V_h.z };
          toast('Perspectiva horizontal calibrada.', 'success');
        } else {
          plane.vanishingPointH = null;
          toast('Rectas paralelas en imagen — sin convergencia.', 'info');
        }
        
        state.isTiltingVirtualPlane = false;
        recomputeVirtualPlane(plane);
        drawGrid();
        drawMeasurements();
      }
    }
  }
  
  function renderVirtualPlaneInteractiveLine(mouseX, mouseY) {
    const canvas = document.getElementById('interactive-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const plane = state.planes[state.activePlaneIndex];
    if (!plane || !plane.isVirtual) return;
    
    if (state.isTiltingVirtualPlane) {
      const pts = plane.virtualPlaneTiltPoints || [];
      // Dibujar la recta base como rayo de referencia (yellow)
      if (plane.points.length === 2) {
        const bp1 = plane.points[0];
        const bp2 = plane.points[1];
        const bdx = bp2.x - bp1.x; const bdy = bp2.y - bp1.y;
        const blen = Math.hypot(bdx, bdy) || 1;
        const bnx = bdx / blen; const bny = bdy / blen;
        ctx.beginPath();
        ctx.moveTo(bp1.x - bnx * 5000, bp1.y - bny * 5000);
        ctx.lineTo(bp2.x + bnx * 5000, bp2.y + bny * 5000);
        ctx.strokeStyle = 'rgba(250, 204, 21, 0.5)';
        ctx.lineWidth = 1.5 / state.viewer.scale;
        ctx.setLineDash([8 / state.viewer.scale, 4 / state.viewer.scale]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      // Dibujar la nueva recta paralela interactiva
      if (pts.length === 1) {
        ctx.beginPath();
        const tdx = mouseX - pts[0].x; const tdy = mouseY - pts[0].y;
        const tlen = Math.hypot(tdx, tdy) || 1;
        const tnx = tdx / tlen; const tny = tdy / tlen;
        ctx.moveTo(pts[0].x - tnx * 5000, pts[0].y - tny * 5000);
        ctx.lineTo(pts[0].x + tnx * 5000, pts[0].y + tny * 5000);
        ctx.strokeStyle = '#a855f7';
        ctx.lineWidth = 2 / state.viewer.scale;
        ctx.setLineDash([5 / state.viewer.scale, 5 / state.viewer.scale]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    } else if (plane.points.length === 1) {
      const b = plane.points[0];
      ctx.beginPath();
      ctx.moveTo(b.x, b.y);
      ctx.lineTo(mouseX, mouseY);
      ctx.strokeStyle = '#facc15'; // yellow for base line
      ctx.lineWidth = 2 / state.viewer.scale;
      ctx.stroke();
    }
  }

  function commitMeasurementPoint(point) {
    if (!point) return;

    // 2D logic
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

  // Interacción 3D: Dibujar línea elástica al mover el mouse
  window.addEventListener('mousemove', e => {
    if (!state.viewer.isDragging) {
      const rect = wrapper.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;
      
      const plane = state.planes[state.activePlaneIndex];
      const isCalibratingVirtual = plane && plane.isVirtual && (plane.points.length < 2 || state.isCalibVertVirtualPlane || state.isTiltingVirtualPlane);
      if (isCalibratingVirtual) {
        renderVirtualPlaneInteractiveLine((clickX - state.viewer.offsetX) / state.viewer.scale, (clickY - state.viewer.offsetY) / state.viewer.scale);
      }
    }
  });

  // Medición - MOUSE
  overlayCanvas.addEventListener('mousedown', e => {
    if (state.dragModeEnabled) return;
    if (e.button !== 0 || e.shiftKey || e.ctrlKey) return; 
    const rect = wrapper.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const imgX = (x - state.viewer.offsetX) / state.viewer.scale;
    const imgY = (y - state.viewer.offsetY) / state.viewer.scale;

    if (state.isSelectingExportArea) {
      handleExportAreaClick(imgX, imgY);
      return;
    }

    const plane = state.planes[state.activePlaneIndex];
    const isCalibratingVirtual = plane && plane.isVirtual && (plane.points.length < 2 || state.isCalibVertVirtualPlane || state.isTiltingVirtualPlane);

    if (isCalibratingVirtual) {
      handleVirtualPlaneClick(imgX, imgY);
    } else {
      if (plane && !plane.H) {
        toast('Calibrá la base del plano virtual primero haciendo clic en el piso.', 'warning');
        return;
      }
      const point = processClickPoint(x, y);
      commitMeasurementPoint(point);
    }
  });

  // Medición - TOUCH (1 dedo)
  overlayCanvas.addEventListener('touchstart', e => {
    if (state.dragModeEnabled) return;
    if (e.touches.length === 1) {
      e.preventDefault(); 
      const rect = wrapper.getBoundingClientRect();
      const touch = e.touches[0];
      const clickX = touch.clientX - rect.left;
      const clickY = touch.clientY - rect.top - 60; // Offset Y arriba del dedo
      
      const plane = state.planes[state.activePlaneIndex];
      if (plane && (plane.isVirtual && plane.points.length < 2)) {
        currentTouchPoint = {
          imgX: (clickX - state.viewer.offsetX) / state.viewer.scale,
          imgY: (clickY - state.viewer.offsetY) / state.viewer.scale
        };
      } else {
        currentTouchPoint = processClickPoint(clickX, clickY);
      }
      drawMeasurements(pendingPoint, currentTouchPoint);
    }
  }, { passive: false });

  overlayCanvas.addEventListener('touchmove', e => {
    if (state.dragModeEnabled) return;
    if (e.touches.length === 1 && currentTouchPoint) {
      e.preventDefault();
      const rect = wrapper.getBoundingClientRect();
      const touch = e.touches[0];
      const clickX = touch.clientX - rect.left;
      const clickY = touch.clientY - rect.top - 60; 
      
      const plane = state.planes[state.activePlaneIndex];
      if (plane && (plane.isVirtual && plane.points.length < 2)) {
        currentTouchPoint = {
          imgX: (clickX - state.viewer.offsetX) / state.viewer.scale,
          imgY: (clickY - state.viewer.offsetY) / state.viewer.scale
        };
      } else {
        currentTouchPoint = processClickPoint(clickX, clickY);
      }
      drawMeasurements(pendingPoint, currentTouchPoint);
    }
  }, { passive: false });

  overlayCanvas.addEventListener('touchend', e => {
    if (state.dragModeEnabled) return;
    if (e.changedTouches.length === 1 && currentTouchPoint) {
      e.preventDefault();
      if (state.isSelectingExportArea) {
        handleExportAreaClick(currentTouchPoint.imgX, currentTouchPoint.imgY);
        currentTouchPoint = null;
        drawMeasurements(pendingPoint);
        return;
      }

      const plane = state.planes[state.activePlaneIndex];
      const isCalibratingVirtual = plane && plane.isVirtual && (plane.points.length < 2 || state.isCalibVertVirtualPlane || state.isTiltingVirtualPlane);

      if (isCalibratingVirtual) {
        handleVirtualPlaneClick(currentTouchPoint.imgX, currentTouchPoint.imgY);
      } else {
        if (plane && !plane.H) {
          toast('Calibrá la base del plano virtual primero haciendo clic en el piso.', 'warning');
          currentTouchPoint = null;
          drawMeasurements(pendingPoint);
          return;
        }
        commitMeasurementPoint(currentTouchPoint);
      }
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

function syncCanvasSizes() {
  const mainCanvas = document.getElementById('main-image-canvas');
  const overlayCanvas = document.getElementById('overlay-canvas');
  const interactiveCanvas = document.getElementById('interactive-canvas');
  if (mainCanvas && overlayCanvas) {
    if (overlayCanvas.width !== mainCanvas.width || overlayCanvas.height !== mainCanvas.height) {
      overlayCanvas.width = mainCanvas.width;
      overlayCanvas.height = mainCanvas.height;
    }
  }
  if (mainCanvas && interactiveCanvas) {
    if (interactiveCanvas.width !== mainCanvas.width || interactiveCanvas.height !== mainCanvas.height) {
      interactiveCanvas.width = mainCanvas.width;
      interactiveCanvas.height = mainCanvas.height;
    }
  }
}

function drawGrid() {
  syncCanvasSizes();
  const overlay = document.getElementById('overlay-canvas');
  if (!overlay) return;
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!state.showGrid) return;

  const plane = state.planes[state.activePlaneIndex];
  if (!plane) return;

  let H_inv = plane.H_inv;
  let useFallbackBase = false;
  if (!H_inv && (plane.isVirtual || plane.isElevated)) {
    const pb = state.planes.find(p => p.id === plane.planeBaseIndex);
    if (pb && pb.H_inv) {
      H_inv = pb.H_inv;
      useFallbackBase = true;
    }
  }
  if (!H_inv) return;
  const H_data = H_inv.data64F;

  const isFlat = state.showFlatView && state.flatViewTransform && plane && plane.id === state.flatViewPlaneId;

  function project(x_mm, y_mm) {
    if (isFlat) {
      return {
        x: (x_mm - state.flatViewTransform.X_min) * state.flatViewTransform.scale,
        y: (y_mm - state.flatViewTransform.Y_min) * state.flatViewTransform.scale,
        w: 1.0
      };
    }
    const w = H_data[6] * x_mm + H_data[7] * y_mm + H_data[8];
    return {
      x: (H_data[0] * x_mm + H_data[1] * y_mm + H_data[2]) / w,
      y: (H_data[3] * x_mm + H_data[4] * y_mm + H_data[5]) / w,
      w: w
    };
  }

  const range = 800; // mm
  const step = 50;
  const w_sign = isFlat ? 1.0 : (Math.sign(H_data[8]) || 1);

  let startX = -range, endX = range;
  let startY = -range, endY = range;

  if (isFlat) {
    startX = Math.floor(state.flatViewTransform.X_min / step) * step;
    endX = Math.ceil(state.flatViewTransform.X_max / step) * step;
    startY = Math.floor(state.flatViewTransform.Y_min / step) * step;
    endY = Math.ceil(state.flatViewTransform.Y_max / step) * step;
  }

  if (plane.isVirtual && !useFallbackBase) {
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.5)'; // Purple for virtual vertical
  } else if (plane.isElevated && !useFallbackBase) {
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.5)'; // Orange for elevated parallel
  } else {
    ctx.strokeStyle = 'rgba(110, 231, 247, 0.4)'; // Cyan for physical or base fallback
  }
  ctx.lineWidth = 1 / state.viewer.scale;
  ctx.beginPath();
  
  // Dibujar grilla en pequeños segmentos para evitar cruces con el horizonte
  for (let x = startX; x <= endX; x += step) {
    for (let y = startY; y < endY; y += step) {
      let p1 = project(x, y), p2 = project(x, y + step);
      if ((Math.sign(p1.w) || 1) === w_sign && (Math.sign(p2.w) || 1) === w_sign) {
        ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
      }
    }
  }
  for (let y = startY; y <= endY; y += step) {
    for (let x = startX; x < endX; x += step) {
      let p1 = project(x, y), p2 = project(x + step, y);
      if ((Math.sign(p1.w) || 1) === w_sign && (Math.sign(p2.w) || 1) === w_sign) {
        ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
      }
    }
  }
  ctx.stroke();

  // Ejes X e Y locales (Origen ArUco)
  ctx.lineWidth = 2 / state.viewer.scale;
  ctx.beginPath(); ctx.strokeStyle = 'rgba(248, 113, 113, 0.8)'; // Red = X
  for (let x = 0; x < 100; x += 10) {
    let p1 = project(x, 0), p2 = project(x + 10, 0);
    if ((Math.sign(p1.w) || 1) === w_sign && (Math.sign(p2.w) || 1) === w_sign) { ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); }
  }
  ctx.stroke();

  ctx.beginPath(); ctx.strokeStyle = 'rgba(52, 211, 153, 0.8)'; // Green = Y
  for (let y = 0; y < 100; y += 10) {
    let p1 = project(0, y), p2 = project(0, y + 10);
    if ((Math.sign(p1.w) || 1) === w_sign && (Math.sign(p2.w) || 1) === w_sign) { ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); }
  }
  ctx.stroke();
  
  // Intersección de plano virtual en la realidad (Cyan/Naranja) para feedback visual
  if (!isFlat && plane.isVirtual && plane.points.length >= 2 && state.planes.length >= 2) {
    const pb = state.planes.find(p => p.id === plane.planeBaseIndex);
    const pw = state.planes.find(p => p.id === plane.planeWallIndex);
    if (pb && pw) {
      const mainCanvas = document.getElementById('main-image-canvas');
      const params = getCameraParams(pb.H_inv, pw.H_inv, mainCanvas.width, mainCanvas.height);
      const H_data_base = pb.H.data64F;
      const projToFloor = (pt) => {
        const w = H_data_base[6] * pt.x + H_data_base[7] * pt.y + H_data_base[8];
        return {
          x: (H_data_base[0] * pt.x + H_data_base[1] * pt.y + H_data_base[2]) / w,
          y: (H_data_base[3] * pt.x + H_data_base[4] * pt.y + H_data_base[5]) / w
        };
      };
      const b1_mm = projToFloor(plane.points[0]);
      const b2_mm = projToFloor(plane.points[1]);

      const calib = calibrateVirtualPlaneScale(
        params, pb.H, pb.H_inv, pw.H, pw.H_inv,
        b1_mm, b2_mm, plane.vanishingPointH, plane.vanishingPointV
      );

      if (calib && calib.P_base_intersect && calib.P_top_intersect) {
        // Dibujar recta de intersección horizontal (Piso)
        ctx.beginPath();
        ctx.strokeStyle = '#06b6d4'; // Cyan
        ctx.lineWidth = 3 / state.viewer.scale;
        
        const dx_mm = b2_mm.x - b1_mm.x;
        const dy_mm = b2_mm.y - b1_mm.y;
        const lenU_mm = Math.hypot(dx_mm, dy_mm) || 1;
        const stepU_mm = 50;
        const nU_extra = 100;
        const iStart = -nU_extra;
        const iEnd = Math.ceil(lenU_mm / stepU_mm) + nU_extra;

        const pBaseL = projectVirtualPlane(params, b1_mm, b2_mm, iStart * stepU_mm, 0, plane.vanishingPointH, plane.vanishingPointV);
        const pBaseR = projectVirtualPlane(params, b1_mm, b2_mm, iEnd * stepU_mm, 0, plane.vanishingPointH, plane.vanishingPointV);
        if (pBaseL && pBaseR) {
          ctx.moveTo(pBaseL.x, pBaseL.y);
          ctx.lineTo(pBaseR.x, pBaseR.y);
          ctx.stroke();
        }

        // Dibujar recta de intersección vertical (Pared)
        const p1 = calib.P_base_intersect;
        const p2 = calib.P_top_intersect;
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const len = Math.hypot(dx, dy) || 1;
        const nx = dx / len;
        const ny = dy / len;

        ctx.beginPath();
        ctx.strokeStyle = '#f97316'; // Orange
        ctx.lineWidth = 3 / state.viewer.scale;
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p1.x + nx * 5000, p1.y + ny * 5000);
        ctx.stroke();
      }
    }
  }
}

function drawMeasurements(pendingPoint = null, currentTouchPoint = null) {
  const overlay = document.getElementById('overlay-canvas');
  if (!overlay) return;
  const ctx = overlay.getContext('2d');
  
  drawGrid();

  const plane = state.planes[state.activePlaneIndex];
  if (!plane) return;

  const isFlat = state.showFlatView && state.flatViewTransform && plane && plane.id === state.flatViewPlaneId;
  const H_inv = plane.H_inv;
  const H_data = H_inv ? H_inv.data64F : null;

  function project(pt) {
    if (isFlat) {
      return {
        x: (pt.x_mm - state.flatViewTransform.X_min) * state.flatViewTransform.scale,
        y: (pt.y_mm - state.flatViewTransform.Y_min) * state.flatViewTransform.scale
      };
    }
    // Si no es vista plana y tiene la coordenada original de la imagen, usarla
    if (pt.imgX !== undefined && pt.imgY !== undefined) {
      return { x: pt.imgX, y: pt.imgY };
    }
    if (!H_data) return { x: 0, y: 0 };
    const w = H_data[6] * pt.x_mm + H_data[7] * pt.y_mm + H_data[8];
    return {
      x: (H_data[0] * pt.x_mm + H_data[1] * pt.y_mm + H_data[2]) / w,
      y: (H_data[3] * pt.x_mm + H_data[4] * pt.y_mm + H_data[5]) / w
    };
  }

  const planeMeasurements = state.measurements.filter(m => m.planeId === plane.id);
  // Medidas del plano virtual (separadas)
  const virtualMeasurements = state.measurements.filter(m => m.isVirtual);
  const strokeW = 2 / state.viewer.scale;
  const r = 4 / state.viewer.scale;

  const drawPt = (p) => {
    const pt = project(p);
    ctx.beginPath(); ctx.arc(pt.x, pt.y, r, 0, 2*Math.PI);
    ctx.fillStyle = '#6ee7f7'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();
  };

  planeMeasurements.forEach(m => {
    if (m.isAuto) return; 
    
    const p1 = m.points[0], p2 = m.points[1];
    const pt1 = project(p1);
    const pt2 = project(p2);
    const x1 = pt1.x, y1 = pt1.y;
    const x2 = pt2.x, y2 = pt2.y;
    
    ctx.beginPath(); ctx.arc(x1, y1, r, 0, 2*Math.PI);
    ctx.fillStyle = '#6ee7f7'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();
    
    ctx.beginPath(); ctx.arc(x2, y2, r, 0, 2*Math.PI);
    ctx.fillStyle = '#6ee7f7'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();

    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.strokeStyle = '#6ee7f7'; ctx.lineWidth = strokeW;
    ctx.setLineDash([8 / state.viewer.scale, 5 / state.viewer.scale]);
    ctx.stroke(); ctx.setLineDash([]);
    
    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;
    ctx.font = `600 ${14 / state.viewer.scale}px 'Inter', sans-serif`;
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    
    const text = `${m.distMm.toFixed(1)} mm`;
    const txtMetrics = ctx.measureText(text);
    const textW = txtMetrics.width;
    const textH = 14 / state.viewer.scale;
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(midX - textW/2 - 4/state.viewer.scale, midY - textH/2 - 12/state.viewer.scale, textW + 8/state.viewer.scale, textH + 4/state.viewer.scale);
    
    ctx.fillStyle = '#fff';
    ctx.fillText(text, midX, midY - (10/state.viewer.scale));
  });

  if (pendingPoint) drawPt(pendingPoint);
  if (currentTouchPoint) drawPt(currentTouchPoint);

  // Dibujar medidas del plano virtual
  virtualMeasurements.forEach(m => {
    const p1 = m.points[0], p2 = m.points[1];
    const pt1 = project(p1);
    const pt2 = project(p2);
    const x1 = pt1.x, y1 = pt1.y, x2 = pt2.x, y2 = pt2.y;

    ctx.beginPath(); ctx.arc(x1, y1, r * 1.2, 0, 2*Math.PI);
    ctx.fillStyle = '#f59e0b'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();

    ctx.beginPath(); ctx.arc(x2, y2, r * 1.2, 0, 2*Math.PI);
    ctx.fillStyle = '#f59e0b'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();

    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = strokeW;
    ctx.setLineDash([8 / state.viewer.scale, 5 / state.viewer.scale]);
    ctx.stroke(); ctx.setLineDash([]);

    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;
    const text = `${m.distMm.toFixed(1)} mm`;
    ctx.font = `600 ${14 / state.viewer.scale}px 'Inter', sans-serif`;
    const txtMetrics = ctx.measureText(text);
    const textW = txtMetrics.width;
    const textH = 14 / state.viewer.scale;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(midX - textW/2 - 4/state.viewer.scale, midY - textH/2 - 12/state.viewer.scale, textW + 8/state.viewer.scale, textH + 4/state.viewer.scale);
    ctx.fillStyle = '#f59e0b';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, midX, midY - (10/state.viewer.scale));
  });

  // Pending virtual plane point
  if (!isFlat && state.virtualPlanePendingPoint && state.mode === 'virtualPlane' && !state.isTiltingVirtualPlane) {
    const pp = state.virtualPlanePendingPoint;
    ctx.beginPath(); ctx.arc(pp.x, pp.y, r * 1.2, 0, 2*Math.PI);
    ctx.fillStyle = '#f59e0b'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();
  }

  if (pendingPoint && currentTouchPoint) {
    const pt1 = project(pendingPoint);
    const pt2 = project(currentTouchPoint);
    ctx.beginPath(); ctx.moveTo(pt1.x, pt1.y); ctx.lineTo(pt2.x, pt2.y);
    ctx.strokeStyle = '#f87171'; ctx.lineWidth = strokeW;
    ctx.stroke();
    
    const distMm = Math.hypot(currentTouchPoint.x_mm - pendingPoint.x_mm, currentTouchPoint.y_mm - pendingPoint.y_mm);
    const midX = (pt1.x + pt2.x) / 2;
    const midY = (pt1.y + pt2.y) / 2;
    ctx.font = `600 ${14 / state.viewer.scale}px 'Inter', sans-serif`;
    ctx.fillStyle = '#f87171';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(`${distMm.toFixed(1)} mm`, midX, midY - (10/state.viewer.scale));
  }

  // Dibujar boxes de YOLO integrados como measurements
  planeMeasurements.forEach(m => {
    if (m.isAuto && m.box) {
      const box = m.box;
      
      let x1, y1, x2, y2;
      if (isFlat) {
        const pt1 = project(m.points[0]);
        const pt2 = project(m.points[1]);
        x1 = Math.min(pt1.x, pt2.x);
        y1 = Math.min(pt1.y, pt2.y);
        x2 = Math.max(pt1.x, pt2.x);
        y2 = Math.max(pt1.y, pt2.y);
      } else {
        x1 = box.x1;
        y1 = box.y1;
        x2 = box.x1 + box.w;
        y2 = box.y1 + box.h;
      }
      
      const imgW = x2 - x1;
      const imgH = y2 - y1;

      ctx.strokeStyle = '#facc15'; // Amarillo
      ctx.lineWidth = 2 / state.viewer.scale;
      ctx.strokeRect(x1, y1, imgW, imgH);
      
      const labelH = 20 / state.viewer.scale;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(x1, y1 - labelH, imgW, labelH);
      ctx.fillStyle = '#facc15';
      ctx.font = `${12 / state.viewer.scale}px 'Inter', sans-serif`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      const scorePct = (box.score * 100).toFixed(0);
      ctx.fillText(`${box.label} [${scorePct}%] - W:${m.distMm.toFixed(0)}mm`, x1 + (4 / state.viewer.scale), y1 - (labelH / 2));
    }
  });

  // Puntos base del plano virtual
  if (!isFlat && plane.isVirtual) {
    const vPts = plane.points || [];
    vPts.forEach((pt, index) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, r * 1.5, 0, 2 * Math.PI);
      ctx.fillStyle = '#facc15';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = strokeW;
      ctx.stroke();
      
      ctx.font = `600 ${12 / state.viewer.scale}px 'Inter', sans-serif`;
      ctx.fillStyle = '#facc15';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(`Base ${index+1}`, pt.x, pt.y - (12/state.viewer.scale));
    });
    
    if (vPts.length === 2) {
      const p1 = vPts[0];
      const p2 = vPts[1];
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const len = Math.hypot(dx, dy) || 1;
      const nx = dx / len;
      const ny = dy / len;
      
      // Extender 5000 pixeles
      const startX = p1.x - nx * 5000;
      const startY = p1.y - ny * 5000;
      const endX = p2.x + nx * 5000;
      const endY = p2.y + ny * 5000;

      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = strokeW;
      ctx.setLineDash([5 / state.viewer.scale, 5 / state.viewer.scale]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Recta de referencia de perspectiva horizontal (paralela a la base)
    const tPts = plane.virtualPlaneTiltPoints || [];
    if (tPts.length >= 2) {
      const tp1 = tPts[0];
      const tp2 = tPts[1];
      const tdx = tp2.x - tp1.x;
      const tdy = tp2.y - tp1.y;
      const tlen = Math.hypot(tdx, tdy) || 1;
      const tnx = tdx / tlen; const tny = tdy / tlen;
      ctx.beginPath();
      ctx.moveTo(tp1.x - tnx * 5000, tp1.y - tny * 5000);
      ctx.lineTo(tp2.x + tnx * 5000, tp2.y + tny * 5000);
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = strokeW;
      ctx.setLineDash([5 / state.viewer.scale, 5 / state.viewer.scale]);
      ctx.stroke();
      ctx.setLineDash([]);
      // Puntos
      [tp1, tp2].forEach(pt => {
        ctx.beginPath(); ctx.arc(pt.x, pt.y, r, 0, 2*Math.PI);
        ctx.fillStyle = '#a855f7'; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();
      });
      const mx = (tp1.x + tp2.x) / 2, my = (tp1.y + tp2.y) / 2;
      ctx.font = `600 ${12 / state.viewer.scale}px 'Inter', sans-serif`;
      ctx.fillStyle = '#a855f7'; ctx.textAlign = 'center';
      ctx.fillText('↔ Fuga H', mx, my - (14/state.viewer.scale));
    }

    // Recta de referencia vertical
    const vvPts = plane.vanishingPointVPoints || [];
    if (vvPts.length >= 2) {
      const vp1 = vvPts[0];
      const vp2 = vvPts[1];
      const vdx = vp2.x - vp1.x;
      const vdy = vp2.y - vp1.y;
      const vlen = Math.hypot(vdx, vdy) || 1;
      const vnx = vdx / vlen; const vny = vdy / vlen;
      ctx.beginPath();
      ctx.moveTo(vp1.x - vnx * 5000, vp1.y - vny * 5000);
      ctx.lineTo(vp2.x + vnx * 5000, vp2.y + vny * 5000);
      ctx.strokeStyle = '#34d399';
      ctx.lineWidth = strokeW;
      ctx.setLineDash([5 / state.viewer.scale, 5 / state.viewer.scale]);
      ctx.stroke();
      ctx.setLineDash([]);
      [vp1, vp2].forEach(pt => {
        ctx.beginPath(); ctx.arc(pt.x, pt.y, r, 0, 2*Math.PI);
        ctx.fillStyle = '#34d399'; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();
      });
      const mx = (vp1.x + vp2.x) / 2, my = (vp1.y + vp2.y) / 2;
      ctx.font = `600 ${12 / state.viewer.scale}px 'Inter', sans-serif`;
      ctx.fillStyle = '#34d399'; ctx.textAlign = 'center';
      ctx.fillText('↕ Fuga V (Recta 1)', mx, my - (14/state.viewer.scale));

      if (vvPts.length >= 4) {
        const vp3 = vvPts[2];
        const vp4 = vvPts[3];
        const vdx2 = vp4.x - vp3.x;
        const vdy2 = vp4.y - vp3.y;
        const vlen2 = Math.hypot(vdx2, vdy2) || 1;
        const vnx2 = vdx2 / vlen2; const vny2 = vdy2 / vlen2;
        ctx.beginPath();
        ctx.moveTo(vp3.x - vnx2 * 5000, vp3.y - vny2 * 5000);
        ctx.lineTo(vp4.x + vnx2 * 5000, vp4.y + vny2 * 5000);
        ctx.strokeStyle = '#34d399';
        ctx.lineWidth = strokeW;
        ctx.setLineDash([5 / state.viewer.scale, 5 / state.viewer.scale]);
        ctx.stroke();
        ctx.setLineDash([]);
        [vp3, vp4].forEach(pt => {
          ctx.beginPath(); ctx.arc(pt.x, pt.y, r, 0, 2*Math.PI);
          ctx.fillStyle = '#34d399'; ctx.fill();
          ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();
        });
        const mx2 = (vp3.x + vp4.x) / 2, my2 = (vp3.y + vp4.y) / 2;
        ctx.font = `600 ${12 / state.viewer.scale}px 'Inter', sans-serif`;
        ctx.fillStyle = '#34d399'; ctx.textAlign = 'center';
        ctx.fillText('↕ Fuga V (Recta 2)', mx2, my2 - (14/state.viewer.scale));
      }
    }
  }

  // Dibujar punto de exportación pendiente
  if (state.isSelectingExportArea && state.exportAreaPoints && state.exportAreaPoints.length > 0) {
    const pt = state.exportAreaPoints[0];
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 6 / state.viewer.scale, 0, 2 * Math.PI);
    ctx.fillStyle = '#ef4444'; // Rojo
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2 / state.viewer.scale;
    ctx.stroke();

    ctx.font = `600 ${12 / state.viewer.scale}px 'Inter', sans-serif`;
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'center';
    ctx.fillText('Esquina 1', pt.x, pt.y - (14 / state.viewer.scale));
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

    const refHtml = m.isAuto 
      ? `<span style="color:#facc15;font-weight:600;font-size:12px;">[🤖 IA]</span> <input type="text" class="ref-input" id="${inputId}" placeholder="Etiquetar..." value="${escHtml(m.refName)}">`
      : `<input type="text" class="ref-input" id="${inputId}" placeholder="Ej. Puerta" value="${escHtml(m.refName)}">`;

    const valHtml = m.isAuto && m.depthMm
      ? `${m.distMm.toFixed(1)} × ${m.depthMm.toFixed(1)} mm`
      : `${m.distMm.toFixed(2)} mm`;

    tr.innerHTML = `
      <td style="color:var(--text-3);font-family:monospace;">${m.id}</td>
      <td>${escHtml(m.planeLabel)}</td>
      <td>${refHtml}</td>
      <td class="meas-value">${valHtml}</td>
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
  if (str === null || str === undefined) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* ─────────────────────────────────────────────────────────────
   EXPORTACIÓN
   ───────────────────────────────────────────────────────────── */

function exportAllPNG() {
  if (state.activeTool === 'virtual_plane') {
    if (state.virtualPlanePoints.length < 2) {
      toast('Definí la base del plano virtual antes de exportar.', 'warning');
      return;
    }
  } else {
    if (state.activePlaneIndex < 0) {
      toast('No hay planos activos para exportar.', 'info');
      return;
    }
  }

  state.isSelectingExportArea = true;
  state.exportAreaPoints = [];
  toast('Marcá el área a exportar: hacé clic en la esquina superior izquierda y luego en la esquina inferior derecha.', 'info');
  drawMeasurements();
}

function handleExportAreaClick(imgX, imgY) {
  if (!state.exportAreaPoints) state.exportAreaPoints = [];
  state.exportAreaPoints.push({ x: imgX, y: imgY });

  if (state.exportAreaPoints.length === 1) {
    toast('Esquina 1 marcada. Marcá la esquina opuesta para definir el área.', 'info');
    drawMeasurements();
  } else if (state.exportAreaPoints.length === 2) {
    const pt1 = state.exportAreaPoints[0];
    const pt2 = state.exportAreaPoints[1];
    state.isSelectingExportArea = false;

    if (state.activeTool === 'virtual_plane') {
      exportVirtualPlanePNG(pt1, pt2);
    } else {
      exportTemplatePlanePNG(pt1, pt2);
    }

    state.exportAreaPoints = null;
    drawMeasurements();
  }
}

function exportTemplatePlanePNG(pt1, pt2) {
  if (state.activePlaneIndex < 0) { toast('No hay planos activos para exportar.', 'info'); return; }
  const plane = state.planes[state.activePlaneIndex];
  if (!plane) return;

  showProcessing('Generando recorte plano...', 'Calculando transformación');
  
  setTimeout(() => {
    try {
      const h_data = plane.H.data64F;
      const p1_mm = applyH(h_data, pt1.x, pt1.y);
      const p2_mm = applyH(h_data, pt2.x, pt2.y);
      if (!p1_mm || !p2_mm) throw new Error("Los puntos seleccionados están fuera del plano");

      const bbox = {
        X_min: Math.min(p1_mm.x, p2_mm.x),
        X_max: Math.max(p1_mm.x, p2_mm.x),
        Y_min: Math.min(p1_mm.y, p2_mm.y),
        Y_max: Math.max(p1_mm.y, p2_mm.y)
      };

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

function exportVirtualPlanePNG(pt1, pt2) {
  const pb = state.planes.find(p => p.id === state.planeBaseIndex);
  const pw = state.planes.find(p => p.id === state.planeWallIndex);
  if (!pb || !pw) {
    toast('Se requieren planos de piso y pared para exportar el plano virtual.', 'warning');
    return;
  }

  showProcessing('Generando recorte de plano virtual...', 'Calculando proyección');

  setTimeout(() => {
    try {
      const mainCanvas = document.getElementById('main-image-canvas');
      const params = getCameraParams(pb.H_inv, pw.H_inv, mainCanvas.width, mainCanvas.height);
      const H_data_base = pb.H.data64F;
      const projToFloor = (pt) => {
        const w = H_data_base[6] * pt.x + H_data_base[7] * pt.y + H_data_base[8];
        return {
          x: (H_data_base[0] * pt.x + H_data_base[1] * pt.y + H_data_base[2]) / w,
          y: (H_data_base[3] * pt.x + H_data_base[4] * pt.y + H_data_base[5]) / w
        };
      };
      const b1_mm = projToFloor(state.virtualPlanePoints[0]);
      const b2_mm = projToFloor(state.virtualPlanePoints[1]);

      // Calibrar la escala vertical en base a la intersección del plano pared con el virtual
      const calib = calibrateVirtualPlaneScale(
        params,
        pb.H, pb.H_inv,
        pw.H, pw.H_inv,
        b1_mm, b2_mm,
        state.vanishingPointH,
        state.vanishingPointV
      );
      params.s_v = calib.s_v;

      // Calcular coordenadas (U, Z) de las dos esquinas seleccionadas
      const u1_3d = intersectRayWithVirtualPlane(params, pt1.x, pt1.y, b1_mm, b2_mm, state.vanishingPointH, state.vanishingPointV);
      const u2_3d = intersectRayWithVirtualPlane(params, pt2.x, pt2.y, b1_mm, b2_mm, state.vanishingPointH, state.vanishingPointV);
      if (!u1_3d || !u2_3d) throw new Error("Las esquinas seleccionadas no intersectan el plano virtual de forma válida.");

      const dx_dir = b2_mm.x - b1_mm.x;
      const dy_dir = b2_mm.y - b1_mm.y;
      const lenU = Math.hypot(dx_dir, dy_dir) || 1;
      const u_dir_obj = { x: dx_dir / lenU, y: dy_dir / lenU };

      const getVirtualPlaneCoords = (p_obj) => {
        const dx_int = p_obj.x - b1_mm.x;
        const dy_int = p_obj.y - b1_mm.y;
        const U = dx_int * u_dir_obj.x + dy_int * u_dir_obj.y;
        const Z = p_obj.z;
        return { U, Z };
      };
      const coords1 = getVirtualPlaneCoords(u1_3d);
      const coords2 = getVirtualPlaneCoords(u2_3d);

      const bbox = {
        X_min: Math.min(coords1.U, coords2.U),
        X_max: Math.max(coords1.U, coords2.U),
        Y_min: Math.min(coords1.Z, coords2.Z),
        Y_max: Math.max(coords1.Z, coords2.Z)
      };

      // Construir la homografía inversa H_inv_virt (U, Z) -> Pixel
      const P1_cam = {
        x: params.r1.x * b1_mm.x + params.r2.x * b1_mm.y + params.t_vec.x,
        y: params.r1.y * b1_mm.x + params.r2.y * b1_mm.y + params.t_vec.y,
        z: params.r1.z * b1_mm.x + params.r2.z * b1_mm.y + params.t_vec.z
      };

      const u_dir_cam = {
        x: params.r1.x * u_dir_obj.x + params.r2.x * u_dir_obj.y,
        y: params.r1.y * u_dir_obj.x + params.r2.y * u_dir_obj.y,
        z: params.r1.z * u_dir_obj.x + params.r2.z * u_dir_obj.y
      };
      const un = Math.hypot(u_dir_cam.x, u_dir_cam.y, u_dir_cam.z) || 1;

      let u_cam_unit;
      if (state.vanishingPointH) {
        const raw = {
          x: (state.vanishingPointH.x - params.Pcx) / params.f,
          y: (state.vanishingPointH.y - params.Pcy) / params.f,
          z: 1
        };
        const norm = Math.hypot(raw.x, raw.y, raw.z) || 1;
        u_cam_unit = { x: raw.x / norm, y: raw.y / norm, z: raw.z / norm };
      } else {
        u_cam_unit = { x: u_dir_cam.x / un, y: u_dir_cam.y / un, z: u_dir_cam.z / un };
      }

      const P2_cam = {
        x: params.r1.x * b2_mm.x + params.r2.x * b2_mm.y + params.t_vec.x,
        y: params.r1.y * b2_mm.x + params.r2.y * b2_mm.y + params.t_vec.y,
        z: params.r1.z * b2_mm.x + params.r2.z * b2_mm.y + params.t_vec.z
      };
      const V2 = {
        x: P2_cam.x - P1_cam.x,
        y: P2_cam.y - P1_cam.y,
        z: P2_cam.z - P1_cam.z
      };
      const U_scaled = V2.x * u_cam_unit.x + V2.y * u_cam_unit.y + V2.z * u_cam_unit.z;
      const s_h = U_scaled / lenU;

      const scale_v = params.s_v || params.vn_default || 1.0;
      let v_cam_unit;
      if (state.vanishingPointV) {
        const raw = {
          x: (state.vanishingPointV.x - params.Pcx) / params.f,
          y: (state.vanishingPointV.y - params.Pcy) / params.f,
          z: 1
        };
        const norm = Math.hypot(raw.x, raw.y, raw.z) || 1;
        v_cam_unit = { x: raw.x / norm, y: raw.y / norm, z: raw.z / norm };
      } else {
        const norm3 = Math.hypot(params.r3.x, params.r3.y, params.r3.z) || 1.0;
        v_cam_unit = { x: -params.r3.x / norm3, y: -params.r3.y / norm3, z: -params.r3.z / norm3 };
      }

      const M = [
        s_h * u_cam_unit.x, scale_v * v_cam_unit.x, P1_cam.x,
        s_h * u_cam_unit.y, scale_v * v_cam_unit.y, P1_cam.y,
        s_h * u_cam_unit.z, scale_v * v_cam_unit.z, P1_cam.z
      ];
      const f = params.f;
      const Pcx = params.Pcx;
      const Pcy = params.Pcy;

      const h00 = f * M[0] + Pcx * M[6];
      const h01 = f * M[1] + Pcx * M[7];
      const h02 = f * M[2] + Pcx * M[8];

      const h10 = f * M[3] + Pcy * M[6];
      const h11 = f * M[4] + Pcy * M[7];
      const h12 = f * M[5] + Pcy * M[8];

      const h20 = M[6];
      const h21 = M[7];
      const h22 = M[8];

      const H_inv_virt = cv.matFromArray(3, 3, cv.CV_64F, [
        h00, h01, h02,
        h10, h11, h12,
        h20, h21, h22
      ]);
      const H_virt = new cv.Mat();
      cv.invert(H_inv_virt, H_virt, cv.DECOMP_LU);

      const tmpCanvas = document.createElement('canvas');
      renderVirtualPlane(state.originalImageMat, H_virt, pb.scale, bbox, tmpCanvas);

      H_inv_virt.delete();
      H_virt.delete();

      tmpCanvas.toBlob(blob => {
        if (!blob) throw new Error("Fallo al generar Blob");
        downloadBlob(blob, `mobius_plano_virtual.png`);
        hideProcessing();
        toast('Vista plana del plano virtual exportada.', 'success');
      }, 'image/png');

    } catch(e) {
      hideProcessing();
      toast('Error generando imagen: ' + e.message, 'error');
    }
  }, 50);
}

function renderVirtualPlane(srcMat, H_virt, scale, bbox, targetCanvas) {
  const { X_min, Y_min, X_max, Y_max } = bbox;
  const s = scale;

  /* Dimensiones del canvas de salida (en px) */
  let canvasW = Math.round((X_max - X_min) * s);
  let canvasH = Math.round((Y_max - Y_min) * s);

  /* Clamp de seguridad para no reventar la memoria */
  const MAX_CANVAS_DIM = 6000;
  if (canvasW > MAX_CANVAS_DIM || canvasH > MAX_CANVAS_DIM) {
    const clampScale = Math.min(MAX_CANVAS_DIM / canvasW, MAX_CANVAS_DIM / canvasH);
    canvasW = Math.round(canvasW * clampScale);
    canvasH = Math.round(canvasH * clampScale);
    console.warn('[Möbius] Virtual plane canvas clamped a', canvasW, '×', canvasH);
  }

  targetCanvas.width  = canvasW;
  targetCanvas.height = canvasH;

  // Mapeo T_inv para girar el eje Z para que el piso esté en la parte inferior
  const T_inv = cv.matFromArray(3, 3, cv.CV_64F, [
    s,  0, -X_min * s,
    0, -s,  Y_max * s,
    0,  0,  1,
  ]);

  const M_final = new cv.Mat();
  const empty   = new cv.Mat();
  cv.gemm(T_inv, H_virt, 1.0, empty, 0.0, M_final, 0);

  const dstMat = new cv.Mat();
  const dsize  = new cv.Size(canvasW, canvasH);

  cv.warpPerspective(
    srcMat, dstMat, M_final, dsize,
    cv.INTER_LINEAR,
    cv.BORDER_CONSTANT,
    new cv.Scalar(0, 0, 0, 0)
  );

  cv.imshow(targetCanvas, dstMat);

  T_inv.delete();
  empty.delete();
  M_final.delete();
  dstMat.delete();

  return { canvasW, canvasH };
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

function updatePlaneListUI() {
  const list = document.getElementById('planes-list');
  if (!list) return;
  list.innerHTML = '';

  state.planes.forEach((p, index) => {
    const btn = document.createElement('button');
    btn.className = 'plane-btn';
    btn.id = `btn-plane-${index}`;
    btn.type = 'button';

    let meta = `${p.scale.toFixed(2)} px/mm`;
    if (p.isVirtual) meta = 'Plano Alzado';
    if (p.isElevated) meta = `Plano Elevado (${p.elevationHeight} mm)`;

    btn.innerHTML = `
      <span class="plane-btn-title">
        <span class="planes-badge">${index + 1}</span>
        ${escHtml(p.label)}
      </span>
      <span class="plane-btn-meta">${meta}</span>
    `;

    btn.addEventListener('click', () => {
      setActivePlane(index);
    });

    list.appendChild(btn);
  });

  const countEl = document.getElementById('planes-count');
  if (countEl) countEl.textContent = state.planes.length;
}

function setActivePlane(index) {
  state.activePlaneIndex = index;
  
  document.querySelectorAll('.plane-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`btn-plane-${index}`);
  if (activeBtn) activeBtn.classList.add('active');

  updatePlaneSelectors3DVisibility();
  if (state.showFlatView) {
    updateFlatView();
  } else {
    const mainCanvas = document.getElementById('main-image-canvas');
    if (mainCanvas && state.originalImageMat) {
      cv.imshow(mainCanvas, state.originalImageMat);
    }
  }
  drawGrid();
  drawMeasurements();
}

function updatePlaneSelectors3DVisibility() {
  const plane = state.planes[state.activePlaneIndex];
  const selectors3d = document.getElementById('plane-selectors-3d');
  const btnResetVirtual = document.getElementById('btn-reset-virtual');
  const btnTiltVirtual = document.getElementById('btn-tilt-virtual');
  const btnCalibVertVirtual = document.getElementById('btn-calib-vert-virtual');

  // Rellenar selectores 3D con planos físicos reales (no virtuales ni elevados)
  const selBase = document.getElementById('select-plane-base');
  const selWall = document.getElementById('select-plane-wall');
  if (selBase && selWall) {
    const activeBaseVal = selBase.value;
    const activeWallVal = selWall.value;
    
    selBase.innerHTML = '';
    selWall.innerHTML = '';

    const physicalPlanes = state.planes.filter(p => !p.isVirtual && !p.isElevated);
    physicalPlanes.forEach(p => {
      const opt1 = document.createElement('option');
      opt1.value = p.id;
      opt1.textContent = p.label;
      selBase.appendChild(opt1);

      const opt2 = document.createElement('option');
      opt2.value = p.id;
      opt2.textContent = p.label;
      selWall.appendChild(opt2);
    });

    if (plane && (plane.isVirtual || plane.isElevated)) {
      selBase.value = plane.planeBaseIndex;
      selWall.value = plane.planeWallIndex;
    } else {
      if (activeBaseVal) selBase.value = activeBaseVal;
      if (activeWallVal) selWall.value = activeWallVal;
    }
  }

  if (plane && plane.isVirtual) {
    if (selectors3d) selectors3d.style.display = 'flex';
    if (btnResetVirtual) btnResetVirtual.style.display = 'inline-block';
    
    // Solo mostrar calibración de perspectiva si tiene al menos 2 puntos
    if (btnTiltVirtual) {
      btnTiltVirtual.style.display = plane.points.length >= 2 ? 'inline-block' : 'none';
    }
    if (btnCalibVertVirtual) {
      btnCalibVertVirtual.style.display = plane.points.length >= 2 ? 'inline-block' : 'none';
    }
  } else if (plane && plane.isElevated) {
    if (selectors3d) selectors3d.style.display = 'flex';
    if (btnResetVirtual) btnResetVirtual.style.display = 'none';
    if (btnTiltVirtual) btnTiltVirtual.style.display = 'none';
    if (btnCalibVertVirtual) btnCalibVertVirtual.style.display = 'none';
  } else {
    if (selectors3d) selectors3d.style.display = 'none';
    if (btnResetVirtual) btnResetVirtual.style.display = 'none';
    if (btnTiltVirtual) btnTiltVirtual.style.display = 'none';
    if (btnCalibVertVirtual) btnCalibVertVirtual.style.display = 'none';
  }

  // Actualizar botones de editar/borrar
  const btnRename = document.getElementById('btn-rename-plane');
  if (btnRename) {
    btnRename.style.display = (plane && (plane.isVirtual || plane.isElevated)) ? 'inline-block' : 'none';
  }
  const btnDelete = document.getElementById('btn-delete-plane');
  if (btnDelete) {
    btnDelete.style.display = (plane && (plane.isVirtual || plane.isElevated)) ? 'inline-block' : 'none';
  }
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
      const { H, scale, sheetCenter } = computeHomographyAndScale(sheet);

      const H_inv = new cv.Mat();
      cv.invert(H, H_inv, cv.DECOMP_LU);

      const label = `Plano Base ${i + 1} (${sheet.template.name})`;
      state.planes.push({ id: i, label, H: H.clone(), H_inv, scale, template: sheet.template, sheetCenter, isVirtual: false, isElevated: false });

      H.delete();
      await yieldToUI();
    }

    state.originalImageMat = srcMat;

    const count = state.planes.length;
    if (count > 0) {
      document.getElementById('upload-view').classList.add('hidden');
      document.getElementById('workspace-view').classList.add('visible');
      document.getElementById('measurements-section').classList.add('visible');
      
      initViewer(width, height);
      updatePlaneListUI();
      
      if (count >= 2) {
        state.planeBaseIndex = state.planes[0].id;
        state.planeWallIndex = state.planes[1].id;
      } else {
        state.planeBaseIndex = -1;
        state.planeWallIndex = -1;
      }
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

  state.showFlatView = false;
  state.flatViewTransform = null;
  state.flatViewPlaneId = -1;
  const toggleFlat = document.getElementById('toggle-flat');
  if (toggleFlat) toggleFlat.checked = false;
  const toggleGrid = document.getElementById('toggle-grid');
  if (toggleGrid) toggleGrid.checked = true;
  state.showGrid = true;
  
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

let cameraInterval = null;

function estimateDistanceToSheet(sheet, width, height) {
  // Coordenadas 3D de los marcadores en el espacio de la hoja (mm)
  const targets = sheet.template.targets;
  const objPtsData = [
    targets[0][0], targets[0][1], 0,
    targets[1][0], targets[1][1], 0,
    targets[2][0], targets[2][1], 0,
    targets[3][0], targets[3][1], 0
  ];
  
  // Coordenadas 2D de los centros detectados (px)
  const c = sheet.markers.map(m => m.center);
  const imgPtsData = [
    c[0].x, c[0].y,
    c[1].x, c[1].y,
    c[2].x, c[2].y,
    c[3].x, c[3].y
  ];

  const objectPoints = cv.matFromArray(4, 1, cv.CV_32FC3, objPtsData);
  const imagePoints = cv.matFromArray(4, 1, cv.CV_32FC2, imgPtsData);

  // Matriz de cámara K aproximada
  const f = 0.85 * width;
  const cx = width / 2;
  const cy = height / 2;
  const cameraMatrix = cv.matFromArray(3, 3, cv.CV_64F, [
    f, 0, cx,
    0, f, cy,
    0, 0, 1
  ]);

  const distCoeffs = new cv.Mat();
  const rvec = new cv.Mat();
  const tvec = new cv.Mat();

  try {
    cv.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv.SOLVEPNP_ITERATIVE);
    const tx = tvec.doubleAt(0, 0);
    const ty = tvec.doubleAt(1, 0);
    const tz = tvec.doubleAt(2, 0);
    return Math.hypot(tx, ty, tz);
  } catch (e) {
    console.error('[Möbius Debug] PnP Error:', e);
    return null;
  } finally {
    objectPoints.delete();
    imagePoints.delete();
    cameraMatrix.delete();
    distCoeffs.delete();
    rvec.delete();
    tvec.delete();
  }
}

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

    const badge = document.getElementById('camera-distance-badge');
    const val = document.getElementById('camera-distance-value');
    if (badge) badge.style.display = 'flex';
    if (val) val.innerText = 'Buscando hoja de referencia...';

    // Iniciar loop de detección en vivo para estimar distancia (a ~3 FPS)
    const hiddenCanvas = document.createElement('canvas');
    const hiddenCtx = hiddenCanvas.getContext('2d');

    if (cameraInterval) clearInterval(cameraInterval);
    cameraInterval = setInterval(() => {
      if (video.paused || video.ended || !state.cameraStream || !state.cvReady) return;

      hiddenCanvas.width = 640;
      hiddenCanvas.height = 480;
      hiddenCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

      let srcMat;
      try {
        srcMat = cv.imread(hiddenCanvas);
        const markers = detectMarkers(srcMat);

        if (markers.length >= 4) {
          const sheets = clusterIntoSheets(markers);
          if (sheets.length > 0) {
            const distanceMm = estimateDistanceToSheet(sheets[0], hiddenCanvas.width, hiddenCanvas.height);
            if (distanceMm) {
              const distM = distanceMm / 1000;
              val.innerText = `Distancia: ${distM.toFixed(2)} m`;
              badge.style.color = '#4ade80'; // Verde si detecta
              badge.style.borderColor = 'rgba(74, 222, 128, 0.4)';
              return;
            }
          }
        }
        val.innerText = 'Buscando hoja de referencia...';
        badge.style.color = '#fff';
        badge.style.borderColor = 'rgba(255, 255, 255, 0.1)';
      } catch (e) {
        console.error('[Möbius Live Preview Error]', e);
      } finally {
        if (srcMat) srcMat.delete();
      }
    }, 300);

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

  if (cameraInterval) {
    clearInterval(cameraInterval);
    cameraInterval = null;
  }

  const badge = document.getElementById('camera-distance-badge');
  if (badge) badge.style.display = 'none';

  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }
  video.srcObject = null;
  overlay.classList.remove('visible');
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

  const btnAddVirtual = document.getElementById('btn-add-virtual');
  if (btnAddVirtual) {
    btnAddVirtual.addEventListener('click', () => {
      const physicalPlanes = state.planes.filter(p => !p.isVirtual && !p.isElevated);
      if (physicalPlanes.length < 2) {
        toast('Para crear planos virtuales se necesitan al menos 2 planos detectados.', 'warning');
        return;
      }
      
      const newPlane = {
        id: 'virtual_' + Date.now(),
        label: 'Plano Alzado (Virtual) ' + (state.planes.filter(p => p.isVirtual).length + 1),
        isVirtual: true,
        isElevated: false,
        points: [],
        vanishingPointH: null,
        vanishingPointV: null,
        vanishingPointVPoints: [],
        planeBaseIndex: physicalPlanes[0].id,
        planeWallIndex: physicalPlanes[1].id,
        H: null,
        H_inv: null,
        scale: physicalPlanes[0].scale,
        sheetCenter: { x: state.originalImageMat.cols / 2, y: state.originalImageMat.rows / 2 },
        template: { w_mm: 500, h_mm: 500, targets: [[0,0], [500,0], [500,500], [0,500]] }
      };

      state.planes.push(newPlane);
      updatePlaneListUI();
      setActivePlane(state.planes.length - 1);
      toast('Hacé clic en dos puntos del piso para definir la línea base del plano virtual.', 'info');
    });
  }

  const btnAddElevated = document.getElementById('btn-add-elevated');
  if (btnAddElevated) {
    btnAddElevated.addEventListener('click', () => {
      const physicalPlanes = state.planes.filter(p => !p.isVirtual && !p.isElevated);
      if (physicalPlanes.length < 2) {
        toast('Para crear planos elevados se necesitan al menos 2 planos detectados.', 'warning');
        return;
      }
      
      const heightInput = prompt('Ingresá la elevación en milímetros (mm):', '150');
      if (heightInput === null) return;
      const heightMm = parseFloat(heightInput);
      if (isNaN(heightMm) || heightMm <= 0) {
        toast('Altura inválida.', 'error');
        return;
      }

      const pb = physicalPlanes[0];
      const pw = physicalPlanes[1];

      const newPlane = {
        id: 'elevated_' + Date.now(),
        label: `Plano Elevado (${heightMm} mm)`,
        isVirtual: false,
        isElevated: true,
        elevationHeight: heightMm,
        planeBaseIndex: pb.id,
        planeWallIndex: pw.id,
        H: null,
        H_inv: null,
        scale: pb.scale,
        sheetCenter: { x: state.originalImageMat.cols / 2, y: state.originalImageMat.rows / 2 },
        template: { w_mm: 500, h_mm: 500, targets: [[0,0], [500,0], [500,500], [0,500]] }
      };

      const res = computeElevatedPlaneH(pb, pw, heightMm);
      if (res) {
        newPlane.H = res.H;
        newPlane.H_inv = res.H_inv;
        newPlane.scale = res.scale;
      } else {
        toast('No se pudo calcular la proyección del plano elevado.', 'error');
        return;
      }

      state.planes.push(newPlane);
      updatePlaneListUI();
      setActivePlane(state.planes.length - 1);
      toast(`Plano elevado a ${heightMm} mm creado con éxito.`, 'success');
    });
  }

  const btnRenamePlane = document.getElementById('btn-rename-plane');
  if (btnRenamePlane) {
    btnRenamePlane.addEventListener('click', () => {
      const plane = state.planes[state.activePlaneIndex];
      if (!plane) return;
      const newName = prompt('Nuevo nombre para el plano:', plane.label);
      if (newName && newName.trim()) {
        plane.label = newName.trim();
        updatePlaneListUI();
        setActivePlane(state.activePlaneIndex);
      }
    });
  }

  const btnDeletePlane = document.getElementById('btn-delete-plane');
  if (btnDeletePlane) {
    btnDeletePlane.addEventListener('click', () => {
      const plane = state.planes[state.activePlaneIndex];
      if (!plane) return;
      if (!plane.isVirtual && !plane.isElevated) {
        toast('No se pueden eliminar planos detectados automáticamente.', 'warning');
        return;
      }
      if (confirm(`¿Estás seguro de que querés eliminar el plano "${plane.label}"?`)) {
        if (plane.H) plane.H.delete();
        if (plane.H_inv) plane.H_inv.delete();

        state.measurements = state.measurements.filter(m => m.planeId !== plane.id);
        
        state.planes.splice(state.activePlaneIndex, 1);
        updatePlaneListUI();
        setActivePlane(Math.max(0, state.activePlaneIndex - 1));
        renderMeasurementTable();
        toast('Plano eliminado.', 'info');
      }
    });
  }

  const btnResetVirtual = document.getElementById('btn-reset-virtual');
  if (btnResetVirtual) {
    btnResetVirtual.addEventListener('click', () => {
      const plane = state.planes[state.activePlaneIndex];
      if (!plane || !plane.isVirtual) return;

      plane.points = [];
      plane.vanishingPointH = null;
      plane.vanishingPointV = null;
      plane.vanishingPointVPoints = [];
      plane.virtualPlaneTiltPoints = [];
      if (plane.H) { plane.H.delete(); plane.H = null; }
      if (plane.H_inv) { plane.H_inv.delete(); plane.H_inv = null; }

      state.isTiltingVirtualPlane = false;
      state.isCalibVertVirtualPlane = false;
      
      toast('Plano virtual reiniciado. Marcá dos puntos nuevos en el piso.', 'info');
      updatePlaneSelectors3DVisibility();
      drawGrid();
      drawMeasurements();
    });
  }

  const btnTiltVirtual = document.getElementById('btn-tilt-virtual');
  if (btnTiltVirtual) {
    btnTiltVirtual.addEventListener('click', () => {
      const plane = state.planes[state.activePlaneIndex];
      if (!plane || !plane.isVirtual) return;
      if (plane.points.length < 2) {
        toast('Primero definí la base del plano en el piso (2 clics).', 'warning');
        return;
      }
      state.isTiltingVirtualPlane = true;
      state.isCalibVertVirtualPlane = false;
      plane.virtualPlaneTiltPoints = [];
      toast('Modo perspectiva: Marcá 2 puntos sobre una línea que sea paralela a la recta base en la realidad.', 'info');
    });
  }

  const btnCalibVertVirtual = document.getElementById('btn-calib-vert-virtual');
  if (btnCalibVertVirtual) {
    btnCalibVertVirtual.addEventListener('click', () => {
      const plane = state.planes[state.activePlaneIndex];
      if (!plane || !plane.isVirtual) return;
      if (plane.points.length < 2) {
        toast('Primero definí la base del plano en el piso (2 clics).', 'warning');
        return;
      }
      state.isCalibVertVirtualPlane = true;
      state.isTiltingVirtualPlane = false;
      plane.vanishingPointVPoints = [];
      toast('Ajuste vertical: Marcá 2 rectas verticales en la realidad. Hacé clic en 2 puntos para la primera, y 2 puntos para la segunda.', 'info');
    });
  }

  const selBase = document.getElementById('select-plane-base');
  if (selBase) {
    selBase.addEventListener('change', (e) => {
      const plane = state.planes[state.activePlaneIndex];
      if (plane && (plane.isVirtual || plane.isElevated)) {
        plane.planeBaseIndex = parseInt(e.target.value);
        recomputeVirtualPlane(plane);
        drawGrid();
        drawMeasurements();
      }
    });
  }

  const selWall = document.getElementById('select-plane-wall');
  if (selWall) {
    selWall.addEventListener('change', (e) => {
      const plane = state.planes[state.activePlaneIndex];
      if (plane && (plane.isVirtual || plane.isElevated)) {
        plane.planeWallIndex = parseInt(e.target.value);
        recomputeVirtualPlane(plane);
        drawGrid();
        drawMeasurements();
      }
    });
  }

  const toggleGrid = document.getElementById('toggle-grid');
  if (toggleGrid) {
    toggleGrid.addEventListener('change', (e) => {
      state.showGrid = e.target.checked;
      drawGrid();
    });
  }

  const toggleFlat = document.getElementById('toggle-flat');
  if (toggleFlat) {
    toggleFlat.addEventListener('change', (e) => {
      state.showFlatView = e.target.checked;
      updateFlatView();
      drawGrid();
      drawMeasurements();
    });
  }

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

function recomputeVirtualPlane(plane) {
  if (plane.isElevated) {
    const pb = state.planes.find(p => p.id === plane.planeBaseIndex);
    const pw = state.planes.find(p => p.id === plane.planeWallIndex);
    if (!pb || !pw) return;
    
    const res = computeElevatedPlaneH(pb, pw, plane.elevationHeight);
    if (res) {
      if (plane.H) plane.H.delete();
      if (plane.H_inv) plane.H_inv.delete();
      plane.H = res.H;
      plane.H_inv = res.H_inv;
      plane.scale = res.scale;
    }
  } else if (plane.isVirtual) {
    if (plane.points.length < 2) {
      plane.H = null;
      plane.H_inv = null;
      return;
    }
    const pb = state.planes.find(p => p.id === plane.planeBaseIndex);
    const pw = state.planes.find(p => p.id === plane.planeWallIndex);
    if (!pb || !pw) return;

    const res = computeVirtualVerticalPlaneH(pb, pw, plane);
    if (res) {
      if (plane.H) plane.H.delete();
      if (plane.H_inv) plane.H_inv.delete();
      plane.H = res.H;
      plane.H_inv = res.H_inv;
      plane.scale = res.scale;
    }
  }
}

function computeVirtualVerticalPlaneH(pb, pw, plane) {
  const mainCanvas = document.getElementById('main-image-canvas');
  const params = getCameraParams(pb.H_inv, pw.H_inv, mainCanvas.width, mainCanvas.height);

  const H_data_base = pb.H.data64F;
  const projToFloor = (pt) => {
    const w = H_data_base[6] * pt.x + H_data_base[7] * pt.y + H_data_base[8];
    return {
      x: (H_data_base[0] * pt.x + H_data_base[1] * pt.y + H_data_base[2]) / w,
      y: (H_data_base[3] * pt.x + H_data_base[4] * pt.y + H_data_base[5]) / w
    };
  };
  const b1_mm = projToFloor(plane.points[0]);
  const b2_mm = projToFloor(plane.points[1]);

  const calib = calibrateVirtualPlaneScale(
    params, pb.H, pb.H_inv, pw.H, pw.H_inv,
    b1_mm, b2_mm, plane.vanishingPointH, plane.vanishingPointV
  );
  params.s_v = calib.s_v;

  const A_px = projectVirtualPlane(params, b1_mm, b2_mm, 0, 0, plane.vanishingPointH, plane.vanishingPointV);
  const B_px = projectVirtualPlane(params, b1_mm, b2_mm, 200, 0, plane.vanishingPointH, plane.vanishingPointV);
  const C_px = projectVirtualPlane(params, b1_mm, b2_mm, 200, 200, plane.vanishingPointH, plane.vanishingPointV);
  const D_px = projectVirtualPlane(params, b1_mm, b2_mm, 0, 200, plane.vanishingPointH, plane.vanishingPointV);

  if (!A_px || !B_px || !C_px || !D_px) return null;

  const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    A_px.x, A_px.y,
    B_px.x, B_px.y,
    C_px.x, C_px.y,
    D_px.x, D_px.y,
  ]);
  const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,
    200, 0,
    200, 200,
    0, 200,
  ]);

  const H = cv.findHomography(srcPts, dstPts);
  const H_inv = new cv.Mat();
  cv.invert(H, H_inv, cv.DECOMP_LU);

  srcPts.delete();
  dstPts.delete();

  return { H, H_inv, scale: pb.scale };
}

function computeElevatedPlaneH(pb, pw, heightMm) {
  const mainCanvas = document.getElementById('main-image-canvas');
  const params = getCameraParams(pb.H_inv, pw.H_inv, mainCanvas.width, mainCanvas.height);

  const A_px = project3D(params, 0, 0, heightMm);
  const B_px = project3D(params, 200, 0, heightMm);
  const C_px = project3D(params, 200, 200, heightMm);
  const D_px = project3D(params, 0, 200, heightMm);

  if (!A_px || !B_px || !C_px || !D_px) return null;

  const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    A_px.x, A_px.y,
    B_px.x, B_px.y,
    C_px.x, C_px.y,
    D_px.x, D_px.y,
  ]);
  const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,
    200, 0,
    200, 200,
    0, 200,
  ]);

  const H = cv.findHomography(srcPts, dstPts);
  const H_inv = new cv.Mat();
  cv.invert(H, H_inv, cv.DECOMP_LU);

  srcPts.delete();
  dstPts.delete();

  return { H, H_inv, scale: pb.scale };
}

function updateFlatView() {
  const mainCanvas = document.getElementById('main-image-canvas');
  if (!mainCanvas || !state.originalImageMat) return;

  if (state.showFlatView) {
    const plane = state.planes[state.activePlaneIndex];
    if (!plane) {
      toast('Seleccioná un plano para ver la vista plana', 'info');
      document.getElementById('toggle-flat').checked = false;
      state.showFlatView = false;
      return;
    }

    showProcessing('Generando vista plana...', 'Esto puede tardar unos segundos');
    setTimeout(() => {
      try {
        let bbox;
        if (!plane.H) {
          throw new Error("El plano seleccionado no está calibrado.");
        }
        const h_data = plane.H.data64F;
        if (state.exportAreaPoints && state.exportAreaPoints.length === 2) {
          const pt1 = applyH(h_data, state.exportAreaPoints[0].imgX, state.exportAreaPoints[0].imgY);
          const pt2 = applyH(h_data, state.exportAreaPoints[1].imgX, state.exportAreaPoints[1].imgY);
          bbox = computeBBox([{mm: pt1}, {mm: pt2}]);
        } else {
          const targets = (plane.template && plane.template.targets && plane.template.targets.length > 0)
            ? plane.template.targets
            : [[-500,-500], [500,-500], [500,500], [-500,500]];
          const validPoints = conformalFilter(plane.H, state.originalImageMat.cols, state.originalImageMat.rows, plane.sheetCenter, targets);
          bbox = computeBBox(validPoints);
        }

        const canvasW = Math.round(bbox.X_max - bbox.X_min);
        const canvasH = Math.round(bbox.Y_max - bbox.Y_min);
        const MAX_DIM = 4000;
        
        let actualScale = plane.scale;
        const widthPx = canvasW * actualScale;
        const heightPx = canvasH * actualScale;
        if (widthPx > MAX_DIM || heightPx > MAX_DIM) {
           actualScale = Math.min(MAX_DIM / canvasW, MAX_DIM / canvasH);
        }

        renderPlane(state.originalImageMat, plane.H, actualScale, bbox, mainCanvas);

        state.flatViewTransform = {
          X_min: bbox.X_min,
          Y_min: bbox.Y_min,
          scale: actualScale,
          H_inv: plane.H_inv
        };
        
        state.flatViewPlaneId = plane.id;

        document.getElementById('btn-zoom-fit').click();
        hideProcessing();
        toast('Vista plana activada. 1 px = ' + (1/actualScale).toFixed(2) + ' mm.', 'success');
      } catch (e) {
        hideProcessing();
        toast('Error al generar vista plana: ' + e.message, 'error');
        document.getElementById('toggle-flat').checked = false;
        state.showFlatView = false;
        cv.imshow(mainCanvas, state.originalImageMat);
      }
    }, 50);

  } else {
    cv.imshow(mainCanvas, state.originalImageMat);
    state.flatViewTransform = null;
    state.flatViewPlaneId = -1;
    document.getElementById('btn-zoom-fit').click();
  }
}

/* ─────────────────────────────────────────────────────────────
   GUARDAR / CARGAR PROYECTO
   ───────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
  const btnSave = document.getElementById('btn-save-project');
  if (btnSave) {
    btnSave.addEventListener('click', async () => {
      if (!state.originalImageMat || state.planes.length === 0) {
        toast('No hay un proyecto activo para guardar.', 'info');
        return;
      }
      showProcessing('Guardando proyecto...', 'Por favor esperá');

      const tmpCanvas = document.createElement('canvas');
      cv.imshow(tmpCanvas, state.originalImageMat);
      const dataUrl = tmpCanvas.toDataURL('image/jpeg', 0.85);

      const planesData = state.planes.map(p => {
        const item = {
          id: p.id,
          label: p.label,
          scale: p.scale,
          isVirtual: !!p.isVirtual,
          isElevated: !!p.isElevated,
          sheetCenter: p.sheetCenter,
        };
        if (p.H) item.H_data = Array.from(p.H.data64F);
        if (p.H_inv) item.H_inv_data = Array.from(p.H_inv.data64F);
        if (p.template) item.templateName = p.template.name;
        
        if (p.isVirtual) {
          item.points = p.points;
          item.vanishingPointH = p.vanishingPointH;
          item.vanishingPointV = p.vanishingPointV;
          item.vanishingPointVPoints = p.vanishingPointVPoints;
          item.virtualPlaneTiltPoints = p.virtualPlaneTiltPoints;
          item.planeBaseIndex = p.planeBaseIndex;
          item.planeWallIndex = p.planeWallIndex;
        }
        
        if (p.isElevated) {
          item.elevationHeight = p.elevationHeight;
          item.planeBaseIndex = p.planeBaseIndex;
          item.planeWallIndex = p.planeWallIndex;
        }

        return item;
      });

      const projectData = {
        version: "2.0",
        image: dataUrl,
        planes: planesData,
        measurements: state.measurements,
        measIdCounter: state.measIdCounter,
        activePlaneIndex: state.activePlaneIndex,
        planeBaseIndex: state.planeBaseIndex,
        planeWallIndex: state.planeWallIndex
      };

      const jsonStr = JSON.stringify(projectData);
      const blob = new Blob([jsonStr], { type: 'application/json' });
      downloadBlob(blob, `mobius_project_${new Date().toISOString().slice(0,10)}.json`);
      hideProcessing();
      toast('Proyecto guardado con éxito.', 'success');
    });
  }

  const btnLoad = document.getElementById('btn-load-project');
  const fileInput = document.getElementById('project-input');
  if (btnLoad && fileInput) {
    btnLoad.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      showProcessing('Cargando proyecto...', 'Leyendo archivo');
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        
        clearResults();
        
        const img = new Image();
        img.onload = () => {
          const tmp = document.createElement('canvas');
          tmp.width = img.width; tmp.height = img.height;
          const ctx = tmp.getContext('2d');
          ctx.drawImage(img, 0, 0);
          state.originalImageMat = cv.imread(tmp);
          
          state.planes = data.planes.map(p => {
            const item = {
              id: p.id,
              label: p.label,
              scale: p.scale,
              isVirtual: !!p.isVirtual,
              isElevated: !!p.isElevated,
              sheetCenter: p.sheetCenter,
              template: p.templateName ? { name: p.templateName } : { w_mm: 500, h_mm: 500, targets: [[0,0], [500,0], [500,500], [0,500]] }
            };
            if (p.H_data) item.H = cv.matFromArray(3, 3, cv.CV_64F, p.H_data);
            if (p.H_inv_data) item.H_inv = cv.matFromArray(3, 3, cv.CV_64F, p.H_inv_data);
            
            if (p.isVirtual) {
              item.points = p.points || [];
              item.vanishingPointH = p.vanishingPointH;
              item.vanishingPointV = p.vanishingPointV;
              item.vanishingPointVPoints = p.vanishingPointVPoints || [];
              item.virtualPlaneTiltPoints = p.virtualPlaneTiltPoints || [];
              item.planeBaseIndex = p.planeBaseIndex;
              item.planeWallIndex = p.planeWallIndex;
            }
            
            if (p.isElevated) {
              item.elevationHeight = p.elevationHeight;
              item.planeBaseIndex = p.planeBaseIndex;
              item.planeWallIndex = p.planeWallIndex;
            }

            return item;
          });
          
          state.measurements = data.measurements || [];
          state.measIdCounter = data.measIdCounter || 0;
          state.planeBaseIndex = data.planeBaseIndex;
          state.planeWallIndex = data.planeWallIndex;
          
          document.getElementById('upload-view').classList.add('hidden');
          document.getElementById('workspace-view').classList.add('visible');
          document.getElementById('measurements-section').classList.add('visible');
          
          updatePlaneListUI();
          initViewer(img.width, img.height);
          setActivePlane(data.activePlaneIndex || 0);
          
          hideProcessing();
          toast('Proyecto cargado.', 'success');
        };
        img.onerror = () => { throw new Error("Fallo al cargar la imagen del proyecto"); };
        img.src = data.image;
        
      } catch (err) {
        hideProcessing();
        toast('Error al cargar proyecto: ' + err.message, 'error');
      }
      fileInput.value = ''; // reset
    });
  }
});
