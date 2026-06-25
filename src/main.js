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
  virtualPlaneTopIntersect: null
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
    if (state.virtualPlanePoints.length < 2) {
      state.virtualPlanePoints.push({ x: imgX, y: imgY });
      if (state.virtualPlanePoints.length === 2) {
        toast('Plano virtual definido. Ahora podés medir sobre él o ajustar perspectiva/vertical.', 'success');
        const btnTilt = document.getElementById('btn-tilt-virtual');
        if (btnTilt) btnTilt.style.display = 'inline-block';
        const btnCalibV = document.getElementById('btn-calib-vert-virtual');
        if (btnCalibV) btnCalibV.style.display = 'inline-block';
        drawGrid();
      } else {
        toast('Hacé clic en otro punto del piso para definir la línea base.', 'info');
      }
    } else if (state.isCalibVertVirtualPlane) {
      if (!state.vanishingPointVPoints) state.vanishingPointVPoints = [];
      state.vanishingPointVPoints.push({ x: imgX, y: imgY });
      const ptsCount = state.vanishingPointVPoints.length;
      if (ptsCount === 1) {
        toast('Primer punto de la primera recta vertical marcado. Marcá el segundo punto de la misma recta.', 'info');
      } else if (ptsCount === 2) {
        toast('Primera recta vertical marcada. Marcá el primer punto de la segunda recta vertical.', 'info');
      } else if (ptsCount === 3) {
        toast('Primer punto de la segunda recta vertical marcado. Marcá el segundo punto de la misma recta.', 'info');
      } else if (ptsCount === 4) {
        const vp1 = state.vanishingPointVPoints[0];
        const vp2 = state.vanishingPointVPoints[1];
        const vp3 = state.vanishingPointVPoints[2];
        const vp4 = state.vanishingPointVPoints[3];

        // L1 = vp1 x vp2
        const L1 = {
          x: vp1.y - vp2.y,
          y: vp2.x - vp1.x,
          z: vp1.x * vp2.y - vp1.y * vp2.x
        };
        // L2 = vp3 x vp4
        const L2 = {
          x: vp3.y - vp4.y,
          y: vp4.x - vp3.x,
          z: vp3.x * vp4.y - vp3.y * vp4.x
        };

        // Intersección V_v = L1 x L2
        const V_v = {
          x: L1.y * L2.z - L1.z * L2.y,
          y: L1.z * L2.x - L1.x * L2.z,
          z: L1.x * L2.y - L1.y * L2.x
        };

        if (Math.abs(V_v.z) > 1e-7) {
          state.vanishingPointV = { x: V_v.x / V_v.z, y: V_v.y / V_v.z };
          toast('Fuga vertical calibrada con éxito mediante 2 rectas.', 'success');
        } else {
          // Rectas perfectamente paralelas en imagen (fuga vertical en el infinito)
          // Usamos la dirección promedio de las dos rectas
          const dx1 = vp2.x - vp1.x;
          const dy1 = vp2.y - vp1.y;
          const dx2 = vp4.x - vp3.x;
          const dy2 = vp4.y - vp3.y;
          const dx = (dx1 + dx2) / 2;
          const dy = (dy1 + dy2) / 2;
          const len = Math.hypot(dx, dy) || 1;
          const farScale = 100000;
          state.vanishingPointV = {
            x: vp1.x + (dx / len) * farScale,
            y: vp1.y + (dy / len) * farScale
          };
          toast('Rectas verticales paralelas detectadas. Fuga vertical establecida en el infinito.', 'info');
        }
        state.isCalibVertVirtualPlane = false;
        drawGrid();
        drawMeasurements();
      }
    } else if (state.isTiltingVirtualPlane) {
      state.virtualPlaneTiltPoints.push({ x: imgX, y: imgY });
      const pts = state.virtualPlaneTiltPoints.length;
      if (pts === 1) {
        toast('Primer punto marcado. Marcá el segundo punto de la recta paralela.', 'info');
      } else if (pts === 2) {
        // Tenemos la recta base (virtualPlanePoints[0,1]) y la nueva recta (tiltPoints[0,1])
        // Calcular punto de fuga horizontal = intersección en imagen de ambas rectas
        const bp1 = state.virtualPlanePoints[0];
        const bp2 = state.virtualPlanePoints[1];
        const tp1 = state.virtualPlaneTiltPoints[0];
        const tp2 = state.virtualPlaneTiltPoints[1];
        
        // Línea L_base en coordenadas homogéneas: bp1 x bp2
        const L_base = {
          x: bp1.y - bp2.y,
          y: bp2.x - bp1.x,
          z: bp1.x * bp2.y - bp1.y * bp2.x
        };
        // Línea L_tilt: tp1 x tp2
        const L_tilt = {
          x: tp1.y - tp2.y,
          y: tp2.x - tp1.x,
          z: tp1.x * tp2.y - tp1.y * tp2.x
        };
        
        // Punto de fuga V_h = L_base x L_tilt
        const V_h = {
          x: L_base.y * L_tilt.z - L_base.z * L_tilt.y,
          y: L_base.z * L_tilt.x - L_base.x * L_tilt.z,
          z: L_base.x * L_tilt.y - L_base.y * L_tilt.x
        };
        
        if (Math.abs(V_h.z) > 1e-7) {
          // Guardar el punto de fuga horizontal en coordenadas de imagen
          state.vanishingPointH = { x: V_h.x / V_h.z, y: V_h.y / V_h.z };
          toast('Perspectiva horizontal calibrada. La grilla ahora fuga correctamente.', 'success');
        } else {
          // Rectas perfectamente paralelas en la imagen => punto de fuga en el infinito
          // Guardar la dirección de la línea (paralelas sin convergencia visible)
          state.vanishingPointH = null; // null = paralelas perfectas
          toast('Rectas paralelas en imagen — grilla con filas paralelas.', 'info');
        }
        
        state.isTiltingVirtualPlane = false;
        drawGrid();
      }
      
      // Limpiar interactive canvas
      const canvas = document.getElementById('interactive-canvas');
      if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      
    } else {
      // Estamos midiendo
      if (!state.virtualPlanePendingPoint) {
        state.virtualPlanePendingPoint = { x: imgX, y: imgY };
        renderMeasurements();
      } else {
        const p1 = state.virtualPlanePendingPoint;
        const p2 = { x: imgX, y: imgY };
        
        const pb = state.planes.find(p => p.id === state.planeBaseIndex);
        const pw = state.planes.find(p => p.id === state.planeWallIndex);
        
        if (pb && pw) {
          const canvas = document.getElementById('main-image-canvas');
          const params = getCameraParams(pb.H_inv, pw.H_inv, canvas.width, canvas.height);
          
          const projToFloor = (pt) => {
            const H_data = pb.H.data64F;
            const w = H_data[6] * pt.x + H_data[7] * pt.y + H_data[8];
            return {
              x: (H_data[0] * pt.x + H_data[1] * pt.y + H_data[2]) / w,
              y: (H_data[3] * pt.x + H_data[4] * pt.y + H_data[5]) / w
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
          
          const p1_3d = intersectRayWithVirtualPlane(params, p1.x, p1.y, b1_mm, b2_mm, state.vanishingPointH, state.vanishingPointV);
          const p2_3d = intersectRayWithVirtualPlane(params, p2.x, p2.y, b1_mm, b2_mm, state.vanishingPointH, state.vanishingPointV);
          
          if (p1_3d && p2_3d) {
            const distMm = Math.hypot(p2_3d.x - p1_3d.x, p2_3d.y - p1_3d.y, p2_3d.z - p1_3d.z);
            addMeasurementVirtualPlane(distMm, p1, p2);
            toast(`Medida virtual: ${distMm.toFixed(1)} mm`, 'success');
          } else {
            toast('Error: Los puntos no intersectan el plano virtual de forma válida.', 'error');
          }
        }
        
        state.virtualPlanePendingPoint = null;
        renderMeasurements();
        
        const canvas = document.getElementById('interactive-canvas');
        if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  }
  
  function addMeasurementVirtualPlane(distMm, p1, p2) {
    const entry = {
      id:         ++state.measIdCounter,
      planeId:    -1, // virtual
      planeLabel: 'Plano Virtual',
      isVirtual:  true,
      refName:    '',
      distMm,
      points:     [p1, p2], // puntos en coordenadas de imagen
      timestamp:  new Date().toISOString(),
    };
    state.measurements.push(entry);
    renderMeasurementTable();
    drawMeasurements();
  }
  
  function renderVirtualPlaneInteractiveLine(mouseX, mouseY) {
    const canvas = document.getElementById('interactive-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (state.isTiltingVirtualPlane) {
      const pts = state.virtualPlaneTiltPoints;
      // Dibujar la recta base como rayo de referencia (yellow)
      if (state.virtualPlanePoints.length === 2) {
        const bp1 = state.virtualPlanePoints[0];
        const bp2 = state.virtualPlanePoints[1];
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
      } else if (pts.length === 0) {
        // Sin puntos todavía — mostrar cursor
      }
    } else if (state.virtualPlanePoints.length === 1) {
      const b = state.virtualPlanePoints[0];
      ctx.beginPath();
      ctx.moveTo(b.x, b.y);
      ctx.lineTo(mouseX, mouseY);
      ctx.strokeStyle = '#facc15'; // yellow for base line
      ctx.lineWidth = 2 / state.viewer.scale;
      ctx.stroke();
    } else if (state.virtualPlanePoints.length === 2 && state.virtualPlanePendingPoint) {
      const p1 = state.virtualPlanePendingPoint;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(mouseX, mouseY);
      ctx.strokeStyle = '#3b82f6'; // blue for measurement
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
      
      if (state.activeTool === 'virtual_plane') {
        renderVirtualPlaneInteractiveLine((clickX - state.viewer.offsetX) / state.viewer.scale, (clickY - state.viewer.offsetY) / state.viewer.scale);
      }
    }
  });

  // Medición - MOUSE
  overlayCanvas.addEventListener('mousedown', e => {
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

    if (state.activeTool === 'virtual_plane') {
      handleVirtualPlaneClick(imgX, imgY);
    } else {
      const point = processClickPoint(x, y);
      commitMeasurementPoint(point);
    }
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
      if (state.isSelectingExportArea) {
        handleExportAreaClick(currentTouchPoint.imgX, currentTouchPoint.imgY);
        currentTouchPoint = null;
        drawMeasurements(pendingPoint);
        return;
      }

      if (state.activeTool === 'virtual_plane') {
        handleVirtualPlaneClick(currentTouchPoint.imgX, currentTouchPoint.imgY);
      } else {
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
  
  // Dibujar plano virtual si existe
  if (state.virtualPlanePoints.length >= 2 && state.planes.length >= 2) {
    const pb = state.planes.find(p => p.id === state.planeBaseIndex);
    const pw = state.planes.find(p => p.id === state.planeWallIndex);
    if (pb && pw) {
      const mainCanvas = document.getElementById('main-image-canvas');
      const params = getCameraParams(pb.H_inv, pw.H_inv, mainCanvas.width, mainCanvas.height);

      // Proyectar los 2 puntos base al piso (coordenadas mm del plano base)
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
      state.virtualPlaneBaseIntersect = calib.P_base_intersect;
      state.virtualPlaneTopIntersect = calib.P_top_intersect;

      // Punto de fuga horizontal: si está calibrado, usarlo; si no, usar la dirección de la recta base (paralelas perfectas)
      const hasVPH = state.vanishingPointH != null;

      const stepV_mm = 50;  // paso vertical en mm
      const nV = 20;        // cantidad de filas (1000 mm total)
      const stepU_mm = 50;  // paso horizontal en mm
      const nU_extra = 100; // cantidad de columnas a cada lado (suficiente para cubrir fuera del lienzo)

      // Calcular la longitud de la recta base en mm
      const dx_mm = b2_mm.x - b1_mm.x;
      const dy_mm = b2_mm.y - b1_mm.y;
      const lenU_mm = Math.hypot(dx_mm, dy_mm) || 1;

      // Índice inicial y final en U (centrado en la recta base)
      const iStart = -nU_extra;
      const iEnd = Math.ceil(lenU_mm / stepU_mm) + nU_extra;

      // Color de la grilla
      const color = hasVPH ? 'rgba(168, 85, 247, 0.4)' : 'rgba(236, 72, 153, 0.4)';
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5 / state.viewer.scale;
      ctx.beginPath();

      // Líneas VERTICALES de la grilla: para cada columna U, trazar línea de Z=0 a Z=nV*stepV
      for (let i = iStart; i <= iEnd; i++) {
        const u = i * stepU_mm;
        const pBot = projectVirtualPlane(params, b1_mm, b2_mm, u, 0, state.vanishingPointH, state.vanishingPointV);
        const pTop = projectVirtualPlane(params, b1_mm, b2_mm, u, nV * stepV_mm, state.vanishingPointH, state.vanishingPointV);
        if (pBot && pTop) {
          ctx.moveTo(pBot.x, pBot.y);
          ctx.lineTo(pTop.x, pTop.y);
        }
      }

      // Líneas HORIZONTALES de la grilla: para cada fila Z, trazar línea de U_min a U_max
      for (let j = 0; j <= nV; j++) {
        const z = j * stepV_mm;
        const pStart = projectVirtualPlane(params, b1_mm, b2_mm, iStart * stepU_mm, z, state.vanishingPointH, state.vanishingPointV);
        const pEnd   = projectVirtualPlane(params, b1_mm, b2_mm, iEnd * stepU_mm, z, state.vanishingPointH, state.vanishingPointV);
        if (pStart && pEnd) {
          ctx.moveTo(pStart.x, pStart.y);
          ctx.lineTo(pEnd.x, pEnd.y);
        }
      }

      ctx.stroke();

      // Dibujar recta de intersección horizontal (Piso)
      const pBaseL = projectVirtualPlane(params, b1_mm, b2_mm, iStart * stepU_mm, 0, state.vanishingPointH, state.vanishingPointV);
      const pBaseR = projectVirtualPlane(params, b1_mm, b2_mm, iEnd * stepU_mm, 0, state.vanishingPointH, state.vanishingPointV);
      if (pBaseL && pBaseR) {
        ctx.beginPath();
        ctx.strokeStyle = '#06b6d4'; // Cyan
        ctx.lineWidth = 3 / state.viewer.scale;
        ctx.moveTo(pBaseL.x, pBaseL.y);
        ctx.lineTo(pBaseR.x, pBaseR.y);
        ctx.stroke();

        ctx.font = `600 ${11 / state.viewer.scale}px 'Inter', sans-serif`;
        ctx.fillStyle = '#06b6d4';
        ctx.textAlign = 'left';
        ctx.fillText('Intersección Horizontal (Piso)', pBaseL.x + 10 / state.viewer.scale, pBaseL.y + 15 / state.viewer.scale);
      }

      // Dibujar recta de intersección vertical (Pared)
      if (state.virtualPlaneBaseIntersect && state.virtualPlaneTopIntersect) {
        const p1 = state.virtualPlaneBaseIntersect;
        const p2 = state.virtualPlaneTopIntersect;
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const len = Math.hypot(dx, dy) || 1;
        const nx = dx / len;
        const ny = dy / len;

        ctx.beginPath();
        ctx.strokeStyle = '#f97316'; // Orange
        ctx.lineWidth = 3 / state.viewer.scale;
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p1.x + nx * 5000, p1.y + ny * 5000); // extender hacia arriba
        ctx.stroke();

        ctx.font = `600 ${11 / state.viewer.scale}px 'Inter', sans-serif`;
        ctx.fillStyle = '#f97316';
        ctx.textAlign = 'left';
        ctx.fillText('Intersección Vertical (Pared)', p1.x + 10 / state.viewer.scale, p1.y - 10 / state.viewer.scale);
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

  const planeMeasurements = state.measurements.filter(m => m.planeId === plane.id);
  // Medidas del plano virtual (separadas)
  const virtualMeasurements = state.measurements.filter(m => m.isVirtual);
  const strokeW = 2 / state.viewer.scale;
  const r = 4 / state.viewer.scale;

  const drawPt = (p) => {
    ctx.beginPath(); ctx.arc(p.imgX, p.imgY, r, 0, 2*Math.PI);
    ctx.fillStyle = '#6ee7f7'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();
  };

  planeMeasurements.forEach(m => {
    if (m.isAuto) return; 
    
    const is3D = m.is3D;
    const p1 = m.points[0], p2 = m.points[1];
    
    // Adaptar formato de puntos
    const x1 = is3D ? p1.x : p1.imgX;
    const y1 = is3D ? p1.y : p1.imgY;
    const x2 = is3D ? p2.x : p2.imgX;
    const y2 = is3D ? p2.y : p2.imgY;
    
    ctx.beginPath(); ctx.arc(x1, y1, r, 0, 2*Math.PI);
    ctx.fillStyle = is3D ? '#00ffcc' : '#6ee7f7'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();
    
    ctx.beginPath(); ctx.arc(x2, y2, r, 0, 2*Math.PI);
    ctx.fillStyle = is3D ? '#00ffcc' : '#6ee7f7'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();

    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.strokeStyle = is3D ? '#00ffcc' : '#6ee7f7'; ctx.lineWidth = strokeW;
    ctx.setLineDash([8 / state.viewer.scale, 5 / state.viewer.scale]);
    ctx.stroke(); ctx.setLineDash([]);
    
    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;
    ctx.font = `600 ${14 / state.viewer.scale}px 'Inter', sans-serif`;
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    
    // Fondo de texto para mejor lectura
    const text = `${m.distMm.toFixed(1)} mm`;
    const txtMetrics = ctx.measureText(text);
    const textW = txtMetrics.width;
    const textH = 14 / state.viewer.scale;
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(midX - textW/2 - 4/state.viewer.scale, midY - textH/2 - 12/state.viewer.scale, textW + 8/state.viewer.scale, textH + 4/state.viewer.scale);
    
    ctx.fillStyle = is3D ? '#00ffcc' : '#fff';
    ctx.fillText(text, midX, midY - (10/state.viewer.scale));
  });

  if (pendingPoint) drawPt(pendingPoint);
  if (currentTouchPoint) drawPt(currentTouchPoint);

  // Dibujar medidas del plano virtual
  virtualMeasurements.forEach(m => {
    const p1 = m.points[0], p2 = m.points[1];
    const x1 = p1.x, y1 = p1.y, x2 = p2.x, y2 = p2.y;

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
  if (state.virtualPlanePendingPoint && state.mode === 'virtualPlane' && !state.isTiltingVirtualPlane) {
    const pp = state.virtualPlanePendingPoint;
    ctx.beginPath(); ctx.arc(pp.x, pp.y, r * 1.2, 0, 2*Math.PI);
    ctx.fillStyle = '#f59e0b'; ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = strokeW/2; ctx.stroke();
  }

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

  // Dibujar boxes de YOLO integrados como measurements
  planeMeasurements.forEach(m => {
    if (m.isAuto && m.box) {
      const box = m.box;
      const imgX = box.x1;
      const imgY = box.y1;
      const imgW = box.w;
      const imgH = box.h;

      ctx.strokeStyle = '#facc15'; // Amarillo
      ctx.lineWidth = 2 / state.viewer.scale;
      ctx.strokeRect(imgX, imgY, imgW, imgH);
      
      const labelH = 20 / state.viewer.scale;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(imgX, imgY - labelH, imgW, labelH);
      ctx.fillStyle = '#facc15';
      ctx.font = `${12 / state.viewer.scale}px 'Inter', sans-serif`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      const scorePct = (box.score * 100).toFixed(0);
      ctx.fillText(`${box.label} [${scorePct}%] - W:${m.distMm.toFixed(0)}mm`, imgX + (4 / state.viewer.scale), imgY - (labelH / 2));
    }
  });

  // Puntos base del plano virtual
  state.virtualPlanePoints.forEach((pt, index) => {
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
  
  if (state.virtualPlanePoints.length === 2) {
    const p1 = state.virtualPlanePoints[0];
    const p2 = state.virtualPlanePoints[1];
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
  if (state.virtualPlaneTiltPoints && state.virtualPlaneTiltPoints.length >= 2) {
    const tp1 = state.virtualPlaneTiltPoints[0];
    const tp2 = state.virtualPlaneTiltPoints[1];
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
  if (state.vanishingPointVPoints && state.vanishingPointVPoints.length >= 2) {
    const vp1 = state.vanishingPointVPoints[0];
    const vp2 = state.vanishingPointVPoints[1];
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

    if (state.vanishingPointVPoints.length >= 4) {
      const vp3 = state.vanishingPointVPoints[2];
      const vp4 = state.vanishingPointVPoints[3];
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
      const { H, scale, sheetCenter } = computeHomographyAndScale(sheet);

      const H_inv = new cv.Mat();
      cv.invert(H, H_inv, cv.DECOMP_LU);

      const label = createPlaneSidebarButton(i, scale, sheet.template.name);
      state.planes.push({ id: i, label, H: H.clone(), H_inv, scale, template: sheet.template, sheetCenter });

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
      
      // Update 3D Selectors
      const selBase = document.getElementById('select-plane-base');
      const selWall = document.getElementById('select-plane-wall');
      selBase.innerHTML = '';
      selWall.innerHTML = '';
      
      // Selectors UI
      if (count >= 2) {
        document.getElementById('plane-selectors-3d').style.display = state.activeTool === 'virtual_plane' ? 'flex' : 'none';
        state.planes.forEach((p, idx) => {
          const opt1 = document.createElement('option');
          opt1.value = p.id;
          opt1.textContent = p.label.replace(/<[^>]*>?/gm, ''); // strip HTML
          selBase.appendChild(opt1);
          
          const opt2 = document.createElement('option');
          opt2.value = p.id;
          opt2.textContent = p.label.replace(/<[^>]*>?/gm, '');
          selWall.appendChild(opt2);
        });
        
        selBase.value = state.planes[0].id;
        selWall.value = state.planes[1].id;
        state.planeBaseIndex = 0;
        state.planeWallIndex = 1;
        
        // Listeners
        selBase.onchange = (e) => {
          state.planeBaseIndex = parseInt(e.target.value);
          if (state.activeTool === 'virtual_plane') {
            setActivePlane(state.planeBaseIndex);
          }
        };
        selWall.onchange = (e) => state.planeWallIndex = parseInt(e.target.value);
      } else {
        document.getElementById('plane-selectors-3d').style.display = 'none';
        state.planeBaseIndex = -1;
        state.planeWallIndex = -1;
      }
      
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

  const btnTool2d = document.getElementById('btn-tool-2d');
  const btnTool3d = null; // eliminado
  const btnToolVirtual = document.getElementById('btn-tool-virtual');
  const btnResetVirtual = document.getElementById('btn-reset-virtual');
  
  if (btnTool2d && btnToolVirtual) {
    const hideSelectors = () => {
      const sel = document.getElementById('plane-selectors-3d');
      if (sel) sel.style.display = 'none';
      if (btnResetVirtual) btnResetVirtual.style.display = 'none';
      const btnTilt = document.getElementById('btn-tilt-virtual');
      if (btnTilt) btnTilt.style.display = 'none';
    };

    btnTool2d.addEventListener('click', () => {
      state.activeTool = '2d';
      btnTool2d.classList.add('active', 'btn-primary');
      btnTool2d.classList.remove('btn-secondary');
      btnToolVirtual.classList.remove('active', 'btn-primary');
      btnToolVirtual.classList.add('btn-secondary');
      
      hideSelectors();
      
      state.meas3dBasePoint = null;
      state.virtualPlanePoints = [];
      state.virtualPlanePendingPoint = null;
      renderMeasurements(); 
    });
    
    // btn-tool-3d eliminado — redundante con Plano Virtual

    btnToolVirtual.addEventListener('click', () => {
      if (state.planes.length < 2) {
        toast('Para planos virtuales necesitás al menos 2 planos detectados.', 'warning');
        return;
      }
      state.activeTool = 'virtual_plane';
      btnToolVirtual.classList.add('active', 'btn-primary');
      btnToolVirtual.classList.remove('btn-secondary');
      btnTool2d.classList.remove('active', 'btn-primary');
      btnTool2d.classList.add('btn-secondary');
      
      const selectors3d = document.getElementById('plane-selectors-3d');
      if (selectors3d) selectors3d.style.display = 'flex';
      if (btnResetVirtual) btnResetVirtual.style.display = 'inline-block';
      
      if (state.planeBaseIndex >= 0) setActivePlane(state.planeBaseIndex);
      
      state.meas3dBasePoint = null;
      state.pendingPoint = null;
      if (state.virtualPlanePoints.length === 0) {
        toast('Hacé clic en dos puntos del piso para definir la línea base del plano virtual.', 'info');
      }
      renderMeasurements();
    });
    
    if (btnResetVirtual) {
      btnResetVirtual.addEventListener('click', () => {
        state.virtualPlanePoints = [];
        state.virtualPlanePendingPoint = null;
        state.virtualPlaneNormal = null;
        state.vanishingPointH = null;
        state.vanishingPointV = null;
        state.vanishingPointVPoints = null;
        state.isTiltingVirtualPlane = false;
        state.isCalibVertVirtualPlane = false;
        state.virtualPlaneTiltPoints = [];
        const btnTilt = document.getElementById('btn-tilt-virtual');
        if (btnTilt) btnTilt.style.display = 'none';
        const btnCalibV = document.getElementById('btn-calib-vert-virtual');
        if (btnCalibV) btnCalibV.style.display = 'none';
        toast('Plano virtual reiniciado. Marcá dos puntos nuevos en el piso.', 'info');
        renderMeasurements();
        drawGrid();
      });
    }

    const btnTiltVirtual = document.getElementById('btn-tilt-virtual');
    if (btnTiltVirtual) {
      btnTiltVirtual.addEventListener('click', () => {
        if (state.virtualPlanePoints.length < 2) {
          toast('Primero definí la base del plano en el piso (2 clics).', 'warning');
          return;
        }
        state.isTiltingVirtualPlane = true;
        state.isCalibVertVirtualPlane = false;
        state.virtualPlaneTiltPoints = [];
        toast('Modo perspectiva: Marcá 2 puntos sobre una línea que sea paralela a la recta base en la realidad.', 'info');
      });
    }

    const btnCalibVertVirtual = document.getElementById('btn-calib-vert-virtual');
    if (btnCalibVertVirtual) {
      btnCalibVertVirtual.addEventListener('click', () => {
        if (state.virtualPlanePoints.length < 2) {
          toast('Primero definí la base del plano en el piso (2 clics).', 'warning');
          return;
        }
        state.isCalibVertVirtualPlane = true;
        state.isTiltingVirtualPlane = false;
        state.vanishingPointVPoints = [];
        toast('Ajuste vertical: Marcá 2 rectas verticales en la realidad. Hacé clic en 2 puntos para la primera, y 2 puntos para la segunda.', 'info');
      });
    }
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
