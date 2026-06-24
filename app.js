/* ============================================================
   MÖBIUS — app.js
   Motor de rectificación de perspectiva con marcadores ArUco.
   Procesamiento 100% local vía OpenCV.js (WASM).

   Fases:
     1. Detección y clustering de marcadores ArUco (DICT_4x4_50)
     2. Calibración: homografía pixel→mm y escala nativa
     3. Filtro conforme de fuga bidimensional (Jacobiano de control)
     4. Bounding Box real + warpPerspective eficiente
     5. Herramienta de medición interactiva + exportación
   ============================================================ */
'use strict';

/* ─────────────────────────────────────────────────────────────
   CONSTANTES
───────────────────────────────────────────────────────────── */
const OPENCV_DICT_4X4_50  = 0;   // cv.DICT_4X4_50 enum value
// Catálogo de Hojas de Referencia
const SHEET_TEMPLATES = [
  {
    name: "A5",
    ids: [0, 1, 2, 3],
    w_mm: 100.0,
    h_mm: 170.0,
    targets: [
      [10, 10],   // SupIzq
      [110, 10],  // SupDer
      [10, 180],  // InfIzq
      [110, 180], // InfDer
    ]
  },
  {
    name: "A4",
    ids: [4, 5, 6, 7],
    w_mm: 170.0,
    h_mm: 260.0,
    targets: [
      [10, 10],
      [180, 10],
      [10, 270],
      [180, 270],
    ]
  },
  {
    name: "A3",
    ids: [8, 9, 10, 11],
    w_mm: 260.0,
    h_mm: 380.0,
    targets: [
      [10, 10],
      [270, 10],
      [10, 390],
      [270, 390],
    ]
  }
];

const GRID_STEP_PX    = 20;   // paso del barrido de la grilla (px)
const STRETCH_LIMIT   = 2.0;  // factor máximo de estiramiento respecto a L_ref
const SAFETY_MM       = 30;   // offset perimetral de seguridad (mm)
const MAX_PLANES      = 12;   // límite de hojas por imagen
const MAX_CANVAS_DIM  = 6000; // límite máximo por eje del canvas de salida (px)

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
  // cv es la variable global inyectada por opencv.js
  state.cvReady = true;
  hideCvErrorPanel();
  setStatus('OpenCV listo ✓', 'ready');
  toast('OpenCV.js cargado. Listo para procesar.', 'success');
});

document.addEventListener('opencvLoadFailed', () => {
  // Todas las fuentes fallaron (local + CDNs)
  setStatus('OpenCV no disponible', 'error');
  showCvErrorPanel();
});

/** Muestra el panel de error de carga de OpenCV. */
function showCvErrorPanel() {
  const panel = document.getElementById('cv-error-panel');
  if (panel) panel.classList.remove('hidden');
}

/** Oculta el panel de error (cuando carga correctamente). */
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
  dot.className  = `status-dot ${type}`;
  span.textContent = text;
}

function enableUI() {
  const dz = document.getElementById('drop-zone');
  dz.classList.remove('disabled');
  // Update drop zone hint text
  const sub = dz.querySelector('.drop-sub');
  if (sub && sub.textContent.includes('Cargando')) {
    sub.textContent = 'o usá los botones de abajo · Formatos: JPG, PNG, WEBP';
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

/** Toast notification. @param {string} msg @param {'info'|'success'|'error'} type @param {number} duration ms */
function toast(msg, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => {
    el.style.animation = 'toastOut .3s ease forwards';
    setTimeout(() => el.remove(), 320);
  }, duration);
}

/** Descarga un Blob o string como archivo. */
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href     = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { a.remove(); URL.revokeObjectURL(url); }, 1000);
}

/** Cede el control al hilo del navegador (evita freezes en procesamiento largo). */
function yieldToUI() {
  return new Promise(r => setTimeout(r, 0));
}

/* ─────────────────────────────────────────────────────────────
   CARGA DE IMAGEN → cv.Mat
───────────────────────────────────────────────────────────── */

/**
 * Carga un File (o un HTMLCanvasElement capturado de la cámara) en un cv.Mat RGBA.
 * @param {File|HTMLCanvasElement} source
 * @returns {Promise<{mat: cv.Mat, width: number, height: number}>}
 */
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
   FASE 1 — DETECCIÓN DE MARCADORES ARUCO
───────────────────────────────────────────────────────────── */

/**
 * Detecta marcadores ArUco en srcMat usando cv.aruco_ArucoDetector (OpenCV.js 4.7+).
 * Prueba primero 3 argumentos (firma oficial JS), luego 4 como fallback.
 * @param {cv.Mat} srcMat - Imagen fuente en RGBA
 * @returns {Array<{id:number, center:{x,y}}>}
 */
function detectMarkers(srcMat) {
  const gray    = new cv.Mat();
  const corners = new cv.MatVector();
  const ids     = new cv.Mat();

  cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);

  if (typeof cv.getPredefinedDictionary !== 'function' ||
      typeof cv.aruco_ArucoDetector !== 'function') {
    gray.delete(); corners.delete(); ids.delete();
    throw new Error(
      'API ArUco no encontrada. Verificá que opencv.js sea el de docs.opencv.org/4.8.0.'
    );
  }

  const dictId      = (typeof cv.DICT_4X4_50 !== 'undefined') ? cv.DICT_4X4_50 : OPENCV_DICT_4X4_50;
  const dict        = cv.getPredefinedDictionary(dictId);
  const params      = new cv.aruco_DetectorParameters();

  /* ── MEJORA DE DETECCIÓN PARA IMÁGENES GRANDES ── */
  // Permitir encontrar marcadores muy chiquitos (0.5% del perímetro total)
  params.minMarkerPerimeterRate = 0.005;
  // Ventana adaptativa más tolerante para sombras
  params.adaptiveThreshWinSizeMin = 3;
  params.adaptiveThreshWinSizeMax = 23;
  params.adaptiveThreshWinSizeStep = 10;
  params.polygonalApproxAccuracyRate = 0.05;

  const refineParams = new cv.aruco_RefineParameters(10, 3, true); // requerido en OpenCV.js 4.7+
  const detector    = new cv.aruco_ArucoDetector(dict, params, refineParams);

  /* Intento 1: 3 argumentos (firma oficial OpenCV.js 4.7/4.8) */
  let err3 = null;
  try {
    detector.detectMarkers(gray, corners, ids);
  } catch (e) {
    err3 = e;
  }

  /* Intento 2: 4 argumentos (con rejected) como fallback */
  if (err3) {
    const rejected = new cv.MatVector();
    try {
      detector.detectMarkers(gray, corners, ids, rejected);
      err3 = null;
    } catch (e4) {
      rejected.delete();
      dict.delete(); params.delete(); refineParams.delete(); detector.delete();
      gray.delete(); corners.delete(); ids.delete();
      throw new Error(
        'detectMarkers fallo en ambas firmas. ' +
        '3-args: ' + (err3.message || err3) + ' | ' +
        '4-args: ' + (e4.message || e4)
      );
    }
    rejected.delete();
  }

  dict.delete();
  params.delete();
  refineParams.delete();
  detector.delete();

  /* Extraer datos antes de liberar memoria */
  const markers    = [];
  const numMarkers = ids.rows; // Puede ser 0 si no detecta nada

  if (numMarkers > 0) {
    for (let i = 0; i < numMarkers; i++) {
      const markerId  = ids.data32S[i];
      const cornerMat = corners.get(i);
      const d         = cornerMat.data32F;
      const cx = (d[0] + d[2] + d[4] + d[6]) / 4;
      const cy = (d[1] + d[3] + d[5] + d[7]) / 4;
      markers.push({ id: markerId, center: { x: cx, y: cy } });
      console.debug('[Mobius] Marcador ID=' + markerId + ' en (' + cx.toFixed(0) + ',' + cy.toFixed(0) + ')');
    }
  }

  gray.delete();
  corners.delete();
  ids.delete();

  return markers;
}

/* ─────────────────────────────────────────────────────────────
   FASE 1b — CLUSTERING: agrupar marcadores en hojas
───────────────────────────────────────────────────────────── */

/**
 * Agrupa los marcadores detectados en hojas de referencia usando los templates.
 *
 * @param {Array<{id:number, center:{x,y}}>} markers
 * @returns {Array<{markers: Array<{id,center}>, template: Object}>}
 */
function clusterIntoSheets(markers) {
  const byId = {};
  for (const m of markers) {
    if (!byId[m.id]) byId[m.id] = [];
    byId[m.id].push(m);
  }

  const usedKeys = new Set();
  const sheets   = [];

  for (const template of SHEET_TEMPLATES) {
    const [id0, id1, id2, id3] = template.ids;
    
    if (!byId[id0]) continue;

    // Ordenar los ID0
    byId[id0].sort((a, b) => a.center.y - b.center.y || a.center.x - b.center.x);

    for (let i0 = 0; i0 < byId[id0].length; i0++) {
      const m0  = byId[id0][i0];
      const key0 = `${id0}-${i0}`;
      if (usedKeys.has(key0)) continue;

      const sheetMarkers = [m0];
      let valid   = true;

      for (let k = 1; k <= 3; k++) {
        const targetId = template.ids[k];
        const candidates = byId[targetId];
        if (!candidates || candidates.length === 0) { valid = false; break; }

        let closest  = null;
        let minDist  = Infinity;
        let closestK = -1;

        for (let j = 0; j < candidates.length; j++) {
          const key = `${targetId}-${j}`;
          if (usedKeys.has(key)) continue;
          const d = Math.hypot(
            candidates[j].center.x - m0.center.x,
            candidates[j].center.y - m0.center.y
          );
          if (d < minDist) { minDist = d; closest = candidates[j]; closestK = j; }
        }

        if (!closest) { valid = false; break; }
        sheetMarkers.push(closest);
        sheetMarkers[`_key${targetId}`] = `${targetId}-${closestK}`;
      }

      if (valid && sheetMarkers.length === 4) {
        usedKeys.add(key0);
        for (let k = 1; k <= 3; k++) usedKeys.add(sheetMarkers[`_key${template.ids[k]}`]);
        
        const cleanSheet = [sheetMarkers[0], sheetMarkers[1], sheetMarkers[2], sheetMarkers[3]];
        sheets.push({ markers: cleanSheet, template });

        if (sheets.length >= MAX_PLANES) break;
      }
    }
  }

  return sheets;
}

/* ─────────────────────────────────────────────────────────────
   FASE 2 — HOMOGRAFÍA Y ESCALA NATIVA
───────────────────────────────────────────────────────────── */

/**
 * Calcula la homografía pixel→mm y la escala nativa del peor caso.
 *
 * @param {Object} sheetObj - { markers: [m0,m1,m2,m3], template: Object }
 * @returns {{ H: cv.Mat, scale: number, sheetCenter: {x,y} }}
 */
function computeHomographyAndScale(sheetObj) {
  const c = sheetObj.markers.map(m => m.center);
  const t = sheetObj.template.targets;

  /* Puntos fuente: centros en píxeles (orden ID0..ID3) */
  const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    c[0].x, c[0].y,
    c[1].x, c[1].y,
    c[2].x, c[2].y,
    c[3].x, c[3].y,
  ]);

  /* Puntos destino: coordenadas teóricas en mm desde el template */
  const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    t[0][0], t[0][1],
    t[1][0], t[1][1],
    t[2][0], t[2][1],
    t[3][0], t[3][1],
  ]);

  /* H mapea pixel → mm */
  const H = cv.findHomography(srcPts, dstPts);

  srcPts.delete();
  dstPts.delete();

  /* Escala nativa: peor caso de fuga máxima */
  const dist_px_H = Math.hypot(c[1].x - c[0].x, c[1].y - c[0].y); 
  const dist_px_V = Math.hypot(c[2].x - c[0].x, c[2].y - c[0].y); 
  const scale     = Math.min(dist_px_H / sheetObj.template.w_mm, dist_px_V / sheetObj.template.h_mm);

  /* Centro geométrico de la hoja (para L_ref) */
  const sheetCenter = {
    x: (c[0].x + c[1].x + c[2].x + c[3].x) / 4,
    y: (c[0].y + c[1].y + c[2].y + c[3].y) / 4,
  };

  return { H, scale, sheetCenter };
}

/* ─────────────────────────────────────────────────────────────
   FASE 3 — FILTRO CONFORME DE FUGA BIDIMENSIONAL
───────────────────────────────────────────────────────────── */

/**
 * Aplica la homografía H (pixel→mm) a un punto (x,y) en coordenadas
 * homogéneas, devolviendo {x, y} en mm y el denominador w.
 * @param {Float64Array} h  - datos row-major de la matriz 3×3
 * @param {number} x
 * @param {number} y
 * @returns {{x:number, y:number, w:number}|null}
 */
function applyH(h, x, y) {
  const w = h[6] * x + h[7] * y + h[8];
  if (Math.abs(w) < 1e-10) return null;
  return {
    x: (h[0] * x + h[1] * y + h[2]) / w,
    y: (h[3] * x + h[4] * y + h[5]) / w,
    w,
  };
}

/**
 * Calcula el estiramiento local en píxeles de la homografía en (px, py):
 * qué tamaño en mm tiene un micro-paso de +1px en X y en Y.
 * @returns {{L_dx:number, L_dy:number, L_max:number, w:number, mm:{x,y}}|null}
 */
function localStretch(h, px, py) {
  const p0   = applyH(h, px,     py);
  const p_dx = applyH(h, px + 1, py);
  const p_dy = applyH(h, px,     py + 1);
  if (!p0 || !p_dx || !p_dy) return null;

  const L_dx = Math.hypot(p_dx.x - p0.x, p_dx.y - p0.y);
  const L_dy = Math.hypot(p_dy.x - p0.x, p_dy.y - p0.y);

  return { L_dx, L_dy, L_max: Math.max(L_dx, L_dy), w: p0.w, mm: p0 };
}

/**
 * Barrido de grilla sobre la imagen original.
 * Descarta puntos con distorsión extrema o que cruzaron el horizonte proyectivo.
 *
 * @param {cv.Mat}      H            - Homografía pixel→mm (CV_64F 3×3)
 * @param {number}      imgW         - Ancho de la imagen fuente (px)
 * @param {number}      imgH         - Alto de la imagen fuente (px)
 * @param {{x,y}}       sheetCenter  - Centro de la hoja en px
 * @returns {Array<{gx,gy,mm:{x,y}}>} validPoints
 */
function conformalFilter(H, imgW, imgH, sheetCenter) {
  const h = H.data64F; // Float64Array row-major

  /* L_ref: estiramiento en el centro de la hoja */
  const refS = localStretch(h, sheetCenter.x, sheetCenter.y);
  if (!refS) throw new Error('Homografía inválida: no se puede evaluar en el centro de la hoja.');

  const L_ref       = refS.L_max;
  const W_ref_sign  = Math.sign(refS.w);
  const threshold   = L_ref * STRETCH_LIMIT;
  const validPoints = [];

  for (let gy = 0; gy <= imgH; gy += GRID_STEP_PX) {
    for (let gx = 0; gx <= imgW; gx += GRID_STEP_PX) {
      const s = localStretch(h, gx, gy);
      if (!s) continue;

      /* FILTRO 1: Singularidad del horizonte (cambio de signo de W) */
      if (Math.sign(s.w) !== W_ref_sign) continue;

      /* FILTRO 2: Estiramiento excesivo */
      if (s.L_max > threshold) continue;

      validPoints.push({ gx, gy, mm: s.mm });
    }
  }

  return validPoints;
}

/* ─────────────────────────────────────────────────────────────
   FASE 4 — BOUNDING BOX Y RENDERIZADO
───────────────────────────────────────────────────────────── */

/**
 * Calcula el bounding box en mm a partir de los puntos válidos del filtro,
 * añadiendo el offset de seguridad perimetral.
 * @param {Array<{mm:{x,y}}>} validPoints
 * @returns {{X_min,Y_min,X_max,Y_max}|null}
 */
function computeBBox(validPoints) {
  if (validPoints.length === 0) return null;

  let X_min =  Infinity, Y_min =  Infinity;
  let X_max = -Infinity, Y_max = -Infinity;

  for (const { mm } of validPoints) {
    if (mm.x < X_min) X_min = mm.x;
    if (mm.y < Y_min) Y_min = mm.y;
    if (mm.x > X_max) X_max = mm.x;
    if (mm.y > Y_max) Y_max = mm.y;
  }

  /* Offset de seguridad perimetral */
  X_min -= SAFETY_MM;
  Y_min -= SAFETY_MM;
  X_max += SAFETY_MM;
  Y_max += SAFETY_MM;

  return { X_min, Y_min, X_max, Y_max };
}

/**
 * Ejecuta el warpPerspective y renderiza el plano rectificado en el canvas.
 *
 * Matriz final: M = T_inv · H   (mapea src_pixel → dst_pixel)
 * donde T_inv = [[s, 0, -X_min·s],
 *                [0, s, -Y_min·s],
 *                [0, 0,  1      ]]
 *
 * @param {cv.Mat}    srcMat        - Imagen fuente RGBA
 * @param {cv.Mat}    H             - Homografía pixel→mm
 * @param {number}    scale         - Escala nativa (px/mm)
 * @param {{X_min,Y_min,X_max,Y_max}} bbox
 * @param {HTMLCanvasElement} targetCanvas
 * @returns {{ canvasW:number, canvasH:number }}
 */
function renderPlane(srcMat, H, scale, bbox, targetCanvas) {
  const { X_min, Y_min, X_max, Y_max } = bbox;
  const s = scale;

  /* Dimensiones del canvas de salida (en px) */
  let canvasW = Math.round((X_max - X_min) * s);
  let canvasH = Math.round((Y_max - Y_min) * s);

  /* Clamp de seguridad para no reventar la memoria */
  if (canvasW > MAX_CANVAS_DIM || canvasH > MAX_CANVAS_DIM) {
    const clampScale = Math.min(MAX_CANVAS_DIM / canvasW, MAX_CANVAS_DIM / canvasH);
    canvasW = Math.round(canvasW * clampScale);
    canvasH = Math.round(canvasH * clampScale);
    console.warn('[Möbius] Canvas clamped a', canvasW, '×', canvasH);
  }

  targetCanvas.width  = canvasW;
  targetCanvas.height = canvasH;

  /*
   * T_inv: escala y traslada del espacio mm al espacio dst_pixel.
   * dst_pixel = s · (mm - offset)
   * [ s  0  -X_min·s ]
   * [ 0  s  -Y_min·s ]
   * [ 0  0      1    ]
   */
  const T_inv = cv.matFromArray(3, 3, cv.CV_64F, [
    s,  0, -X_min * s,
    0,  s, -Y_min * s,
    0,  0,  1,
  ]);

  /* M = T_inv · H  (src_pixel → dst_pixel, sin WARP_INVERSE_MAP) */
  const M_final = new cv.Mat();
  const empty   = new cv.Mat(); // Beta=0, src3 no se usa
  cv.gemm(T_inv, H, 1.0, empty, 0.0, M_final, 0);

  const dstMat = new cv.Mat();
  const dsize  = new cv.Size(canvasW, canvasH);

  cv.warpPerspective(
    srcMat, dstMat, M_final, dsize,
    cv.INTER_LINEAR,
    cv.BORDER_CONSTANT,
    new cv.Scalar(0, 0, 0, 0) // borde transparente
  );

  cv.imshow(targetCanvas, dstMat);

  /* Liberar toda la memoria WASM de esta operación */
  T_inv.delete();
  empty.delete();
  M_final.delete();
  dstMat.delete();

  return { canvasW, canvasH };
}

/* ─────────────────────────────────────────────────────────────
   VISOR Y MODO PROYECCIÓN (OPCIÓN B)
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

  // Dibujar imagen original
  cv.imshow(mainCanvas, state.originalImageMat);

  // Ajustar zoom inicial
  const rect = wrapper.getBoundingClientRect();
  const scaleX = rect.width / width;
  const scaleY = rect.height / height;
  state.viewer.scale = Math.min(scaleX, scaleY) * 0.9;
  
  // Centrar
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
  let currentTouchPoint = null; // Para mostrar el cursor desplazado en touch

  function findNearestEdge(x, y, radius) {
  const src = state.originalImageMat;
  if (!src) return null;
  const ix = Math.round(x);
  const iy = Math.round(y);
  let rx = Math.max(0, ix - radius);
  let ry = Math.max(0, iy - radius);
  let rw = radius * 2;
  let rh = radius * 2;
  if (rx + rw > src.cols) rw = src.cols - rx;
  if (ry + rh > src.rows) rh = src.rows - ry;
  if (rw <= 0 || rh <= 0) return null;
  
  const rect = new cv.Rect(rx, ry, rw, rh);
  const roi = src.roi(rect);
  
  const gray = new cv.Mat();
  cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);
  const blur = new cv.Mat();
  cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);
  const edges = new cv.Mat();
  cv.Canny(blur, edges, 50, 150);
  
  let closestDist = Infinity;
  let closestX = -1;
  let closestY = -1;
  const cx = ix - rx;
  const cy = iy - ry;
  for (let r = 0; r < edges.rows; r++) {
    for (let c = 0; c < edges.cols; c++) {
      if (edges.ucharPtr(r, c)[0] > 128) {
        const dist = Math.hypot(c - cx, r - cy);
        if (dist < closestDist) {
          closestDist = dist;
          closestX = c;
          closestY = r;
        }
      }
    }
  }
  
  roi.delete(); gray.delete(); blur.delete(); edges.delete();
  if (closestDist < radius) {
    return { x: rx + closestX, y: ry + closestY };
  }
  return null;
}

  function processClickPoint(clickX, clickY) {
    let imgX = (clickX - state.viewer.offsetX) / state.viewer.scale;
    let imgY = (clickY - state.viewer.offsetY) / state.viewer.scale;

    const plane = state.planes[state.activePlaneIndex];
    if (!plane) return null;

    if (state.snapEnabled && state.originalImageMat) {
      const snapObj = findNearestEdge(imgX, imgY, 40);
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

  // Medición - TOUCH (1 dedo) con offset para ajuste fino
  overlayCanvas.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
      e.preventDefault(); 
      const rect = wrapper.getBoundingClientRect();
      const touch = e.touches[0];
      const clickX = touch.clientX - rect.left;
      const clickY = touch.clientY - rect.top - 60; // Offset Y: 60px arriba del dedo
      
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
  container.style.transform = `translate(${state.viewer.offsetX}px, ${state.viewer.offsetY}px) scale(${state.viewer.scale})`;
  drawMeasurements(); // Redibujar grosores según el scale
}

function drawGrid() {
  const overlay = document.getElementById('overlay-canvas');
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const plane = state.planes[state.activePlaneIndex];
  if (!plane) return;

  const H_inv = plane.H_inv;
  const H_data = H_inv.data64F;

  // Función auxiliar para proyectar mm a px
  function project(x_mm, y_mm) {
    const w = H_data[6] * x_mm + H_data[7] * y_mm + H_data[8];
    return {
      x: (H_data[0] * x_mm + H_data[1] * y_mm + H_data[2]) / w,
      y: (H_data[3] * x_mm + H_data[4] * y_mm + H_data[5]) / w
    };
  }

  // Dinámicamente calcular rango de grilla visible
  const range = 800; // mm
  const step = 50;

  // Grilla sutil
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
    
    // Etiqueta
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
    ctx.strokeStyle = '#f87171'; ctx.lineWidth = strokeW; // Rojo suave para preview
    ctx.stroke();
    
    // Label provisional
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
  tbody.innerHTML = '';

  if (state.measurements.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="meas-empty">Hacé clic en dos puntos del plano activo para medir.</td></tr>';
    return;
  }

  for (const m of state.measurements) {
    const tr = document.createElement('tr');
    
    // Input de referencia binded al state
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

    // Bind event
    document.getElementById(inputId).addEventListener('input', (e) => {
      m.refName = e.target.value;
    });
  }
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* ─────────────────────────────────────────────────────────────
   EXPORTACIÓN
───────────────────────────────────────────────────────────── */



/** Genera el recorte plano del plano activo bajo demanda. */
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

/** Exporta todas las mediciones como CSV. */
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

  const csv  = [header, ...rows].map(r => r.join(',')).join('rn');
  const blob = new Blob(['uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
  downloadBlob(blob, `mediciones_mobius_${new Date().toISOString().slice(0,10)}.csv`);
  toast('Mediciones exportadas como CSV.', 'success');
}

/* ─────────────────────────────────────────────────────────────
   CONSTRUCCIÓN DEL CARD DE RESULTADO
───────────────────────────────────────────────────────────── */

/**
 * Genera el botón en el sidebar para un plano detectado.
 * @param {number} index
 * @param {number} scale
 * @param {string} templateName
 */
function createPlaneSidebarButton(index, scale, templateName) {
  const label = `Plano ${index + 1} (${templateName})`;
  const list  = document.getElementById('planes-list');

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

/**
 * Establece el plano activo, actualizando la UI y redibujando la cuadrícula.
 */
function setActivePlane(index) {
  state.activePlaneIndex = index;
  
  // Actualizar UI
  document.querySelectorAll('.plane-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`btn-plane-${index}`);
  if (activeBtn) activeBtn.classList.add('active');

  // Redibujar cuadrícula
  drawGrid();
  drawMeasurements();
}


/* ─────────────────────────────────────────────────────────────
   PIPELINE PRINCIPAL
───────────────────────────────────────────────────────────── */

/**
 * Valida un File seleccionado (drag/drop o file-input) y llama a processImage.
 * Acepta archivos por MIME type O por extensión (Windows puede no incluir MIME type).
 * @param {File} file
 */
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

/**
 * Orquesta las 5 fases de procesamiento para una imagen dada.
 * @param {File|HTMLCanvasElement} source
 */
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
    /* ── Cargar imagen ── */
    const { mat, width, height } = await loadSourceAsMat(source);
    srcMat = mat;

    /* ── FASE 1: Detectar marcadores ── */
    updateProcessingSub(`Detectando marcadores ArUco en ${width}×${height} px…`);
    await yieldToUI();

    const markers = detectMarkers(srcMat);

    if (markers.length < 4) {
      throw new Error(
        `Se detectaron solo ${markers.length} marcador(es). ` +
        'Se necesitan al menos 4 de un mismo template (ej. IDs 0-3) para rectificar un plano.'
      );
    }

    /* ── FASE 1b: Clustering ── */
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

    /* ── FASES 2: Calcular Homografías ── */
    for (let i = 0; i < sheets.length; i++) {
      updateProcessingSub(`Procesando plano ${i + 1} de ${sheets.length}…`);
      await yieldToUI();

      const sheet = sheets[i];

      /* FASE 2: Homografía y escala */
      const { H, scale } = computeHomographyAndScale(sheet);

      /* Inversa de Homografía para Proyección (H_inv) */
      const H_inv = new cv.Mat();
      cv.invert(H, H_inv, cv.DECOMP_LU);

      /* Crear botón en el sidebar */
      const label = createPlaneSidebarButton(i, scale, sheet.template.name);

      /* Registrar plano en el estado (mantenemos H y H_inv) */
      state.planes.push({ id: i, label, H: H.clone(), H_inv, scale, template: sheet.template });

      H.delete(); // Borramos el temporal, ya lo clonamos en el estado

      await yieldToUI();
    }

    /* ── Finalizar y Configurar Visor ── */
    state.originalImageMat = srcMat; // Guardar la imagen original
    // No hacemos delete de srcMat porque lo mantenemos en state.originalImageMat

    const count = state.planes.length;
    if (count > 0) {
      document.getElementById('planes-count').textContent = String(count);
      document.getElementById('upload-view').classList.add('hidden');
      document.getElementById('workspace-view').classList.add('visible');
      document.getElementById('measurements-section').classList.add('visible');
      
      initViewer(width, height);
      setActivePlane(0); // Seleccionar el primer plano
      
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

/** Limpia todos los resultados anteriores. */
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
  
  // Limpiar DOM
  document.getElementById('planes-list').innerHTML = '';
  document.getElementById('planes-count').textContent = '0';
  document.getElementById('upload-view').classList.remove('hidden');
  document.getElementById('workspace-view').classList.remove('visible');
  document.getElementById('measurements-section').classList.remove('visible');
  
  // Limpiar visor
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

  /* Habilitar la UI INMEDIATAMENTE — no esperar a OpenCV.
     El procesamiento se bloqueará internamente si OpenCV no está listo,
     pero el usuario puede seleccionar archivos y arrastrar desde el inicio. */
  enableUI();

  /* ── Prevención a nivel documento: evita que el browser navegue a la imagen
     si el usuario suelta fuera del drop-zone ── */
  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop',     e => e.preventDefault());

  /* ── Drag & Drop sobre el drop-zone ── */
  ['dragenter', 'dragover'].forEach(ev =>
    dropZone.addEventListener(ev, e => {
      e.preventDefault();
      e.stopPropagation();
      if (!state.processing) dropZone.classList.add('drag-over');
    })
  );
  ['dragleave', 'dragend'].forEach(ev =>
    dropZone.addEventListener(ev, e => {
      dropZone.classList.remove('drag-over');
    })
  );

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

  /* ── Clic en cualquier parte del drop-zone abre el file-picker
     (excepto si se hace clic en un botón interno, que maneja su propio evento) ── */
  dropZone.addEventListener('click', e => {
    if (!e.target.closest('.btn')) {
      fileInput.click();
    }
  });
  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  /* ── Botón archivo ── */
  document.getElementById('btn-file').addEventListener('click', e => {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (file) { handleFileSelected(file); fileInput.value = ''; }
  });

  /* ── Botón cámara ── */
  document.getElementById('btn-camera').addEventListener('click', e => {
    e.stopPropagation();
    openCamera();
  });

  document.getElementById('btn-capture').addEventListener('click', captureFrame);
  document.getElementById('btn-camera-cancel').addEventListener('click', closeCamera);

  /* ── Nueva imagen ── */
  document.getElementById('btn-reset').addEventListener('click', clearResults);

  /* ── Exportación global ── */
  const btnExportPng = document.getElementById('btn-export-png');
  if (btnExportPng) btnExportPng.addEventListener('click', exportAllPNG);
  
  const btnExportCsv = document.getElementById('btn-export-csv');
  if (btnExportCsv) btnExportCsv.addEventListener('click', exportMeasurementsCSV);

  /* ── Limpiar mediciones ── */
  document.getElementById('btn-clear-meas').addEventListener('click', clearAllMeasurements);

  /* ── Delegación: botones de eliminar medición individuales ── */
  document.getElementById('meas-tbody').addEventListener('click', e => {
    const btn = e.target.closest('[data-meas-id]');
    if (btn) removeMeasurement(Number(btn.dataset.measId));
  });

  /* ── Botón reintentar carga de OpenCV ── */
  const btnRetryCV = document.getElementById('btn-retry-cv');
  if (btnRetryCV) {
    btnRetryCV.addEventListener('click', () => {
      hideCvErrorPanel();
      setStatus('Reintentando carga…', 'loading');
      // Reiniciar el loader dinámico definido en el HTML
      if (typeof _tryLoadCV === 'function') {
        window._cvSourceIdx = 0; // resetear al inicio
        _tryLoadCV();
      } else {
        // fallback: recargar la página completa
        window.location.reload();
      }
    });
  }

  /* ── Paste de imagen desde clipboard ── */
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

  // Exportar CSV
  const btnCsv = document.getElementById('btn-export-csv');
  if(btnCsv) {
    btnCsv.onclick = () => {
      if (state.measurements.length === 0) return toast('No hay mediciones para exportar', 'error');
      let csv = 'ID,Plano,Referencia,Distancia (mm)\\n';
      state.measurements.forEach(m => {
        csv += `${m.id},${m.planeLabel},${m.refName || ''},${m.distMm.toFixed(2)}\\n`;
      });
      const blob = new Blob([csv], { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'mediciones_mobius.csv';
      a.click();
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
          
          // Reconstruir planos
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
          
          renderPlanesList();
          if(state.planes.length > 0) selectPlane(0);
          renderMeasurementTable();
          
          hideProcessing();
          state.processing = false;
          toast('Proyecto cargado', 'success');
        } catch(e) {
          hideProcessing();
          state.processing = false;
          toast('Error al leer proyecto', 'error');
        }
      };
      reader.readAsText(file);
    };
  }

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
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADaklEQVR4Ad3BsQ3AQBDDMGn/oZ0RVByQ4kl5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxC0ggSxpkESSNIGGcSJI0gYZxJkDSChHEmQdIIEsaZBEkjSBhnEiSNIGGcSZA0goRxJkHSCBLGmQRJI0gYZxIkjSBhnEmQNIKEcSZB0ggSxpkESSNIGGcSJI0gYZxJkDSChHEmQdIIEsaZBEkjSBhnEiSNIGGcSZA0goRxJkHSCBLGmQRJI0gYZxIkjSBhnEmQNIKEcSZB0ggSxpkESSNIGGcSJI0gYZxJkDSChHEmQdIIEsaZBEkjSBhnEiSNIGGcSZA0goRxJkHSCBLGmQS5G0HKCBLkbgQpI0iQuxGkjCBB7kaQMoIEuRtByggS5G4EKSNIkLsRpIwgQe5GkDKCBLkbQcoIEuRuBCkjSJC7EaSMIEHuRpAyggS5G0HKCBLkbgQpI0iQuxGkjCBB7kaQMoIEuRtByggS5G4EKSNIkLsRpIwgQe5GkDKCBLkbQcoIEuRuBCkjSJC7EaSMIEHuRpAyggS5G0HKCBLkbgQpI0iQuxGkjCBB7kaQMoIEuRtByggS5G4EKSNIkLsRpIwgQe5GkDKCBLkbQcoIEuQH40yC/GCcSZAfjDMJ8oNxJkF+MM4kyA/GmQT5wTiTID8YZxLkB+NMgvxgnEmQH4wzCfKDcSZBfjDOJMgPxpkE+cE4kyA/GGcS5AfjTIL8YJxJkB+MMwnyg3EmQX4wziTID8aZBPnBOJMgPxhnEuQH40yC/GCcSZAfjDMJ8oNxJkF+MM4kyA/GmQT5wTiTID8YZxLkB+NMgvxgnEmQH4wiV/KDUeRKfjCKXMkPRpEr+cEociU/GEWu5AejyJX8YBS5kh+MIlfyg1HkSn4wilzJD0aRK/nBKHIlPxhFruQHo8iV/GAUuZIfjCJX8oNR5Ep+MIpcyQ9GkSv5wShyJT8YRa7kB6PIlfxgFLmSH4wiV/KDUeRKfjCKXMkPRpEr+cEociU/GEWu5AejyJX8YBS5kh+MIlfyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gj5BHyiA+N8sfJbgus8QAAAABJRU5ErkJggg==",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADBElEQVR4Ad3BMRHAQBDEMJs/6A0EF99kTpIj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOULS+AEJksYPSJA0fkCCpPEDEiSNH5AgafyABEnjByRIGj8gQdL4AQmSxg9IkDR+QIKk8QMSJI0fkCBp/IAESeMHJEgaPyBB0vgBCZLGD0iQNH5AgqTxAxIkjR+QIGn8gARJ4wckSBo/IEHS+AEJksYPSJA0fkCCpPEDEiSNH5AgafyABEnjByRIGj8gQdL4AQmSRpBnI0iQNII8G0GCpBHk2QgSJI0gz0aQIGkEeTaCBEkjyLMRJEgaQZ6NIEHSCPJsBAmSRpBnI0iQNII8G0GCpBHk2QgSJI0gz0aQIGkEeTaCBEkjyLMRJEgaQZ6NIEHSCPJsBAmSRpBnI0iQNII8G0GCpBHk2QgSJI0gz0aQIGkEeTaCBEkjyLMRJEgaQZ6NIEHSCPJsBAmSRpBnI0iQNII8G0GCpBHk2QgSJI0gz0aQIGkEeTaCBEkjyLMRJEgaQZ6NIEHSCPJsBAmSRpBnI0iQNII8G0GCpBEkjCKvJI0gYRR5JWkECaPIK0kjSBhFXkkaQcIo8krSCBJGkVeSRpAwirySNIKEUeSVpBEkjCKvJI0gYRR5JWkECaPIK0kjSBhFXkkaQcIo8krSCBJGkVeSRpAwirySNIKEUeSVpBEkjCKvJI0gYRR5JWkECaPIK0kjSBhFXkkaQcIo8krSCBJGkVeSRpAwirySNIKEUeSVpBEkjCKvJI0gYRR5JWkECaPIK0kjSBhFXkkaQcIo8krSCBJGkVeSRpAwirySNIKEUeSVpBEkjCKv5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkiA9v7KfJsmW8mwAAAABJRU5ErkJggg==",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADJ0lEQVR4Ad3BsQ0AMAzDMOn/o90TPBjIUFI+IZ+QT8gn5BPyCfmEfEI+IZ+QT8gn5BPyCfmEfEI+IZ+QT8gn5BPyCfmEfEI+IZ+QT8gn5BPyCfmEfEI+IZ+QA2EmhRwIMynkQJhJIQfCTAo5EGZSyIEwk0IOhJkUciDMpJADYSaFHAgzKeRAmEkhB8JMCjkQZlLIgTCTQg6EmRRyIMykkANhJoUcCDMp5ECYSSEHwkwKORBmUsiBMJNCDoSZFHIgzKSQA2EmhRwIMynkQJhJIQfCTAo5EGZSyIEwk0IOhJkUciDMpJADYSaFHAgzKeRAmEkhB8JMCjkQZlLIgTCTQg6EmRRyIMykkANhJoUcCDMp5ECYSSEHwkwKORBmUsiBMJNCDoSZFHIgzKSQA2EmhRwIMynkQJhJIQfCTAo5EGZSyIEwk0IOhJkUciDMpJADYSaFHAgzKeRAmEkhB8JMCjkQZlLIgTCTQg6EmRRyIMykkANhJoUcCDMp5EBoZCUHQiMrORAaWcmB0MhKDoRGVnIgNLKSA6GRlRwIjazkQGhkJQdCIys5EBpZyYHQyEoOhEZWciA0spIDoZGVHAiNrORAaGQlB0IjKzkQGlnJgdDISg6ERlZyIDSykgOhkZUcCI2s5EBoZCUHQiMrORAaWcmB0MhKDoRGVnIgNLKSA6GRlRwIjazkQGhkJQdCIyupwkqaUEghVVhJEwoppAoraUIhhVRhJU0opJAqrKQJhRRShZU0oZBCqrCSJhRSSBVW0oRCCqnCSppQSCFVWEkTCimkCitpQiGFVGElTSikkCqspAmFFFKFlTShkEKqsJImFFJIFVbShEIKqcJKmlBIIVVYSRMKKaQKK2lCIYVUYSVNKKSQKqykCYUUUoWVNKGQQqqwkiYUUkgVVtKEQgqpwkqaUEghVVhJEwoppAoraUIhhVRhJU0opJAqrKQJhRRShZU0oZBCqrCSJhRSSBVW0oRCCqnCSppQSCGfkE/IJ+QT8gn5hHxCPiGfkE/IJ+QT8gn5hHxCPiGfkE/IJ+QT8gn5hHxCPiGfkE/IJ+QT8gn5hHxCPiGfeIgSpslm7qh6AAAAAElFTkSuQmCC",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADaUlEQVR4Ad3BsREAMBDCMHv/ockIFFyal+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5QqpQyCwUUkgVCpmFQgqpQiGzUEghVShkFgoppAqFzEIhhVShkFkopJAqFDILhRRShUJmoZBCqlDILBRSSBUKmYVCCqlCIbNQSCFVKGQWCimkCoXMQiGFVKGQWSikkCoUMguFFFKFQmahkEKqUMgsFFJIFQqZhUIKqUIhs1BIIVUoZBYKKaQKhcxCIYVUoZBZKKSQKhQyC4UUUoVCZqGQQqpQyCwUUkgVCpmFQgqpQiGzUEghVShkFgoppAqFzEIhhVShkFkopJAqFDILhRRShUJmoZBCqlDILBRSSBUKmYVCCqlCIbNQSCFVKGQWCimkCoXMQiGFVKGQWSikkCoUMguFFFKFQmahkEKqUMgsFFJIFQqZhUIKqUIhs1BIIVUoZBYKKaQKhcxCIYVUoZBZKKSQKhQyC4UUUoVCZqGQQqpQyCwUUkgVCpmFQgqpQiGzUEghVShkFgoppAqFzEIhhVShkFkopJAqFDILhRRShUJmoZBCqlDILBRSSBUKmYVCCqlCIbNQSCFVKGQWCimkCoXMQiGFVKGQWSikkCoUMguFFFKFQmahkEKqUMgsFFJIFQqZhUIK2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YVC/pNdKOQ/2YWZrGQXZrKSXZjJSnZhJivZhZmsZBdmspJdmMlKdmEmK9mFmaxkF2aykl2YyUp2YSYr2YWZrGQXZrKSXZjJSnZhJivZhZmsZBdmspJdmMlKdmEmK9mFmaxkF2aykl2YyUp2YSYr2YWZrGQXZrKSXZjJSnZhJivZhZmsZBdmspJdmMlKdmEmK9mFmazkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+SIBxUex8mUWmejAAAAAElFTkSuQmCC",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADa0lEQVR4Ad3BMRHAQBDEMJs/6A0EFzdpXpJHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkbgQpI0iQuxGkjCBB7kaQMoIEuRtByggS5G4EKSNIkLsRpIwgQe5GkDKCBLkbQcoIEuRuBCkjSJC7EaSMIEHuRpAyggS5G0HKCBLkbgQpI0iQuxGkjCBB7kaQMoIEuRtByggS5G4EKSNIkLsRpIwgQe5GkDKCBLkbQcoIEuRuBCkjSJC7EaSMIEHuRpAyggS5G0HKCBLkbgQpI0iQuxGkjCBB7kaQMoIEuRtByggS5G4EKSNIkLsRpIwgQe5GkDKCBLkbQcoIEuRuBCkjSJC7EeR/cjeC/E/uRpD/yd0I8j+5G0H+J3cjyP/kbgT5n9yNIP+TuxHkf3I3gvxP7kaQ/8ndCPI/uRtB/id3I8j/5G4E+Z/cjSD/k7sR5H9yN4L8T+5GkP/J3QjyP7kbQf4ndyPI/+RuBPmf3I0g/5O7EeR/cjeC/E/uRpD/yd0I8j+5G0H+J3cjyP/kbgT5n9yNIP+TuxHkf5JGkLMRJEgaQc5GkCBpBDkbQYKkEeRsBAmSRpCzESRIGkHORpAgaQQ5G0GCpBHkbAQJkkaQsxEkSBpBzkaQIGkEORtBgqQR5GwECZJGkLMRJEgaQc5GkCBpBDkbQYKkEeRsBAmSRpCzESRIGkHORpAgaQQ5G0GCpBHkbAQJkkaQsxEkSBpBzkaQIGkEORtBgqQR5GwECZJGkLMRJEgaQc5GkCBpBDkbQYKkEeRsBAmSRpCzESRIGkHORpAgaQQ5G0GCpBHkbAQJkkaQsxEkSBpBzkaQIGn8T64kjf/JlaTxP7mSNP4nV5LG/+RK0vifXEka/5MrSeN/ciVp/E+uJI3/yZWk8T+5kjT+J1eSxv/kStL4n1xJGv+TK0njf3IlafxPriSN/8mVpPE/uZI0/idXksb/5ErS+J9cSRr/kytJ439yJWn8T64kjf/JlaTxP7mSNP4nV5LG/+RK0vifXEka/5MrSeN/ciVp/E+u5BHyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gj5BHyCHmEPEIeIY+QR8gj5BEfDa/IyVknTnIAAAAASUVORK5CYII=",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADaklEQVR4Ad3BMRHAQBDEMJs/6A0EFzdpXpJHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkEfIIeYQ8Qh4hj5BHyCPkbvxPgtyN/0mQu/E/CXI3/idB7sb/JMjd+J8EuRv/kyB3438S5G78T4Lcjf9JkLvxPwlyN/4nQe7G/yTI3fifBLkb/5Mgd+N/EuRu/E+C3I3/SZC78T8Jcjf+J0Huxv8kyN34nwS5G/+TIHfjfxLkbvxPgtyN/0mQu/E/CXI3/idB7sb/JMjd+J8EuRv/kyB3438S5G78T4KkEeRsBAmSRpCzESRIGkHORpAgaQQ5G0GCpBHkbAQJkkaQsxEkSBpBzkaQIGkEORtBgqQR5GwECZJGkLMRJEgaQc5GkCBpBDkbQYKkEeRsBAmSRpCzESRIGkHORpAgaQQ5G0GCpBHkbAQJkkaQsxEkSBpBzkaQIGkEORtBgqQR5GwECZJGkLMRJEgaQc5GkCBpBDkbQYKkEeRsBAmSRpCzESRIGkHORpAgaQQ5G0GCpBHkbAQJkkaQsxEkSBpBzkaQIGkEORtBgqQR5GwECZLGlfxP0riS/0kaV/I/SeNK/idpXMn/JI0r+Z+kcSX/kzSu5H+SxpX8T9K4kv9JGlfyP0njSv4naVzJ/ySNK/mfpHEl/5M0ruR/ksaV/E/SuJL/SRpX8j9J40r+J2lcyf8kjSv5n6RxJf+TNK7kf5LGlfxP0riS/0kaV/I/SeNK/idpXMn/JI0r+Z+kcSX/kzSu5H+SxpX8T9K4kv9JGldSRpAgaVxJGUGCpHElZQQJksaVlBEkSBpXUkaQIGlcSRlBgqRxJWUECZLGlZQRJEgaV1JGkCBpXEkZQYKkcSVlBAmSxpWUESRIGldSRpAgaVxJGUGCpHElZQQJksaVlBEkSBpXUkaQIGlcSRlBgqRxJWUECZLGlZQRJEgaV1JGkCBpXEkZQYKkcSVlBAmSxpWUESRIGldSRpAgaVxJGUGCpHElZQQJksaVlBEkSBpXUkaQIGlcSRlBgqRxJWUECZLGlZQRJEgaV1JGkCCPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGPkEfII+QR8gh5hDxCHiGP+ADOlsfJTarkTAAAAABJRU5ErkJggg==",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAADMElEQVR4Ad3BsRHAQAzDMHL/oZURVOg+hQE5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAj5Ag5Qo6QI+QIOUKOkCPkCDlCjpAqFDILhRRShUJmoZBCqlDILBRSSBUKmYVCCqlCIbNQSCFVKGQWCimkCoXMQiGFVKGQWSikkCoUMguFFFKFQmahkEKqUMgsFFJIFQqZhUIKqUIhs1BIIVUoZBYKKaQKhcxCIYVUoZBZKKSQKhQyC4UUUoVCZqGQQqpQyCwUUkgVCpmFQgqpQiGzUEghVShkFgoppAqFzEIhhVShkFkopJAqFDILhRRShUJmoZBCqlDILBRSSBUKmYVCCqlCIbNQSCFVKGQWCimkCoXMQiGFVKGQWSikkCoUMguFFFKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlVXhPVlKF92QlPwiNrOQHoZGV/CA0spIfhEZW8oPQyEp+EBpZyQ9CIyv5QWhkJT8IjazkB6GRlfwgNLKSH4RGVvKD0MhKfhAaWckPQiMr+UFoZCU/CI2s5AehkZX8IDSykh+ERlbyg9DISn4QGlnJD0IjK/lBaGQlPwiNrOQHoZGV/CA0spIfhEZW8oPQyEp+EBpZyQ9CIyv5QWhkJT8IjazkB6GRlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVlJFd6TlVThPVnJEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRHySwpskwQyoaAAAAAElFTkSuQmCC",
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAAAAACIM/FCAAAC7klEQVR4Ad3BwQ3AQAzDMGn/od0R/AhSHELKEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRcoRUYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2SRWmZJ9UYUr2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ/MhUL2yVwoZJ9U4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIVU4QFSSBUeIIX8IDQyJT8IjUzJD0IjU/KD0MiU/CA0MiU/CI1MyQ9CI1Pyg9DIlPwgNDIlPwiNTMkPQiNT8oPQyJT8IDQyJT8IjUzJD0IjU/KD0MiU/CA0MiU/CI1MyQ9CI1Pyg9DIlPwgNDIlPwiNTMkPQiNT8oPQyJT8IDQyJT8IjUzJD0IjU/KD0MiU/CA0MiU/CI1MyQ9CI1Pyg9DIlPwgNDIlR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHyBFyhBwhR8gRcoQcIUfIEXKEHCFHfCexhcn27L0vAAAAAElFTkSuQmCC",
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

  // Calculamos el offset para centrar el rectángulo de marcadores en la hoja física
  const offsetX = (pageSize.w - tpl.w_mm) / 2;
  const offsetY = (pageSize.h - tpl.h_mm) / 2;
  
  // HTML Content para la ventana de impresión
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
          transform: translate(-50%, -50%); /* Centrar la imagen en la coordenada (x,y) real */
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
          /* Forzamos el tamaño de página y quitamos los márgenes del navegador para controlar todo de forma absoluta */
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
  printWindow.document.open();
  printWindow.document.write(html);
  printWindow.document.close();
}
