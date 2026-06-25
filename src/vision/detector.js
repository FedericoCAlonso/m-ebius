/* ============================================================
   MÖBIUS — detector.js
   Detección de marcadores ArUco y agrupamiento en hojas.
   ============================================================ */

export const OPENCV_DICT_4X4_50  = 0;   // cv.DICT_4X4_50 enum value

export const SHEET_TEMPLATES = [
  {
    name: "A5",
    ids: [0, 1, 2, 3],
    w_mm: 110.0,
    h_mm: 170.0,
    targets: [
      [10, 10],   // SupIzq
      [120, 10],  // SupDer
      [10, 180],  // InfIzq
      [120, 180], // InfDer
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

export const MAX_PLANES = 12;

/**
 * Detecta marcadores ArUco en srcMat usando cv.aruco_ArucoDetector (OpenCV.js 4.7+).
 * @param {cv.Mat} srcMat - Imagen fuente en RGBA
 * @returns {Array<{id:number, center:{x,y}}>}
 */
export function detectMarkers(srcMat) {
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
  params.minMarkerPerimeterRate = 0.005;
  params.adaptiveThreshWinSizeMin = 3;
  params.adaptiveThreshWinSizeMax = 23;
  params.adaptiveThreshWinSizeStep = 10;
  params.polygonalApproxAccuracyRate = 0.05;

  const refineParams = new cv.aruco_RefineParameters(10, 3, true);
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

  const markers    = [];
  const numMarkers = ids.rows;

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

/**
 * Agrupa los marcadores detectados en hojas de referencia usando los templates.
 * @param {Array<{id:number, center:{x,y}}>} markers
 * @returns {Array<{markers: Array<{id,center}>, template: Object}>}
 */
export function clusterIntoSheets(markers) {
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
