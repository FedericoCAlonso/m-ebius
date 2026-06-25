/* ============================================================
   MÖBIUS — math.js
   Módulo matemático: Jacobiano de distorsión, homografías,
   cálculo de bounding box y rectificación de perspectiva.
   ============================================================ */

export const GRID_STEP_PX    = 20;   // paso del barrido de la grilla (px)
export const STRETCH_LIMIT   = 2.0;  // factor máximo de estiramiento respecto a L_ref
export const SAFETY_MM       = 30;   // offset perimetral de seguridad (mm)
export const MAX_CANVAS_DIM  = 6000; // límite máximo por eje del canvas de salida (px)

/**
 * Calcula la homografía pixel→mm y la escala nativa del peor caso.
 * @param {Object} sheetObj - { markers: [m0,m1,m2,m3], template: Object }
 * @returns {{ H: cv.Mat, scale: number, sheetCenter: {x,y} }}
 */
export function computeHomographyAndScale(sheetObj) {
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

/**
 * Aplica la homografía H (pixel→mm) a un punto (x,y) en coordenadas
 * homogéneas, devolviendo {x, y} en mm y el denominador w.
 * @param {Float64Array} h  - datos row-major de la matriz 3×3
 * @param {number} x
 * @param {number} y
 * @returns {{x:number, y:number, w:number}|null}
 */
export function applyH(h, x, y) {
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
export function localStretch(h, px, py) {
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
 * @param {cv.Mat}      H            - Homografía pixel→mm (CV_64F 3×3)
 * @param {number}      imgW         - Ancho de la imagen fuente (px)
 * @param {number}      imgH         - Alto de la imagen fuente (px)
 * @param {{x,y}}       sheetCenter  - Centro de la hoja en px
 * @returns {Array<{gx,gy,mm:{x,y}}>} validPoints
 */
export function conformalFilter(H, imgW, imgH, sheetCenter) {
  const h = H.data64F;

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

/**
 * Calcula el bounding box en mm a partir de los puntos válidos del filtro,
 * añadiendo el offset de seguridad perimetral.
 * @param {Array<{mm:{x,y}}>} validPoints
 * @returns {{X_min,Y_min,X_max,Y_max}|null}
 */
export function computeBBox(validPoints) {
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
 * @param {cv.Mat}    srcMat        - Imagen fuente RGBA
 * @param {cv.Mat}    H             - Homografía pixel→mm
 * @param {number}    scale         - Escala nativa (px/mm)
 * @param {{X_min,Y_min,X_max,Y_max}} bbox
 * @param {HTMLCanvasElement} targetCanvas
 * @returns {{ canvasW:number, canvasH:number }}
 */
export function renderPlane(srcMat, H, scale, bbox, targetCanvas) {
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

  const T_inv = cv.matFromArray(3, 3, cv.CV_64F, [
    s,  0, -X_min * s,
    0,  s, -Y_min * s,
    0,  0,  1,
  ]);

  const M_final = new cv.Mat();
  const empty   = new cv.Mat();
  cv.gemm(T_inv, H, 1.0, empty, 0.0, M_final, 0);

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

/**
 * Encuentra el borde más cercano en un radio determinado (para snap).
 * @param {cv.Mat} srcMat
 * @param {number} x
 * @param {number} y
 * @param {number} radius
 * @returns {{x:number, y:number}|null}
 */
export function findNearestEdge(srcMat, x, y, radius) {
  if (!srcMat) return null;
  const ix = Math.round(x);
  const iy = Math.round(y);
  let rx = Math.max(0, ix - radius);
  let ry = Math.max(0, iy - radius);
  let rw = radius * 2;
  let rh = radius * 2;
  if (rx + rw > srcMat.cols) rw = srcMat.cols - rx;
  if (ry + rh > srcMat.rows) rh = srcMat.rows - ry;
  if (rw <= 0 || rh <= 0) return null;
  
  const rect = new cv.Rect(rx, ry, rw, rh);
  const roi = srcMat.roi(rect);
  
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
