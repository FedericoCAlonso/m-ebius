/* ============================================================
   MÖBIUS — geometry3d.js
   Metrología de vista única (Single View Metrology).
   Cálculo de alturas 3D mediante Razón Doble y puntos de fuga.
   ============================================================ */

/**
 * Producto cruz 3D para vectores homogéneos.
 */
export function crossProduct(a, b) {
  return {
    x: a.y * b.w - a.w * b.y,
    y: a.w * b.x - a.x * b.w,
    w: a.x * b.y - a.y * b.x
  };
}

/**
 * Producto escalar 3D.
 */
export function dotProduct(a, b) {
  return a.x * b.x + a.y * b.y + a.w * b.w;
}

/**
 * Normaliza un punto homogéneo para que w=1 (si no es un punto en el infinito).
 */
export function normalize(p) {
  if (Math.abs(p.w) > 1e-7) {
    return { x: p.x / p.w, y: p.y / p.w, w: 1 };
  }
  return p;
}

/**
 * Calcula la estructura proyectiva cruzando los puntos de fuga.
 * Extrae vX, vY, horizonte l, y vZ.
 */
export function calcularEstructuraEscena(H_base_inv, H_alzado_inv) {
  const hb = H_base_inv.data64F; // Matriz H_base
  const hw = H_alzado_inv.data64F; // Matriz H_alzado

  // Los puntos de fuga de la base son las columnas 1 y 2 de H_base
  // Pero ojo, H_base_inv mapea mm -> píxeles. 
  // La columna 0 es h1, columna 1 es h2.
  const vX = { x: hb[0], y: hb[3], w: hb[6] };
  const vY = { x: hb[1], y: hb[4], w: hb[7] };

  // Horizonte l = vX × vY
  const l = crossProduct(vX, vY);

  // Punto de fuga vertical vZ = columna 1 (Y) de H_alzado
  const vZ = { x: hw[1], y: hw[4], w: hw[7] };

  return { vX, vY, l, vZ };
}

/**
 * Calcula las coordenadas paramétricas (alpha, beta) de un punto p
 * en la línea definida por los vectores base v y vZ.
 */
function getLineParams(p, v, vZ) {
  const vv = dotProduct(v, v);
  const vzvz = dotProduct(vZ, vZ);
  const vvz = dotProduct(v, vZ);
  
  const pv = dotProduct(p, v);
  const pvz = dotProduct(p, vZ);

  // No dividimos por el determinante D porque se cancela en la razón doble
  const alpha = pv * vzvz - pvz * vvz;
  const beta = pvz * vv - pv * vvz;

  return { alpha, beta };
}

/**
 * Evalúa la Razón Doble (Cross-Ratio) de 4 puntos colineales representados
 * como vectores homogéneos en la recta definida por v y vZ.
 * CR(b, t, v, vZ)
 */
function calcularCrossRatio1D(b, t, v, vZ) {
  const pb = getLineParams(b, v, vZ);
  const pt = getLineParams(t, v, vZ);

  const num = Math.abs(pb.alpha * pt.beta - pt.alpha * pb.beta);
  const den = Math.abs(pb.beta * pt.alpha);

  if (den < 1e-10) return 0;
  return num / den;
}

/**
 * Calcula la altura real de un objeto mediante la Razón Doble.
 * Utiliza una medida de referencia virtual extraída de la escala métrica de H_alzado.
 * 
 * @param {{x,y}} b - Base del objeto en el piso (px)
 * @param {{x,y}} t - Techo del objeto (px)
 * @param {Object} vZ - Punto de fuga vertical (homogéneo)
 * @param {Object} l - Línea del horizonte (homogéneo)
 * @param {cv.Mat} H_alzado_inv - Homografía (mm -> px) de la pared
 * @param {number} anguloPlanesDeg - Ángulo de corrección trigonométrica
 * @returns {number} Altura en milímetros
 */
/**
 * Calcula la altura real de un objeto mediante la Razón Doble y calibración focal.
 * 
 * @param {{x,y}} b - Base del objeto en el piso (px)
 * @param {{x,y}} t - Techo del objeto (px)
 * @param {Object} vZ - Punto de fuga vertical (homogéneo)
 * @param {Object} l - Línea del horizonte (homogéneo)
 * @param {cv.Mat} H_base_inv - Homografía (mm -> px) del piso
 * @param {number} width - Ancho del lienzo original
 * @param {number} height - Alto del lienzo original
 * @returns {number} Altura en milímetros
 */
export function calcularAlturaLibre(b, t, vZ, l, H_base_inv, width, height) {
  // Convertir puntos a homogéneos
  const b_hom = { x: b.x, y: b.y, w: 1 };
  const t_hom = { x: t.x, y: t.y, w: 1 };

  // v es la intersección de la línea (b -> t) con el horizonte l
  const line_bt = crossProduct(b_hom, t_hom);
  const v = crossProduct(line_bt, l);

  // Cross Ratio del objeto
  const cr_obj = calcularCrossRatio1D(b_hom, t_hom, v, vZ);

  // --- 1. Calibrar Cámara (Focal Length f) ---
  const Pcx = width / 2;
  const Pcy = height / 2;
  
  // Normalizar horizonte l
  const norm_l = Math.hypot(l.x, l.y) || 1;
  const A = l.x / norm_l;
  const B = l.y / norm_l;
  const C = l.w / norm_l;

  const vZx = vZ.x / vZ.w;
  const vZy = vZ.y / vZ.w;

  // f^2 = - (vZ - Pc) · (v_perp - Pc)
  // v_perp es el punto en l más cercano a Pc
  const d_Pc_l = A * Pcx + B * Pcy + C; 
  const f2 = d_Pc_l * (A * (vZx - Pcx) + B * (vZy - Pcy));
  
  const f = Math.sqrt(Math.abs(f2));

  // --- 2. Descomponer H_base para obtener Zc (Altura de cámara) ---
  const hb = H_base_inv.data64F;
  const h1 = { x: hb[0], y: hb[3], z: hb[6] };
  const h2 = { x: hb[1], y: hb[4], z: hb[7] };
  const h3 = { x: hb[2], y: hb[5], z: hb[8] };

  // Función para aplicar K_inv a un vector h
  const applyKinv = (h) => ({
    x: (h.x - Pcx * h.z) / f,
    y: (h.y - Pcy * h.z) / f,
    z: h.z
  });

  const r1_raw = applyKinv(h1);
  const r2_raw = applyKinv(h2);
  const t_raw  = applyKinv(h3);

  const normR1 = Math.hypot(r1_raw.x, r1_raw.y, r1_raw.z);
  const normR2 = Math.hypot(r2_raw.x, r2_raw.y, r2_raw.z);
  const lambda = 1.0 / Math.sqrt(normR1 * normR2);

  const r1 = { x: r1_raw.x * lambda, y: r1_raw.y * lambda, z: r1_raw.z * lambda };
  const r2 = { x: r2_raw.x * lambda, y: r2_raw.y * lambda, z: r2_raw.z * lambda };
  const t_vec = { x: t_raw.x * lambda, y: t_raw.y * lambda, z: t_raw.z * lambda };

  // r3 = r1 x r2 (Vector Normal al piso)
  const r3 = {
    x: r1.y * r2.z - r1.z * r2.y,
    y: r1.z * r2.x - r1.x * r2.z,
    z: r1.x * r2.y - r1.y * r2.x
  };

  // Zc es la distancia del plano al centro óptico
  const Zc = Math.abs(r3.x * t_vec.x + r3.y * t_vec.y + r3.z * t_vec.z);

  // --- 3. Altura final ---
  // El Cross Ratio de Criminisi nos da (Z / Zc)
  let alturaMm = cr_obj * Zc;

  return alturaMm;
}

export function getCameraParams(H_base_inv, H_alzado_inv, width, height) {
  const hb = H_base_inv.data64F;
  const hw = H_alzado_inv.data64F;

  // Los puntos de fuga de los ejes X y Y del piso son las primeras dos columnas de H_base
  const vX = { x: hb[0], y: hb[3], w: hb[6] };
  const vY = { x: hb[1], y: hb[4], w: hb[7] };
  const l = crossProduct(vX, vY);
  
  // El punto de fuga vertical de la pared es típicamente la columna Y de H_alzado (columna 1)
  const vZ = { x: hw[1], y: hw[4], w: hw[7] };

  const Pcx = width / 2;
  const Pcy = height / 2;
  
  const norm_l = Math.hypot(l.x, l.y) || 1;
  const A = l.x / norm_l;
  const B = l.y / norm_l;
  const C = l.w / norm_l;

  const vZx = vZ.x / vZ.w;
  const vZy = vZ.y / vZ.w;

  const d_Pc_l = A * Pcx + B * Pcy + C; 
  const f2 = d_Pc_l * (A * (vZx - Pcx) + B * (vZy - Pcy));
  const f = Math.sqrt(Math.abs(f2));

  // Las columnas de H_base nos dan r1 y r2, la columna de H_alzado (vZ) nos da r3
  const h1 = { x: hb[0], y: hb[3], z: hb[6] };
  const h2 = { x: hb[1], y: hb[4], z: hb[7] };
  const h3 = { x: hb[2], y: hb[5], z: hb[8] };

  const applyKinv = (h) => ({
    x: (h.x - Pcx * h.z) / f,
    y: (h.y - Pcy * h.z) / f,
    z: h.z
  });

  const r1_raw = applyKinv(h1);
  const r2_raw = applyKinv(h2);
  const t_raw  = applyKinv(h3);
  
  // Vector que apunta hacia el punto de fuga vertical en el espacio de la cámara
  const vZ_raw = applyKinv({ x: hw[1], y: hw[4], z: hw[7] });

  const normR1 = Math.hypot(r1_raw.x, r1_raw.y, r1_raw.z);
  const normR2 = Math.hypot(r2_raw.x, r2_raw.y, r2_raw.z);
  const lambda = 1.0 / Math.sqrt(normR1 * normR2);

  const r1 = { x: r1_raw.x * lambda, y: r1_raw.y * lambda, z: r1_raw.z * lambda };
  const r2 = { x: r2_raw.x * lambda, y: r2_raw.y * lambda, z: r2_raw.z * lambda };
  
  // r3 = r1 × r2: garantiza que [r1|r2|r3] sea una matriz de rotación ortonormal.
  // El punto de fuga de la pared (vZ) se usa SOLO para estimar f, no para r3.
  // Esto es fundamental: R^T = R^{-1} solo si R es ortonormal, y la conversión
  // cámara→objeto en intersectRayWithVirtualPlane depende de esa propiedad.
  const r3 = {
    x: r1.y * r2.z - r1.z * r2.y,
    y: r1.z * r2.x - r1.x * r2.z,
    z: r1.x * r2.y - r1.y * r2.x
  };
  
  const t_vec = { x: t_raw.x * lambda, y: t_raw.y * lambda, z: t_raw.z * lambda };

  // Calcular la escala de la pared (alzado) en el espacio de cámara para calibrar el eje vertical vn_default
  const rw1_raw = applyKinv({ x: hw[0], y: hw[3], z: hw[6] });
  const rw2_raw = applyKinv({ x: hw[1], y: hw[4], z: hw[7] });
  const normRw1 = Math.hypot(rw1_raw.x, rw1_raw.y, rw1_raw.z);
  const normRw2 = Math.hypot(rw2_raw.x, rw2_raw.y, rw2_raw.z);
  const lambda_w = 1.0 / Math.sqrt(normRw1 * normRw2);
  const vn_default = (normRw2 * lambda_w) || Math.hypot(r3.x, r3.y, r3.z) || 1.0;

  return { f, Pcx, Pcy, r1, r2, r3, t_vec, vn_default };
}


/**
 * Proyecta un punto 3D (X, Y, Z) mm al lienzo 2D (píxeles).
 */
export function project3D(params, X_mm, Y_mm, Z_mm) {
  const { f, Pcx, Pcy, r1, r2, r3, t_vec } = params;
  
  // Coordenadas en la cámara (Notar que -Z_mm es porque r3 apunta hacia abajo (hacia adentro del piso) y Z es altura positiva hacia arriba)
  const Xc = r1.x * X_mm + r2.x * Y_mm + r3.x * (-Z_mm) + t_vec.x;
  const Yc = r1.y * X_mm + r2.y * Y_mm + r3.y * (-Z_mm) + t_vec.y;
  const Zc_pt = r1.z * X_mm + r2.z * Y_mm + r3.z * (-Z_mm) + t_vec.z;

  if (Zc_pt < 1e-5) return null; // Detrás de cámara

  const imgX = (f * Xc / Zc_pt) + Pcx;
  const imgY = (f * Yc / Zc_pt) + Pcy;

  return { x: imgX, y: imgY };
}

/**
 * Proyecta un punto en las coordenadas paramétricas (U, Z) del plano virtual
 * a la imagen en píxeles, considerando las correcciones de fuga horizontal y vertical.
 *
 * @param {Object} params - Parámetros de cámara.
 * @param {{x,y}} P1_mm - Punto base 1 del plano virtual en el piso (mm).
 * @param {{x,y}} P2_mm - Punto base 2 del plano virtual en el piso (mm).
 * @param {number} U_mm - Distancia horizontal desde P1_mm a lo largo del plano (mm).
 * @param {number} Z_mm - Altura vertical sobre el piso (mm).
 * @param {{x,y}} vanishingPointH - Punto de fuga horizontal (opcional, en px).
 * @param {{x,y}} vanishingPointV - Punto de fuga vertical (opcional, en px).
 * @returns {{x,y}} Punto proyectado en imagen.
 */
export function projectVirtualPlane(params, P1_mm, P2_mm, U_mm, Z_mm, vanishingPointH = null, vanishingPointV = null) {
  const { f, Pcx, Pcy, r1, r2, r3, t_vec } = params;
  const scale_v = params.s_v || params.vn_default || Math.hypot(r3.x, r3.y, r3.z) || 1.0;

  // 1. Origen en cámara
  const P1_cam = {
    x: r1.x * P1_mm.x + r2.x * P1_mm.y + t_vec.x,
    y: r1.y * P1_mm.x + r2.y * P1_mm.y + t_vec.y,
    z: r1.z * P1_mm.x + r2.z * P1_mm.y + t_vec.z
  };

  // 2. Dirección horizontal en cámara
  const dx = P2_mm.x - P1_mm.x;
  const dy = P2_mm.y - P1_mm.y;
  const lenU = Math.hypot(dx, dy) || 1;
  const u_dir_obj = { x: dx/lenU, y: dy/lenU, z: 0 };
  const u_dir_cam = {
    x: r1.x * u_dir_obj.x + r2.x * u_dir_obj.y,
    y: r1.y * u_dir_obj.x + r2.y * u_dir_obj.y,
    z: r1.z * u_dir_obj.x + r2.z * u_dir_obj.y
  };
  const un = Math.hypot(u_dir_cam.x, u_dir_cam.y, u_dir_cam.z) || 1;

  let u_cam_unit;
  if (vanishingPointH) {
    const raw = {
      x: (vanishingPointH.x - Pcx) / f,
      y: (vanishingPointH.y - Pcy) / f,
      z: 1
    };
    const norm = Math.hypot(raw.x, raw.y, raw.z) || 1;
    u_cam_unit = { x: raw.x / norm, y: raw.y / norm, z: raw.z / norm };
  } else {
    u_cam_unit = { x: u_dir_cam.x / un, y: u_dir_cam.y / un, z: u_dir_cam.z / un };
  }

  // Calibración dinámica del factor de escala horizontal (s_h)
  // Asegura que la proyección del segmento base [P1, P2] mida exactamente lenU mm en el plano virtual
  const P2_cam = {
    x: r1.x * P2_mm.x + r2.x * P2_mm.y + t_vec.x,
    y: r1.y * P2_mm.x + r2.y * P2_mm.y + t_vec.y,
    z: r1.z * P2_mm.x + r2.z * P2_mm.y + t_vec.z
  };
  const V2 = {
    x: P2_cam.x - P1_cam.x,
    y: P2_cam.y - P1_cam.y,
    z: P2_cam.z - P1_cam.z
  };
  const U_scaled = V2.x * u_cam_unit.x + V2.y * u_cam_unit.y + V2.z * u_cam_unit.z;
  const s_h = U_scaled / lenU;

  // 3. Dirección vertical en cámara
  let v_cam_unit;
  if (vanishingPointV) {
    const raw = {
      x: (vanishingPointV.x - Pcx) / f,
      y: (vanishingPointV.y - Pcy) / f,
      z: 1
    };
    const norm = Math.hypot(raw.x, raw.y, raw.z) || 1;
    v_cam_unit = { x: raw.x / norm, y: raw.y / norm, z: raw.z / norm };
  } else {
    const norm3 = Math.hypot(r3.x, r3.y, r3.z) || 1.0;
    v_cam_unit = { x: -r3.x / norm3, y: -r3.y / norm3, z: -r3.z / norm3 };
  }

  // 4. Punto en coordenadas de cámara (escalando con s_h y scale_v para consistencia métrica)
  const P_cam = {
    x: P1_cam.x + (U_mm * s_h) * u_cam_unit.x + (Z_mm * scale_v) * v_cam_unit.x,
    y: P1_cam.y + (U_mm * s_h) * u_cam_unit.y + (Z_mm * scale_v) * v_cam_unit.y,
    z: P1_cam.z + (U_mm * s_h) * u_cam_unit.z + (Z_mm * scale_v) * v_cam_unit.z
  };

  if (P_cam.z < 1e-5) return null;

  // 5. Proyectar a imagen
  const imgX = (f * P_cam.x / P_cam.z) + Pcx;
  const imgY = (f * P_cam.y / P_cam.z) + Pcy;

  return { x: imgX, y: imgY };
}

/**
 * Lanza un rayo desde la cámara por el píxel (u, v) y lo intersecta
 * con el plano virtual.
 *
 * El plano virtual está definido por:
 *   - P1_mm, P2_mm: dos puntos en el piso (el borde base del plano)
 *   - vanishingPointH: punto de fuga horizontal en imagen (si existe)
 *   - vanishingPointV: punto de fuga vertical en imagen (si existe)
 *
 * Retorna {x, y, z} en mm (coordenadas del sistema objeto del piso).
 */
export function intersectRayWithVirtualPlane(params, u, v, P1_mm, P2_mm, vanishingPointH = null, vanishingPointV = null) {
  const { f, Pcx, Pcy, r1, r2, r3, t_vec } = params;
  const scale_v = params.s_v || params.vn_default || Math.hypot(r3.x, r3.y, r3.z) || 1.0;

  // ── 1. Origen del plano en coordenadas de cámara ─────────────────────────
  const P1_cam = {
    x: r1.x * P1_mm.x + r2.x * P1_mm.y + t_vec.x,
    y: r1.y * P1_mm.x + r2.y * P1_mm.y + t_vec.y,
    z: r1.z * P1_mm.x + r2.z * P1_mm.y + t_vec.z
  };

  // ── 2. Direcciones de referencia y sus normas locales ────────────────────
  const dx = P2_mm.x - P1_mm.x;
  const dy = P2_mm.y - P1_mm.y;
  const lenU = Math.hypot(dx, dy) || 1;
  const u_dir_obj = { x: dx/lenU, y: dy/lenU, z: 0 };
  const u_dir_cam = {
    x: r1.x * u_dir_obj.x + r2.x * u_dir_obj.y,
    y: r1.y * u_dir_obj.x + r2.y * u_dir_obj.y,
    z: r1.z * u_dir_obj.x + r2.z * u_dir_obj.y
  };

  // ── 3. Ejes unitarios del plano virtual ──────────────────────────────────
  let u_cam_unit;
  if (vanishingPointH) {
    const raw = {
      x: (vanishingPointH.x - Pcx) / f,
      y: (vanishingPointH.y - Pcy) / f,
      z: 1
    };
    const norm = Math.hypot(raw.x, raw.y, raw.z) || 1;
    u_cam_unit = { x: raw.x / norm, y: raw.y / norm, z: raw.z / norm };
  } else {
    const un = Math.hypot(u_dir_cam.x, u_dir_cam.y, u_dir_cam.z) || 1;
    u_cam_unit = { x: u_dir_cam.x / un, y: u_dir_cam.y / un, z: u_dir_cam.z / un };
  }

  // Calibración dinámica del factor de escala horizontal (s_h)
  const P2_cam = {
    x: r1.x * P2_mm.x + r2.x * P2_mm.y + t_vec.x,
    y: r1.y * P2_mm.x + r2.y * P2_mm.y + t_vec.y,
    z: r1.z * P2_mm.x + r2.z * P2_mm.y + t_vec.z
  };
  const V2 = {
    x: P2_cam.x - P1_cam.x,
    y: P2_cam.y - P1_cam.y,
    z: P2_cam.z - P1_cam.z
  };
  const U_scaled = V2.x * u_cam_unit.x + V2.y * u_cam_unit.y + V2.z * u_cam_unit.z;
  const s_h = U_scaled / lenU;

  let v_cam_unit;
  if (vanishingPointV) {
    const raw = {
      x: (vanishingPointV.x - Pcx) / f,
      y: (vanishingPointV.y - Pcy) / f,
      z: 1
    };
    const norm = Math.hypot(raw.x, raw.y, raw.z) || 1;
    v_cam_unit = { x: raw.x / norm, y: raw.y / norm, z: raw.z / norm };
  } else {
    const norm3 = Math.hypot(r3.x, r3.y, r3.z) || 1.0;
    v_cam_unit = { x: -r3.x / norm3, y: -r3.y / norm3, z: -r3.z / norm3 };
  }

  // ── 4. Normal del plano virtual = u_cam_unit × v_cam_unit ────────────────
  const n_cam = {
    x: u_cam_unit.y * v_cam_unit.z - u_cam_unit.z * v_cam_unit.y,
    y: u_cam_unit.z * v_cam_unit.x - u_cam_unit.x * v_cam_unit.z,
    z: u_cam_unit.x * v_cam_unit.y - u_cam_unit.y * v_cam_unit.x
  };
  const nn = Math.hypot(n_cam.x, n_cam.y, n_cam.z) || 1;
  const n = { x: n_cam.x/nn, y: n_cam.y/nn, z: n_cam.z/nn };

  // ── 5. Rayo por el píxel (u, v) ──────────────────────────────────────────
  const ray = {
    x: (u - Pcx) / f,
    y: (v - Pcy) / f,
    z: 1
  };

  // ── 6. Intersección rayo-plano ───────────────────────────────────────────
  const num = P1_cam.x * n.x + P1_cam.y * n.y + P1_cam.z * n.z;
  const den = ray.x * n.x + ray.y * n.y + ray.z * n.z;

  if (Math.abs(den) < 1e-9) return null;
  const t = num / den;
  if (t < 0) return null;

  const P_cam = { x: ray.x * t, y: ray.y * t, z: ray.z * t };

  // ── 7. Resolver coordenadas U y Z en base a los ejes unitarios ───────────
  const V = {
    x: P_cam.x - P1_cam.x,
    y: P_cam.y - P1_cam.y,
    z: P_cam.z - P1_cam.z
  };

  const V_dot_u = V.x * u_cam_unit.x + V.y * u_cam_unit.y + V.z * u_cam_unit.z;
  const V_dot_v = V.x * v_cam_unit.x + V.y * v_cam_unit.y + V.z * v_cam_unit.z;
  const cos_theta = u_cam_unit.x * v_cam_unit.x + u_cam_unit.y * v_cam_unit.y + u_cam_unit.z * v_cam_unit.z;

  const D = 1 - cos_theta * cos_theta;
  let U_unit = 0;
  let Z_unit = 0;
  if (D > 1e-6) {
    U_unit = (V_dot_u - V_dot_v * cos_theta) / D;
    Z_unit = (V_dot_v - V_dot_u * cos_theta) / D;
  } else {
    U_unit = V_dot_u;
    Z_unit = V_dot_v;
  }

  // Convertir las distancias tridimensionales a la escala métrica de los planos base y alzado
  const U_mm = U_unit / s_h;
  const Z_mm = Z_unit / scale_v;

  // Retornar en el sistema objeto del piso: P1_mm + U_mm * u_dir_obj
  const P_obj = {
    x: P1_mm.x + U_mm * u_dir_obj.x,
    y: P1_mm.y + U_mm * u_dir_obj.y,
    z: Z_mm
  };

  return P_obj;
}

/**
 * Calibra el factor de escala vertical (s_v) calculando la intersección del plano pared con el plano virtual.
 */
export function calibrateVirtualPlaneScale(params, H_base, H_base_inv, H_alzado, H_alzado_inv, b1_mm, b2_mm, vanishingPointH = null, vanishingPointV = null) {
  const { f, Pcx, Pcy, r1, r2, r3, t_vec, vn_default } = params;

  // 1. Obtener la normal del plano pared en cámara
  const hw = H_alzado_inv.data64F;
  const applyKinv = (h) => ({
    x: (h.x - Pcx * h.z) / f,
    y: (h.y - Pcy * h.z) / f,
    z: h.z
  });

  const rw1_raw = applyKinv({ x: hw[0], y: hw[3], z: hw[6] });
  const rw2_raw = applyKinv({ x: hw[1], y: hw[4], z: hw[7] });
  const t_w_raw = applyKinv({ x: hw[2], y: hw[5], z: hw[8] });

  const normRw1 = Math.hypot(rw1_raw.x, rw1_raw.y, rw1_raw.z) || 1.0;
  const normRw2 = Math.hypot(rw2_raw.x, rw2_raw.y, rw2_raw.z) || 1.0;
  const lambda_w = 1.0 / Math.sqrt(normRw1 * normRw2);

  const rw1 = { x: rw1_raw.x * lambda_w, y: rw1_raw.y * lambda_w, z: rw1_raw.z * lambda_w };
  const rw2 = { x: rw2_raw.x * lambda_w, y: rw2_raw.y * lambda_w, z: rw2_raw.z * lambda_w };
  const t_w = { x: t_w_raw.x * lambda_w, y: t_w_raw.y * lambda_w, z: t_w_raw.z * lambda_w };

  // normal de la pared
  const n_w_raw = {
    x: rw1.y * rw2.z - rw1.z * rw2.y,
    y: rw1.z * rw2.x - rw1.x * rw2.z,
    z: rw1.x * rw2.y - rw1.y * rw2.x
  };
  const normNw = Math.hypot(n_w_raw.x, n_w_raw.y, n_w_raw.z) || 1.0;
  const n_w = { x: n_w_raw.x / normNw, y: n_w_raw.y / normNw, z: n_w_raw.z / normNw };
  const d_w = -(n_w.x * t_w.x + n_w.y * t_w.y + n_w.z * t_w.z);

  // 2. Recta del plano virtual en el piso (2D en mm)
  // A_virt * X + B_virt * Y + C_virt = 0
  const A_virt = b2_mm.y - b1_mm.y;
  const B_virt = -(b2_mm.x - b1_mm.x);
  const C_virt = b2_mm.x * b1_mm.y - b1_mm.x * b2_mm.y;

  // 3. Recta de la pared en el piso (2D en mm)
  // Reemplazando la proyección 3D del piso en la ecuación de la pared:
  // A_wall * X + B_wall * Y + C_wall = 0
  const A_wall = n_w.x * r1.x + n_w.y * r1.y + n_w.z * r1.z;
  const B_wall = n_w.x * r2.x + n_w.y * r2.y + n_w.z * r2.z;
  const C_wall = n_w.x * t_vec.x + n_w.y * t_vec.y + n_w.z * t_vec.z + d_w;

  // 4. Intersección de ambas rectas en el piso (mm)
  const delta = A_wall * B_virt - B_wall * A_virt;
  if (Math.abs(delta) < 1e-6) {
    return { s_v: vn_default, P_base_intersect: null, P_top_intersect: null };
  }

  const X_int = (B_wall * C_virt - C_wall * B_virt) / delta;
  const Y_int = (C_wall * A_virt - A_wall * C_virt) / delta;

  // 5. Proyectar este punto base de intersección a 3D cámara
  const P_base_cam = {
    x: r1.x * X_int + r2.x * Y_int + t_vec.x,
    y: r1.y * X_int + r2.y * Y_int + t_vec.y,
    z: r1.z * X_int + r2.z * Y_int + t_vec.z
  };

  // 6. Proyectar a píxeles de imagen
  if (P_base_cam.z < 1e-5) {
    return { s_v: vn_default, P_base_intersect: null, P_top_intersect: null };
  }
  const u_base = (f * P_base_cam.x / P_base_cam.z) + Pcx;
  const v_base = (f * P_base_cam.y / P_base_cam.z) + Pcy;

  // 7. Mapear este pixel a coordenadas del plano pared (mm) usando H_alzado (pixel -> mm)
  const h_alzado_data = H_alzado.data64F;
  const w_base_wall = h_alzado_data[6] * u_base + h_alzado_data[7] * v_base + h_alzado_data[8];
  if (Math.abs(w_base_wall) < 1e-7) {
    return { s_v: vn_default, P_base_intersect: null, P_top_intersect: null };
  }
  const P_base_wall_mm = {
    x: (h_alzado_data[0] * u_base + h_alzado_data[1] * v_base + h_alzado_data[2]) / w_base_wall,
    y: (h_alzado_data[3] * u_base + h_alzado_data[4] * v_base + h_alzado_data[5]) / w_base_wall
  };

  // 8. Crear un punto a altura de referencia (ej: 500 mm) en la pared
  const H_ref = 500.0;
  const X_wall_top = P_base_wall_mm.x;
  const Y_wall_top = P_base_wall_mm.y - H_ref; // Y sube hacia arriba (restando en coordenadas de plantilla)

  // Mapear este punto top a píxeles usando H_alzado_inv (mm -> pixel)
  const w_top_wall = hw[6] * X_wall_top + hw[7] * Y_wall_top + hw[8];
  if (Math.abs(w_top_wall) < 1e-7) {
    return { s_v: vn_default, P_base_intersect: null, P_top_intersect: null };
  }
  const u_top = (hw[0] * X_wall_top + hw[1] * Y_wall_top + hw[2]) / w_top_wall;
  const v_top = (hw[3] * X_wall_top + hw[4] * Y_wall_top + hw[5]) / w_top_wall;

  // Proyectar este pixel top a 3D cámara
  const P_top_cam = {
    x: lambda_w * w_top_wall * (u_top - Pcx) / f,
    y: lambda_w * w_top_wall * (v_top - Pcy) / f,
    z: lambda_w * w_top_wall
  };

  // 9. Calcular la dirección vertical unitaria en cámara
  let v_cam_unit;
  if (vanishingPointV) {
    const raw = {
      x: (vanishingPointV.x - Pcx) / f,
      y: (vanishingPointV.y - Pcy) / f,
      z: 1
    };
    const norm = Math.hypot(raw.x, raw.y, raw.z) || 1;
    v_cam_unit = { x: raw.x / norm, y: raw.y / norm, z: raw.z / norm };
  } else {
    const norm3 = Math.hypot(r3.x, r3.y, r3.z) || 1.0;
    v_cam_unit = { x: -r3.x / norm3, y: -r3.y / norm3, z: -r3.z / norm3 };
  }

  // 10. Despejar s_v
  const v_diff = {
    x: P_top_cam.x - P_base_cam.x,
    y: P_top_cam.y - P_base_cam.y,
    z: P_top_cam.z - P_base_cam.z
  };
  const dot = v_diff.x * v_cam_unit.x + v_diff.y * v_cam_unit.y + v_diff.z * v_cam_unit.z;
  let s_v = Math.abs(dot) / H_ref;

  if (isNaN(s_v) || s_v < 1e-4) {
    s_v = vn_default;
  }

  return {
    s_v,
    P_base_intersect: { x: u_base, y: v_base },
    P_top_intersect: { x: u_top, y: v_top }
  };
}
