/* Urban Radiance Field -- MRT viewer.
 *
 * Three static assets, no backend:
 *   scene.glb        decimated reconstruction, pushed toward grey (backdrop)
 *   mrt_surface.glb  the 1.5 m pedestrian surface, geometry only
 *   attribution.bin  one row per surface VERTEX, in the same order
 *
 * That last identity is the whole trick. `export_gridded_surface_ply` writes the surface's
 * vertices in `valid_mask` order, which is exactly the order the standpoints were computed
 * in, so vertex i of the mesh is row i of the blob. A click becomes a raycast, the raycast
 * gives a triangle, and the triangle gives the row -- no spatial index, no server query.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { COLORMAPS } from './colormaps.js';

const DATA = './data/';
const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- loading */

async function fetchProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  if (!res.body) return res.arrayBuffer();
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onBytes(got, total);
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out.buffer;
}

const TYPES = { float32: Float32Array, uint16: Uint16Array, uint8: Uint8Array };

/** Column-major blocks out of attribution.bin. Each block is copied via slice() rather
 *  than viewed in place: a typed-array view needs its byte offset to be a multiple of the
 *  element size, and that should be a property of this loader, not a standing constraint
 *  on how the packer happens to order its blocks. */
function unpack(buffer, manifest) {
  const out = {};
  for (const b of manifest.blocks) {
    const T = TYPES[b.dtype];
    if (!T) throw new Error(`unknown dtype ${b.dtype} in block ${b.name}`);
    out[b.name] = {
      data: new T(buffer.slice(b.offset, b.offset + b.length)),
      cols: b.cols,
      scale: b.scale,
      /** Value at row i, column c, already de-quantised. */
      at(i, c = 0) { return this.data[i * this.cols + c] * (this.scale ?? 1); },
    };
  }
  return out;
}

/* ------------------------------------------------------------- colormaps */

function sampleCmap(name, t) {
  const stops = COLORMAPS[name];
  const n = stops.length - 1;
  const x = Math.max(0, Math.min(1, t)) * n;
  const i = Math.min(n - 1, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/* The LUT holds sRGB display values, the same bytes matplotlib would write. three.js works
 * in linear space and converts back on output, so feeding sRGB straight in would apply the
 * transfer function twice and the colours would no longer mean what they mean in the
 * paper's figures. */
const srgbToLinear = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));

function cmapCss(name, stops = 12) {
  const parts = [];
  for (let i = 0; i <= stops; i++) {
    const [r, g, b] = sampleCmap(name, i / stops);
    parts.push(`rgb(${r | 0},${g | 0},${b | 0}) ${(i / stops * 100).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(',')})`;
}

/* ------------------------------------------------------------ formatting */

const f0 = (v) => v.toFixed(0);
const f1 = (v) => v.toFixed(1);
const pct = (v) => (v <= 0 ? '0%' : v < 0.5 ? '<1%' : `${v.toFixed(0)}%`);

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** The manifest stores the flight time already local to the site, with its offset
 *  ("2019-03-19 07:30:00+01:00"). Putting that through `new Date().toLocaleString()` would
 *  restate it in whatever timezone the page happens to be opened in -- a 07:30 flight shown
 *  as 10:30 to a reader three hours east, which silently contradicts the sun position the
 *  whole scene was computed from. So the literal local time is what gets rendered. */
function siteTime(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})/.exec(iso);
  return m ? `${+m[3]} ${MONTHS[+m[2] - 1]} ${m[1]}, ${m[4]}:${m[5]} local time` : iso;
}
const signed = (v, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);

/* ------------------------------------------------------------------ main */

const state = {
  M: null, B: null, N: 0, band: 'comb', field: null, picked: -1, fields: [],
};

init().catch((err) => {
  console.error(err);
  $('loading-text').textContent = `Could not load: ${err.message}`;
  $('loading-text').style.color = '#f5876c';
  $('loading').querySelector('.spinner').style.display = 'none';
});

async function init() {
  const setText = (t) => { $('loading-text').textContent = t; };
  const setFill = (f) => { $('loading-fill').style.width = `${Math.round(f * 100)}%`; };

  setText('Loading radiation data…');
  const [manifest, sceneMeta] = await Promise.all([
    fetch(`${DATA}manifest.json`).then((r) => r.json()),
    fetch(`${DATA}scene.json`).then((r) => r.json()),
  ]);
  state.M = manifest;
  state.N = manifest.n_points;

  const attribution = await fetchProgress(`${DATA}attribution.bin`,
    (got, total) => setFill(total ? (got / total) * 0.55 : 0.3));
  state.B = unpack(attribution, manifest);

  setText('Loading geometry…');
  const loader = new GLTFLoader();
  const loadGlb = (file) => new Promise((res, rej) => loader.load(`${DATA}${file}`, res, undefined, rej));
  const [mrtGltf, sceneGltf] = await Promise.all([loadGlb('mrt_surface.glb'), loadGlb('scene.glb')]);
  setFill(0.95);

  buildFields();
  buildViewer(mrtGltf, sceneGltf, sceneMeta);
  buildChrome();

  setFill(1);
  $('loading').classList.add('done');
  setTimeout(() => { $('loading').style.display = 'none'; }, 450);
}

/* ------------------------------------------------------- derived surfaces */

/** Range from percentiles, so one hot cell cannot flatten the whole colour scale.
 *  `exact` uses the true min/max instead -- what the MRT fields want, because those
 *  ranges are the ones the paper quotes. */
function rangeOf(values, exact) {
  if (exact) {
    let lo = Infinity, hi = -Infinity;
    for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
    return [lo, hi];
  }
  const s = Float64Array.from(values).sort();
  return [s[Math.floor(s.length * 0.01)], s[Math.floor(s.length * 0.99)]];
}

function buildFields() {
  const { B, N } = state;
  const derive = (fn) => { const a = new Float32Array(N); for (let i = 0; i < N; i++) a[i] = fn(i); return a; };
  const swTotal = derive((i) => B.e_sw_diffuse.at(i) + B.e_sw_direct.at(i) + B.e_sw_specular.at(i));

  state.fields = [
    { id: 'mrt_combined', label: 'MRT, combined', unit: '°C', cmap: 'inferno',
      values: B.mrt_combined.data, exact: true },
    { id: 'mrt_longwave', label: 'MRT, longwave only', unit: '°C', cmap: 'inferno',
      values: B.mrt_longwave.data, exact: true },
    { id: 'sw_gain', label: 'Shortwave contribution', unit: 'K', cmap: 'YlOrRd',
      values: derive((i) => B.mrt_combined.at(i) - B.mrt_longwave.at(i)) },
    { id: 'e_lw', label: 'Longwave irradiance', unit: 'W/m²', cmap: 'inferno',
      values: B.e_lw.data },
    { id: 'e_sw', label: 'Shortwave irradiance', unit: 'W/m²', cmap: 'YlOrRd',
      values: swTotal },
    { id: 'sunlit', label: 'Direct sun on the body', unit: '', cmap: 'YlOrRd',
      values: derive((i) => B.sunlit.at(i)), exact: true, discrete: true },
  ];
  for (const f of state.fields) f.range = rangeOf(f.values, f.exact);
  state.field = state.fields[0];

  // The project's vertical datum is the COLMAP georegistration's, so raw z is a large
  // negative number here. Reporting it as a height would read as underground; terrain is
  // only meaningful relative to the scene, so elevations are shown against its lowest
  // standpoint.
  let zMin = Infinity;
  for (const z of B.z.data) if (z < zMin) zMin = z;   // spreading 107k args would overflow
  state.groundMin = zMin;
}

/* ------------------------------------------------------------ 3D viewer */

const V = {};

function buildViewer(mrtGltf, sceneGltf, meta) {
  const canvas = $('view');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x0f1216);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.5, 6000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.maxPolarAngle = Math.PI * 0.495;   // never drop below the ground plane
  controls.screenSpacePanning = false;

  scene.add(new THREE.HemisphereLight(0xcadcf0, 0x2a2e35, 2.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(-0.6, 1, 0.45).multiplyScalar(500);
  scene.add(sun);

  // -- backdrop: grey, lit, and pushed behind everything else
  const backdropGeom = firstMeshGeometry(sceneGltf);
  backdropGeom.computeVertexNormals();
  const backdropMat = new THREE.MeshLambertMaterial({
    vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 1,
  });
  const backdrop = new THREE.Mesh(backdropGeom, backdropMat);
  backdrop.renderOrder = 0;
  scene.add(backdrop);

  // -- MRT surface: unlit on purpose. Shading a data surface would multiply the colormap
  // by a lighting term and the colours would stop meaning their values.
  const mrtGeom = firstMeshGeometry(mrtGltf);
  if (mrtGeom.attributes.position.count !== state.N) {
    throw new Error(`mrt_surface.glb has ${mrtGeom.attributes.position.count} vertices but ` +
                    `attribution.bin has ${state.N} rows -- rebuild the web assets`);
  }
  mrtGeom.setAttribute('color',
    new THREE.BufferAttribute(new Uint8Array(state.N * 3), 3, true));
  const mrtMat = new THREE.MeshBasicMaterial({
    vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 1,
  });
  const mrtMesh = new THREE.Mesh(mrtGeom, mrtMat);
  mrtMesh.renderOrder = 1;
  scene.add(mrtMesh);

  // -- picked-standpoint marker: a ring at 1.5 m with a pin down to the ground
  const marker = new THREE.Group();
  marker.visible = false;
  // Drawn last and without depth testing, so the marker is never swallowed by the surface
  // it is standing on. renderOrder has to go on each child: a Group does not pass its own
  // down, and an opaque mesh with depthTest off would otherwise be sorted anywhere at all.
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x6cc4f5, side: THREE.DoubleSide,
    depthTest: false, depthWrite: false, transparent: true,
  });
  const ring = new THREE.Mesh(new THREE.RingGeometry(2.1, 2.9, 40), ringMat);
  ring.rotation.x = -Math.PI / 2;
  const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 1.5, 10), ringMat);
  pin.position.y = -0.75;
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 12), ringMat);
  marker.add(ring, pin, dot);
  for (const m of marker.children) m.renderOrder = 999;
  scene.add(marker);

  Object.assign(V, { renderer, scene, camera, controls, backdrop, backdropMat, mrtMesh, mrtMat, marker,
                     positions: mrtGeom.attributes.position.array, raycaster: new THREE.Raycaster(),
                     pointer: new THREE.Vector2(), meta });

  // Framing, fog and zoom limits all key off the MRT surface's own size rather than fixed
  // numbers, so rebuilding the assets at a different extent cannot leave the camera stranded.
  mrtGeom.computeBoundingBox();
  V.center = mrtGeom.boundingBox.getCenter(new THREE.Vector3());
  V.span = mrtGeom.boundingBox.getSize(new THREE.Vector3()).length();
  scene.fog = new THREE.Fog(0x0f1216, V.span * 0.9, V.span * 3.2);
  controls.minDistance = 3;
  controls.maxDistance = V.span * 2.5;
  resetView();

  applyField(state.field);
  bindPointer(canvas);

  const onResize = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(onResize).observe(canvas);
  onResize();

  renderer.setAnimationLoop(() => {
    controls.update();
    if (marker.visible) {
      // Keep the marker legible from any distance -- 1.5 m is sub-pixel across a 500 m scene.
      const d = camera.position.distanceTo(marker.position);
      const s = Math.max(1, d / 110);
      marker.scale.setScalar(s);
    }
    renderer.render(scene, camera);
  });
}

function firstMeshGeometry(gltf) {
  let geom = null;
  gltf.scene.traverse((o) => { if (!geom && o.isMesh) geom = o.geometry; });
  if (!geom) throw new Error('glTF contained no mesh');
  return geom;
}

function resetView() {
  const { camera, controls, center, span } = V;
  camera.position.set(center.x + span * 0.34, span * 0.44, center.z + span * 0.44);
  controls.target.copy(center);
  controls.update();
}

function applyField(field) {
  state.field = field;
  const [lo, hi] = field.range;
  const inv = hi > lo ? 1 / (hi - lo) : 0;
  const attr = V.mrtMesh.geometry.attributes.color;
  const col = attr.array;
  for (let i = 0; i < state.N; i++) {
    const [r, g, b] = sampleCmap(field.cmap, (field.values[i] - lo) * inv);
    col[i * 3] = srgbToLinear(r / 255) * 255;
    col[i * 3 + 1] = srgbToLinear(g / 255) * 255;
    col[i * 3 + 2] = srgbToLinear(b / 255) * 255;
  }
  attr.needsUpdate = true;

  $('legend-bar').style.background = cmapCss(field.cmap);
  const d = field.unit === '°C' || field.unit === 'K' ? 1 : 0;
  $('legend-lo').textContent = field.discrete ? 'no' : `${lo.toFixed(d)} ${field.unit}`.trim();
  $('legend-hi').textContent = field.discrete ? 'yes' : `${hi.toFixed(d)} ${field.unit}`.trim();
}

/* -------------------------------------------------------------- picking */

function pickAt(clientX, clientY) {
  const { renderer, camera, raycaster, pointer, mrtMesh, positions } = V;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(mrtMesh, false);
  if (!hits.length) return -1;
  // The triangle gives three candidate rows; the nearest of them is the standpoint the
  // user actually pointed at (grid spacing is 0.5 m, so this is never ambiguous).
  const { face, point } = hits[0];
  let best = -1, bestD = Infinity;
  for (const vi of [face.a, face.b, face.c]) {
    const dx = positions[vi * 3] - point.x;
    const dy = positions[vi * 3 + 1] - point.y;
    const dz = positions[vi * 3 + 2] - point.z;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestD) { bestD = d; best = vi; }
  }
  return best;
}

function bindPointer(canvas) {
  let moved = false, downAt = null, hoverJob = 0;

  canvas.addEventListener('pointerdown', (e) => { moved = false; downAt = { x: e.clientX, y: e.clientY }; });
  canvas.addEventListener('pointermove', (e) => {
    if (downAt && Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4) moved = true;
    // Raycasting 186,860 triangles is a few milliseconds -- fine per frame, wasteful per
    // pointer event, so coalesce into one job per animation frame.
    if (hoverJob) return;
    hoverJob = requestAnimationFrame(() => {
      hoverJob = 0;
      const i = pickAt(e.clientX, e.clientY);
      const el = $('hover');
      if (i < 0) { el.classList.remove('on'); canvas.style.cursor = 'grab'; return; }
      canvas.style.cursor = 'pointer';
      const f = state.field;
      const v = f.values[i];
      el.textContent = f.discrete
        ? `${f.label}: ${v ? 'yes' : 'no'}`
        : `${f.label} ${v.toFixed(f.unit === 'W/m²' ? 0 : 2)} ${f.unit}`.trim();
      el.style.left = `${e.clientX - canvas.getBoundingClientRect().left}px`;
      el.style.top = `${e.clientY - canvas.getBoundingClientRect().top}px`;
      el.classList.add('on');
    });
  });
  canvas.addEventListener('pointerleave', () => $('hover').classList.remove('on'));
  canvas.addEventListener('pointerup', (e) => {
    if (moved) return;                       // an orbit drag, not a click
    const i = pickAt(e.clientX, e.clientY);
    if (i >= 0) select(i);
  });
}

function select(i) {
  state.picked = i;
  const p = V.positions;
  V.marker.position.set(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
  V.marker.visible = true;
  $('hint').classList.add('gone');
  $('panel').classList.remove('empty');
  $('panel-body').hidden = false;
  renderPanel();
}

/* ---------------------------------------------------------------- panel */

/** The three levels for the selected standpoint, in the selected band.
 *  Level 1's stored values are PLANE irradiances; its contribution to the body is
 *  F_i times that, which is what the bars must show. Levels 2 and 3 are already
 *  body-weighted contributions, because each ray was folded in as w_r * value_r. */
function levelRows(i, band) {
  const { M, B } = state;
  const lwc = M.body.lw_coeff, swc = M.body.sw_coeff;
  const mix = (lw, sw) => (band === 'lw' ? lw : band === 'sw' ? sw : lwc * lw + swc * sw);

  const L1 = M.levels.l1, L2 = M.levels.l2, L3 = M.levels.l3;

  // Displayed north-first, the way the notebook's table reads; the blob's own order is
  // East, West, North, South, Up, Down.
  const dirOrder = ['North', 'East', 'South', 'West', 'Up', 'Down']
    .map((n) => L1.labels.indexOf(n)).filter((k) => k >= 0);

  const l1 = dirOrder.map((k) => {
    const fi = L1.factors[k];
    const planeLw = B.l1_lw.at(i, k), planeSw = B.l1_sw.at(i, k);
    return {
      name: L1.labels[k], color: L1.colors[k],
      value: mix(fi * planeLw, fi * planeSw),
      sub: band === 'lw' ? `F ${fi.toFixed(2)} · plane ${f1(planeLw)} W/m²`
        : band === 'sw' ? `F ${fi.toFixed(2)} · plane ${f1(planeSw)} W/m²`
        : `F ${fi.toFixed(2)} · L ${f1(planeLw)} · D ${f1(planeSw)} W/m²`,
    };
  });

  const cat = (blockLw, blockSw, level) => level.labels.map((name, k) => ({
    name: (level.pretty ?? level.labels)[k],
    color: level.colors[k],
    value: mix(blockLw.at(i, k), blockSw.at(i, k)),
  }));

  let l2 = cat(B.l2_lw, B.l2_sw, L2);
  let l3 = cat(B.l3_lw, B.l3_sw, L3).filter((r) => r.value > 0.005);
  // Materials are ranked by what they actually contribute here, with sky pinned last --
  // the same ordering the notebook's figure uses.
  const skyName = (L3.pretty ?? L3.labels)[L3.labels.length - 1];
  l3.sort((a, b) => (a.name === skyName) - (b.name === skyName) || b.value - a.value);

  // Two pathways reach the body without touching any surface or ray, so they belong to no
  // category at any level, and are listed on their own -- exactly as A6 does.
  const extras = [];
  if (band !== 'lw') {
    const direct = B.e_sw_direct.at(i), spec = B.e_sw_specular.at(i);
    if (direct > 0) extras.push({ name: 'Direct sun on body', value: mix(0, direct), extra: true });
    if (spec > 0) extras.push({ name: 'Specular glint', value: mix(0, spec), extra: true });
  }

  // The extras belong to no category at ANY level -- including Direction, because the six
  // plane irradiances carry diffuse and reflected shortwave only. So all three levels get
  // them, which is also what makes all three sum to the same total.
  return [
    { key: 'l1', title: L1.name, rows: l1, extras },
    { key: 'l2', title: L2.name, rows: l2.filter((r) => r.value > 0.005), extras },
    { key: 'l3', title: L3.name, rows: l3, extras },
  ];
}

function renderPanel() {
  const i = state.picked;
  if (i < 0) return;
  const { M, B, band } = state;
  const swc = M.body.sw_coeff;

  const mrt = B.mrt_combined.at(i), mrtLw = B.mrt_longwave.at(i);
  const eLw = B.e_lw.at(i), eSwD = B.e_sw_diffuse.at(i);
  const direct = B.e_sw_direct.at(i), spec = B.e_sw_specular.at(i);
  const swAll = eSwD + direct + spec;
  const flux = eLw + swc * swAll;

  $('mrt-value').textContent = (mrt < 0 ? '−' : '') + Math.abs(mrt).toFixed(2);

  const p = V.positions;
  const o = V.meta.origin_enu;
  // Back out of the viewer's glTF frame into the project's ENU metres.
  const east = p[i * 3] + o[0], north = -p[i * 3 + 2] + o[1], up = p[i * 3 + 1] + o[2];

  const zero = (v) => (v > 0 ? '' : ' class="zero"');
  $('readout-grid').innerHTML = `
    <dt>Longwave only</dt><dd>${signed(mrtLw)} °C</dd>
    <dt>Shortwave adds</dt><dd>${signed(mrt - mrtLw)} K</dd>
    <div class="sep"></div>
    <dt>Longwave E<sub>lw</sub></dt><dd>${f1(eLw)} W/m²</dd>
    <dt>Shortwave, diffuse + reflected</dt><dd>${f1(eSwD)} W/m²</dd>
    <dt>Direct sun on body</dt><dd${zero(direct)}>${f1(direct)} W/m²</dd>
    <dt>Specular glint</dt><dd${zero(spec)}>${f1(spec)} W/m²</dd>
    <dt>Absorbed flux</dt><dd>${f1(flux)} W/m²</dd>
    <div class="sep"></div>
    <dt>Position (E, N)</dt><dd>${east.toFixed(1)}, ${north.toFixed(1)} m</dd>
    <dt>Ground, above scene low</dt><dd>${(up - state.groundMin).toFixed(1)} m</dd>
    <dt>Sees the sun</dt><dd${zero(B.sunlit.at(i))}>${B.sunlit.at(i) ? 'yes' : 'no'}</dd>`;

  const total = band === 'lw' ? eLw : band === 'sw' ? swAll : flux;
  const unit = 'W/m²';
  const bandName = { comb: 'Combined', lw: 'Longwave', sw: 'Shortwave' }[band];

  const html = levelRows(i, band).map(({ title, rows, extras }) => {
    const all = rows.concat(extras);
    const max = Math.max(...all.map((r) => r.value), 1e-9);
    const bars = all.map((r) => `
      <div class="bar-row${r.extra ? ' extra' : ''}${r.value / total < 0.005 ? ' muted' : ''}">
        <span class="swatch"${r.color ? ` style="background:${r.color}"` : ''}></span>
        <div class="body">
          <div class="top">
            <span class="name">${r.name}</span>
            <span class="num"><b>${f1(r.value)}</b> ${unit} · ${pct(r.value / total * 100)}</span>
          </div>
          <div class="track"><div class="fill" style="width:${(r.value / max * 100).toFixed(1)}%;background:${r.color ?? 'var(--warn)'}"></div></div>
          ${r.sub ? `<div class="sub">${r.sub}</div>` : ''}
        </div>
      </div>`).join('');
    return `<section class="level"><h3>${title}</h3>
      <p class="total">${bandName} total ${f1(total)} ${unit} · shares of that total</p>
      ${bars}</section>`;
  }).join('');
  $('levels').innerHTML = html;

  $('closure').textContent =
    `Each level is an exact split of the same budget: every set of bars sums back to the ` +
    `${bandName.toLowerCase()} total above. Values are transported quantised to ` +
    `~0.005 W/m², so shares may round by a tenth.`;
}

/* --------------------------------------------------------------- chrome */

function buildChrome() {
  const { M } = state;

  const sel = $('field');
  sel.innerHTML = state.fields
    .map((f, k) => `<option value="${k}">${f.label}${f.unit ? ` (${f.unit})` : ''}</option>`).join('');
  sel.addEventListener('change', () => applyField(state.fields[+sel.value]));

  // A fully opaque mesh left in the transparent pass sorts per-object instead of per-pixel,
  // which on a draped double-sided surface shows up as flicker. Only opt into transparency
  // when it is actually being used.
  const fade = (mat, v) => {
    mat.opacity = v;
    mat.transparent = v < 1;
    mat.depthWrite = v > 0.98;
    mat.needsUpdate = true;
  };
  $('backdrop').addEventListener('input', (e) => {
    const v = +e.target.value / 100;
    fade(V.backdropMat, v);
    V.backdrop.visible = v > 0.01;
  });
  $('surfop').addEventListener('input', (e) => fade(V.mrtMat, +e.target.value / 100));
  $('reset').addEventListener('click', resetView);

  for (const b of $('bands').querySelectorAll('button')) {
    b.addEventListener('click', () => {
      state.band = b.dataset.band;
      for (const o of $('bands').querySelectorAll('button')) o.classList.toggle('on', o === b);
      renderPanel();
    });
  }

  const c = M.conditions;
  $('when').textContent = siteTime(c.when);
  $('conditions').textContent =
    `One instant: ${c.note}. Sun ${c.sun_elevation_deg.toFixed(1)}° above the horizon, ` +
    `DNI ${f0(c.dni)} / DHI ${f0(c.dhi)} W/m², sky downwelling longwave ` +
    `${f0(c.sky_downwelling_lw)} W/m². Body: ${M.body.model}, ${M.body.n_rays} rays.`;

  addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      state.picked = -1;
      V.marker.visible = false;
      $('panel').classList.add('empty');
      $('panel-body').hidden = true;
    }
  });
}

