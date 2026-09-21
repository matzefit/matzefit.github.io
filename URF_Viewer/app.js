/* Urban Radiance Field -- MRT viewer.
 *
 * Static assets, no backend:
 *   cloud.bin        the dense reconstruction, 1 row per point: position, RGB, material,
 *                    element, surface temperature, sky view factor
 *   mrt_surface.glb  the 1.5 m pedestrian surface, geometry only
 *   attribution.bin  one row per surface VERTEX, in the same order
 *   scene.glb        a decimated solid mesh, fetched only if the layer is switched on
 *
 * The attribution identity is the trick that makes a click cheap.
 * `export_gridded_surface_ply` writes the surface's vertices in `valid_mask` order, which
 * is the order the standpoints were computed in, so vertex i of the mesh is row i of the
 * blob. A click becomes a raycast, the raycast gives a triangle, the triangle gives the row
 * -- no spatial index, no server query.
 *
 * The point cloud never leaves the GPU in a decoded form: positions stay uint16 and are
 * expanded in the vertex shader, and colour, class filtering and the ENU -> viewer frame
 * change all happen there too. At 3M points that is the difference between a ~40 MB buffer
 * and a ~120 MB one, and it lets a filter toggle be a uniform write rather than a rebuild.
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

/** Column-major blocks out of a packed .bin. Each block is copied via slice() rather than
 *  viewed in place: a typed-array view needs its byte offset to be a multiple of the element
 *  size, and that should be a property of this loader, not a standing constraint on how the
 *  packer happens to order its blocks. */
function unpack(buffer, meta) {
  const out = {};
  for (const b of meta.blocks) {
    const T = TYPES[b.dtype];
    if (!T) throw new Error(`unknown dtype ${b.dtype} in block ${b.name}`);
    out[b.name] = {
      data: new T(buffer.slice(b.offset, b.offset + b.length)),
      cols: b.cols, scale: b.scale, bias: b.bias,
      at(i, c = 0) { return this.data[i * this.cols + c] * (this.scale ?? 1) + (this.bias ?? 0); },
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
const linearRgb = (hexOrArr) => {
  const [r, g, b] = Array.isArray(hexOrArr)
    ? hexOrArr
    : [1, 3, 5].map((i) => parseInt(hexOrArr.slice(i, i + 2), 16));
  return [srgbToLinear(r / 255), srgbToLinear(g / 255), srgbToLinear(b / 255)];
};

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
const signed = (v, d = 2) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const thousands = (v) => v.toLocaleString('en-US');

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

/* ------------------------------------------------------------------ main */

const RAMP_STOPS = 16;

const state = {
  M: null, B: null, N: 0, C: null, CB: null,
  band: 'comb', field: null, picked: -1, fields: [],
  matOn: null, elemOn: null, cloudMode: 0,
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
  const [manifest, sceneMeta, cloudMeta] = await Promise.all([
    fetch(`${DATA}manifest.json`).then((r) => r.json()),
    fetch(`${DATA}scene.json`).then((r) => r.json()),
    fetch(`${DATA}cloud.json`).then((r) => r.json()),
  ]);
  state.M = manifest;
  state.N = manifest.n_points;
  state.C = cloudMeta;

  const attribution = await fetchProgress(`${DATA}attribution.bin`,
    (g, t) => setFill(t ? (g / t) * 0.2 : 0.1));
  state.B = unpack(attribution, manifest);

  setText(`Loading ${thousands(cloudMeta.n_points)} points…`);
  const cloudBuf = await fetchProgress(`${DATA}cloud.bin`,
    (g, t) => setFill(0.2 + (t ? (g / t) * 0.6 : 0.3)));
  state.CB = unpack(cloudBuf, cloudMeta);

  setText('Loading the MRT surface…');
  const mrtGltf = await new Promise((res, rej) =>
    new GLTFLoader().load(`${DATA}mrt_surface.glb`, res, undefined, rej));
  setFill(0.95);

  buildFields();
  buildViewer(mrtGltf, sceneMeta);
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
      values: derive((i) => B.e_sw_diffuse.at(i) + B.e_sw_direct.at(i) + B.e_sw_specular.at(i)) },
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

/* Colour, class filtering and the ENU -> viewer frame change all happen per vertex, so a
 * filter toggle is a uniform write instead of a 40 MB buffer rebuild. Uniform arrays are
 * indexed dynamically, which GLSL ES 1.00 permits in a vertex shader (the restriction that
 * bites is in fragment shaders), and the colour ramp is a uniform array rather than a
 * texture for the same reason -- no vertex texture fetch to depend on. */
const CLOUD_VERT = `
precision highp float;
// three.js declares 'position' itself, so it must not be redeclared here -- and the
// quantised coordinates have to live in it: THREE.Points takes its draw count from
// geometry.attributes.position, so under any other name nothing is drawn at all.
attribute vec3 aRgb;
attribute float aMat;
attribute float aElem;
attribute float aTemp;
attribute float aSvf;

uniform vec3 uBoxLo, uBoxSpan, uOrigin;
uniform float uSize, uMode, uDpr;
uniform vec3 uMatColor[13];
uniform float uMatOn[13];
uniform vec3 uElemColor[6];
uniform float uElemOn[6];
uniform vec3 uRamp[${RAMP_STOPS}];

varying vec3 vColor;

vec3 srgbToLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

vec3 ramp(float t) {
  float x = clamp(t, 0.0, 1.0) * float(${RAMP_STOPS - 1});
  int i = int(floor(x));
  i = min(i, ${RAMP_STOPS - 2});
  return mix(uRamp[i], uRamp[i + 1], x - float(i));
}

void main() {
  int mi = int(aMat + 0.5);
  int ei = int(aElem + 0.5);

  // Both filters compose, so "vegetation material" and "facade element" can be intersected.
  if (uMatOn[mi] * uElemOn[ei] < 0.5) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // outside clip space: never rasterised
    gl_PointSize = 0.0;
    return;
  }

  if (uMode < 0.5)      vColor = srgbToLinear(aRgb);
  else if (uMode < 1.5) vColor = uMatColor[mi];
  else if (uMode < 2.5) vColor = uElemColor[ei];
  else if (uMode < 3.5) vColor = ramp(aTemp);
  else                  vColor = ramp(aSvf);

  vec3 enu = uBoxLo + position * uBoxSpan;
  vec3 p = vec3(enu.x - uOrigin.x, enu.z - uOrigin.z, -(enu.y - uOrigin.y));

  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp(uSize * uDpr * (10.0 / -mv.z), 1.0, 24.0);
}
`;

const CLOUD_FRAG = `
precision highp float;
varying vec3 vColor;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  if (dot(d, d) > 0.25) discard;     // round points; square ones read as noise at distance
  gl_FragColor = vec4(vColor, 1.0);
  #include <colorspace_fragment>
}
`;

function buildCloud(meta, blocks) {
  const n = meta.n_points;
  const g = new THREE.BufferGeometry();
  const attr = (arr, size, norm) => {
    const a = new THREE.BufferAttribute(arr, size, norm);
    a.gpuType = undefined;
    return a;
  };
  g.setAttribute('position', attr(blocks.pos.data, 3, true));
  g.setAttribute('aRgb', attr(blocks.rgb.data, 3, true));
  g.setAttribute('aMat', attr(blocks.material.data, 1, false));
  g.setAttribute('aElem', attr(blocks.element.data, 1, false));
  g.setAttribute('aTemp', attr(blocks.temp.data, 1, true));
  g.setAttribute('aSvf', attr(blocks.svf.data, 1, true));

  // The shader expands positions, so three.js cannot derive bounds from the attribute --
  // without this the cloud is frustum-culled at the wrong moments and blinks out.
  const lo = meta.bbox_lo, sp = meta.bbox_span, o = meta.origin_enu;
  const corners = [];
  for (const dx of [0, 1]) for (const dy of [0, 1]) for (const dz of [0, 1]) {
    corners.push(new THREE.Vector3(
      lo[0] + dx * sp[0] - o[0], lo[2] + dz * sp[2] - o[2], -(lo[1] + dy * sp[1] - o[1])));
  }
  g.boundingBox = new THREE.Box3().setFromPoints(corners);
  g.boundingSphere = g.boundingBox.getBoundingSphere(new THREE.Sphere());

  const nMat = state.M.levels.l3.labels.length;   // 13, sky included but unreachable here
  const nElem = state.M.levels.l2.labels.length;  // 6, same
  state.matOn = new Array(nMat).fill(1);
  state.elemOn = new Array(nElem).fill(1);

  const mat = new THREE.ShaderMaterial({
    vertexShader: CLOUD_VERT,
    fragmentShader: CLOUD_FRAG,
    uniforms: {
      uBoxLo: { value: new THREE.Vector3(...lo) },
      uBoxSpan: { value: new THREE.Vector3(...sp) },
      uOrigin: { value: new THREE.Vector3(...o) },
      uSize: { value: 18 },
      uMode: { value: 0 },
      uDpr: { value: Math.min(devicePixelRatio, 2) },
      uMatColor: { value: state.M.levels.l3.colors.map((c) => new THREE.Vector3(...linearRgb(c))) },
      uMatOn: { value: state.matOn.slice() },
      uElemColor: { value: state.M.levels.l2.colors.map((c) => new THREE.Vector3(...linearRgb(c))) },
      uElemOn: { value: state.elemOn.slice() },
      uRamp: { value: rampVectors('inferno') },
    },
  });

  const pts = new THREE.Points(g, mat);
  pts.frustumCulled = true;
  return { pts, mat, n };
}

function rampVectors(cmap) {
  const out = [];
  for (let i = 0; i < RAMP_STOPS; i++) {
    const [r, g, b] = sampleCmap(cmap, i / (RAMP_STOPS - 1));
    out.push(new THREE.Vector3(...linearRgb([r, g, b])));
  }
  return out;
}

function buildViewer(mrtGltf, meta) {
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

  const cloud = buildCloud(state.C, state.CB);
  cloud.pts.renderOrder = 0;
  scene.add(cloud.pts);

  // -- MRT surface: unlit on purpose. Shading a data surface would multiply the colormap
  // by a lighting term and the colours would stop meaning their values.
  const mrtGeom = firstMeshGeometry(mrtGltf);
  if (mrtGeom.attributes.position.count !== state.N) {
    throw new Error(`mrt_surface.glb has ${mrtGeom.attributes.position.count} vertices but ` +
                    `attribution.bin has ${state.N} rows -- rebuild the web assets`);
  }
  mrtGeom.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(state.N * 3), 3, true));
  const mrtMat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  const mrtMesh = new THREE.Mesh(mrtGeom, mrtMat);
  mrtMesh.renderOrder = 1;
  scene.add(mrtMesh);

  // -- picked-standpoint marker. Drawn last and without depth testing, so it is never
  // swallowed by the surface it stands on. renderOrder goes on each child: a Group does not
  // pass its own down, and an opaque mesh with depthTest off sorts anywhere at all.
  const marker = new THREE.Group();
  marker.visible = false;
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x6cc4f5, side: THREE.DoubleSide, depthTest: false, depthWrite: false, transparent: true,
  });
  const ring = new THREE.Mesh(new THREE.RingGeometry(2.1, 2.9, 40), ringMat);
  ring.rotation.x = -Math.PI / 2;
  const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 1.5, 10), ringMat);
  pin.position.y = -0.75;
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 12), ringMat);
  marker.add(ring, pin, dot);
  for (const m of marker.children) m.renderOrder = 999;
  scene.add(marker);

  Object.assign(V, {
    renderer, scene, camera, controls, cloud, mrtMesh, mrtMat, marker, meta,
    positions: mrtGeom.attributes.position.array,
    raycaster: new THREE.Raycaster(), pointer: new THREE.Vector2(), mesh: null,
  });

  // Framing, fog and zoom limits all key off the MRT surface's own size rather than fixed
  // numbers, so rebuilding the assets at a different extent cannot strand the camera.
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
      marker.scale.setScalar(Math.max(1, camera.position.distanceTo(marker.position) / 110));
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
  if (!mrtMesh.visible) return -1;
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
      el.innerHTML = f.discrete
        ? `<i>${f.label}</i> <b>${v ? 'yes' : 'no'}</b>`
        : `<i>${f.label}</i> <b>${v.toFixed(f.unit === 'W/m²' ? 0 : 2)}</b> ${f.unit}`;
      const r = canvas.getBoundingClientRect();
      el.style.left = `${e.clientX - r.left}px`;
      el.style.top = `${e.clientY - r.top}px`;
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

  const l2 = cat(B.l2_lw, B.l2_sw, L2);
  const l3 = cat(B.l3_lw, B.l3_sw, L3).filter((r) => r.value > 0.005);
  // Materials are ranked by what they actually contribute here, with sky pinned last --
  // the same ordering the notebook's figure uses.
  const skyName = (L3.pretty ?? L3.labels)[L3.labels.length - 1];
  l3.sort((a, b) => (a.name === skyName) - (b.name === skyName) || b.value - a.value);

  // Two pathways reach the body without touching any surface or ray, so they belong to no
  // category at any level -- including Direction, because the six plane irradiances carry
  // diffuse and reflected shortwave only. All three levels get them, which is also what
  // makes all three sum to the same total.
  const extras = [];
  if (band !== 'lw') {
    const direct = B.e_sw_direct.at(i), spec = B.e_sw_specular.at(i);
    if (direct > 0) extras.push({ name: 'Direct sun on body', value: mix(0, direct), extra: true });
    if (spec > 0) extras.push({ name: 'Specular glint', value: mix(0, spec), extra: true });
  }

  return [
    { title: L1.name, rows: l1, extras },
    { title: L2.name, rows: l2.filter((r) => r.value > 0.005), extras },
    { title: L3.name, rows: l3, extras },
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

  const p = V.positions, o = V.meta.origin_enu;
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

  $('levels').innerHTML = levelRows(i, band).map(({ title, rows, extras }) => {
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

  $('closure').textContent =
    `Each level is an exact split of the same budget: every set of bars sums back to the ` +
    `${bandName.toLowerCase()} total above. Values are transported quantised to ` +
    `~0.005 W/m², so shares may round by a tenth.`;
}

/* --------------------------------------------------------------- chrome */

/** Filter chips for one categorical layer. The label, colour and order all come from the
 *  manifest, so the chips, the point cloud and the attribution bars cannot disagree about
 *  what "Asphalt" is or what colour it should be. */
function buildChips(hostId, level, counts, onState, labelsKey) {
  const host = $(hostId);
  const names = level.labels;
  const pretty = level.pretty ?? names;
  host.innerHTML = '';
  names.forEach((name, k) => {
    const n = counts[name] ?? counts[labelsKey?.[k]] ?? 0;
    if (n === 0) return;      // sky has no points, and neither does an unused material
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.dataset.k = k;
    b.innerHTML = `<span class="dot" style="background:${level.colors[k]}"></span>` +
                  `<span>${pretty[k]}</span><span class="n">${(n / 1000).toFixed(0)}k</span>`;
    b.addEventListener('click', () => {
      onState[k] = onState[k] ? 0 : 1;
      b.classList.toggle('off', !onState[k]);
      V.cloud.mat.uniforms[hostId === 'chips-mat' ? 'uMatOn' : 'uElemOn'].value = onState.slice();
    });
    host.appendChild(b);
  });
}

function buildChrome() {
  const { M, C } = state;

  $('n-cloud').textContent = `${(C.n_points / 1e6).toFixed(2)}M`;

  const sel = $('field');
  sel.innerHTML = state.fields
    .map((f, k) => `<option value="${k}">${f.label}${f.unit ? ` (${f.unit})` : ''}</option>`).join('');
  sel.addEventListener('change', () => applyField(state.fields[+sel.value]));

  buildChips('chips-elem', M.levels.l2, C.counts.element, state.elemOn);
  buildChips('chips-mat', M.levels.l3, C.counts.material, state.matOn, C.material_labels);

  for (const btn of document.querySelectorAll('.all')) {
    btn.addEventListener('click', () => {
      const isMat = btn.dataset.for === 'mat';
      const arr = isMat ? state.matOn : state.elemOn;
      const host = isMat ? 'chips-mat' : 'chips-elem';
      const anyOff = arr.some((v) => !v);
      arr.fill(anyOff ? 1 : 0);
      for (const c of $(host).querySelectorAll('.chip')) c.classList.toggle('off', !anyOff);
      V.cloud.mat.uniforms[isMat ? 'uMatOn' : 'uElemOn'].value = arr.slice();
    });
  }

  // Cloud colour mode, and the ramp/legend that goes with the continuous ones.
  const RAMPS = { 3: ['inferno', 'temperature_range_c', '°C'], 4: ['viridis', null, ''] };
  $('cloudmode').addEventListener('change', (e) => {
    const m = +e.target.value;
    state.cloudMode = m;
    V.cloud.mat.uniforms.uMode.value = m;
    const r = RAMPS[m];
    $('cloud-legend').hidden = !r;
    if (!r) return;
    const [cmap, rangeKey, unit] = r;
    V.cloud.mat.uniforms.uRamp.value = rampVectors(cmap);
    $('cloud-ramp').style.background = cmapCss(cmap);
    const [lo, hi] = rangeKey ? C[rangeKey] : [0, 1];
    $('cloud-lo').textContent = `${lo.toFixed(1)} ${unit}`.trim();
    $('cloud-hi').textContent = `${hi.toFixed(1)} ${unit}`.trim();
  });

  $('ptsize').addEventListener('input', (e) => { V.cloud.mat.uniforms.uSize.value = +e.target.value; });
  $('surfop').addEventListener('input', (e) => {
    const v = +e.target.value / 100;
    // A fully opaque mesh left in the transparent pass sorts per-object rather than
    // per-pixel, which on a draped double-sided surface shows up as flicker.
    V.mrtMat.opacity = v;
    V.mrtMat.transparent = v < 1;
    V.mrtMat.depthWrite = v > 0.98;
    V.mrtMat.needsUpdate = true;
  });

  $('l-cloud').addEventListener('change', (e) => { V.cloud.pts.visible = e.target.checked; });
  $('l-mrt').addEventListener('change', (e) => {
    V.mrtMesh.visible = e.target.checked;
    if (!e.target.checked) $('hover').classList.remove('on');
  });
  // The solid mesh is 8 MB that most sessions never look at, so it is fetched the first
  // time it is actually asked for rather than on load.
  $('l-mesh').addEventListener('change', async (e) => {
    if (!e.target.checked) { if (V.mesh) V.mesh.visible = false; return; }
    if (V.mesh) { V.mesh.visible = true; return; }
    e.target.disabled = true;
    try {
      const gltf = await new Promise((res, rej) =>
        new GLTFLoader().load(`${DATA}scene.glb`, res, undefined, rej));
      const geom = firstMeshGeometry(gltf);
      geom.computeVertexNormals();
      V.mesh = new THREE.Mesh(geom, new THREE.MeshLambertMaterial({
        vertexColors: true, side: THREE.DoubleSide,
      }));
      V.mesh.renderOrder = 0;
      V.scene.add(V.mesh);
    } catch (err) {
      console.error(err);
      e.target.checked = false;
    } finally {
      e.target.disabled = false;
    }
  });

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
    `${f0(c.sky_downwelling_lw)} W/m². Body: ${M.body.model}, ${M.body.n_rays} rays. ` +
    `Cloud: ${thousands(C.n_points)} points at ${C.voxel_m.toFixed(2)} m, ` +
    `from ${thousands(C.source_points)}.`;

  addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      state.picked = -1;
      V.marker.visible = false;
      $('panel').classList.add('empty');
      $('panel-body').hidden = true;
    }
  });
}
