/* Thermal Clouds -- MRT viewer.
 *
 * Static assets, no backend:
 *   cloud.bin        the dense reconstruction, 1 row per point: position, RGB, material,
 *                    element, surface temperature, sky view factor
 *   mrt_surface.glb  the pedestrian-height surface, geometry only
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
import { utci, utciCategory, UTCI_CATEGORIES } from './utci.js';

// Time slots: every slot's data sit in their own folder (the reference slot in data/, others in
// data_<hhmm>/), listed in slots.json; ?slot=<hhmm> picks one. Only digits pass, so the query
// can never point anywhere else.
const SLOT = (() => {
  const s = new URLSearchParams(location.search).get('slot') || '';
  return /^[0-9]{3,4}$/.test(s) ? s : null;
})();
const DATA = SLOT ? `./data_${SLOT}/` : './data/';
const HERE = SLOT ? `data_${SLOT}` : 'data';
// ?cell=<n>: open on that place of the shared standpoint lattice (set by the time links below, so
// switching time keeps the standpoint). Digits only, like ?slot.
const CELL = (() => {
  const s = new URLSearchParams(location.search).get('cell') || '';
  return /^[0-9]{1,8}$/.test(s) ? +s : -1;
})();
const $ = (id) => document.getElementById(id);

/** The page of slot `dir`, opened on lattice cell `cell` (-1: none). */
function slotHref(dir, cell = -1) {
  const q = new URLSearchParams();
  if (dir !== 'data') q.set('slot', dir.replace('data_', ''));
  if (cell >= 0) q.set('cell', String(cell));
  const qs = q.toString();
  return (qs ? `?${qs}` : location.pathname) + location.hash;
}

async function renderSlots() {
  let list;
  try {
    const res = await fetch('./slots.json', { cache: 'no-cache' });
    if (!res.ok) return;
    list = (await res.json()).slots || [];
  } catch { return; }
  if (list.length < 2) return;
  const el = $('slots');
  el.textContent = 'Time: ';
  list.forEach((s, i) => {
    if (i) el.append(' · ');
    const a = document.createElement('a');
    a.textContent = s.label;
    a.dataset.dir = s.dir;
    a.href = slotHref(s.dir);
    if (s.dir === HERE) a.className = 'current';
    el.append(a);
  });
  updateSlotLinks();
}
renderSlots();

/** Keep the time links on the picked place, so the switch shows the same standpoint at another time. */
function updateSlotLinks() {
  const cell = state.picked >= 0 && H.slots ? H.slots[H.here].cells[state.picked] : -1;
  for (const a of document.querySelectorAll('#slots a[data-dir]')) a.href = slotHref(a.dataset.dir, cell);
}

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

/** A data file, pre-compressed where the pipeline wrote one (reconstruction/web_io.py): `<name>.gz`
 *  first, gunzipped here, else `<name>`. A static host does not compress binary files for us,
 *  and gzip cuts the first download by over a third. Decompression is keyed on the gzip magic
 *  bytes, not on the file name, so a host that already decoded the file (Content-Encoding)
 *  works too. Progress counts the bytes actually transferred. */
async function fetchData(name, onBytes = () => {}, base = DATA) {
  for (const url of [`${base}${name}.gz`, `${base}${name}`]) {
    let buf;
    try {
      buf = await fetchProgress(url, onBytes);
    } catch (err) {
      continue;                              // missing: try the uncompressed name
    }
    const head = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
    if (head[0] !== 0x1f || head[1] !== 0x8b) return buf;
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).arrayBuffer();
  }
  throw new Error(`${name}: not found in ${base} (neither .gz nor plain)`);
}

const parseGltf = (buf) => new Promise((res, rej) => new GLTFLoader().parse(buf, DATA, res, rej));

const TYPES = { float32: Float32Array, uint32: Uint32Array, uint16: Uint16Array, int16: Int16Array, uint8: Uint8Array };

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
const SIGMA = 5.670374419e-8;
const M_SWC = () => state.M.body.sw_coeff;       // a_k / eps_p, the shortwave weight in the MRT flux
const NONE_RGB = [92, 99, 110];                   // a log field's "none" (value 0): neutral grey
const cssColor = (c) => (Array.isArray(c) ? `rgb(${c[0]},${c[1]},${c[2]})` : c);

/* The 3D backdrop's colour lives in style.css (--stage) so the canvas and the page around it are
   set in one place; three.js needs it as a number. */
const STAGE_COLOR = () => {
  const v = getComputedStyle(document.documentElement).getPropertyValue('--stage').trim();
  const m = /^#([0-9a-f]{6})$/i.exec(v);
  return m ? parseInt(m[1], 16) : 0x949494;
};

/* Phone drawers: the tab bar is display:none above 720px, so its computed style is what says
   whether this layout is in use -- one source of truth with the media query, no second breakpoint
   in JS to drift from it. One drawer at a time; `toggle` lets a tab close its own drawer. */
const drawersActive = () => getComputedStyle(document.getElementById('drawer-tabs')).display !== 'none';
function openDrawer(which, toggle = false) {
  const open = toggle && document.body.classList.contains(`drawer-${which}`) ? null : which;
  document.body.classList.remove('drawer-sidebar', 'drawer-panel');
  if (open) document.body.classList.add(`drawer-${open}`);
  for (const b of document.querySelectorAll('#drawer-tabs button')) {
    b.setAttribute('aria-expanded', String(b.dataset.drawer === open));
  }
}

const state = {
  M: null, B: null, N: 0, C: null, CB: null,
  band: 'comb', field: null, picked: -1, fields: [],
  matOn: null, elemOn: null, cloudMode: 0, cloudRamp: 'inferno',
  off: new Set(),          // sources switched off in the panel: 'm:<material index>', 'direct', 'glint'
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
    loadComfortMeta(),
  ]);
  state.M = manifest;
  state.N = manifest.n_points;
  state.C = cloudMeta;
  state.matOn = new Array(manifest.levels.l3.labels.length).fill(1);
  state.elemOn = new Array(manifest.levels.l2.labels.length).fill(1);

  // The MRT surface and its attribution first: they are the page's content and usable on
  // their own. The point cloud -- the largest file -- streams in behind them (loadCloud), so
  // the wait before the first view is the surface's, not the whole dataset's.
  let fa = 0, fm = 0;
  const progress = () => setFill(0.8 * fa + 0.2 * fm);   // roughly their share of the bytes
  const [attribution, mrtGltf] = await Promise.all([
    fetchData('attribution.bin', (g, t) => { fa = t ? g / t : 0.5; progress(); }),
    fetchData('mrt_surface.glb', (g, t) => { fm = t ? g / t : 0.5; progress(); }).then(parseGltf),
  ]);
  state.B = unpack(attribution, manifest);

  buildFields();
  buildViewer(mrtGltf, sceneMeta);
  buildChrome();

  setFill(1);
  $('loading').classList.add('done');
  setTimeout(() => { $('loading').style.display = 'none'; }, 450);

  loadCloud(cloudMeta);   // not awaited: the view is already interactive
  loadGlints();           // likewise; optional (older builds have no glints.json)
  loadRays();             // likewise; optional (scripts/viewer/web_rays.py)
  if (location.hash === '#debug') window.urfDebug = { state, V, H, CF, R, select, setRayFocus, pickAt };   // headless tests only
  await loadHistoryMeta();
  if (CELL >= 0) openCell(CELL);   // arrived from another time's link: same place, this time
}

/* ------------------------------------------------------ history over the day */

/* The same place at every time it was flown (scripts/viewer/web_history.py -> web/history.json, history.bin).
 * All slots share one standpoint lattice; a slot's rows are its lattice cells in order, so a cell maps
 * to a row by binary search in that slot's sorted cell list. history.json (8 kB) is read at start, the
 * 13 MB history.bin only the first time a standpoint is picked. */
const H = { meta: null, here: -1, slots: null, loading: null, failed: false };

async function loadHistoryMeta() {
  try {
    const res = await fetch('./history.json', { cache: 'no-cache' });
    if (!res.ok) return;
    const meta = await res.json();
    const here = meta.slots.findIndex((s) => s.dir === HERE);
    // A history built from another build of any slot would show that slot's old values, or put rows
    // at the wrong place: every slot's attribution build stamp must be the one it was made from.
    const stamps = await Promise.all(meta.slots.map((s) => (s.dir === HERE ? state.M
      : fetch(`./${s.dir}/manifest.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null))));
    const stale = meta.slots.filter((s, k) => stamps[k]?.generated !== s.generated).map((s) => s.dir);
    if (here < 0 || meta.slots[here].n_points !== state.N || meta.slots.length < 2 || stale.length) {
      console.warn(`history.json does not match the viewer data (${stale.join(', ') || HERE}) -- ` +
                   'history off; rerun scripts/viewer/web_history.py');
      return;
    }
    H.meta = meta;
    H.here = here;
    if (state.picked >= 0) renderHistory(state.picked);
  } catch (err) {
    console.warn(err);
  }
}

function ensureHistory() {
  if (!H.meta || H.slots || H.failed) return Promise.resolve();
  H.loading ??= fetchData('history.bin', () => {}, './').then((buf) => {
    const { meta } = H;
    const blocks = unpack(buf, { blocks: meta.slots.flatMap((s) => s.blocks) });
    const nCells = meta.lattice.nx * meta.lattice.ny;
    H.slots = meta.slots.map((s) => {
      const bits = blocks[`${s.dir}/valid`].data;
      const cells = new Uint32Array(s.n_points);
      let k = 0;
      for (let c = 0; c < nCells; c++) if (bits[c >> 3] & (1 << (c & 7))) cells[k++] = c;
      if (k !== s.n_points) throw new Error(`history.bin: ${s.dir} has ${k} cells, history.json says ${s.n_points}`);
      const get = (name) => blocks[`${s.dir}/${name}`];
      return { ...s, cells, mrt: get('mrt'), mrtLw: get('mrt_lw'), contrib: get('contrib'), glint: get('glint') };
    });
    updateSlotLinks();
  }).catch((err) => { console.error(err); H.failed = true; });
  return H.loading;
}

function rowOf(cells, c) {
  let lo = 0, hi = cells.length - 1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1, v = cells[m];
    if (v === c) return m;
    if (v < c) lo = m + 1; else hi = m - 1;
  }
  return -1;
}

/** Open the page on lattice cell `cell`: select it and bring the camera over it. */
async function openCell(cell) {
  if (!H.meta) return;
  await ensureHistory();
  if (!H.slots) return;
  const i = rowOf(H.slots[H.here].cells, cell);
  if (i < 0) {
    $('hint').textContent = 'That place has no standpoint at this time (not flown, or no thermal coverage)';
    return;
  }
  select(i);
  const p = V.marker.position, { camera, controls } = V;
  const dir = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(p);
  camera.position.copy(p).addScaledVector(dir, 220);
  controls.update();
}

/** Local time of day in hours, from the slot's literal site time (see siteTime). */
const hourOf = (iso) => {
  const m = /[T ](\d{2}):(\d{2})(?::(\d{2}))?/.exec(iso);
  return m ? +m[1] + m[2] / 60 + (+m[3] || 0) / 3600 : NaN;
};

function renderHistory(i) {
  const el = $('history');
  if (!H.meta) { $('fold-history').hidden = true; return; }
  $('fold-history').hidden = false;
  const head = '';
  if (!H.slots) {
    $('sum-history').textContent = '';
    el.innerHTML = `<p class="note">${H.failed ? 'Could not load the other times.' : 'Loading the other times…'}</p>`;
    ensureHistory().then(() => { if (state.picked === i) renderHistory(i); });
    return;
  }
  const S = H.meta.sources, cell = H.slots[H.here].cells[i];
  const times = H.slots.map((s) => {
    const r = rowOf(s.cells, cell);
    if (r < 0) return { s, r };
    const src = Array.from({ length: s.contrib.cols }, (_, k) => s.contrib.at(r, k));
    const glint = s.glint.at(r);
    return { s, r, mrt: s.mrt.at(r), mrtLw: s.mrtLw.at(r), src, glint,
             total: src.reduce((a, b) => a + b, 0) + glint, h: hourOf(s.when) };
  });
  const have = times.filter((t) => t.r >= 0);
  const names = S.pretty, colors = S.colors, sky = S.sky;

  // -- chart: MRT, its longwave part and the air, at each time this place was flown
  const W = 352, Ht = 140, ml = 30, mr = 10, mt = 20, mb = 30;
  const hs = times.map((t) => hourOf(t.s.when));
  const x0 = Math.min(...hs) - 0.6, x1 = Math.max(...hs) + 0.6;
  const vals = have.flatMap((t) => [t.mrt, t.mrtLw, t.s.conditions.t_atm_c]);
  const y0 = Math.floor((Math.min(...vals) - 2) / 10) * 10, y1 = Math.ceil((Math.max(...vals) + 2) / 10) * 10;
  const X = (h) => ml + (h - x0) / (x1 - x0) * (W - ml - mr);
  const Y = (v) => mt + (1 - (v - y0) / (y1 - y0)) * (Ht - mt - mb);
  let svg = '';
  for (let v = y0; v <= y1; v += 10) {
    svg += `<line class="grid" x1="${ml}" x2="${W - mr}" y1="${Y(v)}" y2="${Y(v)}"/>` +
           `<text x="${ml - 4}" y="${Y(v) + 3}" text-anchor="end">${v}°</text>`;
  }
  times.forEach((t, k) => {
    const x = X(hs[k]), cur = k === H.here, off = t.r < 0;
    if (cur) svg += `<rect class="cur" x="${x - 17}" y="${mt - 8}" width="34" height="${Ht - mt - mb + 12}" rx="4"/>`;
    const label = `<text class="${cur ? 'tcur' : off ? 'toff' : 'tlink'}" x="${x}" y="${Ht - mb + 15}" text-anchor="middle">${t.s.label}</text>` +
      (off ? `<text class="toff" x="${x}" y="${Ht - mb + 26}" text-anchor="middle">not here</text>` : '');
    svg += off || cur ? label : `<a href="${slotHref(t.s.dir, cell)}">${label}</a>`;
  });
  const path = (key) => have.map((t, k) => `${k ? 'L' : 'M'}${X(t.h).toFixed(1)},${Y(t[key]).toFixed(1)}`).join('');
  if (have.length > 1) svg += `<path class="h-lw" d="${path('mrtLw')}"/><path class="h-mrt" d="${path('mrt')}"/>`;
  for (const t of have) {
    const x = X(t.h);
    svg += `<rect class="h-air" x="${x - 3}" y="${Y(t.s.conditions.t_atm_c) - 3}" width="6" height="6"/>` +
           `<circle class="h-lw-dot" cx="${x}" cy="${Y(t.mrtLw)}" r="3.5"/>` +
           `<circle class="h-mrt-dot" cx="${x}" cy="${Y(t.mrt)}" r="4"/>` +
           `<text class="h-val" x="${x}" y="${Y(t.mrt) - 8}" text-anchor="middle">${f1(t.mrt)}</text>`;
  }
  const chart = `<svg viewBox="0 0 ${W} ${Ht}" role="img" aria-label="MRT at each time this place was flown">${svg}</svg>
    <div class="h-legend"><span><i class="k-mrt"></i>MRT</span><span><i class="k-lw"></i>longwave only</span><span><i class="k-air"></i>air</span></div>`;

  // -- sources: the same order in every row, so a colour stays in its place from one time to the next
  const nSrc = names.length;
  const peak = (k) => Math.max(...have.map((t) => t.src[k]));
  const order = [...Array(nSrc).keys()].filter((k) => k !== sky).sort((a, b) => peak(b) - peak(a));
  order.push(sky);
  const maxTotal = Math.max(...have.map((t) => t.total), 1e-9);
  const seg = (v, name, color) => (v > 0 ? `<i style="width:${(v / maxTotal * 100).toFixed(2)}%;background:${color}" title="${name} ${f0(v)} W/m²"></i>` : '');
  const rows = have.map((t) => {
    const cur = t.s.dir === HERE;
    const inner = `<span class="t">${t.s.label}</span><span class="stack">${
      order.map((k) => seg(t.src[k], names[k], colors[k])).join('')}${seg(t.glint, S.glint.pretty, S.glint.color)}</span>` +
      `<span class="v">${f0(t.total)}</span>`;
    return cur ? `<div class="hrow cur">${inner}</div>` : `<a class="hrow" href="${slotHref(t.s.dir, cell)}">${inner}</a>`;
  }).join('');

  // -- the largest sources (sky aside), as numbers, with each time's largest in bold
  const top = order.filter((k) => k !== sky && peak(k) >= 1).slice(0, 6);
  const glintPeak = Math.max(...have.map((t) => t.glint));
  const largest = have.map((t) => order.filter((k) => k !== sky).reduce((a, b) => (t.src[b] > t.src[a] ? b : a)));
  const cellRow = (name, color, vs, boldAt) => `<tr><td><span class="sw" style="background:${color}"></span>${name}</td>${
    vs.map((v, j) => `<td>${boldAt?.[j] ? `<b>${f0(v)}</b>` : f0(v)}</td>`).join('')}</tr>`;
  const table = `<table class="htab"><thead><tr><th>W/m²</th>${have.map((t) =>
      `<th${t.s.dir === HERE ? ' class="cur"' : ''}>${t.s.label}</th>`).join('')}</tr></thead><tbody>${
    top.map((k) => cellRow(names[k], colors[k], have.map((t) => t.src[k]), largest.map((b) => b === k))).join('')}${
    glintPeak >= 0.5 ? cellRow(S.glint.pretty, S.glint.color, have.map((t) => t.glint)) : ''}${
    cellRow(names[sky], colors[sky], have.map((t) => t.src[sky]))}${
    `<tr class="sum"><td>Absorbed, S<sub>str</sub></td>${have.map((t) => `<td>${f0(t.total)}</td>`).join('')}</tr>`}</tbody></table>`;

  // -- one sentence on whether the largest source changes with the sun
  const lname = (k) => names[k];
  let verdict;
  if (have.length < 2) verdict = 'This place was flown only at this time.';
  else if (largest.every((k) => k === largest[0])) verdict = `Largest source at every time: <b>${lname(largest[0])}</b>.`;
  else verdict = `Largest source: ${have.map((t, j) => `<b>${lname(largest[j])}</b> at ${t.s.label}`).join(', ')}.`;
  if (have.length >= 2) {
    const a = have[0], b = have[have.length - 1];
    const ranked = [...Array(nSrc).keys()].filter((k) => k !== sky)
      .map((k) => [k, b.src[k] - a.src[k]]).sort((p, q) => q[1] - p[1]);
    const floor = Math.max(2, 0.1 * (ranked[0]?.[1] ?? 0));          // leave out the trivial ones
    const rise = ranked.slice(0, 3).filter(([, d]) => d >= floor);
    if (rise.length) verdict += ` From ${a.s.label} to ${b.s.label}, the most added: ${
      rise.map(([k, d]) => `${lname(k)} +${f0(d)}`).join(', ')} W/m².`;
  }

  $('sum-history').textContent = have.length > 1
    ? have.map((t) => `${t.s.label} ${f1(t.mrt)}`).join(' · ') + ' °C' : 'only this time';
  el.innerHTML = `${head}${chart}
    <p class="note">What the body absorbs, by source: S<sub>str</sub> = ${state.M.body.eps_p.toFixed(2)} × longwave +
      ${state.M.body.alpha_sw.toFixed(2)} × shortwave (VDI 3787, ISO 7726; the Combined budget). Bar length = total;
      a time links to its own page.</p>
    <div class="hbars">${rows}</div>${table}
    <p class="verdict">${verdict}</p>
    <p class="note">Each time is its own flight: the surfaces' longwave was measured then, the sunlight computed for
      that instant.${H.slots.some((s, k) => k !== H.here && times[k].r < 0) ? ' “Not here”: that flight did not cover this place, or left it without thermal coverage.' : ''}</p>`;
}

/* ------------------------------------------------- switching sources off, and UTCI */

/* scripts/viewer/web_comfort.py -> web/comfort.json (8 kB: each slot's air, humidity and 10 m wind for UTCI) and
 * comfort.bin (each place's view weight per material, fetched with the first pick). Switching a source off
 * replaces what it sends the body by what a surface at air temperature that reflects no sunlight would send:
 *   S_str' = S_str - (a_l l3_lw_k + a_k l3_sw_k) + a_l W_k sigma T_air^4,
 * W_k the share of the standing body's view the material fills. The surface stays where it is, so its shade
 * stays too; only what it emits and reflects changes. Direct sun and glint off: as if the body were shaded from
 * them. Every source off gives MRT = air temperature. */
const CF = { meta: null, slot: null, cells: null, W: null, loading: null, failed: false };

async function loadComfortMeta() {
  try {
    const res = await fetch('./comfort.json', { cache: 'no-cache' });
    if (!res.ok) return;
    const meta = await res.json();
    CF.meta = meta;
    CF.slot = meta.slots.find((s) => s.dir === HERE) ?? null;
  } catch (err) {
    console.warn(err);
  }
}

function ensureComfort() {
  if (!CF.meta || CF.W || CF.failed) return Promise.resolve();
  CF.loading ??= fetchData('comfort.bin', () => {}, './').then((buf) => {
    const b = unpack(buf, CF.meta);
    const bits = b.valid.data, nCells = CF.meta.lattice.nx * CF.meta.lattice.ny;
    const cells = new Uint32Array(CF.meta.n_places);
    let k = 0;
    for (let c = 0; c < nCells; c++) if (bits[c >> 3] & (1 << (c & 7))) cells[k++] = c;
    if (k !== CF.meta.n_places) throw new Error(`comfort.bin: ${k} places, comfort.json says ${CF.meta.n_places}`);
    CF.cells = cells;
    CF.W = b.w;
  }).catch((err) => { console.error(err); CF.failed = true; });
  return CF.loading;
}

/** View weight of material k at standpoint i, or null when it is not (yet) known. */
function viewWeight(i, k) {
  if (!CF.W || !H.slots) return null;
  const r = rowOf(CF.cells, H.slots[H.here].cells[i]);
  return r < 0 ? null : CF.W.at(r, k);
}

/** MRT and UTCI at standpoint i, as stored and with the switched-off sources replaced. */
function comfortAt(i) {
  const { M, B } = state;
  const aL = M.body.eps_p, aK = M.body.alpha_sw;
  const s0 = aL * B.e_lw.at(i) + aK * (B.e_sw_diffuse.at(i) + B.e_sw_direct.at(i) + B.e_sw_specular.at(i));
  const tAir = CF.slot?.t_air_c ?? M.conditions.t_atm_c;
  const ref = aL * SIGMA * (tAir + 273.15) ** 4;            // what a neutral surface sends over the whole view
  let s = s0, unknown = false;
  for (const key of state.off) {
    if (key === 'direct') s -= aK * B.e_sw_direct.at(i);
    else if (key === 'glint') s -= aK * B.e_sw_specular.at(i);
    else {
      const k = +key.slice(2);
      const w = viewWeight(i, k);
      if (w === null) { unknown = true; continue; }
      s += w * ref - (aL * B.l3_lw.at(i, k) + aK * B.l3_sw.at(i, k));
    }
  }
  const mrtOf = (x) => Math.pow(Math.max(x, 0) / (aL * SIGMA), 0.25) - 273.15;
  const mrt0 = B.mrt_combined.at(i), mrt1 = mrt0 + (mrtOf(s) - mrtOf(s0));   // the stored value, moved by the change
  const u = (m) => (CF.slot ? utci(CF.slot.t_air_c, m, CF.slot.wind_10m_m_s, CF.slot.rh_pct) : NaN);
  return { mrt0, mrt1, utci0: u(mrt0), utci1: u(mrt1), s0, s1: s, unknown, changed: state.off.size > 0 };
}

const badge = (u) => {
  if (!Number.isFinite(u)) return '<span class="ubadge none">outside the UTCI range</span>';
  const c = utciCategory(u);
  return `<span class="ubadge" style="--uc:${c.color}"><b>${f1(u)} °C</b> ${c.label}</span>`;
};

function renderComfort(i) {
  const el = $('comfort');
  if (!CF.slot) { el.hidden = true; return; }
  el.hidden = false;
  const r = comfortAt(i);
  const names = state.M.levels.l3.pretty ?? state.M.levels.l3.labels;
  const offNames = [...state.off].map((k) => (k === 'direct' ? 'direct sun' : k === 'glint' ? 'glint'
    : names[+k.slice(2)].toLowerCase()));
  const s = CF.slot;
  let html = `<div class="urow"><span class="ulab">UTCI</span>${badge(r.utci0)}</div>`;
  if (r.changed) {
    const dU = Number.isFinite(r.utci1) && Number.isFinite(r.utci0) ? r.utci0 - r.utci1 : NaN;
    html += `<div class="whatif"><div class="wtitle"><span>Replaced by a neutral surface at air temperature:
        ${offNames.join(', ')}</span><button type="button" id="all-on">all on</button></div>
      <div class="wrow"><span>MRT</span><b>${f1(r.mrt1)} °C</b><em>${signed(r.mrt1 - r.mrt0, 1)} K</em></div>
      <div class="wrow"><span>UTCI</span>${badge(r.utci1)}${Number.isFinite(dU) ? `<em>${signed(-dU, 1)} K</em>` : ''}</div>
      <p class="wlabel">Added above a neutral surface at air temperature: <b>${signed(r.mrt0 - r.mrt1, 1)} K</b> of MRT${
        Number.isFinite(dU) ? `, <b>${signed(dU, 1)} K</b> of UTCI` : ''}.</p>
      ${r.unknown ? '<p class="wnote">Loading what each material fills of the view…</p>' : ''}</div>`;
  }
  html += `<p class="unote">UTCI with air ${f1(s.t_air_c)} °C, RH ${f0(s.rh_pct)} %, wind ${f1(s.wind_10m_m_s)} m/s at
    10 m (${s.wind_source}), all fixed for this time; only MRT varies.</p>`;
  el.innerHTML = html;
  $('all-on')?.addEventListener('click', () => { state.off.clear(); renderPanel(); });
  if (r.unknown) ensureHistory().then(ensureComfort).then(() => { if (state.picked === i) renderComfort(i); });
}

/* Where each standpoint's glint comes from (scripts/viewer/glint_sources.py): per standpoint row, the
 * glass/water patches whose mirrored sunbeam reaches it, strongest first, with the W/m2 each
 * delivers to the body. A click on a lit standpoint draws a line to each; the "Glint sources"
 * layer shows every patch, coloured by what it delivers over the whole grid. */
async function loadGlints() {
  try {
    const res = await fetch(`${DATA}glints.json`);
    if (!res.ok) return;
    const meta = await res.json();
    if (meta.n_points !== state.N) {
      console.warn(`glints.json is for ${meta.n_points} standpoints, the surface has ${state.N} -- ignored`);
      return;
    }
    // A slot can legitimately have none: at 06:24 the sun is 2.9 deg up, DNI 0, so nothing glints.
    // Test the ENERGY, not the count -- that pass still returns ~1,000 sources, every one carrying
    // 0 W/m2, and offering a checkbox that turns on a cloud delivering nothing would be a lie.
    const strongest = Array.isArray(meta.src_total_range) ? meta.src_total_range[1] : 0;
    if (!meta.n_sources || !(strongest > 0)) { $('n-glint').textContent = 'none'; return; }
    const G = unpack(await fetchData('glints.bin'), meta);
    const p = G.src_pos.data, n = meta.n_sources;
    const pos = new Float32Array(n * 3);
    for (let k = 0; k < n; k++) {             // ENU about origin_enu -> the viewer's (e, u, -n)
      pos[k * 3] = p[k * 3]; pos[k * 3 + 1] = p[k * 3 + 2]; pos[k * 3 + 2] = -p[k * 3 + 1];
    }
    state.G = { ...G, pos, meta };
    buildGlintLayer();
    $('l-glint').disabled = false;
    $('n-glint').textContent = `${(n / 1000).toFixed(1)}k`;
    if (state.picked >= 0) { renderGlintRays(state.picked); renderPanel(); }
  } catch (err) {
    console.error(err);
  }
}

function buildGlintLayer() {
  const { G } = state;
  const n = G.meta.n_sources, tot = G.src_total.data;
  const lo = 0.1, hi = 100, llo = Math.log10(lo), lspan = Math.log10(hi) - llo;   // W/m2, log
  // cyan -> white: apart from every MRT and cloud colour ramp, so a source never reads as a value
  const c0 = [0, 140, 200], c1 = [235, 255, 255];
  const col = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) {
    const t = Math.min(1, (Math.log10(Math.max(tot[k], lo)) - llo) / lspan);
    col.set(linearRgb(c0.map((v, j) => v + (c1[j] - v) * t)), k * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(G.pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  V.glintLayer = new THREE.Points(g, new THREE.PointsMaterial({ size: 6, sizeAttenuation: false, vertexColors: true }));
  V.glintLayer.visible = $('l-glint').checked;
  V.glintLayer.renderOrder = 2;
  V.scene.add(V.glintLayer);
}

/** Lines from standpoint `i` to the patches its glint comes from, drawn over everything: the paths
 *  are unobstructed by construction, and a line hidden behind the cloud would explain nothing. */
function renderGlintRays(i) {
  if (V.glintRays) {
    V.scene.remove(V.glintRays);
    V.glintRays.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
    V.glintRays = null;
  }
  const { G } = state;
  if (!G || i < 0) return;
  const a = G.row_start.data[i], b = G.row_start.data[i + 1];
  if (b <= a) return;
  const P = V.positions, sp = [P[i * 3], P[i * 3 + 1], P[i * 3 + 2]];
  const line = new Float32Array((b - a) * 6), dots = new Float32Array((b - a) * 3);
  for (let q = a; q < b; q++) {
    const k = G.pair_src.data[q], o = (q - a);
    line.set(sp, o * 6);
    line.set(G.pos.subarray(k * 3, k * 3 + 3), o * 6 + 3);
    dots.set(G.pos.subarray(k * 3, k * 3 + 3), o * 3);
  }
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.BufferAttribute(line, 3));
  const dg = new THREE.BufferGeometry();
  dg.setAttribute('position', new THREE.BufferAttribute(dots, 3));
  const over = { depthTest: false, depthWrite: false, transparent: true };
  const rays = new THREE.Group();
  rays.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0xffd23f, opacity: 0.95, ...over })));
  rays.add(new THREE.Points(dg, new THREE.PointsMaterial({ color: 0xffd23f, size: 11, sizeAttenuation: false, ...over })));
  for (const c of rays.children) c.renderOrder = 998;
  V.glintRays = rays;
  V.scene.add(rays);
}

/* --------------------------------------------- the picked standpoint's own rays */

/* scripts/viewer/web_rays.py: the standing body's 401 rays at every standpoint -- the attribution's own rays, so binned by
 * material they give the panel's material bars (checked there). Where a ray lands (distance, material, element, and
 * the mirrored leg where glass or water mirrors it) is shared by every slot: web/rays.json + web/rays/g_<tx>_<ty>.bin.
 * What it brings (longwave, shortwave) is the slot's: <data>/rays.json + <data>/rays/v_<tx>_<ty>.bin. Both are cut
 * into 64 m tiles of the shared lattice; a pick fetches one of each (~1 MB). Rays that escape carry the sky's
 * longwave and the diffuse shortwave, the same for every such ray, and are drawn as short fading stubs. */
const R = { geo: null, val: null, tiles: new Map(), cur: null, focus: null, obj: null, segs: null };
const RAY_SKY_LEN = 8;           // m: an escaping ray's stub
const RAY_LEG_SKY_LEN = 20;      // m: a mirrored leg that escapes
const RAY_SUN_LEN = 60;          // m: the direct beam
const RAY_FACES = { East: [1, 0, 0], West: [-1, 0, 0], North: [0, 1, 0], South: [0, -1, 0], Up: [0, 0, 1], Down: [0, 0, -1] };
const enuToView = (x, y, z) => [x, z, -y];
// Ray colours by value: inferno from 0.18 up, because its black end vanishes against the dark stage (YlOrRd whole).
const rayCmap = (band) => (band === 'sw' ? ['YlOrRd', 0] : ['inferno', 0.18]);
const rayColour = (band, t) => {
  const [name, t0] = rayCmap(band);
  return sampleCmap(name, t0 + (1 - t0) * Math.max(0, Math.min(1, t)));
};

async function loadRays() {
  try {
    const [g, v] = await Promise.all([
      fetch('./rays.json').then((r) => (r.ok ? r.json() : null)).catch(() => null),
      fetch(`${DATA}rays.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    ]);
    if (!g || !v) return;                          // not built for this slot: no rays, nothing else changes
    if (g.format !== 'urf-rays-geometry/2' || v.format !== 'urf-rays-values/1') {
      console.warn(`rays: unknown format ${g.format} / ${v.format} -- rays off`);
      return;
    }
    if (v.attribution_generated !== state.M.generated || v.geometry_generated !== g.generated || v.n_rows !== state.N) {
      console.warn('rays.json is from another build of the attribution or the geometry -- rays off; rerun scripts/viewer/web_rays.py');
      return;
    }
    R.geo = g;
    R.val = v;
    R.dirs = new Float32Array(g.n_rays * 3);       // ENU -> the viewer's (e, u, -n)
    g.directions.forEach((d, j) => R.dirs.set(enuToView(d[0], d[1], d[2]), j * 3));
    R.w = Float32Array.from(g.ray_weights);
    R.wMax = Math.max(...R.w);
    $('rays-grp').hidden = false;
    document.body.classList.add('has-rays');
    updateRayLegend();
    if (state.picked >= 0) renderRays(state.picked);
  } catch (err) {
    console.warn(err);
  }
}

/** Typed arrays one after another out of a tile, in the layout rays.json lists. */
function takeFrom(buf) {
  let o = 0;
  const take = (T, count) => {
    const a = new T(buf.slice(o, o + count * T.BYTES_PER_ELEMENT));
    o += count * T.BYTES_PER_ELEMENT;
    return a;
  };
  take.done = () => { if (o !== buf.byteLength) throw new Error(`ray tile: ${buf.byteLength - o} bytes left over`); };
  return take;
}

function loadRayTiles(key) {
  const gi = R.geo.tiles[key], nv = R.val.tiles[key];
  if (!gi || !nv) return Promise.resolve(null);
  if (!R.tiles.has(key)) {
    const NR = R.geo.n_rays;
    R.tiles.set(key, Promise.all([
      fetchData(`g_${key}.bin`, () => {}, './rays/'),
      fetchData(`v_${key}.bin`, () => {}, `${DATA}rays/`),
    ]).then(([gb, vb]) => {
      const [n, m] = gi;
      let take = takeFrom(gb);
      const cells = take(Uint32Array, n);
      const lo = take(Uint8Array, NR * n), hi = take(Uint8Array, NR * n);   // the distance code as two byte planes
      const dist = new Uint16Array(NR * n);
      for (let k = 0; k < dist.length; k++) dist[k] = lo[k] | (hi[k] << 8);
      const g = { n, cells, dist, legStart: take(Uint32Array, n + 1),
                  legDist: take(Uint16Array, m), legRay: take(Uint16Array, m), legOct: take(Int16Array, 2 * m),
                  code: take(Uint8Array, NR * n) };
      take.done();
      take = takeFrom(vb);
      const v = { n: nv, rows: take(Uint32Array, nv), cells: take(Uint32Array, nv), lw: take(Uint8Array, NR * nv),
                  sw: take(Uint8Array, NR * nv), flags: take(Uint8Array, NR * nv) };
      take.done();
      return { g, v };
    }).catch((err) => { console.error(err); R.tiles.delete(key); return null; }));
  }
  return R.tiles.get(key);
}

/* One fixed set of line segments, made once: slot j < 401 is ray j, slot 401 + j its mirrored leg (off unless the
 * engine traced that mirror), the last slot the sun beam. The drawn segments ease toward each standpoint's targets
 * (stepRays, from the render loop), so dragging the marker sweeps the rays from one standpoint to the next instead of
 * redrawing them; a click snaps. Ray j points the same way at every standpoint, so easing slot by slot is easing
 * each direction's ray from where it ended to where it ends now. */
const RAY_EASE_S = 0.07;                 // s: time constant of the easing
const RAY_SNAP_M = 30;                   // m: a move farther than this snaps

function ensureRayObject() {
  if (R.obj) return;
  const NR = R.geo.n_rays, cap = 2 * NR + 1;
  R.tPos = new Float32Array(cap * 6);
  R.tCol = new Float32Array(cap * 8);
  R.segs = Array.from({ length: cap }, (_, k) => (k < NR ? { j: k, kind: 'ray', len: 0, on: false }
    : k < 2 * NR ? { j: k - NR, kind: 'leg', len: 0, sky: false, on: false } : { j: -1, kind: 'sun', on: false }));
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cap * 6), 3).setUsage(THREE.DynamicDrawUsage));
  geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(cap * 8), 4).setUsage(THREE.DynamicDrawUsage));
  // Two passes over one geometry: depth-tested at full strength, and faintly through everything, so a ray behind
  // a facade or under the MRT surface (every downward ray) is still there, just quieter.
  const solid = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true,
                                                                          depthWrite: false }));
  const xray = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.45, depthTest: false, depthWrite: false }));
  solid.renderOrder = 996;
  xray.renderOrder = 1000;      // over the picked-standpoint marker (999), which would hide the shortest rays
  solid.frustumCulled = xray.frustumCulled = false;   // the positions move every frame while easing
  const grp = new THREE.Group();
  grp.add(solid, xray);
  grp.visible = false;
  R.obj = grp;
  R.solid = solid;
  V.scene.add(grp);
}

function clearRays() {
  R.cur = null;
  if (R.obj) R.obj.visible = false;
}

/** Draw the rays of standpoint `i` (-1: none). `smooth`: ease from the rays shown now (a drag step); otherwise the old
 *  rays go at once and the new ones snap in. The tile may still be on its way; a newer pick wins. */
async function renderRays(i, smooth = false) {
  if (!R.geo || i < 0) { clearRays(); renderRayNote(); return; }
  const L = R.geo.lattice, p = V.positions, o = V.meta.origin_enu, T = R.geo.tile;
  const ix = Math.round((p[i * 3] + o[0] - L.x0) / L.spacing);
  const iy = Math.round((-p[i * 3 + 2] + o[1] - L.y0) / L.spacing);
  const cell = ix * L.ny + iy, tx = Math.floor(ix / T), ty = Math.floor(iy / T), key = `${tx}_${ty}`;
  if (!smooth) { clearRays(); $('ray-note').textContent = 'Loading the rays…'; }
  else {
    // dragging toward a tile's edge: fetch the neighbour before the marker gets there
    const ex = ix - tx * T, ey = iy - ty * T, near = 4;
    for (const dx of [-1, 0, 1]) for (const dy of [-1, 0, 1]) {
      if ((dx || dy) && (dx >= 0 || ex < near) && (dx <= 0 || ex >= T - near) && (dy >= 0 || ey < near)
          && (dy <= 0 || ey >= T - near)) loadRayTiles(`${tx + dx}_${ty + dy}`);
    }
  }
  const t = await loadRayTiles(key);
  if (state.picked !== i) return;
  const pv = t ? rowOf(t.v.rows, i) : -1, pg = t ? rowOf(t.g.cells, cell) : -1;
  if (pv < 0 || pg < 0 || t.v.cells[pv] !== cell) {
    console.warn(`no rays stored for standpoint ${i} (cell ${cell}, tile ${key})`);
    clearRays();
    $('ray-note').textContent = 'No rays stored for this standpoint.';
    return;
  }
  const prev = R.cur;
  R.cur = { i, g: t.g, v: t.v, pg, pv };
  ensureRayObject();
  const far = !prev || Math.hypot(p[i * 3] - p[prev.i * 3], p[i * 3 + 2] - p[prev.i * 3 + 2]) > RAY_SNAP_M;
  buildRays(!smooth || far);
  renderRayNote();
}

const rayLen = (q, fallback) => (q ? R.geo.distance.d0 * Math.exp((q - 1) * R.geo.distance.step) : fallback);

function octDecode(qx, qy) {
  let x = qx / 32767, y = qy / 32767;
  const z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) [x, y] = [(1 - Math.abs(y)) * (x >= 0 ? 1 : -1), (1 - Math.abs(x)) * (y >= 0 ? 1 : -1)];
  const s = Math.hypot(x, y, z);
  return [x / s, y / s, z / s];
}

/** What ray j brings, in W/m2: longwave and shortwave radiance (sky values where it escapes). */
function rayValue(j) {
  const { g, v, pg, pv } = R.cur, V8 = R.val;
  const hit = g.code[j * g.n + pg] !== R.geo.code.sky;
  return {
    hit,
    lw: hit ? V8.lw.lo + v.lw[j * v.n + pv] * (V8.lw.hi - V8.lw.lo) / 255 : V8.sky.lw,
    sw: hit ? v.sw[j * v.n + pv] * V8.sw.hi / 255 : V8.sky.sw,
  };
}

/** Targets for every slot: each ray from the standpoint to its first hit (or a stub toward the sky), each traced
 *  mirror's leg from that hit, the sun beam when the body sees the sun; slots not in use collapse to a point with no
 *  colour. `snap`: show the targets at once; otherwise stepRays eases toward them. */
function buildRays(snap) {
  const { i, g, v, pg, pv } = R.cur, NR = R.geo.n_rays, S = R.segs, Tp = R.tPos;
  const P = V.positions, ox = P[i * 3], oy = P[i * 3 + 1], oz = P[i * 3 + 2];
  for (let j = 0; j < NR; j++) {
    const len = rayLen(g.dist[j * g.n + pg], RAY_SKY_LEN);
    const ex = ox + R.dirs[j * 3] * len, ey = oy + R.dirs[j * 3 + 1] * len, ez = oz + R.dirs[j * 3 + 2] * len;
    Tp.set([ox, oy, oz, ex, ey, ez], j * 6);
    Object.assign(S[j], { len, on: true });
    Tp.set([ex, ey, ez, ex, ey, ez], (NR + j) * 6);              // the leg, off until a traced mirror says otherwise
    Object.assign(S[NR + j], { on: false, sky: false, len: 0 });
  }
  for (let k = g.legStart[pg]; k < g.legStart[pg + 1]; k++) {
    const j = g.legRay[k];
    if (!(v.flags[j * v.n + pv] & 1)) continue;   // this slot's engine did not trace the mirror (no own emission)
    const d = enuToView(...octDecode(g.legOct[2 * k], g.legOct[2 * k + 1]));
    const len = rayLen(g.legDist[k], RAY_LEG_SKY_LEN), s = (NR + j) * 6;
    Tp.set([Tp[s] + d[0] * len, Tp[s + 1] + d[1] * len, Tp[s + 2] + d[2] * len], s + 3);
    Object.assign(S[NR + j], { on: true, sky: !g.legDist[k], len });
  }
  const sun = R.val.sun_enu, lit = !!(sun && state.B.sunlit.at(i)), d = enuToView(...(sun ?? [0, 0, 1]));
  const L = lit ? RAY_SUN_LEN : 0;
  Tp.set([ox, oy, oz, ox + d[0] * L, oy + d[1] * L, oz + d[2] * L], 2 * NR * 6);
  S[2 * NR].on = lit;
  colourRays();
  R.obj.visible = $('l-rays').checked;
  if (snap) {
    const geom = R.solid.geometry;
    geom.attributes.position.array.set(Tp);
    geom.attributes.color.array.set(R.tCol);
    geom.attributes.position.needsUpdate = geom.attributes.color.needsUpdate = true;
    geom.computeBoundingSphere();
    R.anim = false;
  }
}

/** One frame of the easing (the render loop calls it): every slot moves a share of the way to its target. */
function stepRays(dt) {
  if (!R.anim || !R.obj) return;
  const f = 1 - Math.exp(-dt / RAY_EASE_S), geom = R.solid.geometry;
  const P = geom.attributes.position.array, C = geom.attributes.color.array, Tp = R.tPos, Tc = R.tCol;
  let left = 0;
  for (let k = 0; k < P.length; k++) { const x = Tp[k] - P[k]; P[k] += x * f; if (Math.abs(x) > left) left = Math.abs(x); }
  for (let k = 0; k < C.length; k++) { const x = Tc[k] - C[k]; C[k] += x * f; if (Math.abs(x) > left) left = Math.abs(x); }
  if (left < 1e-3) { P.set(Tp); C.set(Tc); R.anim = false; }
  geom.attributes.position.needsUpdate = geom.attributes.color.needsUpdate = true;
  geom.computeBoundingSphere();                    // the hover's raycast tests it first
}

/** The band the panel shows decides the value; the sidebar decides material / element / value colours; a hovered
 *  panel row (R.focus) keeps its own rays bright and dims the rest. Writes the colour targets; they ease in too. */
function colourRays() {
  if (!R.cur || !R.obj) return;
  const { g, pg } = R.cur, M = state.M, band = state.band, NR = R.geo.n_rays, S = R.segs, a = R.tCol;
  const mode = $('raymode').value;
  const aL = M.body.eps_p, aK = M.body.alpha_sw;
  const L3 = M.levels.l3, L2 = M.levels.l2;
  const skyMat = L3.labels.length - 1, skyElem = L2.labels.indexOf('Sky');
  const [lo, hi] = R.val.ranges[band];
  const f = R.focus;
  const face = f?.startsWith('d:') ? RAY_FACES[f.slice(2)] : null;
  for (let j = 0; j < NR; j++) {
    const c = g.code[j * g.n + pg], hit = c !== R.geo.code.sky;
    const m = hit ? c & 15 : skyMat, e = hit ? c >> 4 : skyElem;
    let rgb;
    if (mode === 'mat') rgb = linearRgb(L3.colors[m]);
    else if (mode === 'elem') rgb = linearRgb(L2.colors[e]);
    else {
      const val = rayValue(j);
      const x = band === 'lw' ? val.lw : band === 'sw' ? val.sw : aL * val.lw + aK * val.sw;
      rgb = linearRgb(rayColour(band, (x - lo) / (hi - lo)));
    }
    // brighter where the body weighs the ray more (sideways rays: F 0.22 faces; up/down: 0.06)
    let w = 0.45 + 0.55 * R.w[j] / R.wMax;
    if (f) {
      let on = 0;
      if (f.startsWith('m:')) on = +f.slice(2) === m ? 1 : 0;
      else if (f.startsWith('e:')) on = +f.slice(2) === e ? 1 : 0;
      else if (face) {
        const d = R.geo.directions[j];
        on = Math.max(0, d[0] * face[0] + d[1] * face[1] + d[2] * face[2]);
      }
      w = on > 0 ? 0.25 + 0.75 * on : 0.035;
    }
    a.set([...rgb, w, ...rgb, hit ? w : 0], j * 8);                         // escaping: fades out along its stub
    const leg = S[NR + j], wl = leg.on ? 0.8 * w : 0;
    a.set([...rgb, wl, ...rgb, leg.sky ? 0 : wl], (NR + j) * 8);
  }
  const sunA = S[2 * NR].on ? (!f || f === 'direct' ? 1 : 0.08) : 0;
  a.set([...linearRgb('#ffd23f'), sunA, ...linearRgb('#ffd23f'), sunA], 2 * NR * 8);
  R.anim = true;
}

function setRayFocus(key) {
  if (R.focus === key) return;
  R.focus = key;
  colourRays();
}

function updateRayLegend() {
  if (!R.val) return;
  const val = $('raymode').value === 'val';
  $('ray-legend').hidden = !val;
  if (!val) return;
  const band = state.band, [lo, hi] = R.val.ranges[band];
  const stops = Array.from({ length: 13 }, (_, k) => {
    const [r, g, b] = rayColour(band, k / 12);
    return `rgb(${r | 0},${g | 0},${b | 0}) ${(k / 12 * 100).toFixed(1)}%`;
  });
  $('ray-ramp').style.background = `linear-gradient(90deg, ${stops.join(',')})`;
  $('ray-lo').textContent = `${f0(lo)} W/m²`;
  $('ray-hi').textContent = `${f0(hi)} W/m²`;
  $('ray-band').textContent = { comb: 'Combined, a_l·longwave + a_k·shortwave', lw: 'Longwave',
                                sw: 'Shortwave (sunlight reflected or scattered)' }[band];
}

function renderRayNote() {
  const el = $('ray-note');
  if (!R.geo) return;
  if (!R.cur) { el.textContent = state.picked < 0 ? 'Pick a standpoint to draw its rays.' : ''; $('n-rays').textContent = ''; return; }
  const { g, v, pg, pv } = R.cur, NR = R.geo.n_rays;
  let nHit = 0, nMir = 0, near = Infinity;
  const lens = [];
  for (let j = 0; j < NR; j++) {
    const q = g.dist[j * g.n + pg];
    if (!q) continue;
    nHit++;
    const len = rayLen(q, 0);
    lens.push(len);
    near = Math.min(near, len);
    if (v.flags[j * v.n + pv] & 1) nMir++;
  }
  lens.sort((x, y) => x - y);
  const med = lens.length ? lens[lens.length >> 1] : NaN;
  $('n-rays').textContent = `${nHit} hit`;
  el.textContent = `${nHit} of ${NR} rays reach a surface (nearest ${f1(near)} m, median ${f1(med)} m); ` +
    `${NR - nHit} escape to the sky${nMir ? `; ${nMir} ${nMir === 1 ? 'lands' : 'land'} on glass or water, ` +
    `drawn with a second segment to what the mirror shows` : ''}. ` +
    'Brighter rays weigh more for a standing body. Drag the marker (its ring or centre) to move the standpoint. ' +
    'Hover a row in the panel to pick out its rays, or a ray for what it brings. Rays below the MRT surface show ' +
    'faintly; lower its opacity to see them in full.';
}

/** The ray under the pointer, as hover text (null: none). */
function rayHoverText(clientX, clientY) {
  if (!R.obj?.visible || !R.cur) return null;
  const { raycaster, camera, renderer, pointer } = V;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const P = V.positions, i = R.cur.i;
  const dCam = camera.position.distanceTo(new THREE.Vector3(P[i * 3], P[i * 3 + 1], P[i * 3 + 2]));
  raycaster.params.Line.threshold = Math.max(0.05, dCam * 0.004);
  const hit = raycaster.intersectObject(R.solid, false).find((h) => R.segs[Math.floor(h.index / 2)]?.on);
  if (!hit) return null;                           // slots not in use are collapsed points: never a hover
  const s = R.segs[Math.floor(hit.index / 2)];
  if (s.kind === 'sun') return '<i>Direct sun</i> the beam reaches the body (see the panel)';
  const { M, B, band } = state, { g, pg } = R.cur;
  const c = g.code[s.j * g.n + pg], val = rayValue(s.j);
  const L3 = M.levels.l3;
  const mat = val.hit ? (L3.pretty ?? L3.labels)[c & 15] : 'Sky';
  const aL = M.body.eps_p, aK = M.body.alpha_sw, w = R.w[s.j];
  const x = band === 'lw' ? val.lw : band === 'sw' ? val.sw : aL * val.lw + aK * val.sw;
  const total = band === 'lw' ? B.e_lw.at(i)
    : band === 'sw' ? B.e_sw_diffuse.at(i) + B.e_sw_direct.at(i) + B.e_sw_specular.at(i)
    : aL * B.e_lw.at(i) + aK * (B.e_sw_diffuse.at(i) + B.e_sw_direct.at(i) + B.e_sw_specular.at(i));
  const mirrored = R.cur.v.flags[s.j * R.cur.v.n + R.cur.pv] & 1;
  const dist = f1(s.kind === 'ray' ? s.len : rayLen(g.dist[s.j * g.n + pg], 0));
  const head = !val.hit ? '<b>Sky</b> <i>· the ray escapes</i>'
    : s.kind === 'leg' ? `<i>Mirrored by</i> <b>${mat}</b> <i>${dist} m away; this segment points to what it mirrors</i>`
    : `<i>Ray to</i> <b>${mat}</b> <i>${dist} m away</i>`;
  const into = { comb: 'the absorbed S_str', lw: 'the longwave total', sw: 'the shortwave total' }[band];
  return `${head}${mirrored ? '<br><i>glass/water: its own emission plus what it mirrors (second segment)</i>' : ''}<br>` +
    `<i>brings</i> ${f0(val.lw)} <i>longwave</i> · ${f0(val.sw)} <i>shortwave, W/m²</i><br>` +
    `<i>adds</i> <b>${(w * x).toFixed(2)}</b> W/m² <i>to ${into}</i> (${(100 * w * x / total).toFixed(2)} %)`;
}

async function loadCloud(meta) {
  const label = $('n-cloud');
  try {
    const buf = await fetchData('cloud.bin', (g, t) => {
      label.textContent = t ? `loading ${Math.round((g / t) * 100)}%` : 'loading…';
    });
    state.CB = unpack(buf, meta);
    V.cloud = buildCloud(meta, state.CB);
    V.cloud.pts.renderOrder = 0;
    V.cloud.pts.visible = $('l-cloud').checked;
    V.scene.add(V.cloud.pts);
    label.textContent = `${(meta.n_points / 1e6).toFixed(2)}M`;
  } catch (err) {
    console.error(err);
    label.textContent = 'could not load';
  }
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
    // Specular glints. The stored MRT includes them, so what a glint adds is the MRT minus the MRT
    // of the same flux without the intercepted glint: T = (S/sigma)^(1/4), S = sigma T^4 - k e_spec
    // with k = a_k / eps_p (view_factors.equivalent_temperature). Most standpoints get none and the
    // rest span three decades, so both layers are log-scaled and "none" is drawn grey.
    { id: 'glint_mrt', label: 'MRT added by glints', unit: 'K', cmap: 'YlOrRd', log: [0.01, 10],
      none: 'grey: no glint reaches this standpoint', digits: 2,
      values: derive((i) => {
        const e = B.e_sw_specular.at(i);
        if (!(e > 0)) return 0;
        const t = B.mrt_combined.at(i) + 273.15;
        return t - Math.pow(t ** 4 - (M_SWC() * e) / SIGMA, 0.25);
      }) },
    { id: 'glint_body', label: 'Glint on the body', unit: 'W/m²', cmap: 'YlOrRd', log: [0.1, 100],
      none: 'grey: no glint reaches this standpoint', digits: 1, values: B.e_sw_specular.data },
  ];

  // Attribution over the grid: what each standpoint's radiation comes FROM. The l3 blocks hold every
  // material's body-weighted contribution, longwave and shortwave apart; together they enter the MRT
  // flux as lw + (a_k/eps_p) sw -- the same sum the readout panel splits into shares. Glints are NOT
  // in it (they bypass the ray budget); their own two layers are above.
  const L3 = state.M.levels.l3;
  const names3 = L3.pretty ?? L3.labels;
  const mat = (i, k) => B.l3_lw.at(i, k) + M_SWC() * B.l3_sw.at(i, k);
  const iGlass = L3.labels.indexOf('glass');
  if (iGlass >= 0) {
    // What glazing adds, as MRT: the standpoint's MRT minus the MRT of the same flux without the
    // glass contribution -- the same construction as the glint layers, so the two are comparable.
    state.fields.push({
      id: 'glazing_mrt', label: 'MRT from glazing', unit: 'K', cmap: 'YlOrRd', log: [0.05, 20],
      none: 'grey: no glazing in this standpoint’s view', digits: 2,
      values: derive((i) => {
        const c = mat(i, iGlass);
        if (!(c > 0)) return 0;
        const t = B.mrt_combined.at(i) + 273.15;
        return t - Math.pow(Math.max(t ** 4 - c / SIGMA, 0), 0.25);
      }),
    });
  }
  // Dominant contributor: the material sending this standpoint the most. Coloured by the material's
  // own colour, the same key the point cloud and the readout use.
  state.fields.push({
    id: 'dominant_mat', label: 'Dominant material', unit: '', palette: L3.colors, classLabels: names3,
    values: derive((i) => {
      let best = 0, bestV = -Infinity;
      for (let k = 0; k < names3.length; k++) { const v = mat(i, k); if (v > bestV) { bestV = v; best = k; } }
      return best;
    }),
  });
  // UTCI from each standpoint's MRT, with this time's air, humidity and wind (web/comfort.json), coloured by
  // its stress category. NaN (grey) where the index is undefined.
  if (CF.slot) {
    const s = CF.slot;
    const u = derive((i) => utci(s.t_air_c, B.mrt_combined.at(i), s.wind_10m_m_s, s.rh_pct));
    const finite = u.filter(Number.isFinite);
    state.fields.push(
      { id: 'utci', label: 'UTCI', unit: '°C', cmap: 'YlOrRd', utci: true, digits: 1, values: u,
        range: finite.length ? rangeOf(finite) : [0, 1] },
      { id: 'utci_cat', label: 'UTCI, heat stress category', unit: '', bins: true, utci: true, values: u });
  }
  for (const f of state.fields) f.range ??= f.bins ? [0, 1] : f.log ?? rangeOf(f.values, f.exact);
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
/* Array sizes come from the manifest's label lists (Karlsruhe had 13 materials and 6 elements,
 * another site has its own vocabulary): a shader with fixed sizes would read past the end of
 * its colour and filter arrays for every class beyond them. */
const cloudVert = (nMat, nElem) => `
precision highp float;
// three.js declares 'position' itself, so it must not be redeclared here -- and the
// quantised coordinates have to live in it: THREE.Points takes its draw count from
// geometry.attributes.position, so under any other name nothing is drawn at all.
attribute vec3 aRgb;
attribute float aMat;
attribute float aElem;
attribute float aTemp;
attribute float aSvf;
attribute float aSw;

uniform vec3 uBoxLo, uBoxSpan, uOrigin;
uniform float uSize, uMode, uDpr;
uniform vec3 uMatColor[${nMat}];
uniform float uMatOn[${nMat}];
uniform vec3 uElemColor[${nElem}];
uniform float uElemOn[${nElem}];
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
  else if (uMode < 4.5) vColor = ramp(aSvf);
  else                  vColor = ramp(aSw);

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
  // reflected shortwave per point; a cloud built before it existed gets zeros (the menu hides it)
  g.setAttribute('aSw', attr(blocks.sw ? blocks.sw.data : new Uint8Array(n), 1, true));

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

  // Sky is in both label lists but unreachable here. The filter state (state.matOn/elemOn) is
  // set up in init() from the manifest, because the chips exist before the cloud has loaded;
  // the cloud starts from whatever the sidebar says by the time it arrives.
  const nMat = state.M.levels.l3.labels.length;
  const nElem = state.M.levels.l2.labels.length;

  const mat = new THREE.ShaderMaterial({
    vertexShader: cloudVert(nMat, nElem),
    fragmentShader: CLOUD_FRAG,
    uniforms: {
      uBoxLo: { value: new THREE.Vector3(...lo) },
      uBoxSpan: { value: new THREE.Vector3(...sp) },
      uOrigin: { value: new THREE.Vector3(...o) },
      uSize: { value: +$('ptsize').value },
      uMode: { value: state.cloudMode },
      uDpr: { value: Math.min(devicePixelRatio, 2) },
      uMatColor: { value: state.M.levels.l3.colors.map((c) => new THREE.Vector3(...linearRgb(c))) },
      uMatOn: { value: state.matOn.slice() },
      uElemColor: { value: state.M.levels.l2.colors.map((c) => new THREE.Vector3(...linearRgb(c))) },
      uElemOn: { value: state.elemOn.slice() },
      uRamp: { value: rampVectors(state.cloudRamp) },
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
  renderer.setClearColor(STAGE_COLOR());

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

  const cloud = null;   // streamed in afterwards by loadCloud()

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
  const hPed = state.M.body?.height_above_ground ?? 1.5;   // the standpoint height (1.1 m Dubai)
  const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, hPed, 10), ringMat);
  pin.position.y = -hPed / 2;
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 12), ringMat);
  marker.add(ring, pin, dot);
  for (const m of marker.children) m.renderOrder = 999;
  scene.add(marker);

  Object.assign(V, {
    renderer, scene, camera, controls, cloud, mrtMesh, mrtMat, marker, meta,
    positions: mrtGeom.attributes.position.array,
    raycaster: new THREE.Raycaster(), pointer: new THREE.Vector2(), mesh: null,
    markerTarget: new THREE.Vector3(),
  });

  // Framing, fog and zoom limits all key off the MRT surface's own size rather than fixed
  // numbers, so rebuilding the assets at a different extent cannot strand the camera.
  mrtGeom.computeBoundingBox();
  V.center = mrtGeom.boundingBox.getCenter(new THREE.Vector3());
  V.span = mrtGeom.boundingBox.getSize(new THREE.Vector3()).length();
  // Fog must be the stage colour or distant geometry fades to a colour the backdrop never has.
  scene.fog = new THREE.Fog(STAGE_COLOR(), V.span * 0.9, V.span * 3.2);
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

  let last = performance.now();
  renderer.setAnimationLoop(() => {
    const now = performance.now(), dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    controls.update();
    // A dragged marker eases to its standpoint at the same rate as the rays' origin, so the two stay together.
    if (!marker.position.equals(V.markerTarget)) {
      marker.position.lerp(V.markerTarget, 1 - Math.exp(-dt / RAY_EASE_S));
      if (marker.position.distanceTo(V.markerTarget) < 1e-3) marker.position.copy(V.markerTarget);
    }
    stepRays(dt);
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
  // log fields: t = log(v/lo) / log(hi/lo), clamped; v = 0 is "none", drawn grey
  const llo = field.log ? Math.log10(lo) : 0, lspan = field.log ? Math.log10(hi) - llo : 1;
  const attr = V.mrtMesh.geometry.attributes.color;
  const col = attr.array;
  // A categorical field carries its classes' own colours instead of a ramp.
  const hexRgb = (c) => (Array.isArray(c) ? c : [1, 3, 5].map((k) => parseInt(c.slice(k, k + 2), 16)));
  const pal = field.palette && field.palette.map(hexRgb);
  // UTCI: each value in its stress category's colour (utci.js), grey where the index is undefined
  const binRgb = field.bins && UTCI_CATEGORIES.map(([, , c]) => hexRgb(c));
  for (let i = 0; i < state.N; i++) {
    const v = field.values[i];
    const [r, g, b] = binRgb ? (Number.isFinite(v) ? binRgb[utciCategory(v).index < 0 ? binRgb.length - 1
      : utciCategory(v).index] : NONE_RGB)
      : pal ? (pal[v] ?? NONE_RGB)
      : !Number.isFinite(v) || (field.log && !(v > 0)) ? NONE_RGB
      : sampleCmap(field.cmap, field.log ? (Math.log10(Math.max(v, lo)) - llo) / lspan : (v - lo) * inv);
    col[i * 3] = srgbToLinear(r / 255) * 255;
    col[i * 3 + 1] = srgbToLinear(g / 255) * 255;
    col[i * 3 + 2] = srgbToLinear(b / 255) * 255;
  }
  attr.needsUpdate = true;

  // Categorical: swatches of the classes actually present here instead of the bar and its ticks.
  const sw = $('legend-swatches');
  sw.textContent = '';
  $('legend-bar').style.display = pal || binRgb ? 'none' : '';
  $('legend-bar').parentElement.querySelector('.ticks').style.display = pal || binRgb ? 'none' : '';
  if (binRgb) {
    const seen = new Set();
    for (const v of field.values) if (Number.isFinite(v)) seen.add(utciCategory(v).label);
    UTCI_CATEGORIES.forEach(([hi, label, color], k) => {
      if (!seen.has(label)) return;
      const lo = k ? UTCI_CATEGORIES[k - 1][0] : -Infinity;
      const range = !Number.isFinite(lo) ? `≤ ${hi}` : !Number.isFinite(hi) ? `> ${lo}` : `${lo} to ${hi}`;
      const el = document.createElement('span');
      el.className = 'swatch';
      el.innerHTML = `<i style="background:${color}"></i>${label} (${range} °C)`;
      sw.append(el);
    });
    const s = CF.slot;
    $('legend-note').textContent = `UTCI from each standpoint's MRT; air ${f1(s.t_air_c)} °C, RH ${f0(s.rh_pct)} %, ` +
      `wind ${f1(s.wind_10m_m_s)} m/s at 10 m, fixed for this time. Grey: outside the UTCI range.`;
    return;
  }
  if (pal) {
    const seen = new Set(field.values);
    field.classLabels.forEach((name, k) => {
      if (!seen.has(k)) return;
      const el = document.createElement('span');
      el.className = 'swatch';
      el.innerHTML = `<i style="background:${cssColor(field.palette[k])}"></i>${name}`;
      sw.append(el);
    });
    $('legend-note').textContent = 'the material sending this standpoint the most (glints not included)';
    return;
  }
  $('legend-bar').style.background = cmapCss(field.cmap);
  const d = field.log ? (lo < 1 ? String(lo).split('.')[1]?.length ?? 0 : 0)
    : field.unit === '°C' || field.unit === 'K' ? 1 : 0;
  $('legend-lo').textContent = field.discrete ? 'no' : `${lo.toFixed(d)} ${field.unit}`.trim();
  $('legend-hi').textContent = field.discrete ? 'yes' : `${field.log ? '≥ ' : ''}${hi.toFixed(field.log ? 0 : d)} ${field.unit}`.trim();
  $('legend-note').textContent = field.log ? `log scale · ${field.none}`
    : field.utci ? `air ${f1(CF.slot.t_air_c)} °C, RH ${f0(CF.slot.rh_pct)} %, wind ${f1(CF.slot.wind_10m_m_s)} m/s ` +
      `at 10 m, fixed for this time; 26-32 moderate, 32-38 strong, 38-46 very strong, > 46 °C extreme heat stress` : '';
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

/** Is (clientX, clientY) on the picked-standpoint marker: its centre dot or its ring band (not the ring's inside, so
 *  an orbit drag can still start next to the standpoint)? Sizes from the ring's own radii, projected. */
function overMarker(clientX, clientY) {
  if (!V.marker.visible || !V.mrtMesh.visible) return false;
  const { renderer, camera, marker } = V;
  const rect = renderer.domElement.getBoundingClientRect();
  const c = marker.position.clone().project(camera);
  const q = marker.position.clone().add(new THREE.Vector3(2.9 * marker.scale.x, 0, 0)).project(camera);
  const rOut = Math.hypot((q.x - c.x) / 2 * rect.width, (q.y - c.y) / 2 * rect.height);
  const rIn = rOut * 2.1 / 2.9;
  const d = Math.hypot(clientX - (rect.left + (c.x + 1) / 2 * rect.width), clientY - (rect.top + (1 - c.y) / 2 * rect.height));
  return d <= Math.max(14, rIn * 0.35) || (d >= rIn - 6 && d <= rOut + 6);
}

function bindPointer(canvas) {
  let moved = false, downAt = null, hoverJob = 0;
  // Dragging the marker moves the standpoint: the panel and the rays follow, the rays easing from each standpoint to
  // the next. Grabbing it switches the orbit off for that press (a capture listener runs before OrbitControls').
  let drag = null, dragJob = 0, dragAt = null;
  const endDrag = (e) => {
    const d = drag;
    drag = null;
    V.controls.enabled = true;
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    canvas.style.cursor = 'move';
    if (d.moved) { selectDone(state.picked); window.urfTrack?.('standpoint-drag'); }
    return d.moved;
  };
  canvas.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || !overMarker(e.clientX, e.clientY)) return;
    drag = { moved: false };
    V.controls.enabled = false;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
    $('hover').classList.remove('on');
  }, { capture: true });
  canvas.addEventListener('pointercancel', (e) => { if (drag) endDrag(e); });

  canvas.addEventListener('pointerdown', (e) => { moved = false; downAt = { x: e.clientX, y: e.clientY }; });
  canvas.addEventListener('pointermove', (e) => {
    if (downAt && Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4) moved = true;
    if (drag) {
      dragAt = { x: e.clientX, y: e.clientY };
      if (!dragJob) {
        dragJob = requestAnimationFrame(() => {
          dragJob = 0;
          if (!drag) return;
          const i = pickAt(dragAt.x, dragAt.y);
          if (i >= 0 && i !== state.picked) { drag.moved = true; select(i, true); }
        });
      }
      return;
    }
    // Raycasting 186,860 triangles is a few milliseconds -- fine per frame, wasteful per
    // pointer event, so coalesce into one job per animation frame.
    if (hoverJob) return;
    hoverJob = requestAnimationFrame(() => {
      hoverJob = 0;
      const el = $('hover');
      const place = () => {
        const r = canvas.getBoundingClientRect();
        el.style.left = `${e.clientX - r.left}px`;
        el.style.top = `${e.clientY - r.top}px`;
        el.classList.add('on');
      };
      // the marker first (every ray starts there), then the picked standpoint's rays: both are drawn over the surface
      if (!e.buttons && overMarker(e.clientX, e.clientY)) {
        el.classList.remove('wide');
        el.innerHTML = '<i>Drag to move the standpoint</i>';
        canvas.style.cursor = 'move';
        place();
        return;
      }
      const rt = e.buttons ? null : rayHoverText(e.clientX, e.clientY);      // not while orbiting
      el.classList.toggle('wide', !!rt);
      if (rt) { el.innerHTML = rt; canvas.style.cursor = 'default'; place(); return; }
      const i = pickAt(e.clientX, e.clientY);
      if (i < 0) { el.classList.remove('on'); canvas.style.cursor = 'grab'; return; }
      canvas.style.cursor = 'pointer';
      const f = state.field;
      const v = f.values[i];
      el.innerHTML = f.utci
        ? (Number.isFinite(v) ? `<i>UTCI</i> <b>${f1(v)}</b> °C · ${utciCategory(v).label}` : '<i>UTCI</i> outside its range')
        : f.palette ? `<i>${f.label}</i> <b>${f.classLabels[v] ?? '—'}</b>`
        : f.discrete
        ? `<i>${f.label}</i> <b>${v ? 'yes' : 'no'}</b>`
        : f.log && !(v > 0) ? `<i>${f.label}</i> <b>none</b>`
        : `<i>${f.label}</i> <b>${v.toFixed(f.digits ?? (f.unit === 'W/m²' ? 0 : 2))}</b> ${f.unit}`;
      const r = canvas.getBoundingClientRect();
      el.style.left = `${e.clientX - r.left}px`;
      el.style.top = `${e.clientY - r.top}px`;
      el.classList.add('on');
    });
  });
  canvas.addEventListener('pointerleave', () => $('hover').classList.remove('on'));
  canvas.addEventListener('pointerup', (e) => {
    if (drag && endDrag(e)) return;          // the marker was dragged; a press on it without moving is a click
    if (moved) return;                       // an orbit drag, not a click
    const i = pickAt(e.clientX, e.clientY);
    if (i >= 0) { select(i); window.urfTrack?.('standpoint-click'); }   // visits.js; absent = no-op
  });
}

/** Pick standpoint i. `dragging`: a step of a marker drag -- the marker and the rays ease there, the panel follows,
 *  and what can wait (the other times, the time links, the phone drawer) waits for the drag to end (selectDone). */
function select(i, dragging = false) {
  state.picked = i;
  const p = V.positions;
  V.markerTarget.set(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
  if (!dragging) V.marker.position.copy(V.markerTarget);
  V.marker.visible = true;
  $('hint').classList.add('gone');
  $('panel').classList.remove('empty');
  $('panel-body').hidden = false;
  renderPanel();
  renderGlintRays(i);
  renderRays(i, dragging);
  if (dragging) return;
  selectDone(i);
  // Phone: the budget is behind a drawer, so a tap on the surface brings it up.
  if (drawersActive()) openDrawer('panel');
}

function selectDone(i) {
  renderHistory(i);
  updateSlotLinks();
  ensureHistory().then(ensureComfort);      // the material switch needs both; small, fetched once
}

/* ---------------------------------------------------------------- panel */

/** The three levels for the selected standpoint, in the selected band.
 *  Level 1's stored values are PLANE irradiances; its contribution to the body is
 *  F_i times that, which is what the bars must show. Levels 2 and 3 are already
 *  body-weighted contributions, because each ray was folded in as w_r * value_r. */
function levelRows(i, band) {
  const { M, B } = state;
  // Combined = what the body absorbs, S_str = a_l * longwave + a_k * shortwave (VDI 3787, ISO 7726).
  const aL = M.body.eps_p, aK = M.body.alpha_sw;
  const mix = (lw, sw) => (band === 'lw' ? lw : band === 'sw' ? sw : aL * lw + aK * sw);

  const L1 = M.levels.l1, L2 = M.levels.l2, L3 = M.levels.l3;

  // Displayed north-first, the way the notebook's table reads; the blob's own order is
  // East, West, North, South, Up, Down.
  const dirOrder = ['North', 'East', 'South', 'West', 'Up', 'Down']
    .map((n) => L1.labels.indexOf(n)).filter((k) => k >= 0);

  // Direction bars show the irradiance ARRIVING on each plane, not F_i times it. The
  // angular factors are near-constant -- 0.22 on each horizontal plane, 0.06 up and down --
  // so the weighted contribution reads as four bars at ~22% and two at ~6% at every
  // standpoint in the scene, which is a picture of F_i rather than of the place. The
  // shortwave planes routinely span more than a factor of ten and all of that is erased.
  // F_i stays in MRT and is still shown on each row; it is only out of the bar length.
  const planeOf = (k) => {
    const lw = B.l1_lw.at(i, k), sw = B.l1_sw.at(i, k);
    return band === 'lw' ? lw : band === 'sw' ? sw : aL * lw + aK * sw;
  };
  const l1 = dirOrder.map((k) => {
    const fi = L1.factors[k];
    const plane = planeOf(k);
    return {
      name: L1.labels[k], color: L1.colors[k],
      value: plane,
      sub: `F ${fi.toFixed(2)} · contributes ${f1(fi * plane)} W/m²`,
      ray: `d:${L1.labels[k]}`,                 // hover: the rays this face sees, by their cosine
    };
  });

  const cat = (blockLw, blockSw, level, keyed, rayKey) => level.labels.map((name, k) => ({
    name: (level.pretty ?? level.labels)[k],
    color: level.colors[k],
    value: mix(blockLw.at(i, k), blockSw.at(i, k)),
    key: keyed ? `m:${k}` : undefined,          // the material switch (state.off)
    ray: `${rayKey}:${k}`,                      // hover: this category's rays in the view
  }));

  const l2 = cat(B.l2_lw, B.l2_sw, L2, false, 'e');
  const l3 = cat(B.l3_lw, B.l3_sw, L3, true, 'm').filter((r) => r.value > 0.005);
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
    if (direct > 0) extras.push({ name: 'Direct sun on body', value: mix(0, direct), extra: true, key: 'direct', ray: 'direct' });
    if (spec > 0) extras.push({ name: 'Sunlight reflected by glass and water', value: mix(0, spec), extra: true, key: 'glint', ray: 'glint' });
  }

  // Direction carries no extras: the direct beam and the glint are contributions to the
  // body, and this row is now plane irradiance, so listing them together would add two
  // different quantities. They are still shown on Element and Material, which are
  // partitions of the budget and do sum to its total.
  return [
    { title: L1.name, rows: l1, extras: [], irradiance: true },
    { title: L2.name, rows: l2.filter((r) => r.value > 0.005), extras },
    { title: L3.name, rows: l3, extras },
  ];
}

function renderPanel() {
  const i = state.picked;
  if (i < 0) return;
  const { M, B, band } = state;
  const aL = M.body.eps_p, aK = M.body.alpha_sw;

  const mrt = B.mrt_combined.at(i), mrtLw = B.mrt_longwave.at(i);
  const eLw = B.e_lw.at(i), eSwD = B.e_sw_diffuse.at(i);
  const direct = B.e_sw_direct.at(i), spec = B.e_sw_specular.at(i);
  const swAll = eSwD + direct + spec;
  // S_str, the radiation the body absorbs: T_mrt = (S_str / (eps_p sigma))^(1/4). Because a_l = eps_p, this is
  // the same MRT as (sum F_i L_i + (a_k / eps_p) sum F_i K_i) / sigma, the form the pipeline stores.
  const flux = aL * eLw + aK * swAll;

  $('mrt-value').textContent = (mrt < 0 ? '−' : '') + Math.abs(mrt).toFixed(2);

  const p = V.positions, o = V.meta.origin_enu;
  // Back out of the viewer's glTF frame into the project's ENU metres.
  const east = p[i * 3] + o[0], north = -p[i * 3 + 2] + o[1], up = p[i * 3 + 1] + o[2];

  const zero = (v) => (v > 0 ? '' : ' class="zero"');
  // Where the glint comes from: patch count and the strongest one (the lines drawn in the view).
  let glintFrom = '';
  const G = state.G;
  if (G && spec > 0) {
    const a = G.row_start.data[i], b = G.row_start.data[i + 1], k = G.pair_src.data[a];
    const d = Math.hypot(G.pos[k * 3] - p[i * 3], G.pos[k * 3 + 1] - p[i * 3 + 1], G.pos[k * 3 + 2] - p[i * 3 + 2]);
    const what = (M.levels.l3.pretty ?? M.levels.l3.labels)[G.src_mat.data[k]].toLowerCase();
    glintFrom = `<dt>Glint from</dt><dd>${b - a} patch${b - a > 1 ? 'es' : ''}; strongest: ${what}, ` +
                `${f0(d)} m away</dd>`;
  }
  $('readout-grid').innerHTML = `
    <dt>Longwave only</dt><dd>${mrtLw.toFixed(2)} °C</dd>
    <dt>Shortwave adds</dt><dd>${signed(mrt - mrtLw)} K</dd>
    <div class="sep"></div>
    <dt>Longwave E<sub>lw</sub></dt><dd>${f1(eLw)} W/m²</dd>
    <dt>Shortwave, diffuse + reflected</dt><dd>${f1(eSwD)} W/m²</dd>
    <dt>Direct sun on body</dt><dd${zero(direct)}>${f1(direct)} W/m²</dd>
    <dt>Sunlight reflected by glass and water</dt><dd${zero(spec)}>${spec > 0 && spec < 1 ? spec.toFixed(2) : f1(spec)} W/m²</dd>
    ${glintFrom}
    <dt>Absorbed, S<sub>str</sub></dt><dd>${f1(flux)} W/m²</dd>
    <div class="sep"></div>
    <dt>Position (E, N)</dt><dd>${east.toFixed(1)}, ${north.toFixed(1)} m</dd>
    <dt>Ground, above scene low</dt><dd>${(up - state.groundMin).toFixed(1)} m</dd>
    <dt>Sees the sun</dt><dd${zero(B.sunlit.at(i))}>${B.sunlit.at(i) ? 'yes' : 'no'}</dd>`;

  const total = band === 'lw' ? eLw : band === 'sw' ? swAll : flux;
  const unit = 'W/m²';
  const bandName = { comb: 'Combined', lw: 'Longwave', sw: 'Shortwave' }[band];

  const levelHtml = ({ title, rows, extras, irradiance }, inFold) => {
    const all = rows.concat(extras);
    const max = Math.max(...all.map((r) => r.value), 1e-9);
    const switchable = title === M.levels.l3.name && CF.slot;     // the Material level carries the switches
    const bars = all.map((r) => {
      const sw = switchable && r.key, off = sw && state.off.has(r.key);
      return `
      <div class="bar-row${r.extra ? ' extra' : ''}${!irradiance && r.value / total < 0.005 ? ' muted' : ''}${
        sw ? ' toggle' : ''}${off ? ' off' : ''}"${r.ray ? ` data-ray="${r.ray}"` : ''}>
        ${sw ? `<input type="checkbox" class="src-toggle" data-key="${r.key}"${off ? '' : ' checked'}
          title="Untick to replace it by a surface at air temperature" aria-label="${r.name}">` : ''}
        <span class="swatch"${r.color ? ` style="background:${r.color}"` : ''}></span>
        <div class="body">
          <div class="top">
            <span class="name">${r.name}</span>
            <span class="num"><b>${f1(r.value)}</b> ${unit}${
              irradiance ? '' : ` · ${pct(r.value / total * 100)}`}</span>
          </div>
          <div class="track"><div class="fill" style="width:${(r.value / max * 100).toFixed(1)}%;background:${r.color ?? 'var(--warn)'}"></div></div>
          ${r.sub ? `<div class="sub">${r.sub}</div>` : ''}
        </div>
      </div>`;
    }).join('');
    const caption = irradiance
      ? (band === 'comb' ? `Absorbed on each body face, a_l·L + a_k·K, before F_i`
                         : `Received by each body face · what a sensor facing that way reads`)
      : `${band === 'comb' ? 'Absorbed (S_str)' : bandName} total ${f1(total)} ${unit} · shares of that total`;
    const lead = switchable
      ? `<p class="switch-label">Untick a source: it is replaced by a neutral surface at air temperature (a black body
          at the air's temperature that reflects no sunlight; its shade stays). The box above then shows what it adds
          above that surface.</p>` : '';
    return `<section class="level">${inFold ? '' : `<h3>${title}</h3>`}
      <p class="total">${caption}</p>${lead}
      ${bars}</section>`;
  };
  const [l1, l2, l3] = levelRows(i, band);
  $('levels-material').innerHTML = levelHtml(l3, true);
  $('levels').innerHTML = levelHtml(l1, false) + levelHtml(l2, false);
  // one line in each fold's header, so a closed section still says something
  const top = (lv) => lv.rows.concat(lv.extras).filter((r) => r.name !== (M.levels.l3.pretty ?? M.levels.l3.labels).at(-1)
    && r.name !== 'Sky').sort((a, b) => b.value - a.value)[0];
  const t3 = top(l3), t2 = top(l2);
  $('sum-material').textContent = t3 ? `${t3.name} ${pct(t3.value / total * 100)}` : '';
  $('sum-levels').textContent = t2 ? `${t2.name} ${pct(t2.value / total * 100)}` : '';
  $('sum-details').textContent = `S_str ${f0(flux)} W/m² · sun ${B.sunlit.at(i) ? 'yes' : 'no'}`;

  $('formula').innerHTML =
    `MRT as VDI 3787 / ISO 7726, standing person: S<sub>str</sub> = a<sub>k</sub> Σ F<sub>i</sub> K<sub>i</sub> ` +
    `+ a<sub>l</sub> Σ F<sub>i</sub> L<sub>i</sub>, T<sub>mrt</sub> = (S<sub>str</sub> / (ε<sub>p</sub> σ))<sup>1/4</sup> − 273.15, ` +
    `with a<sub>k</sub> ${aK.toFixed(2)}, a<sub>l</sub> = ε<sub>p</sub> ${aL.toFixed(2)}, F<sub>i</sub> 0.22 sideways and 0.06 up and ` +
    `down; K<sub>i</sub> includes the direct sun and glints through f<sub>p</sub>. "Combined" shows S<sub>str</sub>. ` +
    `${f1(flux)} W/m² here gives ${mrt.toFixed(2)} °C.`;
  $('closure').textContent =
    `Element and Material are exact splits of the same budget: each sums back to the ` +
    `${bandName.toLowerCase()} total above. Direction shows what arrives on each plane ` +
    `instead, because the angular factors are near-constant and weighting by them makes ` +
    `every standpoint look alike; F_i is on each row and is still inside MRT. Values are ` +
    `transported quantised to ~0.005 W/m², so shares may round by a tenth.`;
  renderComfort(i);
  colourRays();            // the band may have changed
  updateRayLegend();
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
      if (V.cloud) V.cloud.mat.uniforms[hostId === 'chips-mat' ? 'uMatOn' : 'uElemOn'].value = onState.slice();
    });
    host.appendChild(b);
  });
}

function buildChrome() {
  const { M, C } = state;

  $('n-cloud').textContent = 'loading…';   // loadCloud() replaces it with the count
  if (C.temperature_label) $('cloudmode').querySelector('option[value="3"]').textContent = C.temperature_label;
  $('place').textContent = M.place ?? '';

  const sel = $('field');
  sel.innerHTML = state.fields
    .map((f, k) => `<option value="${k}">${f.label}${f.unit ? ` (${f.unit})` : ''}</option>`).join('');
  sel.addEventListener('change', () => {
    applyField(state.fields[+sel.value]);
    window.urfTrack?.(`field-${state.fields[+sel.value].id}`);
  });

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
      if (V.cloud) V.cloud.mat.uniforms[isMat ? 'uMatOn' : 'uElemOn'].value = arr.slice();
    });
  }

  // Cloud colour mode, and the ramp/legend that goes with the continuous ones.
  const RAMPS = { 3: ['inferno', 'temperature_range_c', '°C'], 4: ['viridis', null, ''],
                  5: ['YlOrRd', 'sw_range_w_m2', 'W/m²'] };
  if (!C.sw_range_w_m2) $('cloudmode').querySelector('option[value="5"]')?.remove();
  $('cloudmode').addEventListener('change', (e) => {
    const m = +e.target.value;
    state.cloudMode = m;
    if (V.cloud) V.cloud.mat.uniforms.uMode.value = m;
    const r = RAMPS[m];
    $('cloud-legend').hidden = !r;
    if (!r) return;
    const [cmap, rangeKey, unit] = r;
    state.cloudRamp = cmap;
    if (V.cloud) V.cloud.mat.uniforms.uRamp.value = rampVectors(cmap);
    $('cloud-ramp').style.background = cmapCss(cmap);
    const [lo, hi] = rangeKey ? C[rangeKey] : [0, 1];
    $('cloud-lo').textContent = `${lo.toFixed(1)} ${unit}`.trim();
    $('cloud-hi').textContent = `${hi.toFixed(1)} ${unit}`.trim();
  });

  $('ptsize').addEventListener('input', (e) => { if (V.cloud) V.cloud.mat.uniforms.uSize.value = +e.target.value; });
  $('surfop').addEventListener('input', (e) => {
    const v = +e.target.value / 100;
    // A fully opaque mesh left in the transparent pass sorts per-object rather than
    // per-pixel, which on a draped double-sided surface shows up as flicker.
    V.mrtMat.opacity = v;
    V.mrtMat.transparent = v < 1;
    V.mrtMat.depthWrite = v > 0.98;
    V.mrtMat.needsUpdate = true;
  });

  $('l-cloud').addEventListener('change', (e) => { if (V.cloud) V.cloud.pts.visible = e.target.checked; });
  $('l-glint').addEventListener('change', (e) => { if (V.glintLayer) V.glintLayer.visible = e.target.checked; });
  $('l-rays').addEventListener('change', (e) => { if (R.obj) R.obj.visible = e.target.checked; });
  $('raymode').addEventListener('change', () => { colourRays(); updateRayLegend(); });
  // Hovering a row of the budget picks out its rays in the view (material, element, a face's cosine, the sun beam).
  $('panel-body').addEventListener('pointerover', (e) => {
    setRayFocus(e.target.closest?.('.bar-row[data-ray]')?.dataset.ray ?? null);
  });
  $('panel-body').addEventListener('pointerleave', () => setRayFocus(null));
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
      const gltf = await parseGltf(await fetchData('scene.glb'));
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

  // Folds: open or closed as the visitor left them. Storage can be refused (private windows, blocked site
  // data), so every read and write is guarded and the defaults in index.html apply without it.
  const FOLD_KEY = 'thermal-clouds-folds';
  let folds = {};
  try { folds = JSON.parse(localStorage.getItem(FOLD_KEY) || '{}') || {}; } catch { folds = {}; }
  for (const d of document.querySelectorAll('details.fold')) {
    if (d.id in folds) d.open = !!folds[d.id];
  }
  $('panel-body').addEventListener('toggle', (e) => {        // toggle does not bubble: listen in the capture phase
    if (!e.target.matches?.('details.fold')) return;
    folds[e.target.id] = e.target.open;
    try { localStorage.setItem(FOLD_KEY, JSON.stringify(folds)); } catch { /* storage refused: fine */ }
  }, true);

  // The material switch: a change re-renders the panel with the source replaced (renderComfort).
  $('levels-material').addEventListener('change', (e) => {
    const t = e.target;
    if (!t.classList?.contains('src-toggle')) return;
    if (t.checked) state.off.delete(t.dataset.key); else state.off.add(t.dataset.key);
    renderPanel();
    window.urfTrack?.('source-switch');
  });

  for (const b of $('bands').querySelectorAll('button')) {
    b.addEventListener('click', () => {
      state.band = b.dataset.band;
      for (const o of $('bands').querySelectorAll('button')) o.classList.toggle('on', o === b);
      renderPanel();
    });
  }

  const c = M.conditions;
  $('when').textContent = siteTime(c.when);
  const sky = c.cloud_cover_tenths === 0 ? 'clear sky' : `cloud ${c.cloud_cover_tenths}/10`;
  $('conds').innerHTML =
    `Air ${c.t_atm_c.toFixed(1)} °C · RH ${f0(c.rh_pct)} % · ${sky}<br>` +
    `Sun ${f1(c.sun_elevation_deg)}° high, azimuth ${f0(c.sun_azimuth_deg)}°<br>` +
    `Global ${f0(c.ghi)} W/m² (direct ${f0(c.dni)}, diffuse ${f0(c.dhi)})`;
  $('conds-src').textContent = c.air_source ? `Weather: ${c.air_source}` : '';
  // A slot flown in two parts maps only the part belonging to this instant; say so, or the
  // missing half of the surface reads as a bug (manifest conditions.coverage, block.toml [web] note).
  $('conds-note').textContent = c.coverage || '';
  $('conditions').textContent =
    `One instant: ${c.note}. Sun ${c.sun_elevation_deg.toFixed(1)}° above the horizon, ` +
    `DNI ${f0(c.dni)} / DHI ${f0(c.dhi)} W/m², sky downwelling longwave ` +
    `${f0(c.sky_downwelling_lw)} W/m². Body: ${M.body.model}, ${M.body.n_rays} rays. ` +
    `Cloud: ${thousands(C.n_points)} points at ${C.voxel_m.toFixed(2)} m, ` +
    `from ${thousands(C.source_points)}.`;

  for (const b of document.querySelectorAll('#drawer-tabs button')) {
    b.addEventListener('click', () => openDrawer(b.dataset.drawer, true));
  }

  addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (document.body.className.includes('drawer-')) openDrawer(null);
      state.picked = -1;
      V.marker.visible = false;
      renderGlintRays(-1);
      renderRays(-1);
      updateSlotLinks();
      $('panel').classList.add('empty');
      $('panel-body').hidden = true;
    }
  });
}
