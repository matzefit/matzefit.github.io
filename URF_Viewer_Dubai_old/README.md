# URF MRT viewer: static site

A self-contained static web page: no server code, no build step. It works on any static
host, including GitHub Pages. three.js is loaded from the jsDelivr CDN; everything else
is in this folder.

```
web/
  index.html  app.js  colormaps.js  style.css
  .nojekyll            tells GitHub Pages to serve the files as they are (no Jekyll)
  visits.js            visitor counting, off until an ID is filled in (see below)
  data/                built by the pipeline (git-ignored in the URF repo)
    manifest.json        labels, colours, conditions, block layout of attribution.bin
    attribution.bin.gz   per-standpoint MRT and its three-level radiation budget
    mrt_surface.glb.gz   the pedestrian-height surface (vertex i = row i of attribution.bin)
    scene.json           shared origin of all layers
    cloud.json           layout of cloud.bin
    cloud.bin.gz         the semantic point cloud (RGB, material, element, temperature, SVF, shortwave)
    scene.glb.gz         grey backdrop mesh; fetched only when "Solid mesh" is switched on
    glints.json          layout of glints.bin
    glints.bin.gz        where each standpoint's glint comes from (glass/water patches)
```

## What the MRT surface can show

One field at a time, from the "MRT surface shows" list:

- **MRT, combined / longwave only / shortwave contribution**, and the longwave and shortwave
  irradiance on the body.
- **Direct sun on the body** — the shadow mask.
- **MRT added by glints** and **glint on the body** — the specular pathway, log-scaled, grey
  where no glint reaches the standpoint.
- **MRT from glazing** — how much of the standpoint's MRT comes from glass, longwave and
  shortwave together (13:00: median 0.6 K, p99 9.9 K, up to 68 K right in front of a facade).
  It is the MRT minus the MRT of the same flux without the glass contribution, the same
  construction as the glint layer, so the two are comparable. Glints are *not* in it: they are
  a separate pathway with its own two layers.
- **Dominant material** — which material sends this standpoint the most, in the material
  colours (13:00: sand 37 %, asphalt 27 %, grass 13 %, render 10 %, trees 8 %).

The last two come from the `l3` blocks already in `attribution.bin`, computed in the browser;
they need no rebuild of the data.

## Phones

Below 720 px the 3D stage fills the screen and the two side panels become drawers, opened from
a tab bar at the bottom ("Layers & field", "Radiation budget"); tapping a standpoint raises the
budget drawer. Above that width the layout is unchanged. The breakpoint lives only in
`style.css` — `app.js` asks for the tab bar's computed `display` rather than re-testing the
width, so the two cannot drift apart.

## Look

The MIT Senseable City Lab house style: pitch-black chrome (`--panel`), the 3D stage in a neutral
dark grey (`--stage`, `#2e2e2e` — deliberately not the panels' black, so the stage still reads as
its own surface; the guideline grey `#949494` and a mid `#7a7a7a` both washed out the pale facades
and the sand plots), Instrument Sans SemiBold from
Google Fonts, and the lab lockup (`assets/mit-scl-logo.svg`, white on transparent) pinned in the
bottom-left corner. `--stage` is the single source of the backdrop colour: `app.js` reads it for
the WebGL clear colour and the distance fog, which must match or far geometry fades to a colour
the page never shows. Swap the SVG for another approved variant and it is picked up as is.

## Load order and compression

The page opens as soon as the attribution and the MRT surface have arrived. The point
cloud, the largest file, then streams in behind the interactive view (the sidebar shows
its progress next to "Point cloud").

The binary files are stored gzipped (`reconstruction/web_io.py`). GitHub Pages does not
compress binaries itself, so this cuts the first download by more than a third. The viewer
fetches `<name>.gz` and decompresses it in the browser. It falls back to a plain `<name>` if
there is no `.gz`, and it also works if a host has already decoded the file.

## Rebuilding `data/`

The last steps of `scripts/mrt_chain.py`, or by hand (with `URF_BLOCK` set):

```
python scripts/precompute_attribution.py   # manifest.json, attribution.bin
python scripts/make_web_assets.py          # mrt_surface.glb, scene.glb, scene.json
python scripts/make_point_cloud.py         # cloud.bin, cloud.json
python scripts/glint_sources.py           # glints.bin, glints.json (after make_web_assets)
```

Sizes are set in `site.toml` `[web]`:
- `mrt_grid_stride`: the viewer shows every n-th standpoint of the 1 m analysis grid.
  The values are the exact grid values, not averages.
- `point_cloud_points`: the cloud's size.
- `scene_voxel_m`: the backdrop mesh's size.
- `place`: the title.

The conditions under the title come from the block's `block.toml` `[atmosphere]`, including its
`source` line: air temperature and humidity, the sky, the sun, and global, direct and diffuse
radiation. When the Kestrel data replaces the NCM stand-in, update that section and rerun the
pipeline from the thermal fusion. Changing only the text needs just `precompute_attribution.py`.

`precompute_attribution.py` and `make_web_assets.py` must use the same stride. Both read it
from `site.toml`, and `make_web_assets.py` checks that the vertex and row counts match.

## Time slots

Each time slot has its own data folder:
- the reference slot (13:00) in `data/`;
- other slots in `data_<hhmm>/`, set by `block.toml` `[web] data_dir` (e.g. `web/data_1100`).

`precompute_attribution.py` lists every slot it builds in `slots.json`. When there are two or
more, the title card shows a time switcher. `?slot=1100` opens that slot; only digits are
accepted, so the query can't point anywhere else. All slots share one geometry
(`scripts/register_slot.py`), so a standpoint is the same place in each. Each folder is about
83 MB; three slots stay far below the 1 GB site limit.

## Try it locally

`fetch()` does not work from `file://`, so serve the folder:

```
python -m http.server 8000 --directory web
```

then open <http://localhost:8000/>.

## Publish on GitHub Pages

1. Copy the whole `web/` folder into your Pages repository: its root, or `docs/`. Include
   `data/` and the hidden `.nojekyll`.
2. Commit the data files as ordinary git files, **not Git LFS**: Pages serves LFS pointer
   files, not their contents.
3. GitHub rejects files over 100 MB and warns above 50 MB, and a Pages site should stay
   under 1 GB. The `[web]` settings keep every file under 50 MB (see the size table below).
4. In the repository settings, go to Pages and choose the branch and folder. Relative paths
   (`./data/...`) work under a project URL such as `https://<user>.github.io/<repo>/`.

Updating: rebuild `data/`, copy it over, and commit. Browsers cache by URL; a hard reload
(Ctrl+F5) shows the new data straight away.

## Glints: where they come from

`scripts/glint_sources.py` repeats notebook 06's glint pass on the viewer's standpoints and records
every source patch → standpoint glint. It checks that these sum to the stored grid's value.

- **Click a standpoint that receives a glint:** yellow lines run to the glass or water patches that
  send it. The panel shows how many patches there are, and the strongest one's material and distance.
- **"Glint sources" layer:** every such patch, coloured cyan to white by the glint it delivers to the
  whole grid (log scale, 0.1–100 W/m²).

A source is a single point of the cloud, with that point's own normal. Because those normals are
noisy, each point acts as a tiny tilted mirror. That is why a weak glint can reach open ground from
a facade patch more than a kilometre away. The 13:00 slot:

- 19,616 source patches (19,129 glass, 487 water);
- the median lit standpoint gets its glint from one patch.

## Visitor counting

GitHub Pages reports nothing about who opens a site, so counting goes through a service.
`visits.js` supports two. It is off until you fill in **one** ID at its top and republish:

- **GoatCounter** (recommended):
  - free for non-commercial use and open source;
  - sets no cookies, so no consent banner is needed;
  - reports visits, country and region, phone/tablet/desktop, browser, operating system and
    referrer.

  To set it up:
  1. Sign up at https://www.goatcounter.com and pick a code, e.g. `urf-dubai`.
  2. Set `GOATCOUNTER_CODE = 'urf-dubai'`.
  3. The dashboard is then https://urf-dubai.goatcounter.com.
- **Google Analytics 4**:
  - more detail (city, device model);
  - but it sets cookies (EU/UK visitors then need a consent banner), and ad blockers stop
    it more often.

  Create a GA4 property with a web data stream and set `GA4_ID = 'G-…'`.

Both report aggregates, never individual people. Ad blockers hide a share of visits, so read
the numbers as a lower bound. Visits from `localhost` are never counted. Besides page views,
two interactions are counted as events: clicking a standpoint (`standpoint-click`) and
switching the MRT field (`field-<id>`).

`visits.js` loads separately from the viewer: if a blocker refuses it, the page still works.

## Current build

Dubai, 13:00 slot, rebuilt 2026-09-29. Changes since 2026-09-27:
- glass with Dubai glazing values;
- glass glints traced as continuous beams off fitted facade planes;
- glass and water mirror their surroundings in the MRT;
- no standpoints on the Greens lake.

- **MRT surface:** 248,856 standpoints, every 2nd cell of the 1 m grid. That grid covers
  995,192 standpoints after dropping those without thermal coverage and those on the lake.
- **Point cloud:** 3,538,624 points, 1.19 m voxel.
- **Backdrop mesh:** 3 m clustering.

| file | size | loaded |
|---|---|---|
| attribution.bin.gz | 22.5 MB (33.1 raw) | first |
| mrt_surface.glb.gz | 3.2 MB (8.1 raw) | first |
| cloud.bin.gz | 38.3 MB (49.5 raw) | streamed after the page opens |
| glints.bin.gz | 0.2 MB (2.2 raw) | after the page opens |
| scene.glb.gz | 18.7 MB (27.4 raw) | only when "Solid mesh" is switched on |
| *.json | < 10 kB each | first |

Total 83 MB. Load times, measured in headless Edge with 40 ms latency on the 85 MB build of
2026-09-27; the files are about the same size:

| connection | page usable | point cloud in |
|---|---|---|
| 30 Mbit/s | 12.8 s | 35.1 s |
| local, unthrottled | 2.1 s | 15.5 s |

The 2026-09-29 build passed the same headless feature and glint-source tests, with no page errors.

Opening the solid mesh adds 18.7 MB.
