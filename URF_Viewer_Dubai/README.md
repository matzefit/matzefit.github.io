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
    cloud.bin.gz         the semantic point cloud
    scene.glb.gz         grey backdrop mesh; fetched only when "Solid mesh" is switched on
```

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
```

Sizes are set in `site.toml` `[web]`:
- `mrt_grid_stride`: the viewer shows every n-th standpoint of the 1 m analysis grid.
  The values are the exact grid values, not averages.
- `point_cloud_points`: the cloud's size.
- `scene_voxel_m`: the backdrop mesh's size.
- `place`: the title.

`precompute_attribution.py` and `make_web_assets.py` must use the same stride. Both read it
from `site.toml`, and `make_web_assets.py` checks that the vertex and row counts match.

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

Dubai, 13:00 slot, built 2026-09-27:

- **MRT surface:** 258,558 standpoints, every 2nd cell of the 1 m grid. That grid covers
  1,034,021 standpoints after dropping those without thermal coverage.
- **Point cloud:** 3,538,624 points, 1.19 m voxel.
- **Backdrop mesh:** 3 m clustering.

| file | size | loaded |
|---|---|---|
| attribution.bin.gz | 23.4 MB (34.4 raw) | first |
| mrt_surface.glb.gz | 3.4 MB (8.4 raw) | first |
| cloud.bin.gz | 35.8 MB (46.0 raw) | streamed after the page opens |
| scene.glb.gz | 18.7 MB (27.4 raw) | only when "Solid mesh" is switched on |
| *.json | < 10 kB each | first |

Total 81 MB. Measured in headless Edge, 40 ms latency:

| connection | page usable | point cloud in |
|---|---|---|
| 30 Mbit/s | 12.9 s | 36.7 s |
| local, unthrottled | 2.1 s | 15.6 s |

Opening the solid mesh adds 18.7 MB.
