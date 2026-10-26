# Thermal Clouds viewer: Abu Dhabi street-video demo

A copy of the Dubai viewer (`web/`, its README explains every layer, field and panel) for the Hamdan Street block
(`upscaling/data/street_video/hamdan_s2`). Same code and data format; changed here only:
- the title and description, and a line under the conditions saying what the page is: an internal demo, geometry
  from a public driving video placed with OpenStreetMap, longwave **predicted** by the Dubai model, not measured;
- no visitor counting (`visits.js` is not copied).

**Internal only.** The video's rights are not cleared (`../README.md`); do not publish this folder.

## Open it

```
python -m http.server 8001 --directory upscaling/street_video/viewer
```

then <http://localhost:8001/>.

## Data

`data/` (not in git) is written by the last steps of `upscaling/street_video/chain.py`, which runs the Dubai export
scripts on the block (`block.toml` `[web] data_dir = "upscaling/street_video/viewer/data"`):
`precompute_attribution.py` (manifest, attribution), `make_web_assets.py` (MRT surface, backdrop mesh, scene),
`glint_sources.py`, `make_point_cloud.py` (the cloud: colour, material, element, predicted brightness temperature,
sky view, shortwave). `slots.json` next to this README is written by the attribution step (one time, so no switcher).
