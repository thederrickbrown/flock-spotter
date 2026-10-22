# Flock Spotter — Technical Spec

**Owner:** Derrick
**Status:** Draft, pre-implementation
**Relationship to other work:** Upstream of the ALPR-avoidance routing app. That project is blocked on trustworthy camera data; this project produces it.

---

## 1. What this is

An offline pipeline that turns dashcam video + a GPS track into geolocated ALPR camera positions, reconciled against what OpenStreetMap already knows.

**In scope (v1):**
- Batch processing of recorded drives. Record now, process later.
- Detection of pole-mounted Flock Safety units and visually similar ALPRs
- Triangulation of detections into lat/lon positions
- Reconciliation against an OSM seed layer → confirmed / new / stale
- A human review step before anything is trusted

**Out of scope (v1):**
- Real-time on-device inference
- Determining which direction a camera faces (see §9)
- Multi-driver fleet management, bounty payouts, contributor tooling
- Automated upload to OSM
- The routing app itself

**Explicit non-goal:** a real-time driving alert. Anything that encourages looking at a screen while driving is out.

---

## 2. Pipeline

```
  capture (video + GPX)
        │
        ▼
  frame extraction @ 2fps ──────────► geotag frames (GPX interpolation)
        │                                     │
        ▼                                     │
  detector (YOLO) ──► tracker (per-camera track IDs)
        │                                     │
        ▼                                     ▼
  bearing computation ◄─────────── position + heading per frame
        │
        ▼
  triangulation (least-squares ray intersection, per track)
        │
        ▼
  clustering (DBSCAN across tracks/drives)
        │
        ▼
  reconciliation vs OSM seed ──► confirmed / new / stale
        │
        ▼
  human review UI ──► verified GeoJSON
```

---

## 3. Capture

### Hardware
Phone or dashcam, windshield-mounted, **fixed rigid mount**. The mount must not shift between calibration and driving — the yaw offset (§6) is baked in per-configuration.

### Recording
- Video: 1080p or better, 30fps, wide but not fisheye. If fisheye, undistortion is required before bearing math.
- GPS: 1Hz minimum, GPX or CSV, recorded simultaneously.

### Time sync
The single most common source of silent error. Video timestamps and GPX timestamps must share a clock.

**Method:** at the start of each drive, record the phone's GPS clock display on video for a few seconds, or hard-stop the car for 10s and align the video's motion-start with the GPX's speed-nonzero transition. Store the resulting offset in the drive manifest.

A 1-second sync error at 40mph is an 18m position error on every detection in that drive.

### Drive manifest
Each drive produces a JSON sidecar:
```json
{
  "drive_id": "2026-08-20-clairemont-01",
  "video": "raw/2026-08-20-clairemont-01.mp4",
  "gpx": "raw/2026-08-20-clairemont-01.gpx",
  "time_offset_sec": -2.4,
  "camera_config": "iphone15-wide-mount-a",
  "notes": "afternoon, clear"
}
```

---

## 4. Frame extraction and geotagging

Extract at 2fps — enough baseline between frames for triangulation, cheap enough to process.

```bash
ffmpeg -i drive.mp4 -vf fps=2 -q:v 2 frames/%06d.jpg
```

Each frame's timestamp = `frame_index / 2.0 + time_offset_sec`.

Interpolate position from the GPX (linear between fixes is fine at 1Hz). Heading comes from the bearing between consecutive GPX points, **not** from the device compass — magnetometer readings inside a car are unreliable.

Discard frames where speed < 3 m/s. Stationary frames give no triangulation baseline and heading-from-GPX is garbage at low speed.

---

## 5. Detection model

### Seed the label set from OSM

Before labeling anything, pull existing ALPR nodes so you know where cameras already are:

```
[out:json][timeout:60];
node["man_made"="surveillance"]["surveillance:type"="ALPR"]
  (32.5,-117.6,33.5,-116.1);
out body;
```

Drive routes that pass known cameras. Frames near a known node are high-probability positives — that's your bootstrap set, and later your recall benchmark.

### Auto-label, then verify

1. Run a VLM over candidate frames with a tight prompt asking for bounding boxes around pole-mounted ALPR units. Expect ~70% usable.
2. Correct by hand in CVAT or Label Studio. This step is non-optional and is most of the initial time cost.
3. Target 300–500 verified positive images for a first model.

### Classes

| Class | Purpose |
|---|---|
| `alpr_flock` | Flock Safety units — slim mast, rectangular head, solar panel, battery box |
| `alpr_other` | Motorola/Vigilant, Genetec, and similar |
| `negative_pole` | Traffic signal cams, cell-site gear, residential security cams |

`negative_pole` is a deliberate hard-negative class. Without it, false positives on utility infrastructure will dominate.

### Training
YOLOv8n or YOLO11n, fine-tuned from COCO weights. Should train in well under an hour on a rented GPU. Export to ONNX for batch inference; CoreML/TFLite export only matters if real-time is revisited later.

### Metrics that matter
- **Recall against the OSM seed set** — of N known cameras driven past, how many were detected in at least one frame? This is the number that determines whether the pipeline is useful.
- Precision on `negative_pole`-heavy footage
- Per-track detection count (a camera seen in only one frame can't be triangulated)

---

## 6. Bearing computation

For each detection in each frame:

```
f_px = (W / 2) / tan(θ_h / 2)          # focal length in pixels
α    = atan2(u - W/2, f_px)            # angular offset from optical axis
β    = normalize(h + yaw_offset + α)   # absolute bearing to target
```

Where `W` = image width, `θ_h` = horizontal FOV, `u` = horizontal center of the bounding box, `h` = vehicle heading from GPX, `yaw_offset` = camera mount rotation relative to vehicle forward.

### Calibration (per camera_config)

**θ_h:** from EXIF/device specs, or measured — photograph an object of known width at known distance and solve.

**yaw_offset:** drive straight at a fixed distant landmark, note the pixel column it sits in when the vehicle heading equals the true bearing to it, solve for offset. A few degrees of error here translates to meters of position error, so measure it rather than assuming zero.

Store both in a config file keyed by `camera_config`, not hardcoded.

---

## 7. Triangulation

A single frame gives a ray, not a position. Two or more rays from different vehicle positions intersect at the camera.

### Track association
Run a lightweight tracker (SORT/ByteTrack, or IoU-matching between consecutive frames) so detections of the same physical camera across frames share a `track_id`. Triangulate per track.

### Least-squares closest point to N rays

Work in a local ENU frame (meters). For each ray `i` with origin `pᵢ` and unit direction `dᵢ` (from bearing β, east/north components `[sin β, cos β]`):

```
Aᵢ = I - dᵢ dᵢᵀ
A  = Σ Aᵢ
b  = Σ Aᵢ pᵢ
x  = A⁻¹ b
```

`x` is the point minimizing squared perpendicular distance to all rays. Convert back to lat/lon.

### Rejection criteria

Discard a track if any of:
- Fewer than 3 detections
- Angular spread between first and last ray < 15° — near-parallel rays produce wildly unstable intersections. This is the main failure mode. A camera dead ahead that you drive straight toward triangulates terribly; a camera you pass, so it drifts to the frame edge, triangulates well.
- Condition number of `A` above threshold
- Resulting point > 60m from the road, or behind the vehicle's path
- Residual (mean perpendicular distance from `x` to the rays) > 8m

Log rejections with reasons. The rejection log is a debugging tool, not noise.

---

## 8. Clustering and reconciliation

### Cluster
DBSCAN over all triangulated points, `eps ≈ 15m`, `min_samples = 1`. Multiple tracks (from multiple passes or multiple drives) collapse into one candidate. Cluster centroid = candidate position; cluster size = confidence signal.

### Reconcile against OSM seed

| Bucket | Condition | Meaning |
|---|---|---|
| **Confirmed** | Candidate within 25m of an OSM ALPR node | Detector works; node validated |
| **New** | Candidate with no OSM node within 25m | The actual product |
| **Stale** | OSM node on a driven road segment, no candidate within 25m, driven ≥2 times | Possibly removed — needs eyes |

Stale detection requires knowing which segments were actually driven. Buffer the GPX track by ~40m and intersect with seed nodes.

### Output
`candidates.geojson` with properties: `status`, `cluster_size`, `n_tracks`, `mean_residual_m`, `source_drives`, `first_seen`, `last_seen`, `review_status`.

---

## 9. Human review

Nothing is trusted until a human confirms it. The review UI is minimal:

- Map of candidates colored by status
- Click a candidate → the best 3 source frames with bounding boxes drawn
- Buttons: **Confirm** / **Reject** / **Not sure**
- Confirmed candidates get a manual facing-direction entry (compass dial)

Facing direction is entered by hand in v1. Estimating it from imagery is a genuinely harder problem than position, and for a few hundred cameras a human clicking a dial is faster than solving it.

Facing direction is what the routing app's directional-edge model depends on, so this field is required before a camera is usable downstream.

---

## 10. Repo layout

```
flock-spotter/
├── config/
│   ├── cameras.yaml           # per-mount FOV + yaw_offset
│   └── regions.yaml           # bboxes for Overpass
├── data/
│   ├── raw/                   # video + gpx (gitignored)
│   ├── frames/                # extracted, geotagged (gitignored)
│   ├── labels/                # YOLO format
│   └── output/                # candidates.geojson
├── src/
│   ├── capture/manifest.py
│   ├── prep/extract_frames.py
│   ├── prep/geotag.py
│   ├── seed/overpass.py
│   ├── label/vlm_prelabel.py
│   ├── detect/infer.py
│   ├── detect/track.py
│   ├── geo/bearing.py
│   ├── geo/triangulate.py
│   ├── geo/cluster.py
│   ├── reconcile/compare_osm.py
│   └── review/app.py
├── tests/
└── notebooks/
```

Python 3.11+. Core deps: `ultralytics`, `opencv-python`, `gpxpy`, `numpy`, `scipy`, `shapely`, `geopandas`, `requests`.

---

## 11. Milestones

**M1 — Geometry harness (no ML)**
Frame extraction, GPX interpolation, bearing math, triangulation, clustering. Validated with **hand-labeled** detections on one drive past 5 physically-verified cameras.

*Exit criterion: triangulated positions land within 15m of surveyed truth.* If the geometry is wrong, no amount of detector accuracy saves it — this is why it comes first.

**M2 — Detector**
Seed from OSM, prelabel, hand-verify 300–500 images, train, evaluate.

*Exit criterion: ≥80% recall against a held-out set of known OSM cameras driven past.*

**M3 — End-to-end + review**
Wire M1 and M2 together, add reconciliation and the review UI, process a full multi-hour San Diego drive.

*Exit criterion: at least one genuinely new camera found, confirmed by driving to it and looking at it.*

**M4 — Coverage strategy**
Route planning that prioritizes high-yield segments (arterial intersections, highway ramps, HOA and business-park entrances, mall/hospital/transit approaches) over exhaustive grid coverage. Re-drive scheduling.

---

## 12. Open questions

- **Undistortion needed?** Depends on lens. Test: photograph a grid, check straight-line fidelity at frame edges. If it bends, undistortion goes in before bearing math.
- **2fps enough?** May need 4fps on fast roads to get 3+ detections per track. Make it a config value from the start.
- **Night and rain performance** — unknown, likely poor. Restrict v1 to daylight, dry.
- **Trailer-mounted units** — different silhouette, mobile, no stable position. Probably a separate class that gets flagged but never written as a permanent node.
- **Verify DeFlock's current OSM tag schema** before writing the Overpass parser. It has changed more than once and my knowledge of it is not current.
- **Purpose-built capture app?** A single phone app that records video and logs GPS simultaneously and stamps `time_offset_sec` into the manifest automatically would eliminate the pipeline's single biggest silent error source (§3 time sync). It doesn't violate the real-time non-goal — it's a recorder with a shared clock, not an alert. Deliberately deferred until M1/M3 prove the pipeline produces trustworthy positions; until then, capture is the §3 procedure with stock apps (same phone for both = shared clock, which already gets the offset to ~1s from file metadata).

---

## 13. Notes on legality and conduct

Recording public fixtures from a vehicle on public roads is legal in California. These cameras are visible from the roadway and many are already publicly documented.

Two operational rules:
1. Mount the phone, start recording before pulling out, don't touch it until parked.
2. Don't trespass onto private property to get a better angle on an HOA or business-park camera. The road is enough.

Data derived from OSM is ODbL — share-alike on derived databases. Contributing verified new cameras back with the DeFlock tag schema is both correct practice and the flywheel that makes the dataset better than any private scrape.
