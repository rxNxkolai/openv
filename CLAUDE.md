# Patron

Agentic shopper-behavior intelligence that plugs into a store's existing security cameras.
Cameras are the sensor. The agent layer is the product.

## Hard constraints (non-negotiable, check before adding any dependency or feature)

### 1. No AGPL dependencies. Ever.

Patron ships as closed-source commercial software. An AGPL-3.0 dependency forces us to
open-source the entire product or buy a commercial license, and it will blow up in
technical due diligence.

**Banned:** `ultralytics` (any YOLO v5/v8/v11/v26 from Ultralytics), and anything
transitively pulling it in.

**Approved:** RF-DETR and D-FINE (Apache 2.0), supervision (MIT), ByteTrack and BoT-SORT
(MIT), MMPose / RTMPose (Apache 2.0), MediaPipe (Apache 2.0), Frigate (MIT),
OpenCV (Apache 2.0).

Before adding a dependency, check its license. If it is AGPL or GPL, find another one.

### 2. Privacy posture is structural, not a setting

This is what keeps us out of BIPA and EU AI Act exposure. Target and Home Depot are in
active BIPA litigation over exactly this category. Damages run $1,000 to $5,000 per
violation.

- **Never** compute, store, or transmit face embeddings or any biometric identifier.
- **Never** persist raw video or raw frames beyond the processing buffer.
- Track IDs are session-scoped. They die when the person leaves frame. No re-identification
  across sessions, across days, or across stores.
- Only anonymized event JSON leaves the edge.
- No emotion or micro-expression inference. It is scientifically contested and legally
  radioactive under EU AI Act Recital 44 and Art. 5.

If a feature request needs any of the above, it is a product decision for Nikolai, not an
implementation detail to quietly add.

### 3. Build the solved 80%, not the demo-flashy 20%

Traffic, path, dwell, zone interactions and the pass-by to pickup funnel are buildable now
and are where the value is. SKU-level identification and gaze are not reliable from a
ceiling camera and are deliberately out of scope until Phase 3 hardware.

## Architecture seam

`pipeline.py` yields a normalized `FrameResult` stream. Everything downstream (zones,
events, agent) consumes that stream and nothing else. Keep that seam clean, it is what
lets the detector swap out without touching the product layer.

Person ground position is `Box.foot_point` (bottom-center), not box center. Zone
membership is always computed on the floor plane.

`floor.py` turns image pixels into store floor coordinates through a single
homography, because the floor is a plane and stores already have floor plans.
That decision to anchor everything on the floor plane is what makes it work with
no rework, and it is the layer a multi-camera bird's-eye view sits on: several
cameras sharing one coordinate space. Only foot points project, points above the
horizon return `None` rather than a plausible lie, and **four correspondences
report no reprojection error at all** because eight equations for eight unknowns
fit exactly whatever the points are. The fifth point is the only automatic check
that the calibration is on the ground plane; `floorview.rectify` is the manual
one.

Stitching a shopper across cameras is a form of re-identification, so it collides
with constraint 2. Spatial-temporal handoff on the floor plane (a track leaving
one frustum where another enters) needs no appearance model and keeps the posture
intact. Appearance-based re-ID does not. That is a product decision, not an
implementation detail.

## Milestones

- **M0** video in, people detected, stable track IDs out  (done)
- **M1** zone polygons, visit records with dwell, per-zone funnel  (done, SQLite)
- **M2** pose and reach detection  (done, true-positive validated on real footage)
- **M3** analysis + agent, diagnosis and recommendation  (done, agent path unrun)
- **M4** live camera end to end  (done for the software path; unrun on a real
  fixed camera, which is the outstanding gate)

`patron live` writes to the same event store the offline pipeline does, so a live
session is analysable afterwards with the same `patron analyze`. It also accepts a
video file as its source, which is what makes the live path testable and demoable
with no camera present. **Looping a file into the event store double-counts
shoppers on every pass** — fine for a demo, meaningless as a measurement, and the
CLI warns rather than producing confident nonsense.

**The agent never computes and never executes.** `analysis.py` computes every number
deterministically and `agent.py` only writes prose over the result. A model doing
arithmetic on raw rows would be unauditable, and a retailer argues with the numbers
before the advice. Recommendations are stored with status `proposed`; no code path
sets `approved`, because approval is the liability gate.

**Patron knows zones, not products.** It cannot say "move SKU-204" — that needs
SKU-level identification, which is out of scope until Phase 3 hardware. The agent's
system prompt forbids inventing SKUs, brands, or prices to paper over that gap.

The deterministic layer must keep working with no API key. It is the floor of the
product, and `patron analyze` is the command that proves it.

Zone `kind` decides which body point membership is tested against, and that is the
whole difference between a visit and a reach. `shelf` zones test **wrists**, every
other kind tests **foot points**. Both go through one `_PresenceMachine`, so the
debounce and backdating logic is written and tested once.

**A flat shelf zone cannot tell a hand at the shelf from a hand between the camera
and the shelf.** A shopper pushing a trolley registers inside the polygon with both
hands. Arm extension (wrist-to-shoulder distance in units of shoulder width) is what
separates them, and it is why `min_arm_extension` exists.

Arm extension divides by **apparent** shoulder width, which collapses when a shopper
turns side-on, so the raw ratio runs away and passes a hand on a trolley handle. The
same grip measured 1.3 facing the camera and 4.1 in profile. Three things fix it and
all three are load-bearing: floor the denominator at 0.20 of standing height, skip
bodies clipped by the frame edge, and set the threshold to 2.5.

Verified on `grocery-store.mp4`: 7 reaches before, 1 after, and the survivor is a
genuine reach into a shelf. That closes true-positive validation. It rests on one
reach episode by one shopper, so 2.5 is provisional on the sensitivity side.

**The threshold is probably calibrated on an artifact, and the likely error is
missed reaches.** The one validated reach is anatomically implausible: the arm
measures 0.52 to 0.57 of box height where a real arm is about 0.44, and apparent
shoulder width is 0.08 to 0.16 where anatomy says about 0.23. The mechanism is
almost certainly that the shopper's lower body is occluded by their trolley, so
the box stops short of the floor, standing height is underestimated, the floored
denominator is too small and the ratio inflates.

That is the normal retail case, not an edge case: trolleys, baskets and low
shelving occlude shoppers below the waist constantly. A shopper whose whole body
is visible while genuinely reaching should measure about 1.9 by anatomy
(arm 0.44 of height over shoulders 0.23), and would be **missed** at 2.5.

Corroborating: 886 wrist measurements across two basketball players, who extend
their arms constantly and are fully visible, top out at 2.16. Nothing reaches
2.5.

So the earlier claim that 2.5 "sits in a gap" holds for those two clips and does
not generalise. Resolving this needs footage of a reach with the whole body
visible. Do not retune on the evidence currently in the repo, and do not treat
the current sensitivity as established.

Specificity is better evidenced. On `people-walking.mp4`, an overhead concourse
where nobody reaches for anything, 29 shoppers crossed a shelf zone and produced
0 reaches at 2.5, against 4 at the old 1.6 and 43 with the gate off. Pooling both
clips, every non-reach measures at most 2.49 and the genuine reach at least 2.59,
so the default sits in a gap rather than flush against the data.

**Never tune this by eye again.** Two fixtures of real keypoints, and
`tests/test_reach_fixture.py` runs the detector against both:
`reach_poses.json` is sensitivity plus specificity on one grocery shopper,
`walking_poses.json` is specificity across 15 bodies, trimmed to the hardest
negatives. Every other reach test builds poses by hand with shoulders square to
the camera, which is precisely the assumption that hid this bug.

Zone JSON is deliberately not gitignored even though `data/` is, because the
polygons are half of how a fixture was produced.

The event store is SQLite for now. The schema is deliberately plain so it ports to
Postgres unchanged when cross-store aggregation lands. A visit is the durable record,
not separate enter/exit rows: enter, exit and dwell are all derivable from it.

## House style

No em dashes in any prose or docs. Use commas, colons, periods.
