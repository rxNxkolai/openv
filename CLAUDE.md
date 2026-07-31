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

## Milestones

- **M0** video in, people detected, stable track IDs out  (done)
- **M1** zone polygons, visit records with dwell, per-zone funnel  (done, SQLite)
- **M2** pose and reach detection  (done, pending true-positive validation on real footage)
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
separates them, and it is why `min_arm_extension` exists. Verified: it took a real
clip from 2 false reaches to 0. True-positive validation still needs footage of
someone actually reaching into a shelf.

The event store is SQLite for now. The schema is deliberately plain so it ports to
Postgres unchanged when cross-store aggregation lands. A visit is the durable record,
not separate enter/exit rows: enter, exit and dwell are all derivable from it.

## House style

No em dashes in any prose or docs. Use commas, colons, periods.
