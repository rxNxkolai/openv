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

- **M0** video in, people detected, stable track IDs out  <- current
- **M1** zone polygons, `entered` / `dwell` / `exited` events into Postgres
- **M2** pose, reach / pickup / put-back, the funnel
- **M3** agent over the event store, diagnosis + recommendation + drafted change
- **M4** live camera end to end

## House style

No em dashes in any prose or docs. Use commas, colons, periods.
