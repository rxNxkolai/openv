# OpenV roadmap

What is built, what is next, and why in that order.

The product is a **recommendation with evidence attached**, not a dataset and not
a dashboard. Cameras are the sensor. The deterministic analysis layer makes the
numbers arguable. The agent layer turns them into something a person can act on.
Every phase below is judged against that: does it get someone closer to making a
decision they would otherwise not have made.

---

## Status at a glance

| Phase | Scope | State |
|---|---|---|
| M0 | Detection and tracking | done |
| M1 | Zones, visits, dwell, event store | done |
| M2 | Pose and reach detection | done, validated both directions |
| M3 | Analysis and agent | done, agent path unrun |
| M4 | Live camera, browser console | done in software, **unrun on a real camera** |
| M5 | Spatial layer, camera fusion, 3D map | built, **fusion unrun on real cameras** |
| M6 | Agent and chat surface | tools and loop built, unrun |
| M7 | Connectors and automation | re-measurement, digest and webhook done |
| M8 | Store map, product and brand numbers, export | built, **unrun on a real planogram** |
| M9 | Multi-store | rollup across exports built, shared database not started |

Five gates are open and none of them are code:

1. **M4 has never run on a real fixed camera.** Everything else is downstream of
   believing this works in a store.
2. **The agent path has never executed.** `openv advise` needs credentials.
3. **Reach sensitivity is probably measuring an artifact.** Specificity is well
   evidenced across 16 bodies and two camera geometries. The positive side is
   one reach whose arm is anatomically too long for its box, almost certainly
   because a trolley hides the shopper's legs and shortens the box. Two
   occlusion-proof denominators were tried; both give believable numbers and no
   discrimination. See CLAUDE.md. This needs footage of a reach with the whole
   body in frame, and a test fails if such footage ever arrives.
4. **Fusion has never joined two real cameras.** It is scored against ground
   truth on a synthetic store, where calibrations agree and clocks are exact.
   Real cameras disagree about where the floor is by some amount nobody has
   measured, and that amount decides the merge radius.
5. **The store map has never held a retailer's real planogram.** M8 is verified
   on a synthetic store and on hand-built cases. What a real planogram export
   looks like, how often it disagrees with the shelf, and how fine the zones
   have to be drawn before a brand gets a number are all still unknown.

---

## Done

### M0, detection and tracking

RF-DETR for people, ByteTrack for identity, normalised into a `FrameResult`
stream. That stream is the architecture seam: zones, events, analysis and the
agent all consume it and nothing else, which is what lets the detector be
swapped without touching the product layer.

### M1, zones and events

Named floor areas, visits with dwell, per-zone funnels, SQLite event store. A
visit is the durable record rather than separate enter and exit rows, because
enter, exit and dwell are all derivable from it. Debounce stops a shopper on a
zone edge becoming forty visits; backdating stops every dwell reading short by
the debounce window.

### M2, pose and reach

Wrists in shelf zones, gated on arm extension. A flat polygon cannot tell a hand
at the shelf from a hand between the camera and the shelf, so a trolley pusher
registers inside it with both hands.

Validated in both directions on real footage:

- **Sensitivity.** The genuine reach in `grocery-store.mp4` is detected.
- **Specificity.** 29 shoppers crossing a shelf zone in overhead concourse
  footage produce zero reaches, against 4 at the old threshold and 43 with the
  gate disabled.

Two fixtures of real MediaPipe keypoints pin it, so this is never tuned by eye
again. That matters because a documented result once claimed here did not
reproduce, and the fixtures are what stop that recurring.

### M3, analysis and the agent

`analysis.py` computes every number deterministically and ranks findings by
severity. `agent.py` writes prose over the result and never calculates. The
benchmark is the store's own median rather than an industry figure, because a
retailer can argue with an external number and cannot argue with their own other
aisles measured the same day. Below 30 observed shoppers a rate is not reported
as a rate.

Recommendations are stored `proposed`. No code path sets `approved`, because
approval is the liability gate.

A shelf nobody reaches is a `dead` finding, ranked high. It used to be the one
shelf the product could not mention, because zero reaches meant zero rows;
the store now records each session's zone set and whether pose ran, so an
absence is readable as a measurement.

### M4, the live path

Camera in, overlay and running numbers in the browser, writing to the same event
store the offline pipeline uses, so a live session is analysable afterwards with
the same `openv analyze`. Accepts a video file as source, which is what makes
the whole path demoable with no camera present.

---

## M5, the spatial layer

**Goal:** every shopper position expressed in store floor coordinates rather than
camera pixels, so distance means something and several cameras can describe one
space.

**Done:**

- `floor.py` maps pixels to floor coordinates through one homography, because
  the floor is a plane and stores already have floor plans.
- `openv floorplan` calibrates by clicking, reads `UNVERIFIABLE` at four points
  rather than a reassuring zero, and warps the floor flat as the manual check.
- Positions persist to the event store, sampled at 1Hz per shopper, so paths and
  speed survive the session and can be asked about later.
- `openv live --floor` puts the plan view in the browser and records to the
  same store the offline path uses.

- `fusion.py` and `openv fuse` join cameras recorded together into store-wide
  paths, on floor position and time only.
- A fused session reads as the whole store: each zone's funnel from the camera
  that measured it, ranked against a store-wide median, with a shopper two
  cameras both saw counted once.
- `openv map3d` draws the store in three dimensions, as a replay in one HTML
  file or live over every camera writing to the store.

**The decision that gated multi-camera, as taken:** stitching one shopper
across cameras is a form of re-identification, which collides with the
session-scoped track ID rule. Spatial-temporal handoff on the floor plane needs
no appearance model and keeps the privacy posture intact. Appearance-based
re-ID does not, and it is the exact surface BIPA litigation targets. Fusion is
the first kind and has no input that could carry the second.

**What that costs, measured on the synthetic store:** no fused track held two
shoppers, 85% of shoppers came out as one track across three cameras, and the
count ran 16% high from refusing to guess. An appearance model would close
most of that 16%. It is not worth the posture.

**Exit criteria:** two real cameras with overlapping coverage produce one
continuous path across the seam, with no appearance model involved. Met on
synthetic cameras. Not yet run on real ones.

**Remaining:**

- Two real cameras, to measure how far apart they place one shopper.
- A live console that runs several cameras in one process. Today each camera
  is its own process against a shared store, and `map3d --live` is where they
  meet.
- Clock sync between cameras that is better than each machine's own clock.

---

## M6, the agent and chat surface

**Goal:** an employee asks a question in plain language and gets an answer they
can act on, with every number traceable to the code that computed it.

This is the product surface where the value actually lands. It is also the
easiest place to destroy the thing that makes OpenV defensible, so the
architecture matters more here than anywhere else.

### The rule that carries over

**The agent never computes and never executes.** A model doing arithmetic on raw
rows is unauditable, and a retailer argues with the numbers before they argue
with the advice. This already holds for `openv advise` and it must hold for
chat.

So chat is **not** text-to-SQL and **not** a model with database access. It is a
tool-calling agent over the deterministic analysis layer. `tools.py` implements
this, with `TOOL_SPECS` ready for a tool-calling API:

| Tool | Returns |
|---|---|
| `list_zones()` | zones, kinds, and what each is paired to |
| `get_findings(zone?, severity?, window?)` | ranked findings, already guarded |
| `get_funnel(zone, window)` | passed, stopped, reached, with the rate guard applied |
| `compare(zone_a, zone_b, metric, window)` | a computed comparison, not two numbers |
| `get_recommendations(status?)` | proposals and their state |
| `measure_change(zone, before, after)` | before and after on the same metric |

### Tools return verdicts, not ingredients

The subtle failure mode, and the one worth designing against explicitly.

If a tool returns `{passed: 12, reached: 1}`, the model can divide and state
"8% reach rate". The deterministic layer would have **refused** to state that
rate, because 12 shoppers is below the confidence threshold. The guard exists in
`analysis.py` and the tool layer would have handed the model exactly what it
needs to bypass it.

So every rate crosses the boundary as a `Rate`: a value that is either a number
or `null`, and when it is `null` the reason travels in the same object. The
counts stay visible, because hiding them would be its own dishonesty, but they
arrive already labelled as not rate-able. A system prompt forbidding division is
not sufficient; the contract has to carry the refusal.

`compare_zones` is the same idea one level up. It computes the comparison rather
than returning two rates, so two withheld numbers cannot be silently subtracted
into a confident difference.

`chat.py` runs the conversation over those tools and keeps every call as a
citation, which is what the evidence panel reads. The cycle is bounded: a model
that keeps asking for tools and never answers is cut off and says so, rather
than returning a partial answer as if it were whole. `openv ask` is the CLI
front end.

The loop is tested against a scripted fake client, so the machinery is verified
with no credentials. **The real path has still never run**, which remains one of
the three open gates.

Threads persist across restarts (`openv threads`, `openv ask --thread`), and
since M8 the agent has two more tools, `list_products` and `get_brand`, which
are its only source of a product or brand name.

**Still to build for M6:** the page itself, and the proposal-from-answer flow.

### Other constraints

- **Every claim carries its provenance.** An answer cites the tool calls behind
  it, and the user can expand them. This is what makes the answer defensible in
  a room with a category manager who disagrees.
- **No invented SKUs, brands or prices.** A product exists for the agent only
  where the store map supplies it. The system prompts forbid papering over a
  gap, and forbid giving a shared zone's numbers to one of its occupants.
- **Chat can propose, never approve.** Same liability gate.
- **The deterministic layer keeps working with no API key.** `openv analyze` is
  the floor of the product and must never depend on a model being reachable.

**Exit criteria:** a store manager asks "why is aisle 6 underperforming" and gets
a ranked, cited answer whose every number can be reproduced by running
`openv analyze` by hand.

---

## M7, connectors and automation

**Goal:** findings reach the person who can act on them, where they already work,
and the loop closes when they do.

### Slack

- **Push.** Findings above a severity threshold post to a channel. This is the
  single highest-leverage connector, because a finding nobody reads is worth
  nothing.
- **Pull.** The same tool API from M6, reachable in a thread.
- **Approval.** A recommendation can be approved from Slack **only** with an
  authenticated identity and an audit record of who approved what and when.
  Approval is the liability gate; it can move to where people work, but it
  cannot become anonymous or implicit.

### Automation

Automations route work to humans. They do not change stores.

- Finding crosses a threshold, so open a ticket, notify an owner, draft the
  change for review.
- Scheduled re-analysis, digests, weekly summaries.

**The automation that matters most is re-measurement.** Built: `openv measure`
compares a zone across two sessions and returns a verdict rather than a delta,
because two rates always differ by something and reporting that as a result
makes noise look like evidence. `indistinguishable` is a real answer.

`openv digest` decides what is worth sending, and says nothing when nothing is,
because a daily "nothing to report" teaches people to mute the channel. Its exit
code carries the decision so a scheduled job branches without parsing. A zone is
re-measured if it is worrying now **or was worrying then**, so a problem that got
fixed still gets its success announced rather than silently dropping out.

Delivery is a plain webhook in the format Slack, Discord, Teams and Google Chat
all accept, so those are configuration rather than code.

`measure` refuses to compare across a redrawn polygon (`not_comparable`),
because the sessions record the zones they were measured against. Moving the
fixture and then the zone is the natural workflow, and it used to yield a
confident p-value over two different boundaries.

**Still to build for M7:** approval routing, ticket creation, and the scheduling
itself.

**Exit criteria:** a recommendation is proposed, approved by a named human,
acted on in the store, and the before-and-after measurement lands in the channel
without anyone asking for it.

---

## M8, the store map

**Goal:** say "the Northfield facing in aisle 6" instead of "shelf-cereal".

OpenV measures zones, not products. An earlier framing called that a hardware
limit awaiting better cameras, and that framing was wrong: the retailer's own
shelf data routes around it entirely. If their system knows which item sits at
a shelf position, and OpenV knows a reach happened in the zone covering it,
that is product-level insight with no product vision and no new hardware.

**This is the phase that makes a brand a buyer.** Retailer value works at zone
granularity. Brand value mostly does not, since a brand cares about its own
facing and not about the endcap in aggregate.

**Built:**

- `storemap.py`: a store's fixtures, the shelf zone each is watched through,
  where each stands on the plan, and the items on it. Checked against the
  zones before anything is recorded, so a typo cannot become a product nobody
  touched.
- The map is recorded with the session, like the zones, so a reach is read
  against what was on the shelf that day and a planogram reset cannot rewrite
  last month.
- `products.py`: brand, category and item numbers, under one rule. A number
  belongs to a brand only where the brand has the zone to itself. A shared
  zone is reported as shared and its reaches are never split, by facings or
  by anything else. A brand's shoppers are counted once across its fixtures.
- `export.py` and `openv export`: the aggregate file a retailer or a brand
  receives, which states in itself what it does not contain. `--brand`
  narrows it to one brand and counts the others without naming them.
- `planview.py` and `openv planview`: the store plan, fixtures coloured by
  finding, foot traffic underneath when positions were recorded.
- `openv demo-store`: two synthetic stores, so all of the above runs with no
  camera, no GPU and no API key.

**Consequence for M1 and M5:** zone granularity is now the granularity of
product insight. One polygon over a whole bay was right when insight was
zone-level. Per-section zones are what give a brand a number, which is an
argument for making zone drawing much cheaper than clicking polygons by hand,
and for generating shelf zones from the store map's own sections.

**Privacy posture is unaffected.** A store map is store data, not shopper data.

**Still to build for M8:**

- An importer for the planogram formats retailers actually export, in place of
  hand-written JSON.
- Shelf zones generated from fixture sections, so the map and the zones cannot
  disagree.
- Time windows inside a session, so a promotion that ran from Thursday can be
  read on its own.

**Exit criteria:** one retailer's real planogram, loaded without hand editing,
produces a brand table that their category manager recognises as their store.

---

## M9, multi-store

**Goal:** compare stores, and support a brand or chain across sites.

**Built:** `openv rollup` pools export files per brand or category. It reads
the exports and nothing else, so it needs no access to any store's database,
footage or cameras, and what crossed the wall to reach it was already
aggregate. It refuses the same session twice, a store with no number adds
nothing to the pool, and every store's own row stays beside the total.

**Remaining:** the event store schema is deliberately plain so it ports to
Postgres unchanged. That migration, and aggregation that runs continuously and
not from files, is where this phase goes next.

**The decision this phase turned on.** Selling a retailer recommendations about
their own store is software. Selling brands continuous cross-store shelf
performance is closer to an insight product, priced on coverage and not on
decisions. The direction is now both, from one builder: the retailer's export
is the whole store, and a brand's is the same file narrowed. The architecture
that follows from it is that everything a brand receives is derived from the
export, never from a store's database.

What the brand product still must not become is a path around the posture. No
row per shopper, no timestamped events, nothing linking one visit to the next.
A brand buys counts per zone per session, and the pool across stores.

Note the existing design deliberately rejected the industry-benchmark angle: the
benchmark is the store's own median, since an external number is arguable. A
category row is the benchmark a brand is shown, and it is still that store's
own. Revisiting that is a strategy decision, not a feature.

---

## Cross-cutting, needed before a paying deployment

Not a phase, but none of the above ships to a customer without these.

- **Edge packaging.** How this actually runs in a store: hardware, install,
  update, and what happens when the network drops.
- **Auth and multi-tenancy.** Currently there is neither.
- **Retention enforcement in code.** Done for the code as it stands:
  `tests/test_privacy_posture.py` fails on a face landmark, a pixel write, an
  identifier that outlives a session, or a track id in an outbound module, and
  `tests/test_export.py` walks every key of the export. What is not yet
  enforced is retention on a deployed box, which belongs with edge packaging.
- **A data agreement.** The export is clean by construction. What a retailer
  permits a brand to receive, and on what terms, is a contract and not a flag.
- **Cost per camera.** Behaviour analytics does not need 30fps; dwell, path and
  conversion are fine at 2 to 5fps, and that is what makes tiled inference
  practical on real hardware. Worth measuring properly before pricing.

---

## Sequencing

**Next: a real planogram.** M8 is the newest layer and the least tested against
the thing it exists for. One retailer's shelf data will show how fine the zones
must be and what the importer has to read.

**Then per-section zones from the map**, since a brand with no number is the
first thing a brand buyer will see, and the fix is zone granularity.

**Then the M6 page**, built around findings, products and approvals and not
around charts.

M5 fusion waits on two real cameras. M9's database waits on more than a
handful of stores.

The five open gates at the top of this document outrank all of it. A live
camera in a real store, with that store's own planogram, tells you more than
any of these phases.
