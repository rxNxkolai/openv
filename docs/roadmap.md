# Patron roadmap

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
| M5 | Spatial layer, floor coordinates | done bar multi-camera |
| M6 | Agent and chat surface | tools and loop built, unrun |
| M7 | Connectors and automation | not started |
| M8 | Planogram integration, SKU level | not started |
| M9 | Multi-store | not started |

Three gates are open and none of them are code:

1. **M4 has never run on a real fixed camera.** Everything else is downstream of
   believing this works in a store.
2. **The agent path has never executed.** `patron advise` needs credentials.
3. **Reach sensitivity rests on one reach episode by one shopper.** Specificity
   is well evidenced across 16 bodies and two camera geometries; the positive
   side is not.

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

### M4, the live path

Camera in, overlay and running numbers in the browser, writing to the same event
store the offline pipeline uses, so a live session is analysable afterwards with
the same `patron analyze`. Accepts a video file as source, which is what makes
the whole path demoable with no camera present.

---

## M5, the spatial layer

**Goal:** every shopper position expressed in store floor coordinates rather than
camera pixels, so distance means something and several cameras can describe one
space.

**Done:**

- `floor.py` maps pixels to floor coordinates through one homography, because
  the floor is a plane and stores already have floor plans.
- `patron floorplan` calibrates by clicking, reads `UNVERIFIABLE` at four points
  rather than a reassuring zero, and warps the floor flat as the manual check.
- Positions persist to the event store, sampled at 1Hz per shopper, so paths and
  speed survive the session and can be asked about later.
- `patron live --floor` puts the plan view in the browser and records to the
  same store the offline path uses.

**Remaining:**

- **Multi-camera fusion.** Each camera gets its own homography into one shared
  floor space.

**The decision that gates multi-camera:** stitching one shopper across cameras is
a form of re-identification, which collides with the session-scoped track ID
rule. Spatial-temporal handoff on the floor plane, where a track leaving one
camera's frustum is matched to one entering another's at the same floor point,
needs no appearance model and keeps the privacy posture intact. Appearance-based
re-ID does not, and it is the exact surface BIPA litigation targets. This is a
product decision, not an implementation detail.

**Exit criteria:** two cameras with overlapping coverage produce one continuous
path across the seam, with no appearance model involved.

---

## M6, the agent and chat surface

**Goal:** an employee asks a question in plain language and gets an answer they
can act on, with every number traceable to the code that computed it.

This is the product surface where the value actually lands. It is also the
easiest place to destroy the thing that makes Patron defensible, so the
architecture matters more here than anywhere else.

### The rule that carries over

**The agent never computes and never executes.** A model doing arithmetic on raw
rows is unauditable, and a retailer argues with the numbers before they argue
with the advice. This already holds for `patron advise` and it must hold for
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
than returning a partial answer as if it were whole. `patron ask` is the CLI
front end.

The loop is tested against a scripted fake client, so the machinery is verified
with no credentials. **The real path has still never run**, which remains one of
the three open gates.

**Still to build for M6:** message persistence across restarts, the page itself,
and the proposal-from-answer flow.

### Other constraints

- **Every claim carries its provenance.** An answer cites the tool calls behind
  it, and the user can expand them. This is what makes the answer defensible in
  a room with a category manager who disagrees.
- **No invented SKUs, brands or prices.** Patron knows zones, not products,
  until M8. The existing system prompt already forbids papering over that gap.
- **Chat can propose, never approve.** Same liability gate.
- **The deterministic layer keeps working with no API key.** `patron analyze` is
  the floor of the product and must never depend on a model being reachable.

**Exit criteria:** a store manager asks "why is aisle 6 underperforming" and gets
a ranked, cited answer whose every number can be reproduced by running
`patron analyze` by hand.

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

**The automation that matters most is re-measurement.** When a change is made,
automatically measure the same funnel over the same zone afterwards and report
the delta. That is what proves Patron works, it is what justifies renewal, and
it carries no liability because it asserts nothing about what to do next. Build
this before anything else in M7.

**Exit criteria:** a recommendation is proposed, approved by a named human,
acted on in the store, and the before-and-after measurement lands in the channel
without anyone asking for it.

---

## M8, planogram integration

**Goal:** say "the Brand X facing on the mid shelf" instead of "shelf-endcap".

Patron knows zones, not products. CLAUDE.md frames this as a hardware limit
awaiting Phase 3 hardware, but that framing is wrong: integrating with the
retailer's own shelf-item system routes around it entirely. If their system knows
which SKU sits at shelf position X, and Patron knows a reach happened in the zone
covering X, that is product-level insight with zero SKU vision and no new
hardware.

**This is the phase that unlocks CPG as a buyer.** Retailer value works fine at
zone granularity. Brand value mostly does not, because a brand cares about their
facing and not about the endcap in aggregate.

**Consequence for M1 and M5:** zone granularity becomes the granularity of
product insight. Today a shelf zone is one polygon over a whole shelf face,
which is right when insight is zone-level. Per-section zones start mattering
here, which is also an argument for making zone drawing much cheaper than
clicking polygons by hand.

**Privacy posture is unaffected.** Shelf-position-to-SKU mapping is store data,
not shopper data.

---

## M9, multi-store

**Goal:** compare stores, and support a brand or chain across sites.

The event store schema is deliberately plain so it ports to Postgres unchanged.
That is the migration this phase begins with.

**Open question that changes the company.** Selling a retailer recommendations
about their own store is unambiguously software. Selling brands continuous
cross-store shelf performance is much closer to an insight product, priced on
coverage rather than on decisions. Both are viable. They are different
businesses with different architectures, and the choice should be deliberate
rather than arrived at.

Note the existing design deliberately rejected the industry-benchmark angle: the
benchmark is the store's own median, precisely because an external number is
arguable. Revisiting that is a strategy decision, not a feature.

---

## Cross-cutting, needed before a paying deployment

Not a phase, but none of the above ships to a customer without these.

- **Edge packaging.** How this actually runs in a store: hardware, install,
  update, and what happens when the network drops.
- **Auth and multi-tenancy.** Currently there is neither.
- **Retention enforcement in code.** The privacy posture is stated in CLAUDE.md
  and honoured by construction. It should be enforced and testable, not
  conventional.
- **Cost per camera.** Behaviour analytics does not need 30fps; dwell, path and
  conversion are fine at 2 to 5fps, and that is what makes tiled inference
  practical on real hardware. Worth measuring properly before pricing.

---

## Sequencing

**Next:** finish M5 calibration, because everything spatial depends on a
calibration someone can actually produce, and because it is the gate for the
bird's-eye view.

**Then M6**, since the chat surface is where the product becomes usable by
someone who is not an analyst, and the dashboard being designed now should be
built around findings and approvals rather than charts.

**Then M7 re-measurement first**, because proving a change worked is what makes
this renewable rather than a one-off study.

M8 waits on a retailer partner willing to expose planogram data. M9 waits on
more than one store.

The three open gates at the top of this document outrank all of it. A live
camera in a real store tells you more than any of these phases.
