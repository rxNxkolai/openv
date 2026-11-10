/* OpenV 3D store map.

   A store drawn in three dimensions from the floor up: the plan as a grid,
   fixtures as blocks coloured by what happened at them, each camera's view as
   an outline on the floor, and shoppers as markers standing where the fused
   tracks put them.

   Drawn on a 2D canvas with its own projection. A scene this size is a few
   dozen polygons and a few dozen markers, which does not need a 3D engine, and
   the console has to load on an edge box with no network and no bundle.

   Coordinates are the store's floor units, x and y on the plan and z up. */

(function(){
'use strict';

const SEVERITY = {high:'#ff5a5f', medium:'#ffb224', low:'#3ddc97', none:'#6b7785'};
const CAMERA_COLOURS = ['#6fd6ff','#b48cff','#ffd166','#5eead4','#ff8fab','#a3e635'];
const FIXTURE_HEIGHT = 1.9;
const SHOPPER_HEIGHT = 1.7;
const TRAIL_SECONDS = 14;

function css(name, fallback){
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
function rgba(hex, a){
  const n = parseInt(hex.replace('#',''), 16);
  return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`;
}

class StoreMap3D {
  constructor(canvas, scene){
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.scene = scene;
    this.people = [];          // [{id, x, y, cameras, trail:[[x,y,age]]}]
    this.hover = null;
    this.showCameras = true;
    this.showTrails = true;

    const [x0,y0,x1,y1] = scene.extent;
    this.target = [(x0+x1)/2, (y0+y1)/2, 0];
    this.span = Math.max(x1-x0, y1-y0, 1);
    this.yaw = -0.62; this.pitch = 0.74; this.dist = this.span * 1.3;
    this.accent = css('--accent', '#6fd6ff');
    this.fg = css('--fg', '#dfe5ec');
    this.dim = css('--dim', '#8492a3');
    this.line = css('--line2', '#2a323d');
    this.mono = css('--mono', 'ui-monospace, Menlo, Consolas, monospace');

    this._bind();
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  /* Swap the scene under the same view. Recentres only when the store's
     extent changed, so a refresh does not yank the camera away from wherever
     somebody had turned it. */
  setScene(scene){
    const before = this.scene.extent.join(',');
    this.scene = scene;
    if (scene.extent.join(',') !== before){
      const [x0,y0,x1,y1] = scene.extent;
      this.target = [(x0+x1)/2, (y0+y1)/2, 0];
      this.span = Math.max(x1-x0, y1-y0, 1);
      this.dist = this.span * 1.3;
    }
    this.draw();
  }

  /* ---------- camera ---------- */
  resize(){
    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = Math.max(2, r.width); this.h = Math.max(2, r.height);
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.draw();
  }

  _basis(){
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    // The eye sits back from the target along the view direction.
    const eye = [
      this.target[0] + this.dist * cp * sy,
      this.target[1] - this.dist * cp * cy,
      this.target[2] + this.dist * sp,
    ];
    const f = [-cp*sy, cp*cy, -sp];                 // forward
    const r = [cy, sy, 0];                          // right
    const u = [r[1]*f[2]-r[2]*f[1], r[2]*f[0]-r[0]*f[2], r[0]*f[1]-r[1]*f[0]];
    return {eye, f, r, u};
  }

  project(p){
    const b = this._b;
    const d = [p[0]-b.eye[0], p[1]-b.eye[1], p[2]-b.eye[2]];
    const z = d[0]*b.f[0] + d[1]*b.f[1] + d[2]*b.f[2];
    if (z < 0.05) return null;
    const x = d[0]*b.r[0] + d[1]*b.r[1] + d[2]*b.r[2];
    const y = d[0]*b.u[0] + d[1]*b.u[1] + d[2]*b.u[2];
    const k = (this.h * 1.25) / z;
    return [this.w/2 + x*k, this.h/2 - y*k, z];
  }

  _bind(){
    const c = this.canvas;
    let drag = null;
    c.addEventListener('pointerdown', e => {
      drag = {x:e.clientX, y:e.clientY, pan:e.shiftKey || e.button === 2};
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener('pointermove', e => {
      if (!drag){ this._hover(e); return; }
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan){
        const k = this.dist / this.h;
        const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
        this.target[0] -= (dx*cy - dy*sy) * k;
        this.target[1] -= (dx*sy + dy*cy) * k;
      } else {
        this.yaw -= dx * 0.006;
        this.pitch = Math.min(1.5, Math.max(0.12, this.pitch + dy * 0.005));
      }
      this.draw();
    });
    c.addEventListener('pointerup', () => { drag = null; });
    c.addEventListener('contextmenu', e => e.preventDefault());
    c.addEventListener('wheel', e => {
      e.preventDefault();
      this.dist = Math.min(this.span*5, Math.max(this.span*0.25, this.dist * Math.exp(e.deltaY*0.0012)));
      this.draw();
    }, {passive:false});
  }

  _hover(e){
    const r = this.canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null;
    for (const f of this._fixtureLabels || []){
      const d = Math.hypot(f.x - mx, f.y - my);
      if (d < 46 && (!best || d < best.d)) best = {d, zone:f.zone};
    }
    const zone = best ? best.zone : null;
    if (zone !== this.hover){ this.hover = zone; this.draw(); }
  }

  /* ---------- drawing ---------- */
  _poly(points, fill, stroke, width, dash){
    const ctx = this.ctx;
    const pts = points.map(p => this.project(p));
    if (pts.some(p => !p)) return null;
    ctx.beginPath();
    pts.forEach((p,i) => i ? ctx.lineTo(p[0],p[1]) : ctx.moveTo(p[0],p[1]));
    ctx.closePath();
    if (fill){ ctx.fillStyle = fill; ctx.fill(); }
    if (stroke){
      ctx.setLineDash(dash || []);
      ctx.strokeStyle = stroke; ctx.lineWidth = width || 1; ctx.stroke();
      ctx.setLineDash([]);
    }
    return pts;
  }
  _line(a, b, stroke, width){
    const p = this.project(a), q = this.project(b);
    if (!p || !q) return;
    const ctx = this.ctx;
    ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]);
    ctx.strokeStyle = stroke; ctx.lineWidth = width || 1; ctx.stroke();
  }
  _text(text, x, y, colour, size, align){
    const ctx = this.ctx;
    ctx.font = `${size || 10}px ${this.mono}`;
    ctx.textAlign = align || 'center';
    ctx.fillStyle = colour;
    ctx.fillText(text, x, y);
  }

  _floor(){
    const [x0,y0,x1,y1] = this.scene.extent;
    const pad = this.span * 0.08;
    const ax = Math.floor(x0 - pad), bx = Math.ceil(x1 + pad);
    const ay = Math.floor(y0 - pad), by = Math.ceil(y1 + pad);
    for (let x = ax; x <= bx; x++)
      this._line([x,ay,0],[x,by,0], rgba(this.line, x % 5 === 0 ? 0.9 : 0.38), 1);
    for (let y = ay; y <= by; y++)
      this._line([ax,y,0],[bx,y,0], rgba(this.line, y % 5 === 0 ? 0.9 : 0.38), 1);
    if (this.scene.outline){
      this._poly(this.scene.outline.map(p => [p[0],p[1],0]),
                 'rgba(255,255,255,0.018)', rgba(this.fg, 0.55), 1.25);
    }
  }

  _cameras(){
    if (!this.showCameras) return;
    (this.scene.cameras || []).forEach((cam, i) => {
      if (!cam.footprint || cam.footprint.length < 3) return;
      const colour = CAMERA_COLOURS[i % CAMERA_COLOURS.length];
      const pts = this._poly(cam.footprint.map(p => [p[0],p[1],0.01]),
                             rgba(colour, 0.045), rgba(colour, 0.75), 1, [6,5]);
      if (!pts) return;
      // Labelled on the edge nearest the viewer, a third of the way along it,
      // so two views that share a corner do not stack their names there.
      let best = 0;
      for (let k = 1; k < pts.length; k++)
        if (pts[k][2] + pts[(k+1) % pts.length][2] < pts[best][2] + pts[(best+1) % pts.length][2]) best = k;
      const a = pts[best], b = pts[(best+1) % pts.length];
      const f = 0.2 + 0.18 * (i % 3);
      this._text(cam.label.toUpperCase(), a[0] + (b[0]-a[0])*f, a[1] + (b[1]-a[1])*f + 13,
                 rgba(colour, 0.95), 9);
    });
  }

  _fixtureFaces(list){
    for (const f of this.scene.fixtures || []){
      if (!f.plan) continue;
      const colour = f.measured ? SEVERITY[f.severity] : '#4a5562';
      const hot = this.hover === f.zone;
      const base = f.plan.map(p => [p[0],p[1],0]);
      const top  = f.plan.map(p => [p[0],p[1],FIXTURE_HEIGHT]);
      const n = base.length;
      const cx = base.reduce((a,p) => a+p[0], 0)/n, cy = base.reduce((a,p) => a+p[1], 0)/n;
      for (let i = 0; i < n; i++){
        const j = (i+1) % n;
        const quad = [base[i], base[j], top[j], top[i]];
        const mid = [(base[i][0]+base[j][0])/2, (base[i][1]+base[j][1])/2, FIXTURE_HEIGHT/2];
        list.push({depth:this._depth(mid), draw:() =>
          this._poly(quad, rgba(colour, hot ? 0.34 : 0.16), rgba(colour, 0.85), 1,
                     f.measured ? null : [4,4])});
      }
      list.push({depth:this._depth([cx,cy,FIXTURE_HEIGHT]) - 0.01, draw:() => {
        this._poly(top, rgba(colour, hot ? 0.5 : 0.28), rgba(colour, 1), 1.25,
                   f.measured ? null : [4,4]);
        const p = this.project([cx,cy,FIXTURE_HEIGHT + 0.5]);
        if (!p) return;
        this._fixtureLabels.push({zone:f.zone, x:p[0], y:p[1]});
        this._text((f.label || f.zone).toUpperCase(), p[0], p[1] - 6, this.fg, 10);
        this._text(f.readout, p[0], p[1] + 7, rgba(colour, 1), 10);
        if (hot && f.brands && f.brands.length)
          this._text(f.brands.join(' / '), p[0], p[1] + 20, this.dim, 9);
      }});
    }
  }

  _depth(p){
    const b = this._b;
    return (p[0]-b.eye[0])*b.f[0] + (p[1]-b.eye[1])*b.f[1] + (p[2]-b.eye[2])*b.f[2];
  }

  _ring(x, y, radius, stroke, width){
    const ctx = this.ctx;
    ctx.beginPath();
    for (let i = 0; i <= 20; i++){
      const a = i / 20 * Math.PI * 2;
      const p = this.project([x + Math.cos(a)*radius, y + Math.sin(a)*radius, 0.02]);
      if (!p) return;
      i ? ctx.lineTo(p[0],p[1]) : ctx.moveTo(p[0],p[1]);
    }
    ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.stroke();
  }

  _trails(){
    if (!this.showTrails) return;
    const ctx = this.ctx;
    for (const person of this.people){
      const t = person.trail;
      for (let i = 1; i < t.length; i++){
        const a = this.project([t[i-1][0], t[i-1][1], 0.02]);
        const b = this.project([t[i][0], t[i][1], 0.02]);
        if (!a || !b) continue;
        ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
        ctx.strokeStyle = rgba(this.accent, 0.55 * (1 - t[i][2]));
        ctx.lineWidth = 1.5; ctx.stroke();
      }
    }
  }

  _people(list){
    for (const person of this.people){
      const fused = (person.cameras || 1) > 1;
      list.push({depth:this._depth([person.x, person.y, SHOPPER_HEIGHT/2]), draw:() => {
        const foot = this.project([person.x, person.y, 0.02]);
        const head = this.project([person.x, person.y, SHOPPER_HEIGHT]);
        if (!foot || !head) return;
        const ctx = this.ctx;
        this._ring(person.x, person.y, 0.34, rgba(this.accent, 0.95), 1.25);
        if (fused) this._ring(person.x, person.y, 0.52, rgba(this.accent, 0.4), 1);
        const glow = ctx.createLinearGradient(foot[0], foot[1], head[0], head[1]);
        glow.addColorStop(0, rgba(this.accent, 0.15));
        glow.addColorStop(1, rgba(this.accent, 0.95));
        ctx.beginPath(); ctx.moveTo(foot[0], foot[1]); ctx.lineTo(head[0], head[1]);
        ctx.strokeStyle = glow; ctx.lineWidth = 2; ctx.stroke();
        const s = Math.max(2.5, 180 / head[2]);
        ctx.beginPath();
        ctx.moveTo(head[0], head[1]-s); ctx.lineTo(head[0]+s, head[1]);
        ctx.lineTo(head[0], head[1]+s); ctx.lineTo(head[0]-s, head[1]); ctx.closePath();
        ctx.fillStyle = fused ? '#ffffff' : this.accent; ctx.fill();
        if (this.dist < this.span * 2.2)
          this._text(person.tag, head[0] + s + 5, head[1] + 3, rgba(this.fg, 0.8), 9, 'left');
      }});
    }
  }

  draw(){
    if (!this.w) return;
    const ctx = this.ctx;
    this._b = this._basis();
    this._fixtureLabels = [];
    ctx.clearRect(0, 0, this.w, this.h);
    this._floor();
    this._cameras();
    this._trails();
    const list = [];
    this._fixtureFaces(list);
    this._people(list);
    list.sort((a,b) => b.depth - a.depth);
    for (const item of list) item.draw();
  }

  /* ---------- data ---------- */

  /* Replay: where every recorded track was at time t. Positions are sampled
     about once a second, so the marker is interpolated between two samples
     and is never drawn past the last one. */
  setTime(t){
    const people = [];
    for (const track of this.scene.tracks || []){
      const ts = track.t, n = ts.length;
      if (t < ts[0] || t > ts[n-1]) continue;
      let i = 1;
      while (i < n - 1 && ts[i] < t) i++;
      const f = ts[i] > ts[i-1] ? (t - ts[i-1]) / (ts[i] - ts[i-1]) : 0;
      const x = track.x[i-1] + f*(track.x[i]-track.x[i-1]);
      const y = track.y[i-1] + f*(track.y[i]-track.y[i-1]);
      const trail = [];
      for (let k = 0; k < i; k++){
        const age = t - ts[k];
        if (age <= TRAIL_SECONDS) trail.push([track.x[k], track.y[k], age / TRAIL_SECONDS]);
      }
      trail.push([x, y, 0]);
      people.push({id:track.id, x, y, trail, tag:track.tag,
                   cameras: track.c ? track.c[i-1] : 1});
    }
    this.people = people;
    this.draw();
    return people.length;
  }

  /* Live: fused positions as they arrive. Trails are kept here, per id, for
     as long as the id keeps arriving, and are dropped with it. */
  setLive(positions, now){
    const known = new Map(this.people.map(p => [p.id, p]));
    const people = [];
    for (const p of positions){
      const old = known.get(p.id);
      const history = old ? old.history : [];
      const last = history[history.length - 1];
      if (!last || Math.hypot(last[0]-p.x, last[1]-p.y) > 0.05) history.push([p.x, p.y, now]);
      while (history.length && now - history[0][2] > TRAIL_SECONDS) history.shift();
      people.push({id:p.id, x:p.x, y:p.y, cameras:p.cameras, tag:p.tag, history,
                   trail: history.map(h => [h[0], h[1], (now - h[2]) / TRAIL_SECONDS])});
    }
    this.people = people;
    this.draw();
    return people.length;
  }
}

window.StoreMap3D = StoreMap3D;
})();
