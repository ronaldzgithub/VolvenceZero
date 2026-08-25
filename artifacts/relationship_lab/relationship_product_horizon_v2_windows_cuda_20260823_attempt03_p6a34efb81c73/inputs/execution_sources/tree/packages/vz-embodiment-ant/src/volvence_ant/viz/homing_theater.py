"""Homing theater — the digital ant's validated strength, made watchable.

Foraging pellet-count is *not* where this system beats a hand-written FSM at toy
scale (the project's own matched-control records ``learned`` delivery ≈ 0 while
``fixed_rule`` delivers). The honest, AntBot-scale strength is **navigation**:
the frozen central-complex analogue (:class:`AntNavigator`) integrates efference
copies + a sky-compass into a home vector, so the ant can wander far and still
compute a straight line back to the nest (Dupeyroux 2019: 0.67% of journey
length). A memoryless / no-compass controller cannot — its heading estimate
drifts as ``sqrt(N)`` and it gets lost.

This module stages that as a side-by-side animation. Many ants random-walk
outbound, then switch to homing and steer along their *internal estimate* of
where home is. Each ant draws a dashed arrow to where it *believes* home lies:

- **path-integration** arm (AntBot-class compass): the arrow keeps pointing at
  the real nest, and the ants come home tightly.
- **dead-reckoning** arm (compass channel ablated): the belief rotates away from
  the true nest and the ants miss it.

Everything here is the frozen substrate + a matched ablation — no fabricated
win, no hand-coded behaviour. The optional route-familiarity panel reuses the
real kernel (``route_learning_experiment``): as a fixed route repeats, the
reducible novelty (epistemic prediction error) falls — the memory/PE main chain
recognising a route, contrasted with a memory-off control that never improves.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from volvence_ant.substrate.navigator import AntNavigator, wrap_angle

# AntBot aggregate homing error: 0.67% of journey length (Dupeyroux 2019).
_ANTBOT_REFERENCE_RATIO = 0.0067

_STEP_SIZE = 0.4
_MAX_TURN_RATE = math.radians(45.0)
_HEADING_NOISE = 0.02
_STEP_NOISE = 0.004
_COMPASS_NOISE = 0.007
_NEST_RETURN_RADIUS = 1.0
# canvas / world extent (nest-centred square), auto-fits the widest journey
_VIEW_MARGIN = 3.0


@dataclass(frozen=True)
class HomingAntFrame:
    x: float
    y: float
    heading: float
    believed_home_x: float  # where the ant THINKS home is (true world frame)
    believed_home_y: float
    returned: bool


@dataclass(frozen=True)
class HomingRoundFrame:
    tick: int
    phase: str  # "outbound" | "homing"
    returned_count: int
    ants: tuple[HomingAntFrame, ...]


@dataclass(frozen=True)
class HomingArmReplay:
    label: str
    kind: str  # "path-integration" | "dead-reckoning"
    frames: tuple[HomingRoundFrame, ...]
    return_rate: float
    mean_normalized_error: float
    passes_antbot_scale: bool


@dataclass(frozen=True)
class RouteFamiliarityPanel:
    novelty_by_exposure: tuple[float, ...]
    first_exposure_novelty: float
    last_exposure_novelty: float
    memory_off_last_novelty: float
    familiarity_improved: bool


@dataclass(frozen=True)
class HomingTheaterReport:
    arms: tuple[HomingArmReplay, ...]
    nest: tuple[float, float]
    n_ants: int
    outbound_steps: int
    home_steps: int
    view_extent: float
    antbot_reference_ratio: float
    route: RouteFamiliarityPanel | None
    html_path: str | None


def _believed_home_point(
    *, true_x: float, true_y: float, true_heading: float, nav_state
) -> tuple[float, float]:
    """Project the ant's egocentric home belief into the true world frame.

    The navigator holds the home vector in its *estimated* frame; the ant only
    ever acts egocentrically. So the honest thing to display is the belief
    expressed relative to the ant's true heading: an arrow the viewer can read
    as "this is where the ant thinks the nest is".
    """

    ego_home = wrap_angle(nav_state.home_bearing - nav_state.h_hat)
    believed_bearing = true_heading + ego_home
    length = nav_state.home_distance
    return (
        true_x + length * math.cos(believed_bearing),
        true_y + length * math.sin(believed_bearing),
    )


def _simulate_arm(
    *,
    label: str,
    kind: str,
    compass_gain: float,
    n_ants: int,
    outbound_steps: int,
    home_steps: int,
    seed: int,
) -> HomingArmReplay:
    navs = [
        AntNavigator(
            step_size=_STEP_SIZE,
            heading_noise=_HEADING_NOISE,
            step_noise=_STEP_NOISE,
            compass_gain=compass_gain,
            compass_noise=_COMPASS_NOISE,
            seed=seed * 1000 + i,
        )
        for i in range(n_ants)
    ]
    # Independent world-process RNG per ant; the outbound turn schedule is keyed
    # to the ant index so both arms share the SAME outbound walk (matched
    # control): only the internal estimate differs by the compass ablation.
    rngs = [np.random.default_rng(seed * 7919 + i) for i in range(n_ants)]
    tx = [0.0] * n_ants
    ty = [0.0] * n_ants
    th = [0.0] * n_ants
    path_len = [0.0] * n_ants
    returned = [False] * n_ants
    normalized_error = [0.0] * n_ants

    for nav in navs:
        nav.reset(initial_heading=0.0)

    frames: list[HomingRoundFrame] = []

    def _snapshot(tick: int, phase: str) -> HomingRoundFrame:
        ants = []
        for i in range(n_ants):
            state = navs[i].state
            bx, by = _believed_home_point(
                true_x=tx[i], true_y=ty[i], true_heading=th[i], nav_state=state
            )
            ants.append(
                HomingAntFrame(
                    x=round(tx[i], 4),
                    y=round(ty[i], 4),
                    heading=round(th[i], 4),
                    believed_home_x=round(bx, 4),
                    believed_home_y=round(by, 4),
                    returned=returned[i],
                )
            )
        return HomingRoundFrame(
            tick=tick,
            phase=phase,
            returned_count=sum(returned),
            ants=tuple(ants),
        )

    frames.append(_snapshot(0, "outbound"))

    # -- outbound random walk (identical schedule across arms) ---------------
    for step in range(outbound_steps):
        for i in range(n_ants):
            turn = float(np.clip(rngs[i].normal(0.0, 0.4), -_MAX_TURN_RATE, _MAX_TURN_RATE))
            th[i] = (th[i] + turn + float(rngs[i].normal(0.0, _HEADING_NOISE))) % (2 * math.pi)
            navs[i].update(turn_command=turn, step_command=_STEP_SIZE, true_heading=th[i])
            true_step = max(0.0, _STEP_SIZE + float(rngs[i].normal(0.0, _STEP_NOISE)))
            tx[i] += true_step * math.cos(th[i])
            ty[i] += true_step * math.sin(th[i])
            path_len[i] += true_step
        frames.append(_snapshot(step + 1, "outbound"))

    # normalized home-vector error at the turning point (phase0 / AntBot metric)
    for i in range(n_ants):
        state = navs[i].state
        err = math.hypot(state.home_dx - (-tx[i]), state.home_dy - (-ty[i]))
        normalized_error[i] = err / path_len[i] if path_len[i] > 0 else 0.0

    # -- homing: steer along the internal home-vector estimate ----------------
    for step in range(home_steps):
        for i in range(n_ants):
            if returned[i]:
                continue
            state = navs[i].state
            ego_home = wrap_angle(state.home_bearing - state.h_hat)
            turn = float(np.clip(ego_home, -_MAX_TURN_RATE, _MAX_TURN_RATE))
            th[i] = (th[i] + turn + float(rngs[i].normal(0.0, _HEADING_NOISE))) % (2 * math.pi)
            navs[i].update(turn_command=turn, step_command=_STEP_SIZE, true_heading=th[i])
            true_step = max(0.0, _STEP_SIZE + float(rngs[i].normal(0.0, _STEP_NOISE)))
            tx[i] += true_step * math.cos(th[i])
            ty[i] += true_step * math.sin(th[i])
            if math.hypot(tx[i], ty[i]) <= _NEST_RETURN_RADIUS:
                returned[i] = True
        frames.append(_snapshot(outbound_steps + step + 1, "homing"))

    mean_norm = float(np.mean(normalized_error)) if normalized_error else 0.0
    return HomingArmReplay(
        label=label,
        kind=kind,
        frames=tuple(frames),
        return_rate=sum(returned) / n_ants,
        mean_normalized_error=mean_norm,
        passes_antbot_scale=mean_norm <= _ANTBOT_REFERENCE_RATIO,
    )


async def _route_familiarity(
    *, exposures: int, route_length: int, seed: int
) -> RouteFamiliarityPanel:
    # Imported lazily: this is the only slow, kernel-backed part of the theater.
    from volvence_ant.experiments.phase0 import route_learning_experiment

    result = await route_learning_experiment(
        exposures=exposures,
        route_length=route_length,
        temporal_latent_dim=16,
        seed=seed,
    )
    return RouteFamiliarityPanel(
        novelty_by_exposure=tuple(round(v, 6) for v in result.novelty_by_exposure),
        first_exposure_novelty=round(result.first_exposure_novelty, 6),
        last_exposure_novelty=round(result.last_exposure_novelty, 6),
        memory_off_last_novelty=round(result.memory_off_last_novelty, 6),
        familiarity_improved=result.familiarity_improved,
    )


async def run_homing_theater(
    *,
    n_ants: int = 18,
    outbound_steps: int = 70,
    home_steps: int = 140,
    seed: int = 0,
    include_route: bool = True,
    route_exposures: int = 8,
    route_length: int = 5,
    out_path: Path | None = None,
) -> HomingTheaterReport:
    """Simulate both navigation arms, optionally the route panel, write HTML."""

    path_integration = _simulate_arm(
        label="数字生命 · 路径积分回巢",
        kind="path-integration",
        compass_gain=0.85,
        n_ants=n_ants,
        outbound_steps=outbound_steps,
        home_steps=home_steps,
        seed=seed,
    )
    dead_reckoning = _simulate_arm(
        label="朴素对照 · 无罗盘死走",
        kind="dead-reckoning",
        compass_gain=0.0,
        n_ants=n_ants,
        outbound_steps=outbound_steps,
        home_steps=home_steps,
        seed=seed,
    )
    extent = _VIEW_MARGIN
    for arm in (path_integration, dead_reckoning):
        for frame in arm.frames:
            for ant in frame.ants:
                extent = max(extent, abs(ant.x), abs(ant.y))
    extent += _VIEW_MARGIN

    route = None
    if include_route:
        route = await _route_familiarity(
            exposures=route_exposures, route_length=route_length, seed=seed
        )

    report = HomingTheaterReport(
        arms=(path_integration, dead_reckoning),
        nest=(0.0, 0.0),
        n_ants=n_ants,
        outbound_steps=outbound_steps,
        home_steps=home_steps,
        view_extent=round(extent, 3),
        antbot_reference_ratio=_ANTBOT_REFERENCE_RATIO,
        route=route,
        html_path=None,
    )
    if out_path is not None:
        written = write_homing_theater_html(report=report, out_path=out_path)
        report = HomingTheaterReport(
            arms=report.arms,
            nest=report.nest,
            n_ants=report.n_ants,
            outbound_steps=report.outbound_steps,
            home_steps=report.home_steps,
            view_extent=report.view_extent,
            antbot_reference_ratio=report.antbot_reference_ratio,
            route=report.route,
            html_path=str(written),
        )
    return report


def write_homing_theater_html(*, report: HomingTheaterReport, out_path: Path) -> Path:
    payload = {
        "arms": [asdict(arm) for arm in report.arms],
        "nest": list(report.nest),
        "n_ants": report.n_ants,
        "outbound_steps": report.outbound_steps,
        "home_steps": report.home_steps,
        "view_extent": report.view_extent,
        "antbot_reference_ratio": report.antbot_reference_ratio,
        "route": asdict(report.route) if report.route is not None else None,
    }
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    html = _HOMING_HTML_TEMPLATE.replace("__HOMING_DATA__", data_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


_HOMING_HTML_TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>数字蚂蚁剧场 · 路径积分回巢</title>
<style>
  :root{--bg:#0c0f14;--panel:#141922;--ink:#e8edf5;--muted:#8a94a6;
        --out:#5cc8ff;--home:#39d98a;--lost:#e06c75;--belief:#ffb454;--nest:#c678dd}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.5 system-ui,-apple-system,"PingFang SC",sans-serif}
  header{padding:16px 22px;border-bottom:1px solid #232b38}
  header h1{margin:0;font-size:18px}
  header p{margin:4px 0 0;color:var(--muted);font-size:13px;max-width:1100px}
  #stage{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:18px 22px}
  .panel{background:var(--panel);border:1px solid #232b38;border-radius:12px;
         padding:14px;display:flex;flex-direction:column;gap:10px}
  .panel h2{margin:0;font-size:15px;display:flex;align-items:center;gap:8px}
  .tag{font-size:11px;padding:2px 8px;border-radius:999px}
  .tag.pi{background:#122a20;color:#39d98a}
  .tag.dr{background:#2a1a1a;color:#e06c75}
  canvas{width:100%;aspect-ratio:1;background:#0a0d12;border-radius:8px;border:1px solid #232b38}
  .stats{display:flex;gap:16px;font-size:12px;color:var(--muted)}
  .stats b{display:block;color:var(--ink);font-size:20px;font-variant-numeric:tabular-nums}
  .ok{color:#39d98a}.bad{color:#e06c75}
  #controls{display:flex;align-items:center;gap:14px;padding:10px 22px 6px;flex-wrap:wrap}
  button{background:#1d2530;color:var(--ink);border:1px solid #2d3745;border-radius:8px;
         padding:8px 16px;cursor:pointer;font-size:13px}
  button:hover{background:#26313f}
  #scrub{flex:1;min-width:200px}
  #tick{font-variant-numeric:tabular-nums;color:var(--muted);min-width:150px}
  #legend{display:flex;gap:18px;flex-wrap:wrap;padding:2px 22px 12px;color:var(--muted);font-size:12px}
  .key{display:inline-flex;align-items:center;gap:6px}
  .dot{width:10px;height:10px;border-radius:50%}
  #route{margin:0 22px 22px;background:var(--panel);border:1px solid #232b38;
         border-radius:12px;padding:14px}
  #route h2{margin:0 0 8px;font-size:15px}
  #route p{margin:0 0 10px;color:var(--muted);font-size:12px}
</style></head>
<body>
<header>
  <h1>数字蚂蚁剧场 · 它到底知不知道家在哪</h1>
  <p>一群蚂蚁从巢里随机游走出去(蓝),然后切换到回巢(绿):每只蚂蚁沿它<b>内部估计的回家方向</b>行动,
  橙色虚线箭头指向它<b>以为</b>家在哪。左:完整路径积分(含 AntBot 级天空罗盘),信念始终指向真巢、能精准回家;
  右:删掉罗盘校正的纯死走,朝向估计随步数漂移、信念偏离真巢、蚂蚁迷路。同一套外出轨迹,唯一差别是罗盘消融。</p>
</header>
<div id="controls">
  <button id="play">⏸ 暂停</button>
  <button id="restart">⟲ 重播</button>
  <label>速度 <input id="speed" type="range" min="1" max="6" value="3"></label>
  <input id="scrub" type="range" min="0" value="0">
  <span id="tick"></span>
</div>
<div id="legend">
  <span class="key"><span class="dot" style="background:var(--out)"></span>外出中</span>
  <span class="key"><span class="dot" style="background:var(--home)"></span>已回巢</span>
  <span class="key"><span class="dot" style="background:var(--lost)"></span>回巢中/迷路</span>
  <span class="key"><span class="dot" style="background:var(--belief)"></span>它以为家在此方向</span>
  <span class="key"><span class="dot" style="background:var(--nest)"></span>真实巢穴</span>
</div>
<div id="stage"></div>
<div id="route" style="display:none">
  <h2>路线熟悉度 · 记忆/预测误差主链(真内核)</h2>
  <p>同一条固定路线反复走,可下降的新奇度(认知型预测误差)随曝光下降——内核在"认出"这条路;
  虚线是记忆关闭对照(每次全新会话),它不会下降。这是数字生命的学习,不是硬编码。</p>
  <canvas id="routecanvas" width="1000" height="240"></canvas>
</div>
<script>
const DATA=__HOMING_DATA__;
const EXT=DATA.view_extent, ANTBOT=DATA.antbot_reference_ratio;
const TAIL=18;
const panels=DATA.arms.map(buildPanel);

function buildPanel(arm){
  const wrap=document.createElement("div");wrap.className="panel";
  const cls=arm.kind==="path-integration"?"pi":"dr";
  const tag=arm.kind==="path-integration"?"AntBot 级":"消融对照";
  const rr=(arm.return_rate*100).toFixed(0);
  const ne=(arm.mean_normalized_error*100).toFixed(2);
  const okrr=arm.return_rate>=0.8?"ok":"bad";
  const okne=arm.passes_antbot_scale?"ok":"bad";
  wrap.innerHTML=`<h2>${arm.label}<span class="tag ${cls}">${tag}</span></h2>`+
    `<canvas width="520" height="520"></canvas>`+
    `<div class="stats">`+
      `<div>回巢率<b class="${okrr}" data-rr>${rr}%</b></div>`+
      `<div>归一化回家误差<b class="${okne}">${ne}%</b></div>`+
      `<div>AntBot 参照<b>${(ANTBOT*100).toFixed(2)}%</b></div>`+
    `</div>`;
  document.querySelector("#stage").appendChild(wrap);
  return {arm,canvas:wrap.querySelector("canvas"),rr:wrap.querySelector("[data-rr]")};
}
function wx(c,x){return (x+EXT)/(2*EXT)*c.width;}
function wy(c,y){return c.height-(y+EXT)/(2*EXT)*c.height;}

function drawPanel(p,idx){
  const c=p.canvas,x=c.getContext("2d"),arm=p.arm;
  const f=arm.frames[Math.min(idx,arm.frames.length-1)];
  x.clearRect(0,0,c.width,c.height);
  // faint grid rings around the nest so distance is legible
  const nx=wx(c,0),ny=wy(c,0);
  x.strokeStyle="#1a2230";x.lineWidth=1;
  for(let r=5;r<=EXT;r+=5){x.beginPath();
    x.arc(nx,ny,r/(2*EXT)*c.width,0,7);x.stroke();}
  // nest
  x.strokeStyle="#c678dd";x.lineWidth=2;x.strokeRect(nx-9,ny-9,18,18);
  x.fillStyle="rgba(198,120,221,.18)";x.fillRect(nx-9,ny-9,18,18);
  // tails
  for(let a=0;a<f.ants.length;a++){
    x.beginPath();
    for(let k=Math.max(0,idx-TAIL);k<=idx;k++){
      const af=arm.frames[k].ants[a];const px=wx(c,af.x),py=wy(c,af.y);
      k===Math.max(0,idx-TAIL)?x.moveTo(px,py):x.lineTo(px,py);
    }
    x.strokeStyle=f.phase==="homing"?"rgba(57,217,138,.22)":"rgba(92,200,255,.20)";
    x.lineWidth=1.5;x.stroke();
  }
  // ants + belief arrow
  for(const ant of f.ants){
    const px=wx(c,ant.x),py=wy(c,ant.y);
    // belief arrow (where it thinks home is)
    const bx=wx(c,ant.believed_home_x),by=wy(c,ant.believed_home_y);
    x.strokeStyle="rgba(255,180,84,.5)";x.lineWidth=1;x.setLineDash([3,3]);
    x.beginPath();x.moveTo(px,py);x.lineTo(bx,by);x.stroke();x.setLineDash([]);
    let col=ant.returned?"#39d98a":(f.phase==="homing"?"#e06c75":"#5cc8ff");
    x.strokeStyle=col;x.lineWidth=1.5;x.beginPath();x.moveTo(px,py);
    x.lineTo(px+Math.cos(ant.heading)*8,py-Math.sin(ant.heading)*8);x.stroke();
    x.fillStyle=col;x.beginPath();x.arc(px,py,3.4,0,7);x.fill();
  }
  p.rr.textContent=(f.returned_count/f.ants.length*100).toFixed(0)+"%";
}

const total=Math.max(...DATA.arms.map(a=>a.frames.length));
const scrub=document.querySelector("#scrub");scrub.max=total-1;
const tickLabel=document.querySelector("#tick");
const playBtn=document.querySelector("#play"),speedInput=document.querySelector("#speed");
let idx=0,playing=true,acc=0,last=performance.now();
function render(){
  for(const p of panels)drawPanel(p,idx);
  scrub.value=idx;
  const phase=idx<=DATA.outbound_steps?"外出探索":"回巢导航";
  tickLabel.textContent=`第 ${idx+1}/${total} 拍 · ${phase}`;
}
function loop(now){const dt=now-last;last=now;
  if(playing){acc+=dt;const step=240-(+speedInput.value)*34;
    while(acc>=step){acc-=step;idx=(idx+1)%total;}}
  render();requestAnimationFrame(loop);}
playBtn.onclick=()=>{playing=!playing;playBtn.textContent=playing?"⏸ 暂停":"▶ 播放";};
document.querySelector("#restart").onclick=()=>{idx=0;playing=true;playBtn.textContent="⏸ 暂停";};
scrub.oninput=()=>{idx=+scrub.value;playing=false;playBtn.textContent="▶ 播放";render();};
requestAnimationFrame(loop);

// route familiarity panel
if(DATA.route){
  document.querySelector("#route").style.display="block";
  const c=document.querySelector("#routecanvas"),x=c.getContext("2d");
  const nov=DATA.route.novelty_by_exposure, memoff=DATA.route.memory_off_last_novelty;
  const maxv=Math.max(...nov,memoff,1e-9);
  const padL=48,padB=28,padT=14,padR=14;
  const W=c.width-padL-padR,H=c.height-padT-padB;
  x.strokeStyle="#2d3745";x.beginPath();x.moveTo(padL,padT);x.lineTo(padL,padT+H);
  x.lineTo(padL+W,padT+H);x.stroke();
  // memory-off reference line
  const yoff=padT+H-(memoff/maxv)*H;
  x.strokeStyle="rgba(224,108,117,.7)";x.setLineDash([5,4]);x.beginPath();
  x.moveTo(padL,yoff);x.lineTo(padL+W,yoff);x.stroke();x.setLineDash([]);
  x.fillStyle="#e06c75";x.font="11px system-ui";
  x.fillText("记忆关闭对照",padL+W-92,yoff-5);
  // novelty curve
  x.strokeStyle="#39d98a";x.lineWidth=2;x.beginPath();
  nov.forEach((v,i)=>{const px=padL+(nov.length<2?0:i/(nov.length-1)*W);
    const py=padT+H-(v/maxv)*H;i?x.lineTo(px,py):x.moveTo(px,py);
    x.fillStyle="#39d98a";x.fillRect(px-2,py-2,4,4);x.beginPath;});
  x.beginPath();nov.forEach((v,i)=>{const px=padL+(nov.length<2?0:i/(nov.length-1)*W);
    const py=padT+H-(v/maxv)*H;i?x.lineTo(px,py):x.moveTo(px,py);});
  x.strokeStyle="#39d98a";x.stroke();
  x.fillStyle="#8a94a6";x.font="11px system-ui";
  x.fillText("新奇度(可下降 PE)",padL,padT-2);
  x.fillText("曝光次数 →",padL+W-70,padT+H+18);
}
</script></body></html>"""


__all__ = [
    "HomingAntFrame",
    "HomingArmReplay",
    "HomingRoundFrame",
    "HomingTheaterReport",
    "RouteFamiliarityPanel",
    "run_homing_theater",
    "write_homing_theater_html",
]
