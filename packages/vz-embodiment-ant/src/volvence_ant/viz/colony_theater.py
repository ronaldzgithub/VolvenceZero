"""Colony theater — a side-by-side, intuitive animation of two ant colonies.

The existing :mod:`volvence_ant.viz.dashboard` is an *evidence* panel: two
trajectory polylines plus a wall of kernel telemetry. Useful for falsification,
but abstract. This module answers a different question — *what does a swarm
actually look like* under two regimes:

- **heuristic** arm: a colony of hardcoded FSM foragers (:class:`FixedRuleAnt`).
  Every decision is an ``if situation then behaviour`` rule the author wrote.
- **digital-life** arm: a colony of kernel-driven :class:`AntSession` bodies.
  Each body reuses the frozen substrate + the learnable controller (``z_t`` /
  ``β_t``, path integration, online writeback) through the ``vz-runtime`` facade.

Both colonies forage the SAME world (shared pheromone snapshot bus so trails
self-organise) and half-way through the food is relocated — an unforeseen
perturbation that lets the viewer watch rigid rules degrade while the adaptive
controller keeps sensing and re-routing.

SSOT: the theater only *consumes* immutable, already-published facts — body
geometry from ``AntWorld`` public getters, the pheromone snapshot from the
``ColonyWorld`` bus, the behaviour label each controller publishes on its own
record (``FixedRuleStep.mode`` / ``AntStepRecord.abstract_action``). It never
reaches into kernel-owner private state and never becomes a second owner.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from volvence_ant.controllers.fixed_rule_ant import FixedRuleAnt, FixedRuleConfig
from volvence_ant.env.ant_world import AntWorldConfig, FoodSource
from volvence_ant.env.colony import ColonyWorld
from volvence_ant.env.pheromone_field import PheromoneBus
from volvence_ant.runtime import AntSessionConfig, KernelColonyRunner

# Food starts here, then relocates here at ``relocate_at`` (the perturbation).
_FOOD_START = (6.0, 0.0)
_FOOD_MOVED = (-4.0, 4.0)

# World / pheromone-field extent (square, nest-centred). Keep in sync with the
# ``PheromoneBus`` grid so the theater canvas and the trail heatmap agree.
_FIELD_SPAN = 24.0
_CELL_SIZE = 1.0


@dataclass(frozen=True)
class TheaterAntFrame:
    """One body's observable pose + the behaviour label it published."""

    x: float
    y: float
    heading: float
    carrying: bool
    mode: str


@dataclass(frozen=True)
class TheaterRoundFrame:
    """A single tick of one colony: every body + the shared trail + food."""

    tick: int
    delivered: int
    ants: tuple[TheaterAntFrame, ...]
    food: tuple[tuple[float, float], ...]
    trail: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class TheaterArmReplay:
    """An immutable replay of one colony arm."""

    label: str
    kind: str  # "heuristic" | "digital-life"
    frames: tuple[TheaterRoundFrame, ...]


@dataclass(frozen=True)
class ColonyTheaterReport:
    arms: tuple[TheaterArmReplay, ...]
    nest: tuple[float, float]
    n_ants: int
    rounds: int
    relocate_at: int
    field_span: float
    cell_size: float
    html_path: str | None


def _world_config(seed: int) -> AntWorldConfig:
    return AntWorldConfig(seed=seed, antenna_offset_deg=30.0, antenna_reach=0.9)


def _food(at: tuple[float, float]) -> FoodSource:
    return FoodSource(x=at[0], y=at[1], strength=1.6, decay=5.0, radius=1.6)


def _bus() -> PheromoneBus:
    return PheromoneBus(
        width=_FIELD_SPAN,
        height=_FIELD_SPAN,
        cell_size=_CELL_SIZE,
        decay=0.02,
        deposit_amount=2.0,
    )


def _colony_world(seed: int, *, n_ants: int) -> ColonyWorld:
    return ColonyWorld(
        config=_world_config(seed),
        food_sources=(_food(_FOOD_START),),
        n_bodies=n_ants,
        bus=_bus(),
    )


def _assemble_frame(world: ColonyWorld, modes: list[str]) -> TheaterRoundFrame:
    ants = tuple(
        TheaterAntFrame(
            x=round(body.x, 4),
            y=round(body.y, 4),
            heading=round(body.heading, 4),
            carrying=body.carrying_food,
            mode=modes[body_id],
        )
        for body_id, body in ((i, world.body(i)) for i in range(world.n_bodies))
    )
    food = tuple(
        (round(src.x, 4), round(src.y, 4))
        for src in world.food_sources()
        if src.remaining > 0.0
    )
    field = world.pheromone
    trail = tuple(
        tuple(round(float(value), 4) for value in row) for row in field.trail
    )
    return TheaterRoundFrame(
        tick=world.tick,
        delivered=world.food_delivered,
        ants=ants,
        food=food,
        trail=trail,
    )


def _run_heuristic_arm(
    *, n_ants: int, rounds: int, relocate_at: int, seed: int
) -> TheaterArmReplay:
    world = _colony_world(seed, n_ants=n_ants)
    ants = [
        FixedRuleAnt(
            world,
            config=FixedRuleConfig(
                seed=seed * 100 + i,
                food_sense_threshold=0.02,
                gradient_gain=6.0,
            ),
            body_id=i,
        )
        for i in range(n_ants)
    ]
    frames: list[TheaterRoundFrame] = []
    for round_index in range(rounds):
        if round_index == relocate_at:
            world.move_food(index=0, x=_FOOD_MOVED[0], y=_FOOD_MOVED[1])
        modes = [ant.step().mode for ant in ants]
        frames.append(_assemble_frame(world, modes))
    return TheaterArmReplay(
        label="启发式 · 硬编码 FSM",
        kind="heuristic",
        frames=tuple(frames),
    )


async def _run_digital_life_arm(
    *,
    n_ants: int,
    rounds: int,
    relocate_at: int,
    seed: int,
    session_config: AntSessionConfig,
) -> TheaterArmReplay:
    world = _colony_world(seed, n_ants=n_ants)
    runner = KernelColonyRunner(world, base_config=session_config)
    frames: list[TheaterRoundFrame] = []
    for round_index in range(rounds):
        if round_index == relocate_at:
            world.move_food(index=0, x=_FOOD_MOVED[0], y=_FOOD_MOVED[1])
        record = await runner.step_round()
        modes = [step.abstract_action for step in record.ant_steps]
        frames.append(_assemble_frame(world, modes))
    return TheaterArmReplay(
        label="数字生命 · 学习控制器",
        kind="digital-life",
        frames=tuple(frames),
    )


async def run_colony_theater(
    *,
    n_ants: int = 8,
    rounds: int = 90,
    relocate_at: int | None = None,
    seed: int = 0,
    session_config: AntSessionConfig | None = None,
    out_path: Path | None = None,
) -> ColonyTheaterReport:
    """Run both colonies, assemble immutable frames, optionally write the HTML.

    ``session_config`` lets the caller (an orchestration script) inject an
    online-writeback ``JointLoopSchedule`` for the digital-life arm without this
    module importing that vz-temporal-internal type (import-boundary rule).
    """

    relocate = rounds // 2 if relocate_at is None else relocate_at
    base_config = session_config or AntSessionConfig(
        temporal_latent_dim=16,
        session_id=f"colony-theater:{seed}",
        seed=seed,
    )
    heuristic = _run_heuristic_arm(
        n_ants=n_ants, rounds=rounds, relocate_at=relocate, seed=seed
    )
    digital = await _run_digital_life_arm(
        n_ants=n_ants,
        rounds=rounds,
        relocate_at=relocate,
        seed=seed,
        session_config=base_config,
    )
    report = ColonyTheaterReport(
        arms=(heuristic, digital),
        nest=(0.0, 0.0),
        n_ants=n_ants,
        rounds=rounds,
        relocate_at=relocate,
        field_span=_FIELD_SPAN,
        cell_size=_CELL_SIZE,
        html_path=None,
    )
    if out_path is not None:
        written = write_colony_theater_html(report=report, out_path=out_path)
        report = ColonyTheaterReport(
            arms=report.arms,
            nest=report.nest,
            n_ants=report.n_ants,
            rounds=report.rounds,
            relocate_at=report.relocate_at,
            field_span=report.field_span,
            cell_size=report.cell_size,
            html_path=str(written),
        )
    return report


def write_colony_theater_html(*, report: ColonyTheaterReport, out_path: Path) -> Path:
    """Render a self-contained (zero-dependency) side-by-side canvas animation."""

    payload = {
        "arms": [asdict(arm) for arm in report.arms],
        "nest": list(report.nest),
        "n_ants": report.n_ants,
        "rounds": report.rounds,
        "relocate_at": report.relocate_at,
        "field_span": report.field_span,
        "cell_size": report.cell_size,
    }
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    html = _THEATER_HTML_TEMPLATE.replace("__THEATER_DATA__", data_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


_THEATER_HTML_TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>数字蚂蚁剧场 · 启发式 vs 数字生命</title>
<style>
  :root{--bg:#0c0f14;--panel:#141922;--ink:#e8edf5;--muted:#8a94a6;
        --forage:#5cc8ff;--carry:#ffb454;--trail:#39d98a;--food:#ffd54a;--nest:#c678dd}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.5 system-ui,-apple-system,"PingFang SC",sans-serif}
  header{padding:16px 22px;border-bottom:1px solid #232b38}
  header h1{margin:0;font-size:18px}
  header p{margin:4px 0 0;color:var(--muted);font-size:13px}
  #stage{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:18px 22px}
  .panel{background:var(--panel);border:1px solid #232b38;border-radius:12px;
         padding:14px;display:flex;flex-direction:column;gap:10px}
  .panel h2{margin:0;font-size:15px;display:flex;align-items:center;gap:8px}
  .tag{font-size:11px;padding:2px 8px;border-radius:999px}
  .tag.h{background:#3a2a1a;color:#ffb454}
  .tag.d{background:#122a3a;color:#5cc8ff}
  canvas{width:100%;aspect-ratio:1;background:#0a0d12;border-radius:8px;
         border:1px solid #232b38}
  .stat{display:flex;justify-content:space-between;font-size:13px;color:var(--muted)}
  .stat b{color:var(--ink);font-size:20px;font-variant-numeric:tabular-nums}
  #controls{display:flex;align-items:center;gap:14px;padding:10px 22px 6px;flex-wrap:wrap}
  button{background:#1d2530;color:var(--ink);border:1px solid #2d3745;
         border-radius:8px;padding:8px 16px;cursor:pointer;font-size:13px}
  button:hover{background:#26313f}
  #scrub{flex:1;min-width:200px}
  #tick{font-variant-numeric:tabular-nums;color:var(--muted);min-width:130px}
  #legend{display:flex;gap:18px;flex-wrap:wrap;padding:2px 22px 18px;color:var(--muted);font-size:12px}
  .key{display:inline-flex;align-items:center;gap:6px}
  .dot{width:10px;height:10px;border-radius:50%}
  .reloc{color:#ff6b6b}
</style></head>
<body>
<header>
  <h1>数字蚂蚁剧场 · 一群蚂蚁，两种大脑</h1>
  <p>左右两个殖民地觅食同一个世界；中途食物被搬走（扰动）。看硬编码规则如何僵化崩溃、学习控制器如何持续感知与改道。</p>
</header>
<div id="controls">
  <button id="play">⏸ 暂停</button>
  <button id="restart">⟲ 重播</button>
  <label>速度 <input id="speed" type="range" min="1" max="6" value="3"></label>
  <input id="scrub" type="range" min="0" value="0">
  <span id="tick"></span>
</div>
<div id="legend">
  <span class="key"><span class="dot" style="background:var(--forage)"></span>觅食中</span>
  <span class="key"><span class="dot" style="background:var(--carry)"></span>搬运食物</span>
  <span class="key"><span class="dot" style="background:var(--trail)"></span>信息素走廊</span>
  <span class="key"><span class="dot" style="background:var(--food)"></span>食物源</span>
  <span class="key"><span class="dot" style="background:var(--nest)"></span>巢穴</span>
  <span class="key reloc">┃ 食物搬迁时刻</span>
</div>
<div id="stage"></div>
<script>
const DATA=__THEATER_DATA__;
const SPAN=DATA.field_span, CELL=DATA.cell_size, HALF=SPAN/2;
const TAIL=14; // fading trail length in frames
const panels=DATA.arms.map((arm,i)=>buildPanel(arm,i));

function buildPanel(arm,i){
  const wrap=document.createElement("div");wrap.className="panel";
  const tagClass=arm.kind==="digital-life"?"d":"h";
  const tagText=arm.kind==="digital-life"?"数字生命":"启发式";
  wrap.innerHTML=`<h2>${arm.label}<span class="tag ${tagClass}">${tagText}</span></h2>`+
    `<canvas width="520" height="520"></canvas>`+
    `<div class="stat"><span>累计投递食物</span><b class="d">0</b></div>`;
  document.querySelector("#stage").appendChild(wrap);
  return {arm,canvas:wrap.querySelector("canvas"),delivered:wrap.querySelector(".d")};
}

// world (x,y) -> canvas pixel (y flipped, nest-centred)
function wx(c,x){return (x+HALF)/SPAN*c.width;}
function wy(c,y){return c.height-(y+HALF)/SPAN*c.height;}

// global max trail (across both arms, all frames) so the heatmap is stable
let TRAILMAX=1e-6;
for(const arm of DATA.arms)for(const f of arm.frames)for(const row of f.trail)
  for(const v of row)if(v>TRAILMAX)TRAILMAX=v;

function drawPanel(p,frameIdx){
  const c=p.canvas,x=c.getContext("2d"),arm=p.arm;
  const f=arm.frames[Math.min(frameIdx,arm.frames.length-1)];
  x.clearRect(0,0,c.width,c.height);
  // pheromone trail heatmap
  const rows=f.trail.length, cols=f.trail[0].length;
  const cw=c.width/cols, ch=c.height/rows;
  for(let r=0;r<rows;r++)for(let col=0;col<cols;col++){
    const v=f.trail[r][col]; if(v<=0)continue;
    x.fillStyle="rgba(57,217,138,"+Math.min(0.5,v/TRAILMAX*0.5)+")";
    // grid row 0 is world-bottom (origin_y=-HALF); flip vertically for canvas
    x.fillRect(col*cw,c.height-(r+1)*ch,cw+0.5,ch+0.5);
  }
  // nest
  const nx=wx(c,DATA.nest[0]),ny=wy(c,DATA.nest[1]);
  x.strokeStyle="#c678dd";x.lineWidth=2;x.strokeRect(nx-9,ny-9,18,18);
  x.fillStyle="rgba(198,120,221,.18)";x.fillRect(nx-9,ny-9,18,18);
  // food sources
  for(const fd of f.food){drawStar(x,wx(c,fd[0]),wy(c,fd[1]),9,"#ffd54a");}
  // fading tails
  for(let a=0;a<f.ants.length;a++){
    x.beginPath();
    for(let k=Math.max(0,frameIdx-TAIL);k<=frameIdx;k++){
      const af=arm.frames[k].ants[a];const px=wx(c,af.x),py=wy(c,af.y);
      k===Math.max(0,frameIdx-TAIL)?x.moveTo(px,py):x.lineTo(px,py);
    }
    const carrying=f.ants[a].carrying;
    x.strokeStyle=carrying?"rgba(255,180,84,.35)":"rgba(92,200,255,.28)";
    x.lineWidth=1.5;x.stroke();
  }
  // ant bodies + heading
  for(const ant of f.ants){
    const px=wx(c,ant.x),py=wy(c,ant.y);
    const col=ant.carrying?"#ffb454":"#5cc8ff";
    x.strokeStyle=col;x.lineWidth=1.5;x.beginPath();
    x.moveTo(px,py);
    x.lineTo(px+Math.cos(ant.heading)*8,py-Math.sin(ant.heading)*8);x.stroke();
    x.fillStyle=col;x.beginPath();x.arc(px,py,3.4,0,7);x.fill();
  }
  p.delivered.textContent=f.delivered;
}

function drawStar(x,cx,cy,r,color){
  x.fillStyle=color;x.beginPath();
  for(let i=0;i<10;i++){const ang=Math.PI/5*i-Math.PI/2;const rad=i%2?r*.45:r;
    const px=cx+Math.cos(ang)*rad,py=cy+Math.sin(ang)*rad;
    i?x.lineTo(px,py):x.moveTo(px,py);}
  x.closePath();x.fill();
}

const total=Math.max(...DATA.arms.map(a=>a.frames.length));
const scrub=document.querySelector("#scrub");scrub.max=total-1;
const tickLabel=document.querySelector("#tick");
const playBtn=document.querySelector("#play");
const speedInput=document.querySelector("#speed");
let idx=0,playing=true,acc=0,last=performance.now();

function render(){
  for(const p of panels)drawPanel(p,idx);
  scrub.value=idx;
  const phase=idx>=DATA.relocate_at?"食物已搬迁":"扰动前";
  tickLabel.textContent=`第 ${idx+1}/${total} 拍 · ${phase}`;
}
function loop(now){
  const dt=now-last;last=now;
  if(playing){acc+=dt;const step=260-(+speedInput.value)*38;
    while(acc>=step){acc-=step;idx=(idx+1)%total;}}
  render();requestAnimationFrame(loop);
}
playBtn.onclick=()=>{playing=!playing;playBtn.textContent=playing?"⏸ 暂停":"▶ 播放";};
document.querySelector("#restart").onclick=()=>{idx=0;playing=true;playBtn.textContent="⏸ 暂停";};
scrub.oninput=()=>{idx=+scrub.value;playing=false;playBtn.textContent="▶ 播放";render();};
requestAnimationFrame(loop);
</script></body></html>"""


__all__ = [
    "ColonyTheaterReport",
    "TheaterAntFrame",
    "TheaterArmReplay",
    "TheaterRoundFrame",
    "run_colony_theater",
    "write_colony_theater_html",
]
