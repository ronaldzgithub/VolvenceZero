import { useEffect, useRef, useState } from "react";
import type {
  AntFrame,
  AppFrame,
  WorldObjectKind,
  WorldObjectSnapshot,
} from "./types";

const WIDTH = 960;
const HEIGHT = 680;
const WORLD_SPAN = 24;
const TAU = Math.PI * 2;

type Point = [number, number];

interface MovePreview {
  objectId: string;
  delta: Point;
}

export function worldToCanvas(
  x: number,
  y: number,
  width = WIDTH,
  height = HEIGHT,
): Point {
  const scale = Math.min(width, height) / WORLD_SPAN;
  return [width / 2 + x * scale, height / 2 - y * scale];
}

export function canvasToWorld(
  clientX: number,
  clientY: number,
  bounds: Pick<DOMRect, "left" | "top" | "width" | "height">,
): Point {
  const canvasX = ((clientX - bounds.left) / bounds.width) * WIDTH;
  const canvasY = ((clientY - bounds.top) / bounds.height) * HEIGHT;
  const scale = Math.min(WIDTH, HEIGHT) / WORLD_SPAN;
  return [(canvasX - WIDTH / 2) / scale, (HEIGHT / 2 - canvasY) / scale];
}

function terrainRandom(index: number): number {
  const value = Math.sin(index * 91.733 + 17.219) * 43758.5453;
  return value - Math.floor(value);
}

function drawTerrain(context: CanvasRenderingContext2D): void {
  const ground = context.createRadialGradient(
    WIDTH * 0.46,
    HEIGHT * 0.43,
    20,
    WIDTH * 0.5,
    HEIGHT * 0.52,
    HEIGHT * 0.8,
  );
  ground.addColorStop(0, "#8b7557");
  ground.addColorStop(0.58, "#6c5a43");
  ground.addColorStop(1, "#493e32");
  context.fillStyle = ground;
  context.fillRect(0, 0, WIDTH, HEIGHT);

  context.save();
  for (let index = 0; index < 380; index += 1) {
    const x = terrainRandom(index * 3) * WIDTH;
    const y = terrainRandom(index * 3 + 1) * HEIGHT;
    const radius = 0.4 + terrainRandom(index * 3 + 2) * 1.4;
    context.globalAlpha = 0.08 + terrainRandom(index + 900) * 0.1;
    context.fillStyle = index % 3 === 0 ? "#e0c89d" : "#251f1a";
    context.beginPath();
    context.arc(x, y, radius, 0, TAU);
    context.fill();
  }
  for (let index = 0; index < 18; index += 1) {
    const x = terrainRandom(index + 1500) * WIDTH;
    const y = terrainRandom(index + 1600) * HEIGHT;
    const radius = 2 + terrainRandom(index + 1700) * 4;
    context.globalAlpha = 0.18;
    context.fillStyle = index % 2 ? "#c3a87d" : "#332b23";
    context.beginPath();
    context.ellipse(x, y, radius * 1.5, radius, index, 0, TAU);
    context.fill();
  }
  context.restore();

  const vignette = context.createRadialGradient(
    WIDTH / 2,
    HEIGHT / 2,
    HEIGHT * 0.28,
    WIDTH / 2,
    HEIGHT / 2,
    HEIGHT * 0.78,
  );
  vignette.addColorStop(0, "rgba(14, 10, 7, 0)");
  vignette.addColorStop(1, "rgba(14, 10, 7, .46)");
  context.fillStyle = vignette;
  context.fillRect(0, 0, WIDTH, HEIGHT);
}

function drawTrail(
  context: CanvasRenderingContext2D,
  trail: number[][],
): void {
  if (!trail.length || !trail[0]?.length) return;
  const rows = trail.length;
  const columns = trail[0].length;
  const cellWidth = WIDTH / columns;
  const cellHeight = HEIGHT / rows;
  const maximum = Math.max(1e-9, ...trail.flat());

  context.save();
  context.globalCompositeOperation = "screen";
  trail.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      if (value / maximum < 0.045) return;
      const opacity = Math.min(0.18, (value / maximum) * 0.18);
      const x = (columnIndex + 0.5) * cellWidth;
      const y = HEIGHT - (rowIndex + 0.5) * cellHeight;
      const radius = Math.max(cellWidth, cellHeight) * 1.4;
      const haze = context.createRadialGradient(x, y, 0, x, y, radius);
      haze.addColorStop(0, `rgba(119, 206, 162, ${opacity})`);
      haze.addColorStop(1, "rgba(119, 206, 162, 0)");
      context.fillStyle = haze;
      context.fillRect(x - radius, y - radius, radius * 2, radius * 2);
    });
  });
  context.restore();
}

function drawNest(
  context: CanvasRenderingContext2D,
  nest: Point,
): void {
  const [x, y] = worldToCanvas(...nest);
  context.save();
  context.translate(x, y);

  context.fillStyle = "rgba(23, 15, 9, .34)";
  context.beginPath();
  context.ellipse(3, 7, 46, 28, -0.08, 0, TAU);
  context.fill();

  const mound = context.createRadialGradient(-8, -10, 4, 0, 0, 44);
  mound.addColorStop(0, "#c39a62");
  mound.addColorStop(0.48, "#9a7043");
  mound.addColorStop(1, "#604329");
  context.fillStyle = mound;
  context.beginPath();
  context.ellipse(0, 0, 43, 27, -0.08, 0, TAU);
  context.fill();

  context.strokeStyle = "rgba(58, 37, 21, .34)";
  context.lineWidth = 2;
  for (let ring = 0; ring < 3; ring += 1) {
    context.beginPath();
    context.ellipse(
      -2,
      1,
      33 - ring * 8,
      20 - ring * 5,
      -0.08,
      Math.PI * 0.15,
      Math.PI * 1.6,
    );
    context.stroke();
  }

  const entrance = context.createRadialGradient(-2, 1, 1, -2, 1, 14);
  entrance.addColorStop(0, "#090706");
  entrance.addColorStop(0.65, "#24170e");
  entrance.addColorStop(1, "rgba(74, 47, 27, .2)");
  context.fillStyle = entrance;
  context.beginPath();
  context.ellipse(-2, 2, 15, 10, -0.08, 0, TAU);
  context.fill();

  context.fillStyle = "rgba(217, 178, 111, .6)";
  for (let index = 0; index < 12; index += 1) {
    const angle = terrainRandom(index + 2000) * TAU;
    const distance = 18 + terrainRandom(index + 2100) * 22;
    context.beginPath();
    context.arc(
      Math.cos(angle) * distance,
      Math.sin(angle) * distance * 0.55,
      1 + terrainRandom(index + 2200) * 1.5,
      0,
      TAU,
    );
    context.fill();
  }
  context.restore();
}

function offsetPoint(point: Point, delta: Point): Point {
  return [point[0] + delta[0], point[1] + delta[1]];
}

function objectDelta(object: WorldObjectSnapshot, preview: MovePreview | null): Point {
  return preview?.objectId === object.object_id ? preview.delta : [0, 0];
}

function drawSelection(
  context: CanvasRenderingContext2D,
  center: Point,
  radius: number,
): void {
  context.save();
  context.strokeStyle = "rgba(255, 245, 201, .92)";
  context.fillStyle = "rgba(255, 245, 201, .92)";
  context.lineWidth = 1.5;
  context.setLineDash([5, 5]);
  context.beginPath();
  context.arc(center[0], center[1], radius, 0, TAU);
  context.stroke();
  context.setLineDash([]);
  for (const angle of [0, Math.PI / 2, Math.PI, Math.PI * 1.5]) {
    context.beginPath();
    context.arc(
      center[0] + Math.cos(angle) * radius,
      center[1] + Math.sin(angle) * radius,
      3,
      0,
      TAU,
    );
    context.fill();
  }
  context.restore();
}

function drawButter(
  context: CanvasRenderingContext2D,
  object: WorldObjectSnapshot,
  center: Point,
  scale: number,
): void {
  const radius = Math.max(10, object.physical_radius * scale);
  context.save();
  context.translate(...center);
  context.fillStyle = "rgba(36, 25, 13, .28)";
  context.beginPath();
  context.ellipse(3, 7, radius * 1.2, radius * 0.72, -0.15, 0, TAU);
  context.fill();

  context.fillStyle = object.active ? "#e6b746" : "#80704f";
  context.beginPath();
  context.moveTo(-radius * 1.15, radius * 0.18);
  context.bezierCurveTo(
    -radius,
    -radius * 0.7,
    -radius * 0.15,
    -radius * 0.9,
    radius * 0.85,
    -radius * 0.42,
  );
  context.bezierCurveTo(
    radius * 1.18,
    -radius * 0.08,
    radius * 0.92,
    radius * 0.72,
    radius * 0.15,
    radius * 0.78,
  );
  context.bezierCurveTo(
    -radius * 0.45,
    radius * 0.86,
    -radius * 1.25,
    radius * 0.6,
    -radius * 1.15,
    radius * 0.18,
  );
  context.fill();

  if (object.active) {
    const gloss = context.createLinearGradient(-radius, -radius, radius, radius);
    gloss.addColorStop(0, "rgba(255, 245, 174, .78)");
    gloss.addColorStop(0.5, "rgba(255, 222, 111, .22)");
    gloss.addColorStop(1, "rgba(119, 73, 17, .28)");
    context.fillStyle = gloss;
    context.beginPath();
    context.ellipse(
      -radius * 0.18,
      -radius * 0.16,
      radius * 0.82,
      radius * 0.48,
      -0.2,
      0,
      TAU,
    );
    context.fill();
    context.fillStyle = "rgba(255, 238, 153, .8)";
    for (let index = 0; index < 4; index += 1) {
      context.beginPath();
      context.arc(
        radius * (1.15 + index * 0.22),
        radius * (0.2 + (index % 2) * 0.3),
        Math.max(1.5, radius * (0.14 - index * 0.015)),
        0,
        TAU,
      );
      context.fill();
    }
  }
  context.restore();
}

function drawWoodStick(
  context: CanvasRenderingContext2D,
  start: Point,
  end: Point,
  width: number,
): void {
  context.save();
  context.lineCap = "round";
  context.strokeStyle = "rgba(26, 17, 10, .36)";
  context.lineWidth = width + 7;
  context.beginPath();
  context.moveTo(start[0] + 3, start[1] + 5);
  context.lineTo(end[0] + 3, end[1] + 5);
  context.stroke();

  context.strokeStyle = "#5b3920";
  context.lineWidth = width;
  context.beginPath();
  context.moveTo(...start);
  context.lineTo(...end);
  context.stroke();

  context.strokeStyle = "#9b6a37";
  context.lineWidth = Math.max(2, width * 0.52);
  context.beginPath();
  context.moveTo(...start);
  context.lineTo(...end);
  context.stroke();

  context.strokeStyle = "rgba(223, 171, 100, .48)";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(start[0] + 2, start[1] - 2);
  context.lineTo(end[0] + 2, end[1] - 2);
  context.stroke();
  context.restore();
}

function drawBurningMatch(
  context: CanvasRenderingContext2D,
  center: Point,
  effectRadius: number,
  scale: number,
  tick: number,
): void {
  const flicker = Math.sin(tick * 1.9) * 2 + Math.sin(tick * 0.47) * 1.5;
  const heatRadius = effectRadius * scale;
  context.save();

  if (heatRadius > 0) {
    const halo = context.createRadialGradient(
      center[0],
      center[1] - 4,
      3,
      center[0],
      center[1],
      heatRadius,
    );
    halo.addColorStop(0, "rgba(255, 143, 45, .34)");
    halo.addColorStop(0.45, "rgba(238, 76, 31, .13)");
    halo.addColorStop(1, "rgba(238, 76, 31, 0)");
    context.fillStyle = halo;
    context.beginPath();
    context.arc(center[0], center[1], heatRadius, 0, TAU);
    context.fill();
  }

  context.translate(...center);
  context.rotate(-0.58);
  context.lineCap = "round";
  context.strokeStyle = "rgba(30, 19, 11, .32)";
  context.lineWidth = 8;
  context.beginPath();
  context.moveTo(-27, 5);
  context.lineTo(18, 5);
  context.stroke();
  context.strokeStyle = "#c69558";
  context.lineWidth = 6;
  context.beginPath();
  context.moveTo(-28, 0);
  context.lineTo(18, 0);
  context.stroke();
  context.strokeStyle = "rgba(255, 222, 169, .52)";
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(-27, -1);
  context.lineTo(13, -1);
  context.stroke();

  context.fillStyle = "#351c14";
  context.beginPath();
  context.ellipse(20, 0, 9, 7, 0, 0, TAU);
  context.fill();
  context.fillStyle = "#e54d20";
  context.beginPath();
  context.ellipse(19, -1, 6, 5, 0, 0, TAU);
  context.fill();

  context.globalCompositeOperation = "screen";
  const flame = context.createLinearGradient(19, 8, 19, -28 - flicker);
  flame.addColorStop(0, "rgba(244, 46, 17, .92)");
  flame.addColorStop(0.45, "rgba(255, 154, 22, .95)");
  flame.addColorStop(0.78, "rgba(255, 228, 104, .92)");
  flame.addColorStop(1, "rgba(255, 251, 207, .1)");
  context.fillStyle = flame;
  context.beginPath();
  context.moveTo(13, 0);
  context.bezierCurveTo(
    5,
    -12,
    18 + flicker,
    -17,
    20,
    -30 - flicker,
  );
  context.bezierCurveTo(
    31,
    -19,
    31 - flicker,
    -8,
    25,
    1,
  );
  context.closePath();
  context.fill();
  context.restore();
}

function drawWorldObjects(
  context: CanvasRenderingContext2D,
  objects: WorldObjectSnapshot[],
  selectedObjectId: string | null,
  preview: MovePreview | null,
  tick: number,
): void {
  const scale = Math.min(WIDTH, HEIGHT) / WORLD_SPAN;
  objects.forEach((object) => {
    const delta = objectDelta(object, preview);
    const centerWorld = offsetPoint(object.center, delta);
    const center = worldToCanvas(...centerWorld);
    const selected = object.object_id === selectedObjectId;

    if (object.kind === "butter") {
      drawButter(context, object, center, scale);
      if (selected) {
        drawSelection(
          context,
          center,
          Math.max(17, object.physical_radius * scale + 9),
        );
      }
      return;
    }

    if (
      object.kind === "wood_stick" &&
      object.segment_start &&
      object.segment_end
    ) {
      const start = worldToCanvas(...offsetPoint(object.segment_start, delta));
      const end = worldToCanvas(...offsetPoint(object.segment_end, delta));
      const width = Math.max(7, object.physical_radius * 2 * scale);
      drawWoodStick(context, start, end, width);
      if (selected) {
        const radius = Math.hypot(end[0] - start[0], end[1] - start[1]) / 2 + 12;
        drawSelection(context, center, radius);
      }
      return;
    }

    drawBurningMatch(
      context,
      center,
      object.effect_radius,
      scale,
      tick,
    );
    if (selected) {
      drawSelection(context, center, 36);
    }
  });
}

function drawFoodSource(context: CanvasRenderingContext2D, point: Point): void {
  const [x, y] = worldToCanvas(...point);
  context.save();
  context.translate(x, y);
  context.fillStyle = "rgba(37, 22, 12, .28)";
  context.beginPath();
  context.ellipse(2, 5, 17, 10, 0.2, 0, TAU);
  context.fill();
  context.fillStyle = "#b34d32";
  for (let index = 0; index < 7; index += 1) {
    const angle = (index / 7) * TAU;
    const distance = index % 2 ? 8 : 4;
    context.beginPath();
    context.arc(
      Math.cos(angle) * distance,
      Math.sin(angle) * distance * 0.7,
      4 + (index % 3),
      0,
      TAU,
    );
    context.fill();
  }
  context.restore();
}

function drawAnt(
  context: CanvasRenderingContext2D,
  ant: AntFrame,
  tick: number,
): void {
  const [x, y] = worldToCanvas(ant.x, ant.y);
  const movement = Math.min(1, Math.abs(ant.step_command) * 1.6);
  const gait = tick * (0.5 + movement * 0.7) + ant.body_id * 1.71;
  const bodyColor = ant.heat_harmful ? "#5b2118" : "#2a1d18";
  const bodyHighlight = ant.heat_harmful ? "#b5472d" : "#6b4531";

  context.save();
  context.translate(x + 2, y + 4);
  context.rotate(-ant.heading);
  context.globalAlpha = 0.3;
  context.fillStyle = "#160f0c";
  context.beginPath();
  context.ellipse(-3, 1, 15, 5, 0, 0, TAU);
  context.fill();
  context.restore();

  context.save();
  context.translate(x, y);
  context.rotate(-ant.heading);

  if (ant.heat_harmful) {
    const danger = context.createRadialGradient(0, 0, 2, 0, 0, 20);
    danger.addColorStop(0, "rgba(231, 68, 30, .25)");
    danger.addColorStop(1, "rgba(231, 68, 30, 0)");
    context.fillStyle = danger;
    context.fillRect(-20, -20, 40, 40);
  }

  context.strokeStyle = "#211512";
  context.lineWidth = 1.8;
  context.lineCap = "round";
  for (let pair = 0; pair < 3; pair += 1) {
    const legX = -4 + pair * 4;
    const phase = Math.sin(gait + pair * 2.05);
    for (const side of [-1, 1]) {
      const reach = 9 + phase * side * 2.8 * movement;
      const forward = (pair - 1) * 3.5 + phase * 2.2 * movement;
      context.beginPath();
      context.moveTo(legX, side * 2.2);
      context.lineTo(legX + forward, side * 7);
      context.lineTo(legX + forward - 1.5, side * reach);
      context.stroke();
    }
  }

  context.strokeStyle = "#2c1b14";
  context.lineWidth = 1.4;
  const antennaWave = Math.sin(gait * 0.7) * 1.5;
  for (const side of [-1, 1]) {
    context.beginPath();
    context.moveTo(8, side * 2);
    context.quadraticCurveTo(
      13,
      side * (5 + antennaWave),
      17,
      side * (7 - antennaWave),
    );
    context.stroke();
  }

  context.fillStyle = bodyColor;
  context.beginPath();
  context.ellipse(-7.5, 0, 6.5, 4.8, 0, 0, TAU);
  context.fill();
  context.beginPath();
  context.ellipse(0, 0, 4.5, 3.6, 0, 0, TAU);
  context.fill();
  context.beginPath();
  context.ellipse(7, 0, 4.1, 3.7, 0, 0, TAU);
  context.fill();

  context.fillStyle = bodyHighlight;
  context.globalAlpha = 0.72;
  context.beginPath();
  context.ellipse(-8.5, -1.3, 3.1, 1.2, -0.1, 0, TAU);
  context.fill();
  context.beginPath();
  context.ellipse(6.4, -1, 1.6, 0.8, 0, 0, TAU);
  context.fill();
  context.globalAlpha = 1;

  if (ant.carrying_food) {
    context.fillStyle = "#f3c755";
    context.strokeStyle = "#8f6422";
    context.lineWidth = 1;
    context.beginPath();
    context.arc(13.5, 0, 4.5, 0, TAU);
    context.fill();
    context.stroke();
    context.fillStyle = "rgba(255, 242, 161, .82)";
    context.beginPath();
    context.arc(12.2, -1.4, 1.3, 0, TAU);
    context.fill();
  }
  context.restore();
}

function drawStickDraft(
  context: CanvasRenderingContext2D,
  draft: [Point, Point] | null,
): void {
  if (!draft) return;
  const start = worldToCanvas(...draft[0]);
  const end = worldToCanvas(...draft[1]);
  context.save();
  context.strokeStyle = "rgba(255, 239, 193, .88)";
  context.lineWidth = 9;
  context.lineCap = "round";
  context.setLineDash([10, 7]);
  context.beginPath();
  context.moveTo(...start);
  context.lineTo(...end);
  context.stroke();
  context.restore();
}

function drawWorld(
  context: CanvasRenderingContext2D,
  frame: AppFrame,
  selectedObjectId: string | null,
  stickDraft: [Point, Point] | null,
  preview: MovePreview | null,
): void {
  context.clearRect(0, 0, WIDTH, HEIGHT);
  drawTerrain(context);
  drawTrail(context, frame.trail);
  drawNest(context, frame.nest);
  frame.food.forEach((point) => drawFoodSource(context, point));
  drawWorldObjects(
    context,
    frame.objects,
    selectedObjectId,
    preview,
    frame.tick,
  );
  drawStickDraft(context, stickDraft);

  frame.ants.forEach((ant) => {
    if (frame.objective === "heading_stability") {
      const [x, y] = worldToCanvas(ant.x, ant.y);
      context.save();
      context.strokeStyle = "rgba(248, 209, 111, .52)";
      context.lineWidth = 1.5;
      context.setLineDash([5, 5]);
      context.beginPath();
      context.moveTo(x, y);
      context.lineTo(
        x + Math.cos(ant.target_heading) * 38,
        y - Math.sin(ant.target_heading) * 38,
      );
      context.stroke();
      context.restore();
    }
    drawAnt(context, ant, frame.tick);
  });

  context.save();
  context.fillStyle = "rgba(255, 245, 221, .72)";
  context.font = "500 12px ui-monospace, monospace";
  context.textAlign = "left";
  context.fillText(
    `真实环境帧 · tick ${frame.tick} · ${frame.ants.length} 只蚂蚁`,
    18,
    26,
  );
  context.restore();
}

export type CanvasTool = "select" | WorldObjectKind;

interface WorldCanvasProps {
  frame: AppFrame | null;
  interactionEnabled: boolean;
  tool: CanvasTool;
  selectedObjectId: string | null;
  onPlaceObject: (
    kind: WorldObjectKind,
    start: Point,
    end?: Point,
  ) => void;
  onMoveObject: (
    object: WorldObjectSnapshot,
    delta: Point,
  ) => void;
  onSelectObject: (objectId: string | null) => void;
}

function pointSegmentDistance(
  point: Point,
  start: Point,
  end: Point,
): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= 1e-12) {
    return Math.hypot(point[0] - start[0], point[1] - start[1]);
  }
  const fraction = Math.max(
    0,
    Math.min(
      1,
      ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) /
        lengthSquared,
    ),
  );
  return Math.hypot(
    point[0] - (start[0] + fraction * dx),
    point[1] - (start[1] + fraction * dy),
  );
}

function hitTestObject(
  objects: WorldObjectSnapshot[],
  point: Point,
): WorldObjectSnapshot | null {
  for (const object of objects.slice().reverse()) {
    if (
      object.segment_start &&
      object.segment_end &&
      pointSegmentDistance(point, object.segment_start, object.segment_end) <=
        Math.max(0.45, object.physical_radius + 0.25)
    ) {
      return object;
    }
    if (
      Math.hypot(
        point[0] - object.center[0],
        point[1] - object.center[1],
      ) <= Math.max(0.75, object.physical_radius + 0.35)
    ) {
      return object;
    }
  }
  return null;
}

export function WorldCanvas({
  frame,
  interactionEnabled,
  tool,
  selectedObjectId,
  onPlaceObject,
  onMoveObject,
  onSelectObject,
}: WorldCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [stickStart, setStickStart] = useState<Point | null>(null);
  const [stickEnd, setStickEnd] = useState<Point | null>(null);
  const [moveStart, setMoveStart] = useState<Point | null>(null);
  const [moveCurrent, setMoveCurrent] = useState<Point | null>(null);
  const [movingObject, setMovingObject] =
    useState<WorldObjectSnapshot | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    if (frame) {
      const movePreview =
        movingObject && moveStart && moveCurrent
          ? {
              objectId: movingObject.object_id,
              delta: [
                moveCurrent[0] - moveStart[0],
                moveCurrent[1] - moveStart[1],
              ] as Point,
            }
          : null;
      drawWorld(
        context,
        frame,
        selectedObjectId,
        stickStart && stickEnd ? [stickStart, stickEnd] : null,
        movePreview,
      );
    } else {
      drawTerrain(context);
      drawNest(context, [0, 0]);
      context.fillStyle = "rgba(255, 245, 221, .74)";
      context.font = "500 17px system-ui, sans-serif";
      context.textAlign = "center";
      context.fillText("创建生态闭环，开始观察蚁群", WIDTH / 2, HEIGHT / 2 + 72);
      context.fillStyle = "rgba(255, 245, 221, .48)";
      context.font = "13px system-ui, sans-serif";
      context.fillText(
        "画面只会显示后端发布的真实环境帧",
        WIDTH / 2,
        HEIGHT / 2 + 98,
      );
    }
  }, [
    frame,
    selectedObjectId,
    stickStart,
    stickEnd,
    moveCurrent,
    moveStart,
    movingObject,
  ]);

  function eventWorldPoint(
    event: React.PointerEvent<HTMLCanvasElement>,
  ): Point {
    return canvasToWorld(
      event.clientX,
      event.clientY,
      event.currentTarget.getBoundingClientRect(),
    );
  }

  function pointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!frame || !interactionEnabled) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = eventWorldPoint(event);
    if (tool === "wood_stick") {
      setStickStart(point);
      setStickEnd(point);
      return;
    }
    if (tool === "butter" || tool === "burning_match") {
      onPlaceObject(tool, point);
      return;
    }
    const hit = hitTestObject(frame.objects, point);
    onSelectObject(hit?.object_id ?? null);
    setMovingObject(hit);
    setMoveStart(hit ? point : null);
    setMoveCurrent(hit ? point : null);
  }

  function pointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    const point = eventWorldPoint(event);
    if (stickStart) {
      setStickEnd(point);
    } else if (movingObject) {
      setMoveCurrent(point);
    }
  }

  function endPointer(event: React.PointerEvent<HTMLCanvasElement>) {
    const point = eventWorldPoint(event);
    if (stickStart) {
      if (Math.hypot(point[0] - stickStart[0], point[1] - stickStart[1]) >= 0.5) {
        onPlaceObject("wood_stick", stickStart, point);
      }
    } else if (movingObject && moveStart) {
      const delta: Point = [
        point[0] - moveStart[0],
        point[1] - moveStart[1],
      ];
      if (Math.hypot(...delta) >= 0.05) {
        onMoveObject(movingObject, delta);
      }
    }
    setStickStart(null);
    setStickEnd(null);
    setMovingObject(null);
    setMoveStart(null);
    setMoveCurrent(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  const cursor =
    !interactionEnabled
      ? "not-allowed"
      : tool === "select"
        ? movingObject
          ? "grabbing"
          : "grab"
        : "crosshair";

  return (
    <canvas
      ref={ref}
      className="world-canvas"
      width={WIDTH}
      height={HEIGHT}
      aria-label="数字蚂蚁真实生态环境，可放置和移动黄油、木棍与燃烧火柴"
      aria-disabled={!interactionEnabled}
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={endPointer}
      onPointerCancel={endPointer}
      style={{ cursor }}
    />
  );
}
