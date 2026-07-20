import { useEffect, useRef, useState } from "react";
import type {
  AppFrame,
  WorldObjectKind,
  WorldObjectSnapshot,
} from "./types";

const WIDTH = 960;
const HEIGHT = 680;
const WORLD_SPAN = 24;

export function worldToCanvas(
  x: number,
  y: number,
  width = WIDTH,
  height = HEIGHT,
): [number, number] {
  const scale = Math.min(width, height) / WORLD_SPAN;
  return [width / 2 + x * scale, height / 2 - y * scale];
}

export function canvasToWorld(
  clientX: number,
  clientY: number,
  bounds: Pick<DOMRect, "left" | "top" | "width" | "height">,
): [number, number] {
  const canvasX = ((clientX - bounds.left) / bounds.width) * WIDTH;
  const canvasY = ((clientY - bounds.top) / bounds.height) * HEIGHT;
  const scale = Math.min(WIDTH, HEIGHT) / WORLD_SPAN;
  return [(canvasX - WIDTH / 2) / scale, (HEIGHT / 2 - canvasY) / scale];
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
  trail.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      if (value <= 0) return;
      const opacity = Math.min(0.48, (value / maximum) * 0.48);
      context.fillStyle = `rgba(48, 206, 156, ${opacity})`;
      context.fillRect(
        columnIndex * cellWidth,
        HEIGHT - (rowIndex + 1) * cellHeight,
        cellWidth + 0.5,
        cellHeight + 0.5,
      );
    });
  });
}

function drawWorldObjects(
  context: CanvasRenderingContext2D,
  objects: WorldObjectSnapshot[],
  selectedObjectId: string | null,
): void {
  const scale = Math.min(WIDTH, HEIGHT) / WORLD_SPAN;
  objects.forEach((object) => {
    const [centerX, centerY] = worldToCanvas(
      object.center[0],
      object.center[1],
    );
    if (object.kind === "burning_match" && object.effect_radius > 0) {
      const halo = context.createRadialGradient(
        centerX,
        centerY,
        2,
        centerX,
        centerY,
        object.effect_radius * scale,
      );
      halo.addColorStop(0, "rgba(255, 96, 52, .34)");
      halo.addColorStop(1, "rgba(255, 96, 52, 0)");
      context.fillStyle = halo;
      context.beginPath();
      context.arc(
        centerX,
        centerY,
        object.effect_radius * scale,
        0,
        Math.PI * 2,
      );
      context.fill();
    }

    if (object.kind === "butter") {
      context.fillStyle = object.active ? "#f5d76e" : "#887a4a";
      context.strokeStyle = "#fff0a3";
      context.lineWidth = object.object_id === selectedObjectId ? 3 : 1;
      context.beginPath();
      context.arc(
        centerX,
        centerY,
        Math.max(7, object.physical_radius * scale),
        0,
        Math.PI * 2,
      );
      context.fill();
      context.stroke();
      return;
    }

    if (object.segment_start && object.segment_end) {
      const [startX, startY] = worldToCanvas(...object.segment_start);
      const [endX, endY] = worldToCanvas(...object.segment_end);
      context.lineCap = "round";
      context.strokeStyle =
        object.kind === "wood_stick" ? "#8d613e" : "#c8a97e";
      context.lineWidth = Math.max(
        4,
        object.physical_radius * 2 * scale +
          (object.object_id === selectedObjectId ? 4 : 0),
      );
      context.beginPath();
      context.moveTo(startX, startY);
      context.lineTo(endX, endY);
      context.stroke();
    }
    if (object.kind === "burning_match") {
      context.fillStyle = "#ff6b35";
      context.beginPath();
      context.arc(centerX, centerY, 8, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#ffd166";
      context.beginPath();
      context.arc(centerX, centerY - 4, 4, 0, Math.PI * 2);
      context.fill();
    }
  });
}

function drawWorld(
  context: CanvasRenderingContext2D,
  frame: AppFrame,
  selectedObjectId: string | null,
  stickDraft: [[number, number], [number, number]] | null,
): void {
  context.clearRect(0, 0, WIDTH, HEIGHT);
  const gradient = context.createLinearGradient(0, 0, 0, HEIGHT);
  gradient.addColorStop(0, "#101c24");
  gradient.addColorStop(1, "#071016");
  context.fillStyle = gradient;
  context.fillRect(0, 0, WIDTH, HEIGHT);

  context.strokeStyle = "rgba(193, 219, 230, .08)";
  context.lineWidth = 1;
  for (let value = -12; value <= 12; value += 2) {
    const [x] = worldToCanvas(value, 0);
    const [, y] = worldToCanvas(0, value);
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, HEIGHT);
    context.moveTo(0, y);
    context.lineTo(WIDTH, y);
    context.stroke();
  }

  drawTrail(context, frame.trail);

  const [nestX, nestY] = worldToCanvas(frame.nest[0], frame.nest[1]);
  context.fillStyle = "#dcb978";
  context.beginPath();
  context.arc(nestX, nestY, 14, 0, Math.PI * 2);
  context.fill();
  context.fillStyle = "#21180d";
  context.font = "700 10px system-ui";
  context.textAlign = "center";
  context.fillText("巢", nestX, nestY + 3);

  frame.food.forEach(([x, y]) => {
    const [foodX, foodY] = worldToCanvas(x, y);
    context.fillStyle = "#f06b57";
    context.beginPath();
    context.arc(foodX, foodY, 11, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "rgba(240,107,87,.28)";
    context.lineWidth = 8;
    context.stroke();
  });

  drawWorldObjects(context, frame.objects, selectedObjectId);

  if (stickDraft) {
    const [startX, startY] = worldToCanvas(...stickDraft[0]);
    const [endX, endY] = worldToCanvas(...stickDraft[1]);
    context.save();
    context.strokeStyle = "rgba(198, 151, 102, .7)";
    context.lineWidth = 8;
    context.setLineDash([8, 5]);
    context.beginPath();
    context.moveTo(startX, startY);
    context.lineTo(endX, endY);
    context.stroke();
    context.restore();
  }

  frame.ants.forEach((ant) => {
    const [x, y] = worldToCanvas(ant.x, ant.y);
    if (frame.objective === "heading_stability") {
      context.save();
      context.strokeStyle = "rgba(255, 209, 102, .72)";
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
    context.save();
    context.translate(x, y);
    context.rotate(-ant.heading);
    context.fillStyle = ant.carrying_food ? "#ffd166" : "#7ed6c0";
    context.strokeStyle = "#071016";
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(13, 0);
    context.lineTo(-8, -7);
    context.lineTo(-5, 0);
    context.lineTo(-8, 7);
    context.closePath();
    context.fill();
    context.stroke();
    context.restore();
    context.fillStyle = "rgba(230,244,248,.7)";
    context.font = "10px ui-monospace";
    context.textAlign = "left";
    context.fillText(`${ant.body_id}`, x + 10, y - 10);
  });

  context.fillStyle = "rgba(223, 239, 245, .78)";
  context.font = "12px ui-monospace";
  context.textAlign = "left";
  context.fillText(
    `authoritative tick ${frame.tick} · ${frame.ants.length} bodies`,
    18,
    26,
  );
}

export type CanvasTool = "select" | WorldObjectKind;

interface WorldCanvasProps {
  frame: AppFrame | null;
  tool: CanvasTool;
  selectedObjectId: string | null;
  onPlaceObject: (
    kind: WorldObjectKind,
    start: [number, number],
    end?: [number, number],
  ) => void;
  onMoveObject: (
    object: WorldObjectSnapshot,
    delta: [number, number],
  ) => void;
  onSelectObject: (objectId: string | null) => void;
}

function pointSegmentDistance(
  point: [number, number],
  start: [number, number],
  end: [number, number],
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
  point: [number, number],
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
      ) <= Math.max(0.55, object.physical_radius + 0.25)
    ) {
      return object;
    }
  }
  return null;
}

export function WorldCanvas({
  frame,
  tool,
  selectedObjectId,
  onPlaceObject,
  onMoveObject,
  onSelectObject,
}: WorldCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [stickStart, setStickStart] = useState<[number, number] | null>(null);
  const [stickEnd, setStickEnd] = useState<[number, number] | null>(null);
  const [moveStart, setMoveStart] = useState<[number, number] | null>(null);
  const [movingObject, setMovingObject] =
    useState<WorldObjectSnapshot | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    if (frame) {
      drawWorld(
        context,
        frame,
        selectedObjectId,
        stickStart && stickEnd ? [stickStart, stickEnd] : null,
      );
    } else {
      context.fillStyle = "#09131a";
      context.fillRect(0, 0, WIDTH, HEIGHT);
      context.fillStyle = "rgba(223, 239, 245, .55)";
      context.font = "16px system-ui";
      context.textAlign = "center";
      context.fillText("创建实验后，这里只绘制真实环境帧", WIDTH / 2, HEIGHT / 2);
    }
  }, [frame, selectedObjectId, stickStart, stickEnd]);

  function eventWorldPoint(
    event: React.PointerEvent<HTMLCanvasElement>,
  ): [number, number] {
    return canvasToWorld(
      event.clientX,
      event.clientY,
      event.currentTarget.getBoundingClientRect(),
    );
  }

  function pointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!frame) return;
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
  }

  function pointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    const point = eventWorldPoint(event);
    if (stickStart) {
      setStickEnd(point);
    }
  }

  function pointerUp(event: React.PointerEvent<HTMLCanvasElement>) {
    const point = eventWorldPoint(event);
    if (stickStart) {
      if (Math.hypot(point[0] - stickStart[0], point[1] - stickStart[1]) >= 0.5) {
        onPlaceObject("wood_stick", stickStart, point);
      }
      setStickStart(null);
      setStickEnd(null);
    } else if (movingObject && moveStart) {
      const delta: [number, number] = [
        point[0] - moveStart[0],
        point[1] - moveStart[1],
      ];
      if (Math.hypot(...delta) >= 0.05) {
        onMoveObject(movingObject, delta);
      }
    }
    setMovingObject(null);
    setMoveStart(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return (
    <canvas
      ref={ref}
      className="world-canvas"
      width={WIDTH}
      height={HEIGHT}
      aria-label="数字蚂蚁真实环境"
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={pointerUp}
      style={{ cursor: tool === "select" ? "grab" : "crosshair" }}
    />
  );
}
