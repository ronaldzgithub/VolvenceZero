import { useEffect, useRef } from "react";
import type { AppFrame } from "./types";

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

function drawWorld(context: CanvasRenderingContext2D, frame: AppFrame): void {
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

export function WorldCanvas({ frame }: { frame: AppFrame | null }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    if (frame) {
      drawWorld(context, frame);
    } else {
      context.fillStyle = "#09131a";
      context.fillRect(0, 0, WIDTH, HEIGHT);
      context.fillStyle = "rgba(223, 239, 245, .55)";
      context.font = "16px system-ui";
      context.textAlign = "center";
      context.fillText("创建实验后，这里只绘制真实环境帧", WIDTH / 2, HEIGHT / 2);
    }
  }, [frame]);

  return (
    <canvas
      ref={ref}
      className="world-canvas"
      width={WIDTH}
      height={HEIGHT}
      aria-label="数字蚂蚁真实环境"
    />
  );
}
