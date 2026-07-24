import "@testing-library/jest-dom/vitest";

const gradient = {
  addColorStop: () => undefined,
};

Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
  value: () => ({
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineCap: "butt",
    globalAlpha: 1,
    globalCompositeOperation: "source-over",
    font: "",
    textAlign: "",
    beginPath: () => undefined,
    bezierCurveTo: () => undefined,
    quadraticCurveTo: () => undefined,
    arc: () => undefined,
    ellipse: () => undefined,
    closePath: () => undefined,
    moveTo: () => undefined,
    lineTo: () => undefined,
    clearRect: () => undefined,
    fillRect: () => undefined,
    fillText: () => undefined,
    fill: () => undefined,
    stroke: () => undefined,
    save: () => undefined,
    restore: () => undefined,
    translate: () => undefined,
    rotate: () => undefined,
    setLineDash: () => undefined,
    createLinearGradient: () => gradient,
    createRadialGradient: () => gradient,
  }),
});
