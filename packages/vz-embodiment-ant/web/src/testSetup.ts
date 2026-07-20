import "@testing-library/jest-dom/vitest";

Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
  value: () => ({
    fillStyle: "",
    font: "",
    textAlign: "",
    fillRect: () => undefined,
    fillText: () => undefined,
  }),
});
