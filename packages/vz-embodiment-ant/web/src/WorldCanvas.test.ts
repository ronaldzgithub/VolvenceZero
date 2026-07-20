import { describe, expect, it } from "vitest";
import { canvasToWorld, worldToCanvas } from "./WorldCanvas";

describe("world projection", () => {
  it("keeps the nest centered and flips the world y axis", () => {
    expect(worldToCanvas(0, 0, 960, 680)).toEqual([480, 340]);
    const [, positiveY] = worldToCanvas(0, 2, 960, 680);
    const [, negativeY] = worldToCanvas(0, -2, 960, 680);
    expect(positiveY).toBeLessThan(340);
    expect(negativeY).toBeGreaterThan(340);
  });

  it("converts pointer coordinates back into world coordinates", () => {
    const bounds = { left: 100, top: 50, width: 480, height: 340 };
    expect(canvasToWorld(340, 220, bounds)).toEqual([0, 0]);
    const projected = worldToCanvas(2, -3);
    const clientX = bounds.left + (projected[0] / 960) * bounds.width;
    const clientY = bounds.top + (projected[1] / 680) * bounds.height;
    const roundTrip = canvasToWorld(clientX, clientY, bounds);
    expect(roundTrip[0]).toBeCloseTo(2);
    expect(roundTrip[1]).toBeCloseTo(-3);
  });
});
