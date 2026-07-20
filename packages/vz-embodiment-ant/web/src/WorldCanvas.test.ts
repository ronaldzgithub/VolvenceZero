import { describe, expect, it } from "vitest";
import { worldToCanvas } from "./WorldCanvas";

describe("world projection", () => {
  it("keeps the nest centered and flips the world y axis", () => {
    expect(worldToCanvas(0, 0, 960, 680)).toEqual([480, 340]);
    const [, positiveY] = worldToCanvas(0, 2, 960, 680);
    const [, negativeY] = worldToCanvas(0, -2, 960, 680);
    expect(positiveY).toBeLessThan(340);
    expect(negativeY).toBeGreaterThan(340);
  });
});
