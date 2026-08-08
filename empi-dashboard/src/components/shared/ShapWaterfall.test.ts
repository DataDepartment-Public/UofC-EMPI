import { describe, expect, it } from "vitest";
import { arrowBarPath } from "./ShapWaterfall";

/** Pull the "x y" coordinate pairs out of an SVG path in order. */
function points(d: string): [number, number][] {
  return [...d.matchAll(/[ML] (-?[\d.]+) (-?[\d.]+)/g)].map((m) => [
    Number(m[1]),
    Number(m[2]),
  ]);
}

const BOX = { x: 100, y: 50, width: 60, height: 20 };
const midY = BOX.y + BOX.height / 2;

describe("arrowBarPath", () => {
  it("points the head right when confidence rose", () => {
    const pts = points(arrowBarPath({ ...BOX, pointsRight: true })!);
    const rightmost = Math.max(...pts.map(([px]) => px));
    expect(rightmost).toBe(BOX.x + BOX.width);
    // The tip stands alone on the centre line.
    expect(pts.filter(([px]) => px === rightmost)).toEqual([[rightmost, midY]]);
  });

  it("points the head left when confidence fell", () => {
    const pts = points(arrowBarPath({ ...BOX, pointsRight: false })!);
    const leftmost = Math.min(...pts.map(([px]) => px));
    expect(leftmost).toBe(BOX.x);
    expect(pts.filter(([px]) => px === leftmost)).toEqual([[leftmost, midY]]);
  });

  it("keeps the trailing edge flat, spanning the bar's full height", () => {
    // A notched tail pulls the visible start inward and breaks the chain.
    const tailXs = { right: BOX.x, left: BOX.x + BOX.width };
    for (const [pointsRight, tailX] of [
      [true, tailXs.right],
      [false, tailXs.left],
    ] as const) {
      const onTail = points(arrowBarPath({ ...BOX, pointsRight })!).filter(
        ([px]) => px === tailX,
      );
      expect(onTail.map(([, py]) => py).sort()).toEqual(
        [BOX.y, BOX.y + BOX.height].sort(),
      );
      // Nothing on the centre line at the tail — that would be a notch.
      expect(onTail.some(([, py]) => py === midY)).toBe(false);
    }
  });

  it("leaves no gap between a bar's tip and the next bar's start", () => {
    // The waterfall reads as one chain: each bar begins exactly where the
    // previous one ended, at every height including the centre line.
    const first = { x: 100, y: 50, width: 60, height: 20 };
    const second = { x: first.x + first.width, y: 74, width: 40, height: 20 };

    const firstPts = points(arrowBarPath({ ...first, pointsRight: true })!);
    const secondPts = points(arrowBarPath({ ...second, pointsRight: true })!);

    const firstEnd = Math.max(...firstPts.map(([px]) => px));
    const secondStart = Math.min(...secondPts.map(([px]) => px));
    expect(secondStart).toBe(firstEnd);
  });

  it("never draws outside the bar's own extent", () => {
    // The head marks where the confidence landed; overhang would overstate it.
    for (const pointsRight of [true, false]) {
      for (const [px] of points(arrowBarPath({ ...BOX, pointsRight })!)) {
        expect(px).toBeGreaterThanOrEqual(BOX.x);
        expect(px).toBeLessThanOrEqual(BOX.x + BOX.width);
      }
    }
  });

  it("clamps the head so a short bar keeps a body instead of becoming a spike", () => {
    const narrow = { x: 0, y: 0, width: 6, height: 20 };
    const pts = points(arrowBarPath({ ...narrow, pointsRight: true })!);
    const shoulderX = Math.max(
      ...pts.filter(([, py]) => py !== 10).map(([px]) => px),
    );
    expect(shoulderX).toBeGreaterThan(0); // body survives
    expect(shoulderX).toBeLessThan(narrow.width); // head still visible
  });

  it("declines to shape a hairline bar", () => {
    expect(arrowBarPath({ x: 0, y: 0, width: 3, height: 20, pointsRight: true })).toBeNull();
  });

  it("closes a five-point arrow so it fills as one solid shape", () => {
    const d = arrowBarPath({ ...BOX, pointsRight: true })!;
    expect(d).toMatch(/Z$/);
    expect(points(d)).toHaveLength(5);
  });
});
