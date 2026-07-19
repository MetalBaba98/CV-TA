#!/usr/bin/env python3
"""
Task 02 -- Single-view metrology by the cross-ratio (June 03 / May 25 lecture).

Four world points lie on the MB->Arch line, ordered by distance from camera:
    A' = MB / red line        s = 0           (reference)
    B' = bicycle              s = unknown
    C' = intersection (pole3) s = d_int        (reference)
    V' = vanishing point      s = +infinity   (image of the point at infinity)

The cross-ratio is a projective invariant, so
    CR(A',B';C',V')  in the image  ==  CR(A,B;C,V)  in the world.

Because cross-ratio is invariant to any affine reparametrisation of the
image line, we may use each point's y-pixel as its 1-D coordinate -- no need
to measure x. With the vanishing point as the image of infinity the world
side collapses to a simple ratio and we solve for the bicycle's distance.

Reading points in MS Paint: read the y-pixel of A', B', C', V' along your
dark-blue line. d_int and d_arch are the real-world reference distances
(intersection and arch, from Google Maps; round to nearest 10 m).
"""
import argparse


def cross_ratio(p1, p2, p3, p4):
    """CR(p1,p2;p3,p4) = (p1-p3)(p2-p4) / ((p2-p3)(p1-p4))."""
    return ((p1 - p3) * (p2 - p4)) / ((p2 - p3) * (p1 - p4))


def bicycle_distance(yA, yB, yC, yV, d_int):
    """Return s_B = world distance of bicycle from MB (metres)."""
    cr_img = cross_ratio(yA, yB, yC, yV)
    # world: CR = d_int / (d_int - s_B)  ->  s_B = d_int * (1 - 1/CR)
    s_B = d_int * (1.0 - 1.0 / cr_img)
    return s_B, cr_img


def main():
    p = argparse.ArgumentParser()
    # y-pixels along the blue line (refine A',C',V' in MS Paint)
    p.add_argument("--yA", type=float, default=3498, help="MB / red line (measured)")
    p.add_argument("--yB", type=float, default=2153, help="bicycle ground contact (bbox bottom)")
    p.add_argument("--yC", type=float, default=2050, help="intersection / 3rd pole base  [ESTIMATE - refine]")
    p.add_argument("--yV", type=float, default=1650, help="vanishing point             [ESTIMATE - refine]")
    # real-world reference distances (verify on Google Maps; nearest 10 m)
    p.add_argument("--d_int", type=float, default=50.0, help="MB -> intersection (m)")
    p.add_argument("--d_arch", type=float, default=160.0, help="MB -> arch (m)")
    a = p.parse_args()

    s_B, cr = bicycle_distance(a.yA, a.yB, a.yC, a.yV, a.d_int)
    d_bike_arch = a.d_arch - s_B

    print("=" * 56)
    print("  INPUTS (y-pixels)")
    print(f"   A' MB={a.yA:.0f}  B' bike={a.yB:.0f}  C' int={a.yC:.0f}  V' vp={a.yV:.0f}")
    print(f"   references: d_int={a.d_int} m   d_arch={a.d_arch} m")
    print("-" * 56)
    print(f"  image cross-ratio CR        : {cr:.4f}")
    print(f"  bicycle distance from MB    : {s_B:6.1f} m")
    print(f"  >> BICYCLE -> ARCH distance : {d_bike_arch:6.1f} m")
    print("=" * 56)

    # ---- Q5: does the reference point inside the bbox matter? quantify ----
    print("\n  Q5  Sensitivity to reference point within the YOLO bbox")
    print("      (bicycle box spans y = 2013 top .. 2153 bottom)")
    print("      ref point        y_B    bike->arch")
    for name, yB in [("top (saddle)", 2013), ("center", 2083),
                     ("bottom (wheel contact)", 2153)]:
        sb, _ = bicycle_distance(a.yA, yB, a.yC, a.yV, a.d_int)
        print(f"      {name:22s} {yB:4.0f}   {a.d_arch - sb:6.1f} m")
    sb_top, _ = bicycle_distance(a.yA, 2013, a.yC, a.yV, a.d_int)
    sb_bot, _ = bicycle_distance(a.yA, 2153, a.yC, a.yV, a.d_int)
    spread = abs((a.d_arch - sb_top) - (a.d_arch - sb_bot))
    print(f"      -> spread across the 140-px box: {spread:.1f} m "
          f"(use the wheel/ground-contact point, not the box centre)")


if __name__ == "__main__":
    main()
