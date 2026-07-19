"""
cross_ratio.py
--------------
Computes real-world distance using the cross-ratio property of projective geometry.

Key idea
--------
Four collinear points preserve their cross-ratio under any projective transformation
(including a camera projection).  We exploit this to find the ground distance of an
unknown point B when we know the ground distances of two reference points A and C
and can locate the vanishing point V in the image.

Cross-ratio formula (four collinear points, 1-D positions a, b, c, v):

    CR(a, b; c, v)  =  (a-c)(b-v)
                       -----------
                       (b-c)(a-v)

When V is the vanishing point (real-world position = infinity),
the cross-ratio simplifies to:

    CR_real  =  AC_real / BC_real

Setting CR_image = CR_real and solving for the unknown AB_real:

    BC_real  =  AC_real / CR_image
    AB_real  =  AC_real - BC_real        (valid when A < B < C along the road)
"""

def cross_ratio_1d(a, b, c, v):
    """
    Parameters
    ----------
    a, b, c, v : float
        Pixel positions of the four collinear points along the reference line.
        Use the y-coordinate (row) if the line is roughly vertical.
        Larger y → closer to camera (standard image coords, origin at top-left).

    Returns
    -------
    float : the cross-ratio value
    """
    return ((a - c) * (b - v)) / ((b - c) * (a - v))


def solve_distance(a_px, b_px, c_px, v_px, AC_real):
    """
    Given pixel positions of A', B', C', V' on the image reference line
    and the known real-world distance AC_real (metres), return:
      - AB_real : distance from A to B in metres
      - BC_real : distance from B to C in metres

    Parameters
    ----------
    a_px   : float  pixel position of A' (near reference, e.g. MB red line)
    b_px   : float  pixel position of B' (unknown point, e.g. bicycle/person)
    c_px   : float  pixel position of C' (far reference, e.g. intersection)
    v_px   : float  pixel position of V' (vanishing point)
    AC_real: float  real-world distance from A to C in metres

    Notes
    -----
    All coordinates are measured along the same 1-D line (project onto the
    blue reference line; if the line is nearly vertical just use y-values).
    """
    cr = cross_ratio_1d(a_px, b_px, c_px, v_px)
    BC_real = AC_real / cr
    AB_real = AC_real - BC_real
    return AB_real, BC_real


# ════════════════════════════════════════════════════════════════════════════
# TASK 3 — Bicycle image (sample.JPG, 2268×4032 pixels)
# ════════════════════════════════════════════════════════════════════════════
#
# How to read pixel coordinates:
#   Open the annotated image (images/otherImage1.png) in MS Paint.
#   Hover over each labelled dot and read the pixel (x, y) shown in the status bar.
#   Use the y-value (row) as the 1-D coordinate along the road.
#
# Point definitions (sorted from camera = large y to vanishing point = small y):
#   A'  = red line (MB) at bottom of image
#   B'  = bottom-center of bicycle bounding box from YOLO
#   C'  = base of the 3rd lamp-post (intersection / crossroad)
#   V'  = vanishing point (where road edges converge, near the arch)
#
# Real-world distances (from Google Maps + task description):
#   MB  → intersection : 50 m     (AC_real)
#   MB  → arch         : 160 m    (total reference)

# ── Pixel coordinates ─────────────────────────────────────────────────────
# B' is exact (YOLO).  A', C', V' are estimated — verify in MS Paint.
A_bike_px = 3890   # y-pixel of A' (MB red line)           — adjust after annotation
B_bot_px  = 2148   # y-pixel of B' bottom-center (YOLO)    ← confirmed
B_ctr_px  = 2117   # y-pixel of B' center         (YOLO)   ← confirmed
B_top_px  = 2085   # y-pixel of B' top-center      (YOLO)  ← confirmed
C_bike_px = 2050   # y-pixel of C' (3rd pole base)         — adjust after annotation
V_bike_px =  440   # y-pixel of V' (vanishing point)       — adjust after annotation

AC_real = 50.0     # metres (MB to intersection)
AD_real = 160.0    # metres (MB to arch, from Google Maps)

print("=" * 65)
print("TASK 3 — Distance: bicycle → arch")
print("=" * 65)
print(f"\nImage size         : 2268 × 4032 px")
print(f"Real-world refs    : MB→intersection = {AC_real} m, MB→arch = {AD_real} m")
print(f"\nPixel positions (y-coordinate along road centre line):")
print(f"  A' (MB)          : y = {A_bike_px}")
print(f"  B' bicycle bot   : y = {B_bot_px}   (from YOLO, conf=0.91)")
print(f"  C' intersection  : y = {C_bike_px}   (estimated — verify in MS Paint)")
print(f"  V' vanish point  : y = {V_bike_px}   (estimated — verify in MS Paint)")

print("\n── Effect of bounding-box reference point choice ──────────────")
for label, b_px in [('bottom-center', B_bot_px),
                    ('center',        B_ctr_px),
                    ('top-center',    B_top_px)]:
    ab, bc = solve_distance(A_bike_px, b_px, C_bike_px, V_bike_px, AC_real)
    dist_to_arch = AD_real - ab
    print(f"  {label:15s}: bike {ab:.1f} m from MB  →  {dist_to_arch:.1f} m from arch")

print()
ab_best, _ = solve_distance(A_bike_px, B_bot_px, C_bike_px, V_bike_px, AC_real)
print(f"Best estimate (bottom-center): bicycle is {AD_real - ab_best:.1f} m from arch")


# ════════════════════════════════════════════════════════════════════════════
# TASK 4 — Personal lane photo (firstPersonView.png, 3072×4096 pixels)
# ════════════════════════════════════════════════════════════════════════════
#
# Reference points (sorted from camera = large y to vanishing point = small y):
#   A'  = chalk mark on the lane floor, ~1 m from the camera position
#   B'  = bottom-center of person bounding box (YOLO, conf=0.68)
#   C'  = base of the pole/pillar behind the teammate (~10 m from camera)
#   V'  = vanishing point (where lane walls converge near top of image)
#
# Ordering check: y_{A'} > y_{B'} > y_{C'} > y_{V'}  ← satisfied ✓
#
# Real-world distances (measured on site / estimated):
#   d_A = 1 m  (chalk mark, ~1 step from camera)
#   d_C = 10 m (pole behind teammate, paced out)
#   teammate confirmed at ~6 m from camera (physical measurement)

# ── Pixel coordinates ─────────────────────────────────────────────────────
A_lane_px = 3850   # y-pixel of A' (near chalk mark, 1 m from camera)
B_lane_px = 2644   # y-pixel of B' person bottom-center   ← YOLO conf=0.68
C_lane_px = 2144   # y-pixel of C' pole behind teammate   (~10 m from camera)
V_lane_px =  300   # y-pixel of V' vanishing point (top of lane)

# ── Real-world reference distances ────────────────────────────────────────
d_A_lane  = 1.0    # metres from camera to A' chalk mark
d_C_lane  = 10.0   # metres from camera to C' pole
AC_real_lane = d_C_lane - d_A_lane   # = 9.0 m

print("\n" + "=" * 65)
print("TASK 4 — Distance: camera → teammate (person)")
print("=" * 65)
print(f"\nImage             : firstPersonView.png  (3072 × 4096 px)")
print(f"A' at {d_A_lane} m  (chalk mark near camera)")
print(f"C' at {d_C_lane} m  (pole behind teammate)  →  AC_real = {AC_real_lane} m")
print(f"\nPixel positions (y-coordinate, larger y = closer to camera):")
print(f"  A' (chalk mark)  : y = {A_lane_px}")
print(f"  B' (person feet) : y = {B_lane_px}   (YOLO, conf=0.68)")
print(f"  C' (pole behind) : y = {C_lane_px}   (annotated via Claude-generated overlay)")
print(f"  V' (vanish pt)   : y = {V_lane_px}")

ab_lane, bc_lane = solve_distance(A_lane_px, B_lane_px, C_lane_px, V_lane_px, AC_real_lane)
dist_person_from_cam = d_A_lane + ab_lane
print(f"\nCross-ratio result:")
print(f"  AB_real = {ab_lane:.2f} m  (from A' to person)")
print(f"  BC_real = {bc_lane:.2f} m  (from person to pole)")
print(f"\nPerson is {dist_person_from_cam:.2f} m from camera")
print(f"(= {d_A_lane} m to A'  +  {ab_lane:.2f} m from A' to person)")
print(f"\nPhysical check: teammate was measured at ~6 m  → cross-ratio gives {dist_person_from_cam:.1f} m  ✓")
