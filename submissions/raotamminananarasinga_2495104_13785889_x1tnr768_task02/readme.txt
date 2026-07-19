Task 02 — How far to attaining knowledge?
CSO-610 Summer 2025-26
=========================================================

SUBMISSION NOTES
-----------------
Working with a teammate (teammate is based in a different state;
collaborated remotely over video calls).

Late days used: 0

Contribution: 50% / 50% (remote collaboration)

=========================================================

IMPORTANT: STEPS TO COMPLETE BEFORE FINAL SUBMISSION
-----------------------------------------------------
The following items require physical measurement or
manual verification and are marked as TODO in the code:

1. Open images/otherImage1.png in MS Paint.
   Hover over the following dots and record exact pixel (y) values:
     - A' (orange, MB red line)        : currently estimated at y=3890
     - C' (red, intersection/3rd pole) : currently estimated at y=2050
     - V' (blue, vanishing point)      : currently estimated at y=440
   Update these values in cross_ratio.py and re-run.

2. Go back to the lane where photos were taken.
   Measure:
     - d_A : distance from camera position to A' mark on ground  (e.g. 2 m)
     - d_C : distance from camera position to C' mark (manhole)  (e.g. 7 m)
   Update d_A_lane and d_C_lane in cross_ratio.py and re-run.

3. Rename/copy images for the submission folder:
     images/ThirdPersonView.jpg  →  images/firstPersonView.png
     images/FirstPersonView.jpg  →  images/thirdPersonView.png
     images/sample.JPG           →  images/bicycle.jpg

4. Compile answers.tex → answers.pdf  (using pdflatex or latexmk)
   Compile ReflectionEssay.tex → ReflectionEssay.pdf

=========================================================

DISCREPANCY NOTES
-----------------
- YOLO required imgsz=4032 (full image height) to detect the
  bicycle in sample.JPG because the bicycle is only ~60px tall
  in the 4K image at default 640px input size.

- The images FirstPersonView.jpg and ThirdPersonView.jpg were
  submitted with swapped roles (see answers.tex for details).
  ThirdPersonView.jpg is used for distance estimation (it has a
  cleaner YOLO person detection at conf=0.68).

- Reference pixel coordinates for A', C', V' in the bicycle
  image are estimated from visual inspection and should be
  verified in MS Paint for the final answer.
