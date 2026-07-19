TASK 02 - Single View Geometry and Distance Estimation
======================================================

This readme is intentionally free of identifying information (per the
instruction to keep the zip payload anonymous). Name, roll number, late
days and team contribution are submitted separately in the second .txt
payload.

CONTENTS
--------
answers.tex          Main document (compile this).
answers.pdf          Compiled answers (8 pages).
task-body.tex        All answers, written inline in the student spaces.
02-task-header.tex   Provided header (unmodified).
HW.sty               Provided style file (unmodified).
ReflectionEssay.tex  Reflection essay source.
ReflectionEssay.pdf  Compiled reflection essay.
readme.txt           This file.
images/
  bicycle.jpg           YOLOv8 detection of the bicycle (Part 1).
  otherImage1.png       Dark-blue MB-arch axis with A',B',C',V' (Part 1).
  otherImage2.png       Google Maps "Measure distance" screenshot (Q1).
  firstPersonView.png   Part 2 first-person view (teammate on walkway, dusk).
  thirdPersonView.png   Part 2 third-person view (both people, ~7-8 m apart).
code/
  detect_bicycle.py        YOLOv8 detection -> bounding box + ground point.
  crossratio_distance.py   Cross-ratio distance solve + sensitivity sweep.
  requirements.txt         Python dependencies.

HOW TO COMPILE
--------------
  pdflatex answers.tex
  pdflatex answers.tex      (run twice for cross-references)

HOW TO RUN THE CODE
-------------------
  pip install -r code/requirements.txt
  python3 code/detect_bicycle.py images/sample.JPG     # detection
  python3 code/crossratio_distance.py                  # distance + Q5 table
Model weights (yolov8x.pt) are NOT included; Ultralytics downloads them
on first run, so nothing large is shipped in this zip.

RESULT (Part 1)
---------------
MB -> arch (Google Maps) = 160.35 m ~ 160 m.
Bicycle -> arch ~ 124 m (with vanishing point) / ~121 m (three finite
references); reported as ~120 m. Four pixel rows used (image 2268x4032):
A'=3500 (MB), B'=2150 (bicycle wheel), C'=2050 (intersection), V'=1720
(vanishing point).

RESULT (Part 2)
---------------
First/third-person photos taken at dusk (~7:25 PM) in the society
walkway outside B-6. Walkway depth to vanishing region ~100 m.
Teammate estimated ~7-8 m from the camera by cross ratio, matching the
~7-8 m separation visible in the third-person view.
