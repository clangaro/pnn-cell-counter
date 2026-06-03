# Decision Log

Append-only record of non-obvious design and methodological choices, plus
validation results that shift the plan. Routine execution notes (what script
ran, what it printed) belong elsewhere.

Format:

    ## YYYY-MM-DD — <short title>
    **Decision:** what we chose
    **Rationale:** why
    **Alternatives rejected:** what else we considered and why not
    **Status:** active | superseded by <entry>

---

## 2026-05-28 — Build a bespoke detector instead of using the off-the-shelf PNN counter
**Decision:** Implement our own PV/PNN colocalization detector rather than adopt the Ciampi/Lupori 2022 PNN counter.
**Rationale:** Our task is PV–PNN colocalization classification (PV soma with or without surrounding PNN halo), not PNN-only detection. We also already have manually annotated labels tied to PV-positive neurons, so a downstream tool tuned for PNN segmentation does not match the input/output shape we need.
**Alternatives rejected:** Ciampi/Lupori 2022 off-the-shelf PNN counter — solves a different task (segment PNNs) and does not consume our PV-anchored labels.
**Status:** active

## 2026-05-28 — Train on merged RGB images, not separated fluorescence channels
**Decision:** Feed the network the merged RGB confocal image containing both fluorescence channels in one frame.
**Rationale:** The biological signal of interest — orange PV soma co-located with a green PNN halo — only exists when both channels are visible together. Splitting channels removes the colocalization cue that defines the "double" class.
**Alternatives rejected:** Per-channel inputs (e.g., two-stream model on PV and PNN channels separately) — would force the network to relearn colocalization from disjoint inputs and discards the natural co-registration provided by the microscope.
**Status:** active

## 2026-05-28 — Read class labels from ZEN <Stroke> hex color
**Decision:** Map arrow class labels directly from the <Stroke> attribute in ZEN's `metadata.xml`: `#FFFF0000` → `single_pv` (no PNN), `#FF9900CC` → `double` (has PNN). Blue/unsure arrows are excluded from training.
**Rationale:** The annotator's color choice in ZEN is the ground-truth label. Reading it from XML is exact and avoids drift from re-derivation. Excluding blue/unsure arrows keeps the training set clean.
**Alternatives rejected:** HSV color thresholding on rendered overlays (current `detect_arrows.py` / `extract_arrow_coords.py` approach) — lossy, depends on rendering, and conflates ambiguous arrows with real labels.
**Status:** active

## 2026-05-28 — Use mosaic→scene coordinate transform anchored on bounds containment
**Decision:** Convert arrow tip coordinates from mosaic space to per-scene JPG space with `jpg_x = mosaic_x − scene.StartX`, `jpg_y = mosaic_y − scene.StartY`. The hosting scene is the one whose `Bounds` (from `info.xml`) contain the arrow tip.
**Rationale:** Arrows in `metadata.xml` are stored in mosaic coordinates, but the exported JPGs are per-scene crops. Choosing the host scene by bounds containment is unambiguous when scenes do not overlap and is robust to scene ordering.
**Alternatives rejected:** Matching scenes by filename heuristics (e.g., scene index in filename) — fragile to renames and does not validate that the point actually lies inside the scene.
**Status:** active

## 2026-06-03 — Parser stroke-handling policy
**Decision:** `parse_metadata.py` keeps red (`#FFFF0000` → single_pv) and violet (`#FF9900CC` → double) as the only labels. Blue (`#FF0000FF`) is excluded silently (counted per animal only). Any other stroke value, including missing `<Stroke>`, is excluded **and** printed as an anomaly with arrow tip coordinates so it can be located in the source CZI.
**Rationale:** Blue is an annotator-intentional "skip this one" signal — it's expected to appear and we don't want it to spam the run log. Everything else is unexpected: either an annotator typo, a default that needs interpretation (no-`<Stroke>`), or a future class we haven't accounted for. Surfacing those with locations makes them actionable instead of silently dropped.
**Alternatives rejected:** Hard-erroring on any unknown stroke (would block builds during the RA correction window); silently dropping unknowns (loses visibility into label drift).
**Status:** active

## 2026-06-03 — Coordinate transform verified end-to-end on id47
**Decision:** Accept `jpg_x = mosaic_x − scene.StartX`, scene assignment by bounds containment, as the canonical mosaic→scene transform. Pipeline build can proceed on this assumption.
**Rationale:** Rendered full-resolution overlays on freshly-exported Box JPGs for id47 (all 573 red+magenta arrows across 4 scenes). 100% of tips landed inside a scene, 100% stayed inside the JPG frame after transform, and every burned-in arrow in the matching JPG had its overlay on the soma. A separate spanning-arrow check showed every arrow in id47 has tail and tip in the same scene, ruling out boundary-crossing concerns. Stale local JPGs (older ZEN export than the current XML) initially showed unmarked arrows, which resolved when we re-rendered onto the Box JPGs — confirming the transform was right and the local exports were just out of sync.
**Alternatives rejected:** Filename-based scene matching, or affine recalibration via fiducial points — neither was needed since bounds containment was sufficient.
**Status:** active

## 2026-05-28 — Retire OpenCV arrow detection in favor of XML metadata
**Decision:** Replace the color-thresholding arrow-detection pipeline (`detect_arrows.py`, `extract_arrow_coords.py`) with direct parsing of `metadata.xml`. Tail = (X1, Y1), tip = (X2, Y2); the tip is the neuron location.
**Rationale:** Coordinates and labels are already in the XML and are exact. The OpenCV path was a workaround to recover labels from rendered overlays and introduces detection error (mixed-hue filtering, boundary heuristics, contour tip estimation) that the XML eliminates.
**Alternatives rejected:** Extending the HSV-based detector (additional color ranges, better filtering) — adds complexity to solve a problem we no longer have once we read XML.
**Status:** active
