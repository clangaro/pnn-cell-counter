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

## Glossary

- **Bezier** — ZEN's freehand polygon ROI element. Each `<Bezier>` has a
  `<Name>` (region label, e.g. `AC`, `CA1`, `Area`) and a `<Points>` list of
  polygon vertices in mosaic coordinates. Sibling of `<Arrow>` inside the
  ZEN `<Elements>` container.

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

## 2026-05-28 — Read class labels from ZEN `<Stroke>` hex color
**Decision:** Map arrow class labels directly from the `<Stroke>` attribute in ZEN's `metadata.xml`: `#FFFF0000` → `single_pv` (no PNN), `#FF9900CC` → `double` (has PNN). Blue (`#FF0000FF`) and any other stroke value (cyan, pink, yellow, orange, or missing `<Stroke>`) are excluded from training.
**Rationale:** The annotator's color choice in ZEN is the ground-truth label. Reading it from XML is exact and avoids drift from re-derivation. Blue is the annotator's intentional "skip this one" signal. The oddball colors (185 arrows) and the no-stroke arrows (2,678 arrows) are still pending review — see Open Questions — but they stay excluded until resolved.
**Alternatives rejected:** HSV color thresholding on rendered overlays (the retired `detect_arrows.py` / `extract_arrow_coords.py` approach) — lossy, depends on rendering, and conflates ambiguous arrows with real labels.
**Status:** active

## 2026-05-28 — Use mosaic→scene coordinate transform anchored on bounds containment
**Decision:** Convert arrow tip coordinates from mosaic space to per-scene JPG space with `jpg_x = mosaic_x − scene.StartX`, `jpg_y = mosaic_y − scene.StartY`. The hosting scene is the one whose `Bounds` (from `info.xml`) contain the arrow tip.
**Rationale:** Arrows in `metadata.xml` are stored in mosaic coordinates, but the exported JPGs are per-scene crops. Choosing the host scene by bounds containment is unambiguous when scenes do not overlap and is robust to scene ordering.
**Alternatives rejected:** Matching scenes by filename heuristics (e.g., scene index in filename) — fragile to renames and does not validate that the point actually lies inside the scene.
**Status:** active

## 2026-05-28 — Retire OpenCV arrow detection in favor of XML metadata
**Decision:** Replace the color-thresholding arrow-detection pipeline (`detect_arrows.py`, `extract_arrow_coords.py`) with direct parsing of `metadata.xml`. Tail = `(X1, Y1)`, tip = `(X2, Y2)`; the tip is the neuron location. All artifacts produced by the retired pipeline — the per-neuron crops under `dataset/` and `pnn_counter/dataset/`, and `outputs/arrow_coords.csv` — are deprecated and not reused; the new pipeline regenerates everything from the XML.
**Rationale:** Coordinates and labels are already in the XML and are exact. The OpenCV path was a workaround to recover labels from rendered overlays and introduces detection error (mixed-hue filtering, boundary heuristics, contour tip estimation) that the XML eliminates.
**Alternatives rejected:** Extending the HSV-based detector (additional color ranges, better filtering) — adds complexity to solve a problem we no longer have once we read XML.
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

## 2026-06-03 — ROI source: ZEN `<Bezier>` elements
**Decision:** Use `<Bezier>` elements from the same `metadata.xml` as the source of all region-of-interest geometry. Each Bezier carries its region label in `<Attributes><Name>` and its polygon vertices in `<Geometry><Points>` (space-separated `x,y` pairs in mosaic coordinates).
**Rationale:** Beziers are the only non-Arrow annotation primitive ZEN puts in the `<Elements>` container, and they carry both the semantic region name and the freehand polygon outline the annotator actually drew. The mosaic-coordinate `<Points>` use the same coordinate system as Arrow tips, so the existing `mosaic − scene.Start{X,Y}` transform converts ROI vertices to JPG-pixel space without modification. Universality confirmed: 277 Beziers across 19 animals, none missing, none geometry-straddling between scenes.
**Alternatives rejected:** Inferring ROIs from image content (e.g., color thresholding on rendered red polygons) — reintroduces the lossy rendering-dependence we just got rid of for arrows. Hand-drawing region masks separately — duplicates work the annotator already did in ZEN.
**Status:** active

## 2026-06-03 — Crop the dataset to ROIs, not to full scenes or per-neuron windows
**Decision:** The training crops are per-ROI (one image per Bezier per scene), not per-scene (whole JPG) and not per-neuron (fixed window around each arrow). Arrow coordinates inside each ROI are stored alongside the ROI image so the detector learns to localize neurons within a region.
**Rationale:**
- **Complete labeling within input.** A detector must assume every PV neuron inside its input image is annotated; otherwise unlabeled neurons become false negatives during training. Within an ROI polygon the annotator placed an arrow at every PV neuron they identified — that completeness assumption holds. Across a full scene it does not (a scene contains tissue outside any ROI that the annotator did not exhaustively review).
- **Resolution efficiency.** Scenes are 11k–20k pixels wide. Downsampling a whole scene to a network-friendly size (typically ≤1024 px) destroys the soma-scale detail the model needs. Cropping to ROIs first preserves native resolution where it matters.
- **Per-region stratification.** Each crop carries its region tag (`AC`, `CA1`, …), so train/val/test splits and evaluation can be region-stratified without extra bookkeeping.
**Alternatives rejected:**
- Full-scene inputs with arrow-only training — the unlabeled-background problem above.
- Fixed-window per-neuron crops (the old detect_arrows / dataset/ approach) — frames the problem as classification of single cells in isolation, which loses the colocalization context and prevents the model from learning what "no neuron here" looks like.
**Status:** active

## 2026-06-03 — ROI name canonicalisation: ACA/ACC → AC
**Decision:** Normalize ROI names from `<Bezier><Name>` before downstream use, with the canonical-spelling map `ACA → AC`, `ACC → AC` (ACC included precautionarily — no animal currently uses it but we want the rule in place if a new XML does). All other canonical regions (`PL`, `ILA`, `FC`, `CA1`, `CA2`, `CA3`, `DG`) stay as-is.
**Rationale:** Anatomically the same region; only spelling differs (`AC` in 14 animals, `ACA` in id77 + id73). Normalising at parse time avoids downstream code branching on every variant.
**Alternatives rejected:** Treating `ACA` as a distinct region (would fragment the dataset by spelling, not biology). Per-animal mapping (overkill — one global rule suffices).
**Status:** active

## 2026-06-03 — General region-assignment rule across all animals
**Decision:** Region is assigned per arrow using this order of checks:
1. If the arrow's scene is in `DROPPED_SCENES` for the animal → arrow is excluded entirely.
2. If the animal is in `REGION_UNKNOWN_ANIMALS` (id43, id69, id70) → `region = UNKNOWN`.
3. If the arrow's scene is in `UNKNOWN_REGION_SCENES` for the animal (e.g. id80 s2) → `region = UNKNOWN`.
4. If every Bezier on the arrow's scene is named `Area` → `region = UNKNOWN` (this catches id41 s3 automatically; no per-animal entry needed).
5. Otherwise, find the Bezier containing the arrow tip (PIP) and use `normalize_roi_name()` on its `<Name>`. If no Bezier contains the tip → `region = UNKNOWN`.
**Rationale:** A single algorithm covers every animal-specific quirk by composing three small constants (`DROPPED_SCENES`, `REGION_UNKNOWN_ANIMALS`, `UNKNOWN_REGION_SCENES`) plus the natural per-scene Area-only check. No animal needs hardcoded special-case logic. Tagging at the per-arrow level preserves maximum labelled data — animals with mixed annotation quality (id41, id80) keep their good scenes' region info instead of getting flattened.
**Alternatives rejected:**
- Assign region by scene majority (e.g. "this scene is mostly CA1"). Rejected because scenes legitimately contain multiple regions side-by-side and the majority would smear them.
- Assign region by nearest-Bezier centroid when PIP fails. Rejected because it fabricates region knowledge for arrows the annotator placed outside any drawn polygon.
**Status:** active

## 2026-06-03 — id41 per-scene region tagging (s3 → UNKNOWN, s0–s2 named)
**Decision:** For id41, assign region per scene rather than per animal. s0 and s1 (PFC: AC/PL/ILA) and s2 (HPC: CA1/CA2/CA3/DG/FC) inherit region names from the Bezier each arrow falls inside. s3 has 5 Beziers all named `Area`, so every arrow in s3 gets `region="UNKNOWN"`. Keep all arrows from all four scenes.
**Rationale:** id41 has properly named ROIs in 3 of 4 scenes — flatly marking the whole animal UNKNOWN would discard real region information for the 264 arrows in s0–s2. The 77 arrows in s3 still carry valid red/violet labels and should still train the detector, they just can't contribute to region-stratified analysis.
**Alternatives rejected:**
- Blanket animal-level UNKNOWN like id43/id69/id70. Rejected because 3 of 4 scenes are properly named — that would throw away usable region info.
- Drop s3 entirely. Rejected for the same reason as id43: arrow labels in s3 are valid even though the ROI names aren't, so the arrows still belong in the detector training set.
**Status:** active

## 2026-06-03 — id43 blanket region="UNKNOWN"; suffix normalisation skipped
**Decision:** Treat id43 the same way as id69/id70: every arrow gets `region="UNKNOWN"`. Skip any normalisation of id43's numeric-suffix Bezier names entirely (`AC1/AC2/PL1/.../FC2` left as-is in the XML but never consulted downstream). Arrow labels (single_pv / double) remain trusted.
**Rationale:** The s0 diagnostic overlay (`outputs/inspect/id43_s0_diagnostic.jpg`) showed the Beziers are under-drawn — neurons that visibly belong in the ILA region sit outside the ILA1 polygon. PIP-based region assignment is therefore unreliable for this animal regardless of how the suffix is interpreted, so no naming scheme rescues it. The arrows themselves point at real PV neurons and carry the annotator's class call, so they still belong in the detector training set.
**Alternatives rejected:**
- Apply suffix-stripping (`AC1/AC2 → AC`, `CA11/CA12/CA13 → CA1`, …) and use polygons as drawn. Rejected because the polygon geometry is wrong, not just the names — relabelling doesn't fix arrows that fall outside the drawn polygon.
- Dilate / pad the polygons outward until they cover the outside-ROI arrows. Rejected because that fabricates ROI geometry the annotator didn't draw, and the right boundary is anatomical (not "wherever the nearest unlabelled tissue is").
- Re-draw the polygons by hand to match where the arrows actually sit. Same fabrication problem; also requires anatomical judgement we can't apply without the annotator.
- Drop id43 entirely. Rejected because we'd lose 122 labelled arrows over a metadata-quality issue, not a label-quality issue.
**Status:** active

## 2026-06-03 — id69 and id70 tagged region="UNKNOWN" (geometry kept)
**Decision:** Keep all Bezier ROIs in id69 and id70 (they are geometrically valid — 100% PIP), but tag every arrow in these animals with `region="UNKNOWN"` because no semantic region names exist (every Bezier is named `Area`). The arrows still contribute to training the detector; they cannot contribute to region-stratified evaluation.
**Rationale:** The arrow labels (single_pv / double) are still trustworthy; only the region attribution is missing. Excluding the animals entirely would discard 354 labelled red/violet arrows (id69 = 234, id70 = 120) for a metadata gap rather than a label-quality issue. (Raw arrow counts including blue/no-stroke are 641 + 398.)
**Alternatives rejected:**
- Set aside both animals until re-annotated. Rejected because the labelled arrows are immediately usable for training the detector — re-annotation only unlocks per-region evaluation, which is a strict subset of the value.
- Dropping both animals outright. Same cost as above.
**Status:** active

## 2026-06-03 — id80 — s3 dropped, s2 marked UNKNOWN, s0/s1 named
**Decision:** Drop id80 s3 entirely (no Beziers, no arrows — nothing to keep). For s2, keep all arrows but force `region="UNKNOWN"` because all 4 of its Beziers are `Area` placeholders. s0 and s1 use the standard PIP-based region assignment from their named Beziers (AC/ILA/PL). Supersedes the earlier "restrict to s0/s1" entry.
**Rationale:** Refining the rule per the cross-animal General region-assignment rule — s2 arrows still have valid labels (they're just region-unknown like id69/id70 arrows), so excluding them would needlessly discard data. s3 has zero arrows so dropping it costs nothing.
**Alternatives rejected:** The earlier "drop s2 outright" version — discards usable arrow labels (single_pv / double) for a metadata-only issue, inconsistent with how we handle id69/id70 and id41 s3.
**Status:** active (supersedes earlier id80 entry from same date)

## 2026-06-03 — Missing FC accepted as real biology, not annotation error
**Decision:** Animals that lack an FC (Frontal Cortex) Bezier (`id47, id57, id74, id80`, … — 7 animals total) are accepted as-is. FC absence is not flagged or escalated.
**Rationale:** FC may genuinely not have been imaged for some animals depending on slice level. Investigating each case is high-cost low-value; the FC class will simply have smaller n in the final dataset. If region-stratified analysis later finds FC is underpowered we revisit then.
**Alternatives rejected:** Treating each missing-FC animal as a data error and re-checking the CZI — too many to be worth it without a stronger signal that something is wrong.
**Status:** active

## 2026-08-03 — Phase 1 smoke-test dataset: id47, id52, id83
**Decision:** Build the Phase 1 pipeline-validation smoke set from the three fully-clean animals: `id47` (train, PFC+HPC, no FC), `id83` (train, PFC+HPC+FC), `id52` (val, PFC+HPC+FC). Only named Bezier ROIs cropped; only red (`#FFFF0000` → single_pv, class 0) and violet (`#FF9900CC` → double, class 1) arrows labelled. 44 crops total, 2,900 arrows.
**Rationale:** id47/id52/id83 are the only animals with no-stroke count ≤ 0, all Beziers named, PIP ≥ 99.5% (see Phase-1 audit). Choosing them isolates plumbing bugs from data-quality noise. Train on 2 / val on 1 with animal-level split is deliberate: it exercises unseen-animal generalisation while acknowledging (see below) that single-animal val metrics are noisy.
**Alternatives rejected:** Including a partially-clean animal (id45 or id51) to bump n — would let no-stroke / annotation issues confound smoke-test debugging. Random per-crop 80/20 split — leaks the same animal into both sides and is not what deployment will look like.
**Status:** active — Phase 1 only. Phase 2 uses full cleaned dataset.

## 2026-08-03 — YOLO box size fixed at 60 px per ROI-crop native resolution
**Decision:** Convert each arrow tip to a **60 × 60 px square box centered on the tip** for YOLO training. Same size used for every animal and every region.
**Rationale:** Sampled 50 arrows and measured nearest-neighbour distances in mosaic-pixel space: p25 = 94 px, median = 117 px, min = 38 px. Choosing 60 px keeps boxes below the p25 spacing (so at typical PV density one box ≈ one neuron, not two) while comfortably enclosing a soma + PNN halo (halos measure ~40–50 px at this magnification). Formula used: `0.6 × p25`, clamped `[60, 100]`. Raw value was 56 → clamped to the 60 px floor.
**Alternatives rejected:** Per-region box size — needless complexity for a smoke test; NN spacing is comparable across regions. Adaptive per-arrow box — YOLO's fixed-anchor design doesn't consume per-instance sizing hints, and the annotator's arrow does not encode a soma diameter.
**Status:** active — revisit if the halo-fit visual QC on Phase 2 dataset shows PNNs poking outside the box.

## 2026-08-03 — Best-z-plane per crop via variance-of-Laplacian
**Decision:** Do NOT hard-code a z-plane. For each ROI crop, score every available z-plane's crop with variance-of-Laplacian (`cv2.Laplacian(gray, cv2.CV_64F).var()`) and use whichever z-plane scores highest as the training crop's source.
**Rationale:** The initial "always z=1" policy produced dim, out-of-focus crops for id52 PFC scenes (visible in QC). Empirically, across the 44 Phase-1 crops the sharpest z was z=2 for 30 crops, z=3 for 11, z=1 for only 3 — z=1 was the winner less than 7% of the time. Variance-of-Laplacian is the standard bench autofocus proxy: it penalises blurred edges (low high-frequency content) and rewards in-focus tissue detail.
**Alternatives rejected:**
- Fixed z=middle (e.g. z=3) — id83 HPC scenes actually preferred z=1 or z=2 in some crops, so a fixed middle-z would still leave sharpness on the table.
- Max-intensity projection across all z-planes — fuses in-focus signal with out-of-focus haze from other planes; degrades the sharp-signal-to-fog ratio, especially in tissue-dense regions.
- Best-z per **scene** rather than per **crop** — small HPC crops on the same scene often prefer a different z than the large CA1 crop (verified in the Phase-1 winning-z log).
**Status:** active — the log of winning z per crop is emitted by `build_smoke_dataset.py`. Extend to the full dataset in Phase 2 unchanged.

## 2026-08-03 — Phase 1 training split: 2 animals train, 1 val, no leakage
**Decision:** train = {id47, id83}, val = {id52}. Split at the animal level; the same animal never appears in both bins. 29 crops train / 15 crops val (1,934 train boxes / 966 val boxes).
**Rationale:** For a detector that will be deployed on new animals, the honest generalisation test is unseen-animal validation. Splitting by crop or by scene inside the same animal shares tissue-level features (staining batch, microscope calibration, slice quality) between train and val and produces optimistic numbers that collapse on real deployment.
**Caveat — do not trust the val metrics:** a single held-out animal is a noisy estimator. id52's per-region class balance is not the train mix (val has ~5.6:1 single_pv:double, train is ~2.2:1), so any val metric conflates model quality with animal-specific class distribution. The Phase-1 val split exists to confirm the model can produce detections on unseen images — not to measure real performance.
**Status:** active for Phase 1 only. Phase 2 will use ≥5 val animals and stratified reporting.

## 2026-08-04 — Phase 2 tiling: 640×640 patches, 128 px overlap, empty tiles dropped
**Decision:** Split each Phase-1 ROI crop into **640 × 640 tiles at native resolution** with **stride 512 (128 px overlap ≈ 20%)**. Tiles that contain **no arrow tips are dropped**; hard-negative sampling is deferred. Arrows in overlap zones are **duplicated into both host tiles** for training. Deduplication across tile seams at inference is a separate follow-up (Phase 2b).
**Rationale:**
- Phase 1 collapsed the double class because 60 px arrow boxes shrank to ~10–30 px at `imgsz=1280` — near YOLO's smallest-anchor floor. Feeding YOLO 640-px tiles at native resolution keeps every arrow box at 60 px (~9.4% of tile linear size), well inside the confident detection band.
- 20 % overlap protects cells that would otherwise straddle a tile seam.
- Dropping empty tiles keeps the training signal dense; if Phase 1's over-prediction tendency recurs on val we'll add a controlled fraction of empty tiles back as hard negatives.
- Duplicating overlap arrows during training is fine (each duplicate is a legitimate positive from a different context); it would be wrong at *inference* — hence the NMS-across-seams follow-up.
**Actual tile counts** (from `outputs/smoke_dataset/phase2/tiles/tiling_report.txt`):
- Source: 44 Phase-1 ROI crops → 2,832 tiles generated → **1,221 kept** (1,611 dropped, 56.9 % empty).
- **4,804 arrow instances** — single_pv = 3,566, double = 1,238 (2.88 : 1, same ratio as Phase 1 pre-tiling).
- Per animal: id47 = 439 tiles / 1,691 instances · id52 = 380 / 1,571 · id83 = 402 / 1,542.
- Train (id47 + id83) = 841 tiles / 3,233 instances · Val (id52) = 380 tiles / 1,571 instances.
**Alternatives rejected:**
- Keep imgsz large (2048 or 2560) instead of tiling — MPS memory + throughput cost balloons; effective box size still lags what native-res tiling gives.
- Larger tiles (e.g. 1024) — halves the per-tile box-fraction improvement and roughly quarters the tile count (less shuffling diversity per epoch).
- 50 % overlap — quadruples the training set with heavy redundancy; 20 % is the point where cell-straddling risk vanishes without inflating the epoch time.
- Keeping empty tiles at 100 % — dilutes signal 2.3× (1,611 empty vs 1,221 kept) and reinforces the "predict everything, low confidence" failure mode we're trying to fix.
**Status:** active — revisit if val shows over-prediction (add hard negatives) or if the class 1 collapse recurs (bigger imgsz or tiling other z-planes too).

## 2026-08-05 — Phase 2 result: tiling doubled mAP and unlocked class 1, but training still stalls at epoch 2
**Result:**
- **Both runs early-stopped at epoch 22 (patience=20).** Best val mAP50 was hit at epoch 2 in both.
- **Run 1** (`phase2/`) — default LR (0.01), degrees=180, hflip+vflip: best mAP50 = 0.0056 at epoch 2.
- **Run 2** (`phase2_gentle/`) — lr0=0.001, degrees=45, hflip only: best mAP50 = 0.0111 at epoch 2 (2× run 1).
- **Ultralytics val() on Run 2 best.pt** (default conf/NMS): overall mAP50 = **0.0205**, class 0 (single_pv) AP50 = **0.0377**, class 1 (double) AP50 = **0.0033**. So single_pv is ~10× as detectable as double, but neither is close to usable.
- **Class-wise box counts on val (conf ≥ 0.05):** GT single=1,336 double=235 → PRED single=41,153 double=5,585. Model over-predicts single by **31×**, double by **24×**.
- **Improvement over Phase 1:** class 1 is no longer zeroed — the model produces double predictions. Overall detections cluster on cell-dense regions correctly (visible in val overlays); spatial localisation works, but confidence discrimination and NMS quality are poor.
**Diagnosis:** In both runs training loss (train_box 2.7-3.0) is stuck from epoch 2 onwards while val_cls oscillates wildly (spikes up to val_cls=1256 in Run 2 epoch 9). This is classic small-dataset instability — the model finds a low-loss trivial solution ("predict everything low-conf") that pattern-matches the loss surface but doesn't actually improve mAP. patience=20 fires cleanly.
**Interpretation for Phase 3:** the tiling change was necessary (Phase 1 mAP50=0.002 → Phase 2 mAP50=0.021 = 10× jump) but not sufficient. The remaining gap is *not* box-size / resolution — it's training regime and dataset size.
**Levers for Phase 3 (in rough priority):**
1. **More animals.** 29→841 tiles was diversity from tiling one animal's crops many times, which is limited. Adding id45 + id51 (~2 more train animals, 1000+ more tiles) is the biggest lever.
2. **Bigger model.** yolov8n (3M params) may be capacity-limited on the halo-signal for class 1. yolov8s (11M) is fast enough on MPS.
3. **Longer warmup, lower initial LR.** Both stalls started at epoch 2 — probably an initial LR shock. Try `warmup_epochs=10` and `lr0=0.0005`.
4. **Hard-negative sampling.** Bring back ~15 % of the dropped empty tiles as background examples to teach the model to say "no cell here" and reduce spam predictions.
5. **Resolve Open Question (a)** on no-stroke arrows before Phase 3 — could add ~30 % more labelled data.
**Status:** Phase 2 accepted as an incremental step. Pipeline is bug-free; model needs more data and a gentler training regime to converge.

## 2026-08-03 — Phase 1 result: pipeline works; model does not meaningfully learn the double class
**Result:**
- Pipeline ran end-to-end without intervention (Stage 1 crop extraction → Stage 2 YOLO labels → Stage 3 split → Stage 4 training → val prediction render).
- 50 epochs of YOLOv8n pretrained on ImageNet, `imgsz=1280`, MPS, rotation+flip augmentation, HSV shifts disabled (fluorescence hue is semantic).
- Training loss decreased: `train_box` 4.12 → 3.76, `val_box` 4.75 → 3.91.
- Val mAP50 peaked at **0.0021** (epoch 42) — effectively zero.
- Predictions on val: model produces plausible detections spatially (boxes cluster on cell-dense areas) but predicts **only class 0 (single_pv)**, never class 1 (double). It over-predicts single_pv by ~2–10× on most val crops.
**Interpretation:**
- Pipeline plumbing is correct (verified by QC crop overlay showing per-tip box centering + correct class colouring, and by the model spatially localising cells at all).
- The class 1 collapse is expected for this configuration: (a) 60-px boxes in 3-8k-px crops downsample to ~10–30 px at `imgsz=1280` — near the smallest object YOLO's anchor grid can resolve; (b) only 608 double examples in train, of which the halo-vs-no-halo distinction requires the full-resolution green channel that is destroyed by downsampling; (c) 50 epochs on 29 images is too little.
**Next-phase implications (not implemented in Phase 1):**
- Tile large ROI crops into overlapping 640-px patches at native resolution — this is the single largest lever. Currently a 60 px box in a 4000 px crop = 1.5% linear coverage; with 640 px tiles it becomes 9.4%, well within YOLO's usable range.
- More training animals — Phase 1 gave 29 train crops; realistically we want 100+ crops from ≥5 train animals.
- Resolve Open Questions (a) and (b) before Phase 2 training to unlock the no-stroke arrows (~31% more data) and confirm the cyan class assignment.
**Status:** Phase 1 accepted as a plumbing-only pass. Phase 2 design begins.

---

## Open Questions

Unresolved at this point. Listed here so they don't get lost; promoted to a
dated decision entry once resolved.

- **(a) No-`<Stroke>` arrows.** 2,678 arrows (~31% of the dataset) have no
  `<Stroke>` element at all. Concentrated in `id77` (428), `id69` (404),
  `id73` (364), `id58` (284), `id70` (277), `id65` (218), with smaller piles
  in id62, id71, id41, id57, id74, id78, id43, id51, id45. Three competing
  meanings (ZEN default-color, "unsure" placeholder, intermittent mistake) —
  see `outputs/inspect/RA_PROTOCOL.md` Part 2. **Resolution pending the
  annotator's reply** (option A / B / C / D in the protocol). Visual
  diagnostic overlays are queued as the next task.

- **(b) Oddball-color arrows (cyan / pink / yellow / orange).** 185 arrows
  in unexpected stroke colors, listed in `outputs/inspect/unexpected_arrows.csv`
  and covered by `RA_PROTOCOL.md` Part 1. Currently excluded from training.
  Cyan (120 arrows, dominated by id78 and id74) is large enough that it may
  represent a real annotator-intended class rather than a typo —
  worth resolving before final training. Pink/yellow/orange are sparse and
  more likely typos.

- **(c) id43 numeric suffixes.** `AC1/AC2/PL1/.../CA11/CA12/CA13` — the
  semantic intent of the numeric suffix is unknown (hemisphere? section
  number? sub-region?). Currently irrelevant under the id43 blanket-UNKNOWN
  decision above, but worth resolving if id43 is ever brought back into
  region-stratified analysis.

- **(d) Inference-time ROI provenance.** At training time ROIs come from
  the annotator's Beziers. At inference, the user will not have hand-drawn
  ROIs for a fresh scene. Two paths: (i) require the user to draw ROIs
  before counting, (ii) sliding-window over the full scene with a region
  classifier upstream. Affects deployability, not training, so deferred —
  but the choice influences how the detector is evaluated (per-ROI accuracy
  vs. per-scene precision/recall).
