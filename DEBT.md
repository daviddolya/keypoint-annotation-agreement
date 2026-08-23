# Debt on code I did not write myself

To be cleared before the first job application. Anything I cannot explain in
three minutes gets rewritten by hand or dropped from the CV.

| Date | What was written | File | What I must be able to explain |
|---|---|---|---|
| 2026-08-21 | reading the COCO Keypoints format | `common/keypoints.py` | what v=0, v=1, v=2 mean and why the order of the 17 points cannot be changed |
| **2026-08-21** | **OKS** | `common/oks.py` | **asked in interviews.** Why the normalisation by area and why a per-joint sigma; why an offset of exactly one sigma gives exp(-1/2) |
| **2026-08-21** | **person matching** | `common/oks.py` | **asked in interviews.** Why matching on the same quantity you measure with is a bad idea, and what happens to the worst pairs when you do |
| 2026-08-21 | per-joint PCK, kappa on the flag | `common/oks.py` | why the tolerance is expressed in sigmas rather than pixels |
| 2026-08-21 | summary metrics and breakdowns | `annotation/keypoint_agreement.py` | how "mine without a match" differs from "dropped by the ground-truth filter" |
| 2026-08-21 | download, frame selection, skeleton config, rehearsal, rendering, README | `tools/` | nothing, this is scaffolding |
| 2026-08-23 | the two decisions on the price of a pixel | `reports/notes.md` | the decisions are mine (4000 px² threshold, zoom on face and wrists), the wording and the supporting numbers are not; I must be able to state the OKS ceiling by person size and the one-sigma tolerance by joint |
| 2026-08-23 | `GUIDELINES.md` version 1, five sections | `annotation/GUIDELINES.md` | four decisions are mine (point beyond the border: do not place it; hidden point: v=1 when it can be estimated to within a sigma; minimum eight points; 4000 px² threshold). The text and the supporting numbers were assembled from my answers. I must be able to explain why the border rule here is the opposite of the tracking stage |
| 2026-08-23 | disagreement analysis and the report text | `reports/keypoint_report.md`, `annotation/GUIDELINES.md` v2 | the decisions are mine: do not re-annotate the swapped-sides figure, adopt the COCO rule for ears, do not pretend the occluded boundary is fixable. The text, the numbers and three diagnostics (side swap tested across all pairs, distance to the image border, error on hidden points) are not. I must be able to explain why matching by OKS hides the worst pairs |
