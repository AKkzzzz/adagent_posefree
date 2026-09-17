# Source and model attribution

This project integrates VGGT-Omega camera/depth inference, MoGe-2 metric depth,
GCA scale estimation and the existing R9/UFO overlap alignment and execution
optimizations. It does not claim these pretrained models as new models.

- `vendor/omega/` is copied from the user's installed VGGT-Omega working tree.
  Its FAIR Noncommercial Research License must remain at `vendor/omega/LICENSE`.
- `vendor/moge/` is copied from the user's installed MoGe working tree.
  Retain its MIT license and all nested DINOv2 source notices.
- `adagent_posefree/backend/` extracts/modifies functions from the captured
  `h200_adapter/`, `source_ufo/`, and `omega_adapter/` trees. Preserve those
  trees and their licenses and file-level notices. The captured UFO repository
  includes a CC BY-NC 4.0 license; some files carry their own Apache notices.
- `vendor/provenance.json` identifies copied model source by per-file SHA-256
  and source Git revision, including any uncommitted installed modifications.
- `checkpoints/fingerprints.json` records locally copied/downloaded weights.
  Weights are not distributed through this Git repository. Upstream access
  conditions and weight licenses continue to apply.

No blanket MIT/commercial-use grant is asserted for this combined distribution.
