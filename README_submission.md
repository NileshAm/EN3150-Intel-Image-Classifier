# EN3150 Assignment 03 - Pattern Forge

| Member | Index |
|---|---|
| AMARATHUNGE A.M.N.L. | 230038E |
| FERNANDO H.M.D. | 230180F |
| KEERAWELLA K.P.C.P. | 230332B |
| NETTIKUMARA N.A.H.G. | 230436X |

Repository: https://github.com/NileshAm/EN3150-Intel-Image-Classifier

## Environment

Use Python 3.11. The recorded runs used Python 3.11.17 and the package versions in
`requirements_submission.txt`. For a fresh environment:

```console
python -m pip install -r requirements_submission.txt
```

The reported inference measurements are Windows CPU measurements. A GPU is not
required. Platform, versions, class mapping and hashes are in the saved evaluation
manifest. Installation into a new environment has not been independently audited.

## Dataset

Download Intel Image Classification from
https://www.kaggle.com/datasets/puneet6060/intel-image-classification
and extract the class folders directly as follows:

```text
dataset/
  seg_train/
    buildings/ forest/ glacier/ mountain/ sea/ street/
  seg_test/
    buildings/ forest/ glacier/ mountain/ sea/ street/
```

Avoid a duplicate `seg_train/seg_train/` or `seg_test/seg_test/` nesting. The optional
unlabelled `seg_pred/` is used only by exploratory prediction code and is not needed
for the final evaluation. The images are not bundled in the code archive.

The loader combines both labelled folders and creates the documented stratified
70/15/15 split. Evaluation verifies ordered image/label fingerprints against saved
records, not only counts. A different directory iteration order may produce a
fingerprint mismatch; do not bypass it or silently evaluate different membership.

## Final evaluation without training

From the extracted code directory:

```console
python evaluate_saved_models.py
```

Use `python evaluate_saved_models.py --check` for an input/checksum preflight
without loading images or running predictions. This is not a full numerical rerun.

This entry point runs the imports, data preparation, split checks and four-model
evaluation cells from `CNN.ipynb` directly. It does not run training, grid searches
or the exploratory appendices. Matplotlib is non-interactive. Results are written to
a new `sota_results/evaluation/run_.../` directory; timing measurements can vary.
The original reference results and checkpoints are not overwritten.

Alternatively, use the notebook's documented evaluation-only sequence: imports,
Sections 1.1-1.2, shared evaluation in Section 5.3, and Section 6.1's table.

## Reproduce the optimizer audit

```console
python -m Utils.optimizer_audit
```

This starts a fresh three-optimizer, 20-epoch matched experiment in a separate run
directory. It is not needed just to inspect the completed audit. The code records
training-only initialization probes, the common initial-weight checksum, each
optimizer's full history, and validation-selected checkpoints. Test data is excluded
from fitting, initialization selection and optimizer comparison. The audit uses an
explicit depthwise fan-in-aware initialization shared by all three optimizers and
does not replace the default-initialized final Model B.

## Reproduce timing and architecture calculations

The notebook's Section 4.3.2 recomputes the initial/continuation epoch-time average
from saved history CSVs. Section 6.3 contains the complete code to load all four
saved architectures, verify checksums, and compute convolution/dense MAC counts
and largest single logical FP32 output sizes. No data fitting or test prediction is
performed by that analysis. It writes `sota_results/analysis/model_cost_analysis.json`.
Largest logical tensor size is not peak runtime memory.

## Scope of the submission

All original notebook cells and historical results remain. The supplementary search
and legacy original-folder evaluation cells are outside the final workflow; Run All
is not the reproduction entry point. The reference evaluation explicitly excludes
their test scores. Reproducing those optional historical searches needs additional
search/candidate artifacts not included in the compact submission package.

The final-model exports, their metadata/histories, custom-model registration files,
and the source checkpoint required by Model B continuation are included in the code
archive. Dataset-sheet completion must be confirmed separately by the group.
