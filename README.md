# NPAM inference

This repository provides inference code for the nodule–pleural anatomy model (NPAM), developed for research on preoperative CT-based visceral pleural invasion assessment in lung adenocarcinoma.

## Scope

The release accepts an already prepared three-channel NumPy array. It does not include DICOM preprocessing, nodule segmentation, pleural-region segmentation, mask review, or model-training code.

## Input

The input must be a `float32` `.npy` array with shape `[3, 96, 96, 96]` and the following channel order:

1. CT grayscale, clipped to −1000 to 400 HU and linearly scaled to `[0, 1]`;
2. binary nodule mask;
3. binary pleural-region mask.

The cube corresponds to 1-mm isotropic voxels and is centered on the target nodule. The pleural-region mask is an operational approximately 3-mm imaging band and is not a microscopic pleural segmentation.

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CUDA environment if the default package is unsuitable.

## Weights

Download all five fold weights from the repository's release page and place them in `weights/`. See [`weights/README.md`](weights/README.md) for the required filenames.

After downloading, the files can be verified against [`WEIGHTS_SHA256SUMS.txt`](WEIGHTS_SHA256SUMS.txt).

## Example

```bash
python infer.py \
  --input examples/example_input.npy \
  --weights-dir weights \
  --device cuda \
  --output outputs/example_result.json
```

The output contains five fold scores and their arithmetic mean. The ensemble score is not a calibrated probability. The fixed score cutoff of 0.5 reproduces the threshold used for threshold-based reporting in the accompanying study; it is not validated for clinical decision-making outside that study cohort.

Small numeric differences may occur between CPU and GPU inference or across software builds.

## Intended use and limitations

This software is provided for research and reproducibility purposes only. It is not a medical device and must not be used as the sole basis for diagnosis or treatment. The published study evaluated manually reviewed nodule and pleural-region masks; performance with unreviewed automatic masks has not been established. External validation is required before use in other institutions or populations.

## Data availability

Patient-level imaging data are not included because of privacy and institutional restrictions. The example input must be distributed only if its release is permitted by the applicable ethics approval and institutional policy.

## Citation

Please cite the associated article after publication. **[Add final citation and DOI.]**

## License

**[Select and add the final code and model-weight licenses before public release.]**
