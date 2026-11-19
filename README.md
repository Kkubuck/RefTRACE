# RefTRACE

Official repository of **RefTRACE: Reference-Faithful Evaluation and Dense Evidence Routing for Referring
Camouflaged Object Detection**.

The repository provides the fixed P/W/C protocol for R2C7K, three alternative C lists, and the evaluation code.
The code of RefTRACE and RefGuard will be added upon publication.

## P/W/C protocol

Each of the 1,987 test queries of R2C7K (64 classes) is evaluated with three fixed sets of five reference images.

| Condition | Reference set |
|---|---|
| P, matched | the five test references of the query class |
| W, random mismatch | the five test references of another class, drawn for each query with a fixed seed |
| C, confusable mismatch | the five test references of the nearest other class in the ICON-R prototype space |

A class prototype is the mean of the L2-normalised ICON-R descriptors of the five test references of the class.
Every evaluated model uses the same list.

`protocol/r2c7k_reference_faithful_triplets.csv` has one row per query.

| Column | Content |
|---|---|
| `query_stem`, `query_class` | query image and its class |
| `operational_regime` | `single_obj` or `multi_obj` |
| `P_support_class`, `W_support_class`, `C_support_class` | class of each reference set |
| `P_support_names`, `W_support_names`, `C_support_names` | five reference identifiers separated by `\|` |
| `P_payload_note`, `W_payload_note`, `C_payload_note` | condition tag |

### Alternative C lists

`protocol/alternative_C/` repeats the list with the C condition rebuilt from the prototypes of other encoders:
DINOv2-B/14 (pooled CLS token, 224 px), CLIP ViT-L/14 (projected image embedding, 224 px), and DINOv3
ConvNeXt-B (global average of the last stage, 384 px). Only the C columns differ from the main list.

### Checksums

```bash
cd protocol
sha256sum -c SHA256SUMS
```

## Evaluation

```bash
python -m pip install -r requirements.txt
```

Save the predicted probability maps as 8-bit PNG files at the original image size.

```text
predictions/
  P/<query_stem>.png
  W/<query_stem>.png
  C/<query_stem>.png
```

Matched-reference segmentation (S-measure, adaptive E-measure, weighted F-measure, MAE, and PSQ):

```bash
python scripts/evaluate_matched.py \
  --pred-root predictions/P \
  --gt-root /path/to/R2C7K/Camo/test/GT
```

Reference-faithful evaluation (PSQ, RRQ-W, RRQ-C, RRQ, and RFQ):

```bash
python scripts/evaluate_reference_faithful.py \
  --pred-root predictions \
  --gt-root /path/to/R2C7K/Camo/test/GT \
  --triplets protocol/r2c7k_reference_faithful_triplets.csv
```

Maps are evaluated in their original probability space without per-image min-max normalisation. PSQ is the
harmonic mean of S-measure, weighted F-measure, 1 - MAE, and adaptive E-measure. RRQ is the harmonic mean of the
size-balanced rejection qualities under W and C, and RFQ is the harmonic mean of PSQ and RRQ. To evaluate under an
alternative C definition, pass the corresponding list to `--triplets`.

## Data

The R2C7K images and masks are available from the dataset authors at https://github.com/zhangxuying1004/RefCOD.

## Citation

The citation will be added upon publication.
