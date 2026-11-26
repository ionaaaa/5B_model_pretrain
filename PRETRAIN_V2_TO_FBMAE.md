# pretrain_V2 → legacy FBMAE processed Lance

Entry point:

```text
build_fbmae_processed_from_pretrain_v2.py
```

The COS bucket must already be mounted as a filesystem. The converter has no
S3/COS storage-option path and never writes to an input Lance.

Each source dataset is written independently by default:

```text
OUTPUT_ROOT/<dataset_key>/data.lance
```

Rows from different datasets are never mixed, and `sample_id` / `global_idx`
restart from zero for each output Lance.

## Data contract

For each source row, only `eeg[:, :valid_samples]` participates in resampling
and waveform normalization. The order is:

```text
200 Hz real samples
→ standard 64-channel mapping
→ scipy.signal.resample_poly to 100 Hz
→ per-present-channel z-score over real samples only
→ right-pad to 1000 with exact zeros
→ FBMAE targets from the unmasked [64,1000] waveform
→ channel-major [640,100] / [640,5,5]
```

Temporal padding does not create a new time-validity mask because the reference
Lance has no such field. `token_valid_mask` is therefore the reference behavior:
the 64-channel presence mask repeated for all ten temporal tokens.

Targets reproduce the reference algorithm: 1 s Hann spectrogram, 0.1 s hop,
`nfft=128`, PSD density, five mean frequency bands, `log(power + 1e-6)`,
interpolation to 1000 samples, per-channel/per-band temporal z-score, and five
samples per 1-second patch.

## Channel catalogs

- `ds002722`: has `channel_names` in each Lance row; no extra catalog required.
- `ds002723`: use its canonical preflight JSON as the recording-keyed catalog.
- `ds002718`: the catalog contains only device labels such as `EEG001`; an
  authoritative source-to-10–10 mapping is required. The converter deliberately
  fails instead of treating device indices as electrode names.

Channel admission is checked across every row before an output Lance is
created. Directly matched source channels are retained; active source channels
outside the reference vocabulary are explicitly counted and dropped. Missing
standard slots remain zero/invalid. A dataset is marked
`SKIPPED_CHANNEL_MISMATCH` only if one or more rows have zero mapped active
channels, or if multiple active source channels ambiguously map to the same
standard slot. No partial output is written for a rejected dataset.

The `ds002718` mapping file format is:

```json
{
  "source_to_standard": {
    "EEG001": "Fp1",
    "EEG002": "Fpz"
  }
}
```

The example above only demonstrates syntax. It is **not** a valid or complete
mapping and must not be copied as scientific metadata.

## Dataset manifest

For regular multi-dataset processing, use a JSON manifest. Environment
variables are expanded and relative paths are resolved relative to the manifest.

```json
{
  "datasets": [
    {
      "dataset_key": "ds002722",
      "lance_path": "${COS_MOUNT_ROOT}/datasets/ds002722/releases/pretrain_V2/v022/data.lance"
    },
    {
      "dataset_key": "ds002723",
      "lance_path": "${COS_MOUNT_ROOT}/datasets/ds002723/releases/pretrain_V2/v001/data.lance",
      "catalog_path": "${COS_MOUNT_ROOT}/datasets/ds002723/releases/complete-recordings-eeg-native/v001/preflight.json"
    }
  ]
}
```

Use repeated `--dataset DATASET_KEY` to select a subset from the manifest. The
existing repeated `--input DATASET_KEY=PATH` CLI remains available for ad-hoc
runs.

The published paths explicitly present in `docs/STATE.md` are materialized as:

```text
pretrain_v2_published_all.json       63 datasets / 810,599 expected rows
pretrain_v2_published_shard_01.json  20 datasets / 269,443 expected rows
pretrain_v2_published_shard_02.json  21 datasets / 269,405 expected rows
pretrain_v2_published_shard_03.json  22 datasets / 271,751 expected rows
```

The three shards are disjoint and their union is exactly the all-manifest.
They are balanced by expected row count, not merely dataset count. `STATE.md`
claims 96 published datasets in its heading, but this checkout contains only
63 unique explicit `pretrain_V2` / `pretrain_V2_partial` Lance paths and the
linked 205-row CSV is absent; no path was invented for the missing entries.

## Tencent Cloud dry run

```bash
export COS_MOUNT_ROOT=/opt/ml/omni-eeg-processed
cd /mnt/cfs/jianyu/5B_model_pretrain

python build_fbmae_processed_from_pretrain_v2.py \
  --input ds002722="${COS_MOUNT_ROOT}/datasets/ds002722/releases/pretrain_V2/v022/data.lance" \
  --input ds002723="${COS_MOUNT_ROOT}/datasets/ds002723/releases/pretrain_V2/v001/data.lance" \
  --catalog ds002723="${COS_MOUNT_ROOT}/datasets/ds002723/releases/complete-recordings-eeg-native/v001/preflight.json" \
  --output-root /mnt/cfs/jianyu/fbmae_processed_by_dataset \
  --limit 128 \
  --dry-run
```

`--limit` is applied independently to each selected dataset. Channel preflight
still scans channel metadata for every row so a bad catalog cannot hide beyond
the limited conversion sample.

## Full conversion

Run only after the dry run reports the expected valid/dropped channel counts:

```bash
python build_fbmae_processed_from_pretrain_v2.py \
  --input ds002722="${COS_MOUNT_ROOT}/datasets/ds002722/releases/pretrain_V2/v022/data.lance" \
  --input ds002723="${COS_MOUNT_ROOT}/datasets/ds002723/releases/pretrain_V2/v001/data.lance" \
  --catalog ds002723="${COS_MOUNT_ROOT}/datasets/ds002723/releases/complete-recordings-eeg-native/v001/preflight.json" \
  --output-root /mnt/cfs/jianyu/fbmae_processed_by_dataset \
  --read-batch-size 128
```

This produces separate outputs:

```text
/mnt/cfs/jianyu/fbmae_processed_by_dataset/ds002722/data.lance
/mnt/cfs/jianyu/fbmae_processed_by_dataset/ds002723/data.lance
/mnt/cfs/jianyu/fbmae_processed_by_dataset/conversion_summary_pretrain_v2_datasets.example.json
```

The summary records admission status, source/processed row counts, tail-window
counts, valid-sample distributions, mapped/missing standard channels, catalog
sources, valid-channel-count distributions, output paths and processing time.

To retain historical `part_*.lance` layout while still keeping datasets in
separate subdirectories, add `--layout parts --rows-per-part 10000`.

When `--manifest` is used, the summary filename includes the manifest stem.
This lets disjoint manifests write different dataset subdirectories under the
same CFS output root without racing on one summary file. Parallel jobs must not
contain the same `dataset_key`, must not use `--overwrite`, and must not write
through a COS FUSE mount that lacks Lance's atomic-commit filesystem semantics.
Use `--summary-name NAME.json` to override the generated name.

Do not add `--overwrite` for the first production run. If it is explicitly used,
only the exact generated part paths are overwritten; unrelated files in the
output directory are not deleted.
