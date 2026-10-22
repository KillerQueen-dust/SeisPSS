# SeisPSS

**Latent Predictive State-Space Pretraining for Label-Efficient Seismic Phase Picking**

SeisPSS learns representations of three-component seismic waveforms by predicting
masked latent features from an exponential moving average (EMA) teacher. A
convolutional stem feeds a bidirectional Mamba-2 encoder; a cross-attention
predictor uses the visible context. The pretrained stem and encoder initialize
a dense P-wave / S-wave / background picker.

The online encoder retains the full time grid, with mask tokens at target and
guard positions. Predictor memory contains all valid visible tokens outside
those positions. Waveform-derived onset confidence controls multi-horizon target
sampling. Catalog phase picks assist pretraining crop construction.

## Release contents

This repository contains the SeisPSS implementation, pretraining and transfer
configurations, method ablations, checkpoint export, picking evaluation, and
encoder/state analysis tools. The paired 1% scratch/EMA experiments include their
selected training configurations and six frozen training trace-ID lists.

External baseline implementations and their training configurations are excluded
from this release. It therefore covers the main method and its paired
initialization comparison; the complete baseline comparison is outside this
release. Waveforms, model weights, experiment logs, manuscript drafts, and
internal experiment-management files are also excluded. Pretrained weights are
not currently hosted; train and export a backbone using the commands below.

## Installation

Use Linux, Python 3.12, an NVIDIA GPU, and a CUDA environment compatible with
PyTorch and the native Mamba extensions. The lockfile selects PyTorch's CUDA 12.8
index. Mamba-SSM 2.3.1 and causal-conv1d 1.6.1 are required for the main model.

```bash
git clone https://github.com/KillerQueen-dust/SeisPSS.git
cd SeisPSS
uv sync --locked --extra mamba
```

Native extensions may require a matching CUDA toolkit and compiler. After
installation, use `uv run --no-sync` to preserve the installed CUDA environment.
CPU execution uses a feed-forward structural fallback and does not implement
the Mamba recurrence used in the experiments.

## Data

Obtain the datasets through [SeisBench](https://github.com/seisbench/seisbench)
under their respective access conditions and licenses. Set the cache location
before starting Python:

```bash
export SEISBENCH_CACHE_ROOT=/path/to/seisbench
```

The expected files are `datasets/<dataset>/metadata.csv` and
`datasets/<dataset>/waveforms.hdf5` below that cache root. Dataset directories
include `instancecounts`, `pnw`, `geofon`, and `stead`. Dataset construction may
download missing data through SeisBench.

Pretraining uses event traces from InstanceCounts, PNW, and GEOFON with dataset
sampling temperature exponent 0.25. STEAD is excluded from main pretraining.
Inputs are ordered ZNE at 100 Hz, cropped to 6,000 samples without padding, and
standardized per component. Downstream tasks use native STEAD / InstanceCounts
train, development, and test splits. The 1% budget refers to training prefix
blocks, with the complete labeled development split used for selection.

## Pretraining and export

All commands run from the repository root. CSV logging is the default and
requires no W&B account. W&B is optional through `logger=pretrain` or
`logger=finetune`.

```bash
# Inspect the resolved configuration without training or downloading data.
uv run --no-sync python main.py --config-name pretrain/seispss --cfg job --resolve

# Main 90,000-update pretraining recipe.
uv run --no-sync python main.py --config-name pretrain/seispss

# Export the EMA stem and encoder from your Lightning checkpoint.
uv run --no-sync python scripts/utils/export_backbone.py \
  /path/to/pretraining.ckpt outputs/backbone_ema.pt --variant ema
```

Training outputs, checkpoints, configuration snapshots, and backbone exports are
written under `outputs/`. Automatic exports are stored in each run's `exports/`
directory. Use an exported backbone explicitly when fine-tuning:

```bash
uv run --no-sync python main.py --config-name finetune/stead \
  data.training_fraction=0.01 seed=42 \
  transfer.backbone_export_path=outputs/backbone_ema.pt

uv run --no-sync python main.py --config-name finetune/instance \
  data.training_fraction=0.01 seed=42 \
  transfer.backbone_export_path=outputs/backbone_ema.pt
```

These general recipes select a training subset at runtime. Use the following
frozen recipes for the paired 1% experiments instead.

## Paired 1% experiments

The `reproduce/` configurations preserve selected learning rates, fixed subsets,
loader seeds, batch size, schedules, and development-checkpoint selection for
the paired scratch/EMA experiments. See [the protocol](protocols/r1/README.md)
for all three seeds, dataset identity checks, and the selection rule.

```bash
# STEAD: same model architecture and training IDs, different initialization.
uv run --no-sync python main.py --config-name reproduce/stead_scratch seed=42
uv run --no-sync python main.py --config-name reproduce/stead_ema seed=42 \
  transfer.backbone_export_path=outputs/backbone_ema.pt

# InstanceCounts: repeat with seed=3407 and seed=114514 as well.
uv run --no-sync python main.py --config-name reproduce/instancecounts_scratch seed=42
uv run --no-sync python main.py --config-name reproduce/instancecounts_ema seed=42 \
  transfer.backbone_export_path=outputs/backbone_ema.pt
```

These recipes defer testing until development-based choices are fixed. Evaluate
the selected complete picking checkpoint explicitly:

```bash
uv run --no-sync python scripts/evaluate/phase_picking.py \
  /path/to/best-picking.ckpt --accelerator gpu \
  --tolerance-seconds 0.2 --output outputs/test_metrics.json
```

The evaluator restores the trained backbone from the complete checkpoint; the
pretraining export is not needed at evaluation time. Other tolerances can be
evaluated using the same command. Default decoding uses P/S thresholds 0.4/0.5,
50-sample peak separation, and 20-sample (0.2 s) matching tolerance. Longer
records use 6,000-sample windows with stride 1,000 and probability averaging.

## Ablations and analysis

`configs/ablation/` contains uniform/shuffled target sampling, waveform
reconstruction, Transformer latent prediction, and scratch controls. These are
method comparison recipes; their scope and budgets are described in
[configs/README.md](configs/README.md).

```bash
uv run --no-sync python main.py --config-name ablation/uniform_targets --cfg job --resolve
uv run --no-sync python scripts/analysis/bench_encoder_efficiency.py --help
uv run --no-sync python scripts/analysis/extract_state_dynamics.py --help
```

State diagnostics describe internal model dynamics, not a recovered physical
wavefield. See [scripts/README.md](scripts/README.md) for the available tools.

## Dependencies and license

SeisPSS uses [Mamba](https://github.com/state-spaces/mamba),
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d), SeisBench,
PyTorch, Lightning, and Hydra. Those projects and the datasets retain their own
licenses. SeisPSS code is released under the [MIT License](LICENSE).
