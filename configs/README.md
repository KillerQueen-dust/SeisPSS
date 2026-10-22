# Configuration guide

`main.py` is the Hydra entry point. Use `--config-name <name>` without the
`.yaml` suffix. Add `--cfg job --resolve` to inspect configuration without
starting a run.

| Purpose | Configuration |
| --- | --- |
| Main latent pretraining, 90k updates | `pretrain/seispss` |
| General STEAD / InstanceCounts transfer | `finetune/stead`, `finetune/instance` |
| Frozen paired 1% experiments | `reproduce/stead_{scratch,ema}`, `reproduce/instancecounts_{scratch,ema}` |
| Uniform or shuffled target placement | `ablation/uniform_targets`, `ablation/misaligned_targets` |
| Waveform reconstruction, 30k updates | `ablation/waveform_reconstruction` |
| Transformer latent prediction, 30k updates | `ablation/transformer_latent` |
| Scratch architecture controls | `ablation/bimamba_scratch`, `ablation/transformer_scratch` |

The reconstruction and Transformer pretraining ablations retain their existing
30k-update budgets and 60k-step learning-rate schedules. They are separate
recipes, not equal-budget substitutes for the 90k-update main configuration.
The paired `reproduce/` recipes describe the frozen scratch/EMA experiments;
the general and ablation recipes do not imply that every combination was used
in those experiments.

`data`, `model`, `mask`, `loss`, `optim`, `trainer`, `head`, `transfer`, and
`eval` contain shared groups. `finetune/base` is a shared configuration and is
not a standalone model recipe. `debug/fast_debug` is a shorter development
configuration, not a no-data smoke test.

CSV logging is the default. Select W&B explicitly with `logger=pretrain` or
`logger=finetune`. Full, frozen, partial, and scratch transfer protocols are
available through the `transfer` group. Set `transfer.backbone_export_path`
when using a pretrained backbone; weights are not distributed in Git.
