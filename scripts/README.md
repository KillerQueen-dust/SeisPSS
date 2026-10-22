# Scripts

Run scripts from the repository root in the installed project environment.
`--help` shows each Python script's arguments.

| Script | Purpose |
| --- | --- |
| `train/pretrain.sh` | Main pretraining; forwards Hydra overrides |
| `train/finetune.sh {stead,instance}` | General fine-tuning; forwards Hydra overrides |
| `utils/export_backbone.py` | Export EMA or online backbone from a pretraining checkpoint |
| `evaluate/phase_picking.py` | Test a complete picking checkpoint with optional tolerance override |
| `analysis/bench_encoder_efficiency.py` | GPU encoder/step benchmark against the Transformer ablation |
| `analysis/extract_state_dynamics.py` | Extract state diagnostics from a supplied backbone export |
| `analysis/check_finetuned_dynamics.py` | Inspect dynamics in a supplied picking checkpoint |
| `analysis/intervene_state_dynamics.py` | Run state interventions using a supplied picking checkpoint |

Analysis requires compatible CUDA/Mamba extensions. Data-dependent tools need
locally available datasets, and checkpoint-dependent tools require paths to
your own trained weights. Analysis scripts do not ship previously measured
arrays or reproduce manuscript figures from missing local archives.
