# Paired 1% scratch/EMA protocol

These recipes describe the paired SeisPSS initialization experiments on STEAD
and InstanceCounts. They omit the external baseline system, frozen-head probe,
and internal queue machinery. The scientific model, loss, data preprocessing,
and optimizer implementations are retained from the research source.

## Data identity and subset selection

The six `subsets/<dataset>/train_1pct_seed<seed>.csv.gz` files contain only
`trace_name` and `split`. They preserve the exact ordered training IDs used in
the paired experiments. Waveforms, catalog arrival values, and other dataset
columns are obtained from the original dataset, not redistributed here.
Dataset access and reuse remain subject to the original dataset licenses.

`subsets.json` records each file's SHA-256, trace count, ordered-name hash,
source metadata SHA-256, and sampling definition. Check your original
`metadata.csv` against its recorded hash before claiming an identical subset
protocol. The exported ID-only files have different byte hashes from the
original research manifests; their ordered trace-name hashes are identical.

Selection shuffles first-seen training trace-name prefix blocks (the portion
before `$`) using `numpy.random.RandomState(seed)` and takes
`floor(0.01 * number_of_blocks)` blocks. Eligibility then requires a finite
supported phase pick and at least 6,000 samples at 100 Hz. Seeds are **42, 3407,
114514**. The percentage applies to prefix blocks, not an exact trace or event
percentage. Full native development/test splits use the same eligibility
filter. STEAD's native splits contain event overlap; these are archive-split
experiments, not event-disjoint generalization experiments. InstanceCounts
train data also participate in pretraining; STEAD does not.

The DataModule validates that every listed training ID exists in the native
training split and remains eligible. `data.training_fraction=1.0` in these
recipes means use the supplied list directly, without subsampling it again.

## Training and selection

| Dataset | Updates | Warmup updates | Scratch backbone LR | EMA backbone LR |
| --- | ---: | ---: | ---: | ---: |
| STEAD | 35,000 | 3,500 | 0.0001 | 0.00001 |
| InstanceCounts | 40,000 | 4,000 | 0.0001 | 0.00003 |

All runs use head LR 0.0003, effective batch size 192, AdamW, no waveform
augmentation, and development evaluation every 2,000 updates. Both
initializations use the same architecture, training ID list, training seed,
and loader seed. Checkpoints maximize full-development macro P/S F1.

The original seed-42 development search considered scratch LRs
`[0.0001, 0.0003]` and EMA LRs `[0.00003, 0.00001]`. The selected rate was reused
for the other seeds. Development choices were fixed before testing. The
`reproduce/` configurations encode the selected settings; rerunning them does
not repeat that search. Test reports use the mean and sample standard
deviation across the three seeds.

EMA experiments require the 90k-step pretrained EMA backbone. The reference
research export SHA-256 is
`1e99a619686444dc2aaa7c30301490eb5280393608c15c2b583cd9988a4ac300`.
That weight file is not hosted in this repository. A newly trained export can
be substituted, but its numerical results are a new pretraining realization.
Training resume restores optimizer/scheduler state; worker/sampler trajectories
are not guaranteed to resume bit for bit.

Use the commands in the root README with each of the three seeds. Evaluate the
best development checkpoint only after fixing the training choices. The main
matching tolerance is 0.2 s; additional tolerances are 0.05, 0.1, and 0.5 s.
