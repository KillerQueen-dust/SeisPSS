"""Export the online or EMA SeisPSS backbone from a Lightning checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("ema", "online"), default="ema")
    args = parser.parse_args()
    from seispss.export import export_backbone_from_checkpoint

    print(export_backbone_from_checkpoint(
        source_ckpt_path=args.checkpoint, output_path=args.output, variant=args.variant,
    ))


if __name__ == "__main__":
    main()
