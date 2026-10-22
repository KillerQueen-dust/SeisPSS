# uv run py.main
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from seispss.cli import run


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
