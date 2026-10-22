"""Runtime helpers for experiment directories, resume and logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@dataclass
class ResumeInfo:
    ckpt_path: Path
    run_dir: Path
    run_id: str
    state: dict[str, Any]


@dataclass
class ExperimentRuntime:
    run_id: str
    run_name: str
    run_dir: Path
    hydra_output_dir: Path | None
    checkpoints_dir: Path
    history_dir: Path
    exports_dir: Path
    configs_dir: Path
    wandb_dir: Path
    run_state_path: Path
    latest_ckpt_path: Path
    is_resume: bool
    resume_ckpt_path: str | None
    wandb_run_id: str | None


def timestamp_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def prepare_experiment_runtime(cfg: DictConfig) -> tuple[ExperimentRuntime, ResumeInfo | None]:
    hydra_output_dir = None
    if HydraConfig.initialized():
        hydra_output_dir = Path(HydraConfig.get().runtime.output_dir).resolve()

    resume_info = resolve_resume_info(cfg)
    runs_root = Path(cfg.run.output_root).expanduser().resolve() / cfg.run.runs_subdir
    runs_root.mkdir(parents=True, exist_ok=True)

    if resume_info is not None:
        run_id = resume_info.run_id
        is_managed = _is_managed_run_dir(resume_info.run_dir, cfg)
        run_dir = (
            resume_info.run_dir
            if cfg.resume.reuse_run_dir and is_managed
            else _allocate_fresh_run_dir(runs_root, run_id)
        )
        state = resume_info.state
    else:
        requested_run_id = cfg.run.run_id
        if requested_run_id is None:
            run_id = _allocate_unique_run_id(runs_root, timestamp_run_id())
        else:
            run_id = str(requested_run_id)
            if (runs_root / run_id).exists():
                raise FileExistsError(
                    f"Run directory already exists: {runs_root / run_id}. "
                    "Use resume.mode to continue this run, or choose a different run.run_id."
                )
        run_dir = runs_root / run_id
        state = {}

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    history_dir = checkpoints_dir / "history"
    exports_dir = run_dir / "exports"
    configs_dir = run_dir / "configs"
    wandb_dir = run_dir / "wandb"
    for path in (checkpoints_dir, history_dir, exports_dir, configs_dir, wandb_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_name = str(cfg.run.name) if cfg.run.name is not None else run_id
    wandb_run_id = None
    if cfg.get("logger") is not None and cfg.logger.get("_target_") == "lightning.pytorch.loggers.WandbLogger":
        if resume_info is not None and cfg.resume.restore_wandb:
            wandb_run_id = state.get("wandb", {}).get("id", run_id)
            if state.get("wandb", {}).get("name"):
                run_name = state["wandb"]["name"]
        else:
            wandb_run_id = run_id

    runtime = ExperimentRuntime(
        run_id=run_id,
        run_name=run_name,
        run_dir=run_dir,
        hydra_output_dir=hydra_output_dir,
        checkpoints_dir=checkpoints_dir,
        history_dir=history_dir,
        exports_dir=exports_dir,
        configs_dir=configs_dir,
        wandb_dir=wandb_dir,
        run_state_path=run_dir / "run_state.json",
        latest_ckpt_path=checkpoints_dir / "last.ckpt",
        is_resume=resume_info is not None,
        resume_ckpt_path=str(resume_info.ckpt_path) if resume_info is not None else None,
        wandb_run_id=wandb_run_id,
    )
    return runtime, resume_info


def inject_runtime_into_cfg(cfg: DictConfig, runtime: ExperimentRuntime) -> DictConfig:
    OmegaConf.set_struct(cfg, False)
    cfg.run.run_id = runtime.run_id
    cfg.run.run_dir = str(runtime.run_dir)
    cfg.run.checkpoints_dir = str(runtime.checkpoints_dir)
    cfg.run.exports_dir = str(runtime.exports_dir)
    cfg.run.hydra_output_dir = str(runtime.hydra_output_dir) if runtime.hydra_output_dir is not None else None
    cfg.run.name = runtime.run_name
    OmegaConf.set_struct(cfg, True)
    return cfg


def save_runtime_configs(cfg: DictConfig, runtime: ExperimentRuntime) -> None:
    resolved_cfg_path = runtime.configs_dir / "resolved.yaml"
    OmegaConf.save(config=cfg, f=str(resolved_cfg_path), resolve=True)
    if runtime.hydra_output_dir is not None:
        hydra_cfg_path = runtime.configs_dir / "hydra_runtime.json"
        hydra_cfg_path.write_text(
            json.dumps({"output_dir": str(runtime.hydra_output_dir)}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def load_run_state(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "run_state.json"
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def resolve_resume_info(cfg: DictConfig) -> ResumeInfo | None:
    mode = str(cfg.resume.mode).lower()
    if mode == "none":
        return None

    if mode == "path":
        if cfg.resume.ckpt_path is None:
            raise ValueError("resume.ckpt_path must be set when resume.mode=path.")
        ckpt_path = Path(cfg.resume.ckpt_path).expanduser().resolve()
        run_dir = resolve_run_dir_for_resume(cfg=cfg, ckpt_path=ckpt_path)
    elif mode == "last":
        run_dir = resolve_run_dir_for_resume(cfg=cfg, ckpt_path=None)
        ckpt_path = run_dir / "checkpoints" / "last.ckpt"
    else:
        raise ValueError(f"Unsupported resume.mode: {cfg.resume.mode}")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    state = load_run_state(run_dir)
    run_id = state.get("run_id", run_dir.name)
    return ResumeInfo(ckpt_path=ckpt_path, run_dir=run_dir, run_id=run_id, state=state)


def resolve_run_dir_for_resume(cfg: DictConfig, ckpt_path: Path | None) -> Path:
    if cfg.resume.run_dir is not None:
        return Path(cfg.resume.run_dir).expanduser().resolve()
    if cfg.resume.run_id is not None:
        return (Path(cfg.run.output_root).expanduser().resolve() / cfg.run.runs_subdir / cfg.resume.run_id).resolve()
    if ckpt_path is not None:
        return infer_run_dir_from_ckpt(ckpt_path)
    raise ValueError("resume.run_dir or resume.run_id must be set when resume.mode=last.")


def infer_run_dir_from_ckpt(ckpt_path: Path) -> Path:
    ckpt_path = ckpt_path.resolve()
    for parent in ckpt_path.parents:
        if parent.name == "checkpoints":
            return parent.parent
        if parent.name == "history" and parent.parent.name == "checkpoints":
            return parent.parent.parent
    raise ValueError(f"Could not infer run directory from checkpoint path: {ckpt_path}")


def write_run_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _allocate_fresh_run_dir(runs_root: Path, base_run_id: str) -> Path:
    candidate = runs_root / f"{base_run_id}_resume"
    suffix = 1
    while candidate.exists():
        candidate = runs_root / f"{base_run_id}_resume{suffix}"
        suffix += 1
    return candidate


def _allocate_unique_run_id(runs_root: Path, base_run_id: str) -> str:
    candidate = base_run_id
    suffix = 1
    while (runs_root / candidate).exists():
        candidate = f"{base_run_id}_{suffix}"
        suffix += 1
    return candidate


def _is_managed_run_dir(run_dir: Path, cfg: DictConfig) -> bool:
    runs_root = (Path(cfg.run.output_root).expanduser().resolve() / cfg.run.runs_subdir).resolve()
    if (run_dir / "run_state.json").exists():
        return True
    try:
        run_dir.relative_to(runs_root)
        return True
    except ValueError:
        return False
