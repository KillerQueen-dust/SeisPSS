"""CLI helpers for SeisPSS training and export."""

from __future__ import annotations

import logging

import lightning as L
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from seispss.callbacks import BackboneExportCallback, RunStateCallback, StepProgressBar
from seispss.downstream.phase_picking.task import PhasePickingModule
from seispss.runtime import inject_runtime_into_cfg, prepare_experiment_runtime, save_runtime_configs
from seispss.tasks.pretrain_module import SeisPSSPretrainModule


def configure_external_logging() -> None:
    logging.getLogger("seisbench").setLevel(logging.WARNING)


def run(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    configure_external_logging()
    if cfg.mode == "train":
        run_training(cfg)
        return
    if cfg.mode == "train_phase_picking":
        run_phase_picking_training(cfg)
        return
    raise ValueError(f"Unsupported mode: {cfg.mode}")


def run_training(cfg: DictConfig) -> None:
    run_pretraining(cfg, module_class=SeisPSSPretrainModule)


def run_pretraining(cfg: DictConfig, *, module_class) -> None:
    """Assemble a pretraining run using the shared data and runtime contracts."""
    runtime, _ = prepare_experiment_runtime(cfg)
    cfg = inject_runtime_into_cfg(cfg, runtime)
    save_runtime_configs(cfg, runtime)

    L.seed_everything(cfg.seed, workers=True)
    if cfg.trainer.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.trainer.matmul_precision)

    datamodule = instantiate(cfg.data)
    model = module_class(cfg=cfg)

    callbacks = build_callbacks(cfg=cfg, runtime=runtime)
    logger = build_logger(cfg=cfg, runtime=runtime)

    trainer = build_trainer(cfg=cfg, runtime=runtime, callbacks=callbacks, logger=logger)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=runtime.resume_ckpt_path, weights_only=False)


def run_phase_picking_training(cfg: DictConfig) -> None:
    runtime, _ = prepare_experiment_runtime(cfg)
    cfg = inject_runtime_into_cfg(cfg, runtime)
    save_runtime_configs(cfg, runtime)

    L.seed_everything(cfg.seed, workers=True)
    if cfg.trainer.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.trainer.matmul_precision)

    datamodule = instantiate(cfg.data)
    model = PhasePickingModule(cfg=cfg)

    callbacks = build_callbacks(cfg=cfg, runtime=runtime)
    logger = build_logger(cfg=cfg, runtime=runtime)

    trainer = build_trainer(cfg=cfg, runtime=runtime, callbacks=callbacks, logger=logger)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=runtime.resume_ckpt_path, weights_only=False)
    if bool(OmegaConf.select(cfg, "eval.final_test_after_fit", default=False)):
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


def build_trainer(*, cfg: DictConfig, runtime, callbacks: list, logger) -> L.Trainer:
    return L.Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        max_steps=cfg.trainer.max_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        default_root_dir=str(runtime.run_dir),
        deterministic=cfg.trainer.deterministic,
        benchmark=cfg.trainer.benchmark,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=OmegaConf.select(cfg, "trainer.check_val_every_n_epoch", default=1),
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=OmegaConf.select(cfg, "trainer.limit_test_batches", default=None),
    )


def build_callbacks(*, cfg: DictConfig, runtime) -> list:
    progress_refresh_rate = int(
        OmegaConf.select(
            cfg,
            "trainer.progress_bar.refresh_rate",
            default=1,
        )
    )
    progress_process_position = int(OmegaConf.select(cfg, "trainer.progress_bar.process_position", default=0))
    progress_leave = bool(OmegaConf.select(cfg, "trainer.progress_bar.leave", default=False))
    progress_ncols = OmegaConf.select(cfg, "trainer.progress_bar.ncols", default=None)
    progress_dynamic_ncols = bool(OmegaConf.select(cfg, "trainer.progress_bar.dynamic_ncols", default=True))
    callbacks = [
        StepProgressBar(
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            refresh_rate=progress_refresh_rate,
            process_position=progress_process_position,
            leave=progress_leave,
            ncols=None if progress_ncols is None else int(progress_ncols),
            dynamic_ncols=progress_dynamic_ncols,
        ),
        RunStateCallback(
            run_id=runtime.run_id,
            run_name=runtime.run_name,
            run_dir=str(runtime.run_dir),
            state_path=str(runtime.run_state_path),
            latest_ckpt_path=str(runtime.latest_ckpt_path),
            resume_ckpt_path=runtime.resume_ckpt_path,
            hydra_output_dir=str(runtime.hydra_output_dir) if runtime.hydra_output_dir is not None else None,
        ),
        ModelCheckpoint(
            dirpath=str(runtime.checkpoints_dir),
            filename="best-step={step}",
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=int(cfg.checkpoint.save_top_k),
            save_last=bool(cfg.checkpoint.save_last),
            save_weights_only=bool(cfg.checkpoint.save_weights_only),
            save_on_exception=bool(cfg.checkpoint.save_on_exception),
            auto_insert_metric_name=False,
        ),
    ]
    every_n_train_steps = int(cfg.checkpoint.every_n_train_steps)
    if every_n_train_steps > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(runtime.history_dir),
                filename="step={step}",
                monitor=None,
                save_top_k=-1,
                save_last=False,
                save_weights_only=bool(cfg.checkpoint.save_weights_only),
                save_on_exception=bool(cfg.checkpoint.save_on_exception),
                auto_insert_metric_name=False,
                every_n_train_steps=every_n_train_steps,
                save_on_train_epoch_end=False,
                enable_version_counter=False,
            )
        )
    pretrain_mode = cfg.mode == "train"
    export_kind = "seispss"
    if pretrain_mode and bool(OmegaConf.select(cfg, "checkpoint.auto_export_backbone", default=True)):
        callbacks.append(
            BackboneExportCallback(
                checkpoints_dir=str(runtime.checkpoints_dir),
                history_dir=str(runtime.history_dir),
                exports_dir=str(runtime.exports_dir),
                latest_ckpt_path=str(runtime.latest_ckpt_path),
                variant=str(OmegaConf.select(cfg, "checkpoint.export_variant", default="ema")),
                export_kind=export_kind,
            )
        )
    callbacks.extend(instantiate(callback_cfg) for callback_cfg in cfg.trainer.callbacks)
    return callbacks


def build_logger(*, cfg: DictConfig, runtime):
    logger_cfg = OmegaConf.to_container(cfg.logger, resolve=True)
    target = logger_cfg.pop("_target_")
    if target == "lightning.pytorch.loggers.WandbLogger":
        logger_cfg["save_dir"] = str(runtime.wandb_dir)
        logger_cfg["dir"] = str(runtime.wandb_dir)
        logger_cfg["id"] = runtime.wandb_run_id
        logger_cfg["name"] = runtime.run_name
        if runtime.is_resume and cfg.resume.restore_wandb:
            logger_cfg["resume"] = "allow"
        return WandbLogger(**logger_cfg)
    return instantiate({"_target_": target, **logger_cfg})
