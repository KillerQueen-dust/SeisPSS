"""Minimal pkg_resources compatibility shim for ObsPy on modern setuptools."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Iterable


DistributionNotFound = metadata.PackageNotFoundError


@dataclass
class Distribution:
    key: str
    version: str

    @property
    def name(self) -> str:
        return self.key


@dataclass
class EntryPoint:
    name: str
    module_name: str
    attrs: tuple[str, ...]
    dist: Distribution
    _entry_point: metadata.EntryPoint

    def load(self):
        return self._entry_point.load()

    def __str__(self) -> str:
        attrs = ".".join(self.attrs)
        return f"{self.name} = {self.module_name}:{attrs}"


def get_distribution(name: str) -> Distribution:
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise DistributionNotFound(name) from exc
    return Distribution(key=name, version=version)


def iter_entry_points(group: str, name: str | None = None) -> Iterable[metadata.EntryPoint]:
    entry_points = metadata.entry_points()
    selected = entry_points.select(group=group)
    if name is not None:
        selected = [entry_point for entry_point in selected if entry_point.name == name]
    wrapped = []
    for entry_point in selected:
        dist_name = getattr(entry_point.dist, "name", "unknown")
        try:
            dist_version = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            dist_version = "unknown"
        wrapped.append(
            EntryPoint(
                name=entry_point.name,
                module_name=entry_point.module,
                attrs=(entry_point.attr,) if entry_point.attr else tuple(),
                dist=Distribution(key=dist_name, version=dist_version),
                _entry_point=entry_point,
            )
        )
    return wrapped


def get_entry_info(dist: str, group: str, name: str):
    for entry_point in iter_entry_points(group=group, name=name):
        distribution_name = getattr(entry_point.dist, "key", None)
        if distribution_name is None or distribution_name == dist:
            return entry_point
    return None


def load_entry_point(dist: str, group: str, name: str):
    entry_point = get_entry_info(dist=dist, group=group, name=name)
    if entry_point is None:
        raise DistributionNotFound(dist)
    return entry_point.load()
