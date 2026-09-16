"""SandboxVectorizer pipeline configuration (pipelines.toml)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines.toml"
)


class ConfigError(Exception):
    pass


@dataclass
class PipelineConfig:
    path: str
    pipelines: dict[str, str]
    defaults: dict[str, str]

    def resolve(self, value: str | None, command: str) -> tuple[str, str]:
        """Map `--pipeline` (or the per-command default) to (label, passes).

        A name must exist in [pipelines]; anything that looks like a pass
        pipeline (contains '<' or '(') is used verbatim.
        """
        value = value or self.defaults.get(command)
        if value is None:
            raise ConfigError(
                f"{self.path}: no --pipeline given and no [defaults].{command}"
            )
        if value in self.pipelines:
            return value, self.pipelines[value]
        if "<" in value or "(" in value:
            return "(literal)", value
        raise ConfigError(
            f"unknown pipeline '{value}' (defined in {self.path}: "
            f"{', '.join(sorted(self.pipelines))})"
        )


def load(path: str | None = None) -> PipelineConfig:
    path = os.path.abspath(os.path.expanduser(path or DEFAULT_CONFIG))
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None

    pipelines = data.get("pipelines")
    if not isinstance(pipelines, dict) or not pipelines:
        raise ConfigError(f"{path}: missing or empty [pipelines] table")
    for name, passes in pipelines.items():
        if not isinstance(passes, str) or not passes.strip():
            raise ConfigError(f"{path}: pipelines.{name} must be a non-empty string")
        # Whitespace inside -sbvec-passes is not accepted by the pass parser.
        pipelines[name] = "".join(passes.split())

    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ConfigError(f"{path}: [defaults] must be a table")
    for command, name in defaults.items():
        if name not in pipelines:
            raise ConfigError(
                f"{path}: defaults.{command} = '{name}' is not in [pipelines]"
            )
    return PipelineConfig(path, pipelines, defaults)
