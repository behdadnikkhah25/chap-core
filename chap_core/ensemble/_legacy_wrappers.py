"""Legacy compatibility wrappers for ensemble model templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chap_core.models.model_template import ModelTemplate


@dataclass
class BaseModelSpec:
    template: ModelTemplate
    config: Any | None = None


class _TemplateWithConfig:
    def __init__(self, template: ModelTemplate, config: Any | None) -> None:
        self._template = template
        self._config = config

    def get_model(self, _: Any) -> Any:
        if self._config is None or isinstance(self._config, dict):
            return self._template.get_model(None)
        return self._template.get_model(self._config)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._template, item)
