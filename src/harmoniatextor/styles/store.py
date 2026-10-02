# SPDX-FileCopyrightText: 2026 xhdlphzr
# SPDX-License-Identifier: MIT

"""JSON persistence for user-defined style kits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harmoniatextor import config
from harmoniatextor.styles.base import StyleKit

__all__ = ["StyleKitStore"]

_FILE_NAME = "style_kits.json"


class StyleKitStore:
    """Read and write custom style kits under the configuration directory.

    Attributes:
        root: Directory holding the JSON file.
        path: Path to the JSON file.
    """

    def __init__(self, root: Path | None = None) -> None:
        """Initialise the store.

        Args:
            root: Directory for the JSON file; defaults to ``config_dir()``.
        """
        self.root = root if root is not None else config.config_dir()
        self.path = self.root / _FILE_NAME

    def load(self) -> dict[str, StyleKit]:
        """Load every stored custom kit.

        Returns:
            A mapping of kit id to kit; empty when the file is missing or
            malformed.
        """
        if not self.path.exists():
            return {}
        try:
            data: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError, OSError:
            return {}
        raw = data.get("kits", []) if isinstance(data, dict) else []
        kits: dict[str, StyleKit] = {}
        for item in raw:
            if isinstance(item, dict) and "id" in item and "name" in item:
                kit = StyleKit.from_dict({**item, "builtin": False})
                kits[kit.id] = kit
        return kits

    def save(self, kits: dict[str, StyleKit]) -> None:
        """Persist every custom kit.

        Args:
            kits: Mapping of kit id to kit.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"kits": [kit.to_dict() for kit in kits.values()]}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
