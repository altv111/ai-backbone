import json
import logging
from pathlib import Path
from typing import Any


class AuditService:
    def __init__(self, enabled: bool = True, log_path: str = "") -> None:
        self.enabled = enabled
        self.log_path = log_path.strip()
        self._logger = logging.getLogger("app.audit")

    def emit(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return

        line = json.dumps(event, default=str)
        if self.log_path:
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file_handle:
                file_handle.write(line + "\n")
            return

        self._logger.info(line)
