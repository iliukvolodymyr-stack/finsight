import json
import sys
from pathlib import Path

from app.main import app


def export_openapi(output_path: str = "openapi.json") -> None:
    schema = app.openapi()
    content = json.dumps(schema, indent=2)
    Path(output_path).write_text(content, encoding="utf-8")
    if "--print" in sys.argv:
        print(content)
    else:
        print(f"Exported openapi schema to {output_path}")


if __name__ == "__main__":
    export_openapi()
