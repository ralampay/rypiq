"""Check Markdown examples against runnable snapshots and exercise overlays."""
from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def blocks(text: str):
    opening = None
    lines = []
    heading = None
    for line in text.splitlines():
        if opening is None:
            if line.startswith("### `") and line.endswith("`"):
                heading = line[5:-1]
            match = re.fullmatch(r"(`{3,})(\w*)", line)
            if match:
                opening = (match[1], match[2], heading)
                lines = []
        elif line == opening[0]:
            yield opening[1], opening[2], "\n".join(lines) + "\n"
            opening = None
            heading = None
        else:
            lines.append(line)
    assert opening is None, "Unclosed code fence"


def main() -> None:
    modules = sorted((ROOT / "modules").iterdir())
    assert len(modules) == 5
    examples = 0
    diagrams = []
    for number, module in enumerate(modules, 1):
        stage = ROOT / "reference" / f"module-{number:02d}"
        starter = ROOT / "starters" / f"module-{number:02d}"
        for filename in ("ModuleDiscussion.md", "Activity.md", "ActivitySolution.md"):
            path = module / filename
            text = path.read_text()
            assert all(word in text for word in ("Red", "Green", "Refactor")), path
            assert "Working directory" in text, path
            assert "python -m pytest tests/unit" in text, path
            assert "python -m pytest tests/integration" in text, path
            for language, name, code in blocks(text):
                if language == "python":
                    ast.parse(code)
                    examples += 1
                if language == "mermaid":
                    diagrams.append({"file": str(path.relative_to(ROOT)), "code": code})
                if filename == "ActivitySolution.md" and name:
                    assert (stage / name).is_file(), (path, name)
                    assert (stage / name).read_text().strip() == code.strip(), (path, name)
                if filename == "Activity.md" and name:
                    assert (starter / name).is_file(), (path, name)
                    assert (starter / name).read_text().strip() == code.strip(), (path, name)
        for source in stage.rglob("*.py"):
            ast.parse(source.read_text())
        project = tomllib.loads((stage / "pyproject.toml").read_text())
        assert project["project"]["scripts"] == {"rag-workshop": "ragworkshop.cli:main"}
        declared = project["project"]["dependencies"] + project["project"]["optional-dependencies"]["dev"]
        assert all(any(dep.startswith(name + "==") for dep in declared)
                   for name in ("chromadb", "PyYAML", "python-dotenv", "python-docx", "pypdf", "pytest", "reportlab"))
    import json
    Path("/tmp/ragworkshop-diagrams.json").write_text(json.dumps(diagrams))
    print(f"15 module documents checked; {examples} Python blocks parse; {len(diagrams)} Mermaid blocks extracted")


if __name__ == "__main__":
    main()
