"""Upload this repository to a Hugging Face Docker Space (run by .github/workflows/deploy-space.yml).

Creates the Space if needed, copies GROQ_API_KEY / GEMINI_API_KEY into its secrets when they are
set, and uploads the tracked files with the Space's YAML header prepended to README.md (the header
is Space-only, so the GitHub README stays clean). Hugging Face then builds the Dockerfile.

    HF_TOKEN=... SPACE_NAME=msme-scheme-assistant python scripts/deploy_space.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_HEADER = """\
---
title: MSME & NABARD Scheme Assistant
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
short_description: Cited answers on MSME schemes, state policies and NABARD
---

"""
SECRETS = ("GROQ_API_KEY", "GEMINI_API_KEY")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return [f for f in out.splitlines() if f and not f.startswith(".github/")]


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    owner = api.whoami()["name"]
    repo_id = f"{owner}/{os.environ.get('SPACE_NAME') or 'msme-scheme-assistant'}"
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    for name in SECRETS:
        if os.environ.get(name):
            api.add_space_secret(repo_id, name, os.environ[name])
            print(f"set Space secret {name}")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for rel in tracked_files():
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dst)
        readme = stage / "README.md"
        readme.write_text(SPACE_HEADER + readme.read_text(encoding="utf-8"), encoding="utf-8")
        sha = os.environ.get("GITHUB_SHA", "local")[:7]
        api.upload_folder(folder_path=stage, repo_id=repo_id, repo_type="space", delete_patterns="*",
                          commit_message=f"Deploy {sha}")
    print(f"deployed: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
