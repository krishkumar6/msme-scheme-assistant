"""Build a self-contained Colab notebook: the training script and the SFT data
are embedded (gzip + base64), so nothing has to be uploaded separately.

    python scripts/make_colab_notebook.py
    -> colab_upload/finetune_colab_selfcontained.ipynb
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "training/finetune_lora.py": ROOT / "training" / "finetune_lora.py",
    "data/training/sft_train.jsonl": ROOT / "data" / "training" / "sft_train.jsonl",
    "data/training/sft_val.jsonl": ROOT / "data" / "training" / "sft_val.jsonl",
}


def main() -> None:
    nb = json.loads((ROOT / "training" / "finetune_colab.ipynb").read_text(encoding="utf-8"))
    payload = {}
    for dest, src in FILES.items():
        if not src.exists():
            raise SystemExit(f"missing {src} - run training/build_sft_dataset.py first")
        payload[dest] = base64.b64encode(gzip.compress(src.read_bytes(), 9)).decode("ascii")

    unpack = [
        "# Writes the embedded training script + dataset to disk (no upload needed)\n",
        "import base64, gzip, json, os\n",
        "WORK = '/kaggle/working/nabard' if os.path.isdir('/kaggle/working') else '/content/nabard'\n",
        "os.makedirs(WORK, exist_ok=True)\n",
        "os.chdir(WORK)\n",
        f"FILES = json.loads({json.dumps(json.dumps(payload))})\n",
        "for path, blob in FILES.items():\n",
        "    os.makedirs(os.path.dirname(path), exist_ok=True)\n",
        "    with open(path, 'wb') as fh:\n",
        "        fh.write(gzip.decompress(base64.b64decode(blob)))\n",
        "    print(f'wrote {path} ({os.path.getsize(path) // 1024} KB)')\n",
        "!python training/finetune_lora.py --validate-only\n",
    ]
    for cell in nb["cells"]:
        if cell.get("id") == "upload":
            cell["source"] = unpack
            cell["outputs"] = []
        if cell.get("id") == "intro":
            cell["source"] = [
                "# Fine-tune the MSME & NABARD QA model on a free GPU (self-contained)\n",
                "\n",
                "The training script and dataset are **embedded in this notebook**: no file upload needed.\n",
                "\n",
                "- **Colab:** `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all` (~40-50 min).\n",
                "- **Kaggle:** `File → Import Notebook`; in the right panel set **Accelerator: GPU T4 x2** and\n",
                "  **Internet: on** (needs a verified phone number), then `Run All`.\n",
                "\n",
                "The last cell gives you `adapter.zip` (~70 MB): Colab downloads it, on Kaggle take it from the\n",
                "Output panel (`/kaggle/working`).\n",
            ]
    out = ROOT / "colab_upload" / "finetune_colab_selfcontained.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
