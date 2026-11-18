"""
Updates the text of the executed notebooks without re-running them:

1. markdown cells are re-synced from the template in scripts/build_experiment_notebooks.py (code cells and
   their outputs are never touched; the script refuses to run if the code cells differ from the template);
2. the markdown cell tagged `interpretation` gets the written conclusion from notebooks/interpretations/<key>.md.

    python scripts/insert_interpretations.py
"""

import sys
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_experiment_notebooks as B  # noqa: E402

NB_DIR = REPO / "notebooks"
TEXT_DIR = NB_DIR / "interpretations"

TARGETS = {slug: (fname, lambda s=slug, t=title: B.dataset_notebook(s, t)) for slug, fname, title in B.DATASETS}
TARGETS["cross_dataset"] = ("05_cross_dataset_comparison.ipynb", B.cross_notebook)

for key, (fname, make_template) in TARGETS.items():
    nb_file = NB_DIR / fname
    if not nb_file.exists():
        print(f"skip {key}: notebook missing")
        continue
    nb = nbf.read(nb_file, as_version=4)
    template = make_template()
    if [c.cell_type for c in nb.cells] != [c.cell_type for c in template.cells] or any(
            a.source != b.source for a, b in zip(nb.cells, template.cells) if a.cell_type == "code"):
        raise RuntimeError(f"{fname}: code cells differ from the template; rebuild and re-execute instead")
    for cell, tcell in zip(nb.cells, template.cells):
        if cell.cell_type == "markdown":
            cell.source = tcell.source
            cell.metadata = tcell.metadata

    text_file = TEXT_DIR / f"{key}.md"
    if text_file.exists():
        cells = [c for c in nb.cells if "interpretation" in c.metadata.get("tags", [])]
        if len(cells) != 1:
            raise RuntimeError(f"{fname}: expected one cell tagged 'interpretation', found {len(cells)}")
        cells[0].source = text_file.read_text(encoding="utf-8").strip()
    nbf.write(nb, nb_file)
    print(f"updated {fname}" + ("" if text_file.exists() else " (no interpretation text yet)"))
