"""
Structural guards for the Colab notebook. These don't execute it (that needs a
Jupyter kernel), but they catch the failure modes that have broken it before:
missing cell ids, a setup cell that can't handle a private repo, or a pip line
that reinstalls Colab's preinstalled numpy/scipy (which forces a runtime
restart). Executing the notebook end to end is done separately with
`nbconvert --execute`.
"""

from __future__ import annotations
import json
import os

NB = os.path.join(os.path.dirname(__file__), "..", "notebooks",
                  "Oblsk_Negotiation_Agent_Demo.ipynb")


def _nb():
    with open(NB, encoding="utf-8") as f:
        return json.load(f)


def _setup_cell(nb):
    # The first code cell is the environment setup.
    return next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code")


def test_notebook_is_valid_and_every_cell_has_an_id():
    nb = _nb()
    assert nb["nbformat"] == 4
    assert nb["cells"], "notebook has no cells"
    missing = [i for i, c in enumerate(nb["cells"]) if not c.get("id")]
    assert not missing, f"cells missing an id (nbformat hard error): {missing}"


def test_setup_handles_private_repo_without_hanging():
    setup = _setup_cell(_nb())
    # Private-repo clone must support a token and must not block on a prompt.
    assert "GITHUB_TOKEN" in setup, "setup cell can't auth a private-repo clone"
    assert "GIT_TERMINAL_PROMPT" in setup, "clone can hang without prompt disabled"
    # A failed clone must explain what to do, not raise a bare CalledProcessError.
    assert "check=True" not in setup.replace("check=False", ""), \
        "setup cell uses check=True; a clone failure would be a cryptic traceback"


def test_setup_does_not_reinstall_preinstalled_colab_packages():
    setup = _setup_cell(_nb())
    # Reinstalling these in Colab can trigger 'restart runtime to use new
    # versions'. They ship with Colab; only anthropic should be installed.
    pip_region = setup[setup.find("pip"):] if "pip" in setup else ""
    for pkg in ("numpy", "scipy", "matplotlib", "pyyaml"):
        assert pkg not in pip_region, f"setup pip-installs {pkg}; leave it to Colab"
