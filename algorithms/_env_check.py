"""Guard against conda base / NumPy 2.x breaking matplotlib binary wheels."""

from __future__ import annotations

import sys


def require_numpy1_for_matplotlib() -> None:
    import numpy as np

    major = int(np.__version__.split(".", maxsplit=1)[0])
    if major >= 2:
        print(
            "\nERROR: NumPy 2.x is active "
            f"(numpy {np.__version__}). Matplotlib in many envs was built for NumPy 1.x "
            "and will fail with '_ARRAY_API not found'.\n\n"
            "Use the project venv (see README):\n"
            "  source cs224r/bin/activate\n"
            "  python -c \"import numpy; print(numpy.__version__)\"  # expect 1.24.x\n"
            "  python algorithms/extended_policy_comparison.py ...\n\n"
            "Or:  ./cs224r/bin/python algorithms/extended_policy_comparison.py ...\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
