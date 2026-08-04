"""Test-suite wide settings.

Matplotlib must be non-interactive before any test imports pyplot: several
modules under test draw figures, and on a machine with a GUI backend selected
that either opens windows or fails outright on a headless runner. CI sets
MPLBACKEND=Agg, which meant a green CI could hide a locally broken test.
"""

import matplotlib

matplotlib.use("Agg", force=True)
