"""Make a generation run reproducible.

Until 2026-08-02 nothing in the production path was seeded, which is why five
runs of a byte-identical configuration scored 4, 2, 1, 3 and 5 of the six
evaluation axes. Three independent RNGs feed the decode:

* ``torch`` — nucleus sampling in :mod:`beatsaber_automapper.generation.beam_search`
  (``torch.multinomial``, temp 0.9 / top-p 0.97) and the anti-repeat pick in
  :mod:`beatsaber_automapper.models.layout_model`. These choose note positions
  and directions, so they move flow, idiom and hand-role.
* ``random`` — post-processing shuffles the candidate order when it deletes
  notes to hit the NPS target, and picks replacement cut directions. Deleting a
  *different* note changes the surviving note times, so this moves alignment too.
* ``numpy`` — seeded for completeness; the audio front end is deterministic
  today but nothing enforces that.

``postprocess_beatmap`` takes its own ``seed`` argument but no caller has ever
passed one. It seeds the same global ``random`` module this function does, so
seeding once per process covers it without threading a seed through every
signature.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed every RNG the generation path draws from.

    Args:
        seed: The seed to apply to python ``random``, ``numpy`` and ``torch``
            (including all CUDA devices).
    """
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Seeded random/numpy/torch with %d", seed)
