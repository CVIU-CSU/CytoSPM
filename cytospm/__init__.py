# Copyright (c) Tencent Inc. All rights reserved.
# Modified by the CytoSPM Authors, 2026.
__version__ = '1.0.0'

# Register the checkpoint loading schemes before any model is built.
from .utils import checkpoint as _checkpoint  # noqa: F401,E402
from .datasets import *  # noqa: E402,F401,F403
from .evaluation import *  # noqa: E402,F401,F403
from .models import *  # noqa: E402,F401,F403
