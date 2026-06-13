"""Music_Making: compose original music from a scene.

A storyboard (the scene) drives parallel lyric/composition/beat/vocal workflows
that are mixed into one track and checked by an autonomous, scene-aware QC gate.
"""

from .contracts import Storyboard, Track
from .orchestrator import produce
from .storyboard import from_text, from_video

__all__ = ["Storyboard", "Track", "produce", "from_text", "from_video"]
__version__ = "0.1.0"
