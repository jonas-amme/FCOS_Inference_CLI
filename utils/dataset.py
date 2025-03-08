from typing import Tuple, Union
import numpy as np
import openslide 

from openslide import OpenSlide
from dataclasses import dataclass, field
from pathlib import Path


Coords = Tuple[int, int]



@dataclass
class SlideObject:
    """A class for handling whole slide images and extracting patches.

    This class provides an interface for loading and accessing whole slide images (WSI),
    with methods to extract patches at specified coordinates and pyramid levels.

    Attributes:
        slide_path (Union[str, Path]): Path to the whole slide image file
        size (Union[int, float]): Size of patches to extract (width=height). Defaults to 512
        level (Union[int, float]): Pyramid level for patch extraction. Defaults to 0
        slide (OpenSlide): OpenSlide object for the WSI (automatically initialized)
    """

    slide_path: Union[str, Path]
    size: Union[int, float] = 512
    level: Union[int, float] = 0

    slide: OpenSlide = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize the OpenSlide object after instance creation.

        This method is automatically called after the dataclass initialization
        to create the OpenSlide object from the provided slide path.
        """
        self.slide = openslide.open_slide(str(self.slide_path))

    @property
    def patch_size(self) -> Coords:
        """Get the dimensions of patches to be extracted.

        Returns:
            Coords: Tuple of (width, height) for patches, both equal to self.size
        """
        return (self.size, self.size)

    @property
    def slide_size(self) -> Coords:
        """Get the dimensions of the slide at the current pyramid level.

        Returns:
            Coords: Tuple of (width, height) of the slide at the specified level
        """
        return self.slide.level_dimensions[self.level]

    def load_image(self, coords: Coords) -> np.ndarray:
        """Extract a patch from the slide at the specified coordinates.

        Args:
            coords (Coords): Tuple of (x, y) coordinates in the slide's coordinate space

        Returns:
            np.ndarray: RGB image patch as a numpy array with shape (H, W, 3)
        """
        patch = self.slide.read_region(location=coords, level=self.level, size=self.patch_size).convert('RGB')
        return np.array(patch)