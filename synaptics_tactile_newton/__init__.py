# Copyright (C) 2026 Synaptics Incorporated.  All rights reserved.

"""Synaptics Newton tactile sensor package."""

from .sensor import CTSSensor, centroid_scale_to_meters
from .output import CTSOutput
from .display import format_forces, print_forces
