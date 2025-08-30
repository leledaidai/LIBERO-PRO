# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""SRL object base class module."""

# Standard Library
from abc import ABC
from typing import Dict, Optional

# NVIDIA
from lightwheel.srl.tools.logger import Logger


class SRL(ABC):
    """Base class for all SRL classes."""

    # Count the number of initialized classes
    _initialized_count: Dict[str, int] = {}

    def __init__(
        self,
        logger_name: Optional[str] = None,
        log_level: Optional[int] = None,
        no_color: bool = False,
    ):
        """Initialize a new :class:`~srl.abc.srl.SRL` object.

        Args:
            logger_name: Set the name of the logger (default: "{class name}.{class count}").
            log_level: Set logging level for this class (default: logging.DEBUG).
            no_color: If true, disable logging text colors.
        """
        # Update class count
        class_name = self.__class__.__name__
        if class_name not in SRL._initialized_count:
            SRL._initialized_count[class_name] = 1
        else:
            SRL._initialized_count[class_name] += 1
        class_cnt = SRL._initialized_count[class_name]

        # Set default argument
        if logger_name is None:
            logger_name = f"{class_name}.{class_cnt-1}"

        # Initialize logger
        self.logger = Logger(name=logger_name, log_level=log_level, no_color=no_color)
