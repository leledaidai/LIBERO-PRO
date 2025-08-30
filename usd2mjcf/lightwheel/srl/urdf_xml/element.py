# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Python dataclasses to represent the URDF XML schema.

* URDF XML Schema definition:
    https://raw.githubusercontent.com/ros/urdfdom/master/xsd/urdf.xsd
* ROS.org URDF package:
    http://wiki.ros.org/urdf
"""


# Standard Library
import collections
import re
from collections.abc import Iterable
from dataclasses import MISSING, dataclass
from typing import Any, Optional, Tuple

# Third Party
import numpy as np
from lxml import etree


@dataclass
class UrdfXmlElement:
    """Base class for each xml element that is defined in the URDF XML schema."""

    def __post_init__(self) -> None:
        """Additional initialization after the __init__ function has run."""
        self._element: Optional[etree.Element] = None

    @staticmethod
    def _to_str(input_raw: Any) -> str:
        if isinstance(input_raw, str):
            output_str = input_raw
        elif isinstance(input_raw, np.ndarray) or isinstance(input_raw, list):
            array_0 = np.array(input_raw)
            array_1 = np.round(array_0, 12)  # Round the values to the nearest 12th decimal place
            array_2 = array_1 + np.zeros(
                len(array_1)
            )  # Add zero to normalize negative zero to zero
            array_3 = array_2.astype(np.float32)
            array_str_0 = np.array2string(
                array_3, separator=" ", precision=7, suppress_small=True
            ).strip("[]")
            array_str_1 = array_str_0.strip("[]")
            array_str_2 = re.sub(r"\s+", " ", array_str_1)
            array_str_3 = array_str_2.lstrip()
            output_str = array_str_3
        elif isinstance(input_raw, collections.abc.Sequence):
            input_00 = [
                elem + 0.0 for elem in input_raw
            ]  # Add zero to normalize negative zero to zero
            output_str = str(input_00).replace(", ", " ")[1:-1]
        elif isinstance(input_raw, int) or isinstance(input_raw, float):
            input_10 = input_raw
            input_11 = input_10 + 0.0  # Add zero to normalize negative zero to zero
            output_str = "{:.7f}".format(input_11).rstrip("0")
        else:
            output_str = str(input_raw)

        return output_str

    def build_etree(self, name: Optional[str] = None) -> None:
        """Build the lxml etree object.

        Args:
            name: The elements name. (default: the element's class name)
        """

        def _select_metadata(
            key_val: Tuple[str, Any], metadata_key: str, metadata_val: Any
        ) -> bool:
            _, val = key_val
            if (
                len(val.metadata) > 0
                and metadata_key in val.metadata.keys()
                and val.metadata[metadata_key] == metadata_val
            ):
                return True
            else:
                return False

        def _get_attributes(key_val: Tuple[str, Any]) -> bool:
            return _select_metadata(key_val, metadata_key="type", metadata_val="attribute")

        def _get_elements(key_val: Tuple[str, Any]) -> bool:
            return _select_metadata(key_val, metadata_key="type", metadata_val="element")

        def _get_required(key_val: Tuple[str, Any]) -> bool:
            return _select_metadata(key_val, metadata_key="required", metadata_val=True)

        if name is None:
            class_name = self.__class__.__name__
            # camel case to snake case
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()

        required_fields = dict(filter(_get_required, self.__dataclass_fields__.items()))
        if len(required_fields) > 0:
            for key in required_fields.keys():
                val = getattr(self, key)
                if val is MISSING or val is None:
                    raise TypeError(
                        f"`{key}` must be provided to the `{self.__class__.__name__}` element."
                    )

        self._element = etree.Element(name)

        attributes = dict(filter(_get_attributes, self.__dataclass_fields__.items()))
        if len(attributes) > 0:
            # for key in sorted(attributes.keys(), reverse=True):
            for key in attributes.keys():
                val = getattr(self, key)
                self._element.set(key, UrdfXmlElement._to_str(val))

        elements = dict(filter(_get_elements, self.__dataclass_fields__.items()))
        if len(elements) > 0:
            # for key in sorted(elements.keys()):
            for key in elements.keys():
                vals = getattr(self, key)
                if vals is None:
                    continue

                if not isinstance(vals, Iterable):
                    vals = [getattr(self, key)]

                # child_name = None if key == "choice" else key
                if "name" in elements[key].metadata:
                    child_name = elements[key].metadata["name"]
                elif key == "choice":
                    child_name = None
                else:
                    child_name = key

                if len(vals) > 0 and hasattr(vals[0], "name"):
                    vals.sort(key=lambda val_: val_.name)
                for val in vals:
                    val.build_etree(child_name)
                    self._element.append(val._element)

    def to_xml_str(self, **kwargs: Any) -> str:
        """Return the full XML string for this element.

        Args:
            kwargs: Additional keyword arguments are pass to :func:`etree.tostring`.
        """
        kwargs.setdefault("pretty_print", True)
        kwargs.setdefault("method", "xml")
        kwargs.setdefault("xml_declaration", True)
        kwargs.setdefault("encoding", "UTF-8")
        if self._element is None:
            self.build_etree()
        etree_str = etree.tostring(self._element, **kwargs).decode()
        etree_str = etree_str.replace("'", '"')
        return etree_str

    def __str__(self) -> str:
        return self.to_xml_str()
