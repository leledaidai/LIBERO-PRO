# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Miscellaneous tools."""

# Standard Library
import importlib
from typing import Any, List, Mapping, Optional

# NOTE (roflaherty): `importlib_metadata` is needed to support Python < 3.10.
try:
    # Standard Library
    from importlib.metadata import packages_distributions
except ImportError:
    # Third Party
    from importlib_metadata import packages_distributions


def optional_import(
    extras_require_name: str, module_name: str, attr_name: Optional[str] = None
) -> Any:
    """Optionally import modules or module attributes.

    If the module fails to be imported an error will be thrown when the module is attempted to be
    used, not when the import takes place. The error will state that the given "extras require"
    dependency option needs to be used when distribution package is installed.

    Args:
        extras_require_name: Name of "options.extras_require" option that is required when SimPLER
            is pip installed.
        module_name: Name of the module to try to import.
        attr_name: Name of the module attribute (e.g. function, class, variable) to import.

    Raises:
        RuntimeError: The error is thrown when the failed imported module is attempted to be used.
    """
    try:
        module = importlib.import_module(module_name)
        return module if attr_name is None else getattr(module, attr_name)

    except (ImportError, OSError) as error:
        import_name = module_name if attr_name is None else f"{module_name}.{attr_name}"
        try:
            dist_pkg_name = get_dist_pkg_from_module(module_name)
        except ValueError:
            dist_pkg_name = "<distribution package>"
        msg = (
            f"To use '{import_name}' the '{extras_require_name}' extras required option needs to be"
            " given when the package is pip installed. The install command will look something like"
            f" this 'pip install {dist_pkg_name}[{extras_require_name}]'."
        )
        optional_import_error = error

        def _failed_import(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(msg) from optional_import_error

        return _failed_import


def get_dist_pkg_from_module(module_name: str) -> str:
    """Return the best guess distribution package name for the given module name.

    Args:
        module_name: Name of module to try find the distribution package for.

    Raises:
        ValueError: If there is not a distribution package for the given module name.

    Returns:
        Name of distribution package.
    """
    pkg_dists = packages_distributions()
    return _get_dist_pkg_from_module_for_pkg_dist_list(module_name, pkg_dists)


def _get_dist_pkg_from_module_for_pkg_dist_list(
    module_name: str, pkg_dists: Mapping[str, List[str]]
) -> str:
    """Helper function for the `get_dist_pkg_from_module` function.

    This function is needed so that the `get_dist_pkg_from_module` can be fully tested.

    Args:
        module_name: Name of module to try find the distribution package for.
        pkg_dists: List of distribution packages.

    Raises:
        ValueError: If there is not a distribution package for the given module name.

    Returns:
        Name of distribution package.
    """
    module_name_parts = module_name.split(".")
    module_name_idx = 0
    root_module_name = module_name_parts[module_name_idx]
    try:
        dist_names = list(set(pkg_dists[root_module_name]))
    except KeyError:
        raise ValueError(
            f"Unable to find the distribution package associated with the module '{module_name}'."
        )

    while len(dist_names) > 1 and module_name_idx < len(module_name_parts) - 1:
        module_name_idx += 1
        filtered_dist_names = list(
            filter(
                lambda name_: module_name_parts[module_name_idx].replace("_", "-") in name_,
                dist_names,
            )
        )
        if len(filtered_dist_names) > 0 and len(filtered_dist_names) < len(dist_names):
            dist_names = filtered_dist_names

    return dist_names[0]
