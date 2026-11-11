from __future__ import annotations

from typing import Any


def describe_api_component(
    component: Any,
    description: str,
    *,
    input_info: dict[str, Any] | None = None,
) -> None:
    for method_name in ("api_info_as_input", "api_info_as_output"):
        api_info = getattr(component, method_name, None)
        if api_info is None:
            continue
        info = (
            input_info
            if method_name == "api_info_as_input" and input_info is not None
            else None
        )
        setattr(
            component,
            method_name,
            lambda api_info=api_info, info=info: {
                **(info if info is not None else api_info()),
                "additional_description": description,
            },
        )
