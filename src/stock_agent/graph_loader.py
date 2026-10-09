from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GRAPH_PATH = PROJECT_ROOT / "graph.yaml"


def load_graph_config(
    path: str | Path | None = None,
) -> dict[str, Any]:

    graph_path = (
        Path(path)
        if path is not None
        else DEFAULT_GRAPH_PATH
    )

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph config not found: {graph_path}"
        )

    with graph_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "graph.yaml must contain a YAML object."
        )

    if "nodes" not in config:
        raise ValueError(
            "graph.yaml must contain 'nodes'."
        )

    if "edges" not in config:
        raise ValueError(
            "graph.yaml must contain 'edges'."
        )

    validate_graph_config(config)

    return config


def load_handler(
    handler_path: str,
) -> Callable:

    if ":" not in handler_path:
        raise ValueError(
            "Handler must have format "
            "'module.path:function_name'"
        )

    module_name, function_name = (
        handler_path.split(":", 1)
    )

    module = import_module(module_name)

    try:
        handler = getattr(
            module,
            function_name,
        )
    except AttributeError as exc:
        raise ImportError(
            f"Function '{function_name}' "
            f"not found in '{module_name}'"
        ) from exc

    if not callable(handler):
        raise TypeError(
            f"{handler_path} is not callable"
        )

    return handler


def validate_graph_config(
    config: dict[str, Any],
) -> None:

    nodes = config["nodes"]
    edges = config["edges"]

    if not isinstance(nodes, dict):
        raise ValueError(
            "'nodes' must be a dictionary."
        )

    if not isinstance(edges, list):
        raise ValueError(
            "'edges' must be a list."
        )

    node_names = set(nodes.keys())

    for node_name, node_config in nodes.items():

        if not isinstance(node_config, dict):
            raise ValueError(
                f"Invalid config for node: {node_name}"
            )

        if "handler" not in node_config:
            raise ValueError(
                f"Node '{node_name}' has no handler."
            )

    for edge in edges:

        if "from" not in edge or "to" not in edge:
            raise ValueError(
                "Every edge needs 'from' and 'to'."
            )

        source = edge["from"]
        target = edge["to"]

        sources = (
            source
            if isinstance(source, list)
            else [source]
        )

        for item in sources:

            if item == "END":
                raise ValueError(
                    "END cannot be an edge source."
                )

            if (
                item != "START"
                and item not in node_names
            ):
                raise ValueError(
                    f"Unknown source node: {item}"
                )

        if target == "START":
            raise ValueError(
                "START cannot be an edge target."
            )

        if (
            target != "END"
            and target not in node_names
        ):
            raise ValueError(
                f"Unknown target node: {target}"
            )