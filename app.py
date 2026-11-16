"""Entry point that launches the local Gradio WebUI."""

import argparse
from collections.abc import Sequence

from webui.app import create_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enable-annotation",
        action="store_true",
        help="enable the image annotation page and annotation APIs",
    )
    parser.add_argument(
        "--enable-training",
        action="store_true",
        help="enable the model training page and manual training API",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="port for the local Gradio server (default: 7860)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    global demo
    args = build_parser().parse_args(argv)
    demo = create_demo(
        enable_annotation=args.enable_annotation,
        enable_training=args.enable_training,
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        show_error=True,
        css=demo.css,
        head=demo.head,
        footer_links=["api"],
    )


if __name__ == "__main__":
    main()
