"""Entry point that launches the local Gradio WebUI."""

import argparse
from collections.abc import Sequence

from webui.app import create_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="start only the image inference interface",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="port for the Gradio server (default: 7860)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    global demo
    args = build_parser().parse_args(argv)
    demo = create_demo(inference_only=args.inference_only)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        show_error=True,
        css=demo.css,
        head=demo.head,
    )


if __name__ == "__main__":
    main()
