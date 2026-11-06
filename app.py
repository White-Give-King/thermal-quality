"""Entry point that launches the local Gradio WebUI."""

from thermal_focus.webui.app import demo

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        css=demo.css,
        head=demo.head,
    )
