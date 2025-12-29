# helper.py

from typing import Optional
from IPython.display import display, Markdown


def ewriter(text: str, title: Optional[str] = None) -> None:
    """
    Simple text writer utility.
    Used for printing or rendering model output.
    """
    if title:
        print(f"\n=== {title} ===\n")
    print(text)


def writer_gui(text: str, title: Optional[str] = None) -> None:
    """
    GUI-style writer for Jupyter notebooks.
    Renders markdown output nicely.
    """
    if title:
        display(Markdown(f"## {title}"))
    display(Markdown(text))
