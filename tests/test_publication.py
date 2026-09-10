"""Release-hygiene tests for documentation and the static project page."""

import ast
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tokenize
import unittest

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _ProjectPageParser(HTMLParser):
    """Collect identifiers, references, and image alternatives from HTML."""

    def __init__(self) -> None:
        """Create empty collections for parsed page attributes."""
        super().__init__()
        self.identifiers: set[str] = set()
        self.references: list[str] = []
        self.image_alternatives: list[str | None] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Record local-navigation and accessibility attributes."""
        attributes = dict(attrs)
        identifier = attributes.get("id")
        if identifier:
            self.identifiers.add(identifier)
        for name in ("href", "src"):
            reference = attributes.get(name)
            if reference:
                self.references.append(reference)
        if tag == "img":
            self.image_alternatives.append(attributes.get("alt"))


class PublicationTests(unittest.TestCase):
    """Verify repository documentation and website publication artifacts."""

    def test_every_python_scope_has_documentation(self) -> None:
        """Every module, class, and function supplies a useful docstring."""
        undocumented: list[str] = []
        for path in PROJECT_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if ast.get_docstring(tree) is None:
                undocumented.append(f"module {path}")
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ) and ast.get_docstring(node) is None:
                    undocumented.append(f"{path}:{node.lineno}:{node.name}")
        self.assertEqual(undocumented, [])

    def test_source_comments_use_present_tense_descriptions(self) -> None:
        """Source comments do not contain implementation-history language."""
        history_pattern = re.compile(
            r"\b(previously|former|legacy|retained|unchanged|"
            r"single[- ]file|old version)\b",
            re.IGNORECASE,
        )
        matches: list[str] = []
        for path in PROJECT_ROOT.rglob("*.py"):
            with path.open("rb") as source:
                for token in tokenize.tokenize(source.readline):
                    if token.type == tokenize.COMMENT and history_pattern.search(
                        token.string
                    ):
                        matches.append(f"{path}:{token.start[0]}:{token.string}")
        self.assertEqual(matches, [])

    def test_static_page_references_resolve(self) -> None:
        """Local links and images resolve and all images include alt text."""
        page_path = PROJECT_ROOT / "docs" / "index.html"
        parser = _ProjectPageParser()
        parser.feed(page_path.read_text(encoding="utf-8"))

        unresolved: list[str] = []
        for reference in parser.references:
            if reference.startswith("#"):
                if reference[1:] not in parser.identifiers:
                    unresolved.append(reference)
            elif not re.match(r"^[a-z]+:", reference):
                if not (page_path.parent / reference).exists():
                    unresolved.append(reference)

        self.assertEqual(unresolved, [])
        self.assertTrue(parser.image_alternatives)
        self.assertTrue(all(parser.image_alternatives))

    def test_website_preview_is_publication_sized(self) -> None:
        """The demonstration graph is large enough for high-density displays."""
        preview_path = PROJECT_ROOT / "docs" / "assets" / "decay-preview.png"
        with Image.open(preview_path) as image:
            self.assertGreaterEqual(image.width, 1_600)
            self.assertGreaterEqual(image.height, 1_600)

    def test_jupyter_launchers_are_valid_and_compilable(self) -> None:
        """Every Jupyter launcher contains valid JSON and Python source."""
        notebook_paths = list(PROJECT_ROOT.glob("*.ipynb"))
        self.assertTrue(notebook_paths)
        for notebook_path in notebook_paths:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            code_cells = [
                cell for cell in notebook["cells"] if cell["cell_type"] == "code"
            ]
            self.assertTrue(code_cells)
            for cell in code_cells:
                compile("".join(cell["source"]), str(notebook_path), "exec")


if __name__ == "__main__":
    unittest.main()
