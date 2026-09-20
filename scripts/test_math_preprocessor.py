#!/usr/bin/env python3
import unittest

from math_preprocessor import preprocess_markdown


class MathPreprocessorTests(unittest.TestCase):
    def test_protects_setext_like_equals_inside_display_math(self):
        source = "text\n$$\na\n=\nb\n$$\n"
        expected = "text\n$$\na\n\\mathrel{=}\nb\n$$\n"
        self.assertEqual(preprocess_markdown(source), expected)

    def test_preserves_source_outside_math(self):
        source = "Title\n=====\n\ntext\n=\n\n$$\nx = y\n$$\n"
        self.assertEqual(preprocess_markdown(source), source.replace("x = y", "x = y"))

    def test_ignores_fenced_code(self):
        source = "```\n$$\na\n=\n$$\n```\n"
        self.assertEqual(preprocess_markdown(source), source)

    def test_supports_latex_bracket_delimiters(self):
        source = "\\[\na\n=\nb\n\\]\n"
        expected = "\\[\na\n\\mathrel{=}\nb\n\\]\n"
        self.assertEqual(preprocess_markdown(source), expected)


if __name__ == "__main__":
    unittest.main()

