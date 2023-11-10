#!/usr/bin/env python
# coding=utf-8
from tree_sitter import Language

def build_language_lib():
    for lang in ["java", "c", "cpp"]:
        git_dir = f"ts_package/tree-sitter-{lang}"
        Language.build_library(f'build/{lang}-lang-parser.so', [git_dir])

if __name__ == "__main__":
    build_language_lib()