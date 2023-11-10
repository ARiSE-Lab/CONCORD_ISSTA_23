mkdir ts_package;
cd ts_package;
# Download the tree-sitter package
git clone https://github.com/tree-sitter/tree-sitter-c.git;
git clone https://github.com/tree-sitter/tree-sitter-cpp.git;
git clone https://github.com/tree-sitter/tree-sitter-java.git;
cd ..;
# Build tree-sitter
python build_ts_lib.py