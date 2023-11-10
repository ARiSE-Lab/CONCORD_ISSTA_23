from tree_sitter import Language, Parser
import argparse
import sentencepiece as spm
import json
from tqdm import tqdm

def check_string(tokenString):
    """
    Check if the token string is empty or not for corner cases
    """
    tokenString = ["".join(t.split()) for t in tokenString]
    ts = " ".join(tokenString)
    if ts.strip() == "":
        return None
    ts = " ".join(ts.split("\n"))
    return ts

def tree_to_token(root_node, string):
    """
    Traverse the tree and return the token list
    """
    if (len(root_node.children) == 0 or "string" in root_node.type) and root_node.type != 'comment':
        str_const = string[root_node.start_byte:root_node.end_byte].decode("utf-8")
        return [str_const]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens += tree_to_token(child, string)
    return code_tokens

def parse_tree(parser, s, all_tokens):
    """
    Parse the code snippet and return the token list
    """
    tree = parser.parse(s)
    root_node = tree.root_node
    tokens = tree_to_token(root_node, s)
    if all_tokens is not None:
        all_tokens += tokens
    return tokens, all_tokens

def gen_vuldetect(input_file, tokenized_file, spm_model):
    # Initialize the parser. Vulnerability detection task only needs C parser.
    LANGUAGE = Language("./build/c-lang-parser.so", 'c')
    parser = Parser()
    parser.set_language(LANGUAGE)

    # Write the tokenized file
    wf = open(tokenized_file, 'w')
    wf.write("code\tlabel\n")  # write header

    # read the raw input file from CodeXGLUE
    with open(input_file, 'r') as f:
        lines = f.readlines()
        samples = [json.loads(l) for l in lines]
        for i, s in enumerate(tqdm(samples, total=len(samples))):
            func = s["func"]
            label = s["target"]
            tokens, _ = parse_tree(parser, func.encode(), None)

            token_string = [token for token in tokens if token.strip() != ""]
        
            orig_str = check_string(token_string)

            if orig_str is None:
                print(f"Error: No. {i} sample")
                orig_str = "Error"
            processed_str = " ".join(spm_model.encode(orig_str, out_type=str))

            wf.write(processed_str + "\t" + str(label) + "\n")

        print("Processed # Samples:", i)

    wf.close()

def gen_clone(input_file, write_file, spm_model):
    # Initialize the parser. Clone detection task of POJ-104 needs both C and C++ parsers.
    C_LANGUAGE = Language('./build/c-lang-parser.so', 'c')
    c_parser = Parser()
    c_parser.set_language(C_LANGUAGE)
    CPP_LANGUAGE = Language('./build/cpp-lang-parser.so', 'cpp')
    cpp_parser = Parser()
    cpp_parser.set_language(CPP_LANGUAGE)
    wf = open(write_file, 'w')
    with open(input_file, 'r') as f:
        samples = f.readlines()
    for s in tqdm(samples):
        tw = dict()
        s = json.loads(s)

        # simple heuristic to determine the language is C or C++
        if "cin" in s["code"] or "cout" in s["code"]:
            parser = cpp_parser
        else:
            parser = c_parser
        tokens, _ = parse_tree(parser, s["code"].encode(), None)
        token_string = [token for token in tokens if token.strip() != ""]
        orig_str = check_string(token_string)
        if orig_str is None:
            print(f"None token error: {s}")
            continue
        if spm_model is None:
            processed_str = orig_str
        else:
            processed_str = " ".join(spm_model.encode(orig_str, out_type=str))

        tw["label"] = s["label"]
        tw["index"] = s["index"]
        tw["code"] = processed_str
        wf.write(json.dumps(tw, ensure_ascii=False) + "\n")
    wf.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', choices=["cxg_vuldetect", "cxg_clone"], help="choose the task name")
    parser.add_argument('--input_file', help="raw input file")
    parser.add_argument('--output_file', help="output file for the model input")
    parser.add_argument('--spm_model', help="pre-trained bpe model")
    args = parser.parse_args()

    spm_model = spm.SentencePieceProcessor(model_file=args.spm_model) if args.spm_model is not None else None

    if args.task_name == "cxg_vuldetect":
        gen_vuldetect(args.input_file, args.output_file, spm_model)
    elif args.task_name == "cxg_clone":
        gen_clone(args.input_file, args.output_file, spm_model)



if __name__ == '__main__':
    main()
