import argparse
import ast
import json
import os
import re
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from routellm.controller import ModelPair

"""
The core code is based heavily on the original SGLang implementation.
https://github.com/sgl-project/sglang/blob/main/benchmark/gsm8k/bench_sglang.py
"""

INVALID = -9999999


def select_sglang_backend(args):
    if args.backend.startswith("gpt") or args.backend.startswith("router-"):
        from sglang.lang.backend.openai import OpenAI as SGLangOpenAI

        backend = SGLangOpenAI(
            args.backend,
            base_url=f"{args.host}:{args.port}/v1",
            api_key="dummy",
        )
    else:
        raise ValueError(f"Invalid backend: {args.backend}")
    return backend


def read_jsonl(filename: str):
    """Read a JSONL file."""
    rets = []
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            rets.append(json.loads(line))
    return rets


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def main(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lines = read_jsonl(f"{current_dir}/test.jsonl")
    train = read_jsonl(f"{current_dir}/train.jsonl")
    routed_pair = ModelPair(strong=args.strong_model, weak=args.weak_model)

    if args.limit is not None:
        lines = lines[: args.limit]

    # Construct prompts
    k = args.ntrain
    few_shot_examples = get_few_shot_examples(train, k)

    questions = []
    labels = []
    for i in range(len(lines)):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += sgl.user(few_shot_examples + question)
        s += sgl.assistant(sgl.gen("answer", max_tokens=1024, stop=["Question"]))

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Select backend
    backend = select_sglang_backend(args)

    # Run requests
    tic = time.time()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=0,
        backend=backend,
        num_threads=args.parallel,
        progress_bar=True,
    )

    preds = []
    responses = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))
        responses.append(states[i]["answer"])

    # Compute accuracy
    print(
        f"{args.backend}: accuracy={np.mean(np.array(preds) == np.array(labels)) * 100:.2f}% "
        f"on {len(lines)} GSM8K questions in {time.time() - tic:.2f}s"
    )
    return np.array(preds) == np.array(labels), responses, questions, routed_pair


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate GSM8K strong/weak responses through a running RouteLLM server."
    )
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=str, default="6060")
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--ntrain", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--strong-model",
        type=str,
        default="gpt-4-1106-preview",
    )
    parser.add_argument(
        "--weak-model",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="CSV path for generated GSM8K responses. Defaults to gsm8k_responses.csv.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = args.output_file or f"{current_dir}/gsm8k_responses.csv"
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    evaluate_args_base = {
        "parallel": args.parallel,
        "host": args.host,
        "port": args.port,
        "ntrain": args.ntrain,
        "limit": args.limit,
        "strong_model": args.strong_model,
        "weak_model": args.weak_model,
    }

    weak_cors, weak_responses, prompts, routed_pair = main(
        SimpleNamespace(**evaluate_args_base, backend="router-random-1.0"),
    )
    strong_cors, strong_responses, _, _ = main(
        SimpleNamespace(**evaluate_args_base, backend="router-random-0.0"),
    )

    assert len(weak_cors) == len(strong_cors)

    result_df = pd.DataFrame(
        zip(prompts, weak_cors, strong_cors, weak_responses, strong_responses),
        columns=[
            "prompt",
            routed_pair.weak,
            routed_pair.strong,
            f"{routed_pair.weak}_response",
            f"{routed_pair.strong}_response",
        ],
    )

    result_df.to_csv(output_file, index=False)
    print(f"Saved GSM8K responses to {output_file}")
