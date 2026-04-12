import argparse
import json
import os
import pickle
import time

from dspy.clients import configure_cache
from dspy.dsp.utils.settings import settings
from synthesizeme.personalrm.synthesizeme import SynthesizeMe
from synthesizeme.utils.dspy_methods import LLMAsAJudgeProgram
from synthesizeme.utils.format_conv import format_conversation
from tqdm import tqdm

configure_cache(enable_disk_cache=False, enable_memory_cache=False)

dataset = "LaMP_4"  # "LaMP_4", "abstract_generation", "pwab_pos"
percent = 1
port = 8010
k = 3 if dataset == "abstract_generation" else 5
form = "json" if dataset == "pwab_pos" else "raw"
FAST_MODE = True
USE_RM_CACHE = True
DEFAULT_BEST_OF_N = 4
RM_CACHE_DIR = f"pa_back/output/synthesizeme_cache/{dataset}/"
if percent < 1:
    RM_CACHE_DIR = f"pa_back/output/synthesizeme_cache/{dataset}_{percent}/"
RM_NUM_SEARCH_CANDIDATES = 3 if FAST_MODE else 10
RM_MAX_BOOTSTRAPPED_DEMOS = 3 if FAST_MODE else -1
RM_MAX_LABELED_DEMOS = 2 if FAST_MODE else 4
RM_NUM_WORKERS = 4 if FAST_MODE else 8
RM_MAX_TOKENS = 1024 if FAST_MODE else 4096
RM_STOP_AT_SCORE = 60.0 if FAST_MODE else 80.0

os.makedirs(RM_CACHE_DIR, exist_ok=True)


def configure_runtime(args):
    global dataset, percent, port, k, form, RM_CACHE_DIR

    dataset = args.dataset
    percent = args.percent
    port = args.port
    k = 3 if dataset == "abstract_generation" else 5
    form = "json" if dataset == "pwab_pos" else "raw"
    RM_CACHE_DIR = f"pa_back/output/synthesizeme_cache/{dataset}/"
    if percent < 1:
        RM_CACHE_DIR = f"pa_back/output/synthesizeme_cache/{dataset}_{percent}/"
    os.makedirs(RM_CACHE_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use the SynthesizeMe reward model to select the best response from N candidates."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=dataset,
        choices=["LaMP_4", "abstract_generation", "pwab_pos"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=percent,
        help="Training data percentage suffix used by some datasets, e.g. 0.2.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=port,
        help="Port of the local RM model server.",
    )
    parser.add_argument(
        "--best-of-n",
        type=int,
        default=DEFAULT_BEST_OF_N,
        help="Number of candidate generation files to compare. Defaults to 2.",
    )
    parser.add_argument(
        "--candidate-indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit candidate suffix indices to load, e.g. --candidate-indices 0 2 4. Overrides --best-of-n.",
    )
    return parser.parse_args()


def resolve_candidate_indices(args):
    if args.candidate_indices is not None:
        if not args.candidate_indices:
            raise ValueError("candidate_indices cannot be empty.")
        if len(set(args.candidate_indices)) != len(args.candidate_indices):
            raise ValueError("candidate_indices must be unique.")
        return args.candidate_indices

    if args.best_of_n < 1:
        raise ValueError("best_of_n must be at least 1.")

    return list(range(args.best_of_n))


def synme_format(message):
    for key, value in message.items():
        if isinstance(value, dict):
            if "tool_call" not in value:
                value = {"tool_call": value}
            message[key] = json.dumps(value)

    return {
        "context": [{"role": "user", "content": message["question"]}],
        "chosen": {"role": "assistant", "content": message["chosen"]},
        "rejected": {"role": "assistant", "content": message["rejected"]},
    }


def load_synme_data():
    if dataset == "LaMP_4":
        with open("pa_back/caa_data/caa_python_LaMP_4_0.15_qwen3_others.json", "r") as f:
            data = json.load(f)
    elif dataset == "abstract_generation":
        train_file = "pa_back/caa_data/caa_python_abstract_generation_0.2_qwen3_others_3.json"
        if percent < 1:
            train_file = f"pa_back/caa_data/caa_python_abstract_generation_0.2_qwen3_others_3_{percent}.json"
        with open(train_file, "r") as f:
            data = json.load(f)
    elif dataset == "pwab_pos":
        with open("pa_back/caa_data/caa_json_pwab_pos_0_llama-3.1_others_5.json", "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    synme_data = {}
    for uid, messages in data.items():
        uid = str(uid)
        synme_data[uid] = [synme_format(message) for message in messages]

    return synme_data


def build_rm(user_id):
    rm = SynthesizeMe(
        user_id=str(user_id),
        model_id="/home/jxliu/ext0/models/Qwen3-8B",
        model_url=f"http://localhost:{port}/v1",
        num_search_candidates=RM_NUM_SEARCH_CANDIDATES,
        max_bootstrapped_demos=RM_MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=RM_MAX_LABELED_DEMOS,
        num_workers=RM_NUM_WORKERS,
        stop_at_score=RM_STOP_AT_SCORE,
    )
    if rm.lm.kwargs.get("temperature") is None:
        rm.lm.kwargs["temperature"] = 0.1
    rm.lm.kwargs["max_tokens"] = RM_MAX_TOKENS
    rm.lm.cache = False
    rm.lm.cache_in_memory = False
    return rm


def prepare_user_rm(user_id, train_data):
    rm = build_rm(user_id)
    cache_program_path = f"{RM_CACHE_DIR}{user_id}.json"
    cache_persona_path = f"{RM_CACHE_DIR}{user_id}_persona.txt"

    if USE_RM_CACHE and os.path.exists(cache_program_path) and os.path.exists(cache_persona_path):
        try:
            rm.program = rm.load(path=RM_CACHE_DIR)
            rm.persona = getattr(rm.program, "persona", None)
            return rm, "personalized_cached"
        except Exception as error:
            print(f"User {user_id} cache load failed, retraining: {error}")

    if not train_data:
        rm.program = LLMAsAJudgeProgram()
        return rm, "generic_empty_train"

    try:
        if len(train_data) == 1:
            rm.fit(train_data, val_data=train_data)
            if callable(getattr(rm, "program", None)):
                if USE_RM_CACHE:
                    rm.save(path=RM_CACHE_DIR)
                return rm, "personalized_single_example"
            rm.program = LLMAsAJudgeProgram()
            return rm, "generic_fit_fallback"

        rm.fit(train_data)
        if callable(getattr(rm, "program", None)):
            if USE_RM_CACHE:
                rm.save(path=RM_CACHE_DIR)
            return rm, "personalized"
    except Exception as error:
        print(f"User {user_id} fit failed, falling back to generic judge: {error}")

    rm.program = LLMAsAJudgeProgram()
    return rm, "generic_fit_fallback"


def init_usage():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def add_usage(total_usage, new_usage):
    total_usage["input_tokens"] += new_usage["input_tokens"]
    total_usage["output_tokens"] += new_usage["output_tokens"]
    total_usage["total_tokens"] += new_usage["total_tokens"]


def extract_token_usage(prediction):
    usage = init_usage()

    if not hasattr(prediction, "get_lm_usage"):
        return usage

    lm_usage = prediction.get_lm_usage()
    if not lm_usage:
        return usage

    for model_usage in lm_usage.values():
        input_tokens = model_usage.get("prompt_tokens", model_usage.get("input_tokens", 0)) or 0
        output_tokens = model_usage.get("completion_tokens", model_usage.get("output_tokens", 0)) or 0
        total_tokens = model_usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens

        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        usage["total_tokens"] += total_tokens

    return usage


def predict_pairwise_compat(rm, context, option1, option2):
    with settings.context(track_usage=True):
        if not callable(getattr(rm, "program", None)):
            raise RuntimeError("RM program is not initialized; no train data or fit failed.")
        try:
            response = rm.predict_pairwise(context, option1, option2)
        except AttributeError as error:
            if "predict" not in str(error):
                raise
            response = rm.program(
                conversation=format_conversation(context),
                completion_one=format_conversation(option1),
                completion_two=format_conversation(option2),
            )

    return response, extract_token_usage(response)


def prefers_first(preference):
    if isinstance(preference, str):
        normalized = preference.strip().lower()
        if normalized in {"1", "first", "option1", "option_1", "completion_one"}:
            return True
        if normalized in {"2", "second", "option2", "option_2", "completion_two"}:
            return False

    if preference == 1:
        return True
    if preference == 2:
        return False

    raise ValueError(f"Unexpected RM preference output: {preference!r}")


def to_string(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def build_generation_path(candidate_idx):
    return (
        f"pa_back/output/generation/rag_{k}_llama3-8b_-1_False/"
        f"llama-3.1_{dataset}_20_{form}_1.0_1.0_1.0_False_{candidate_idx}/generation.json"
    )


def load_generation_dicts(candidate_indices):
    generation_dicts = {}
    for candidate_idx in candidate_indices:
        generation_path = build_generation_path(candidate_idx)
        with open(generation_path, "r") as f:
            generation_data = json.load(f)
        generation_dicts[candidate_idx] = {
            str(item["id"]): item for item in generation_data["golds"]
        }
    return generation_dicts


def build_candidate_options(sample_id, generation_dicts, candidate_indices):
    candidate_options = []
    for candidate_idx in candidate_indices:
        if sample_id not in generation_dicts[candidate_idx]:
            raise KeyError(
                f"Sample id {sample_id} not found in candidate generation {candidate_idx}."
            )
        candidate_options.append(
            (
                candidate_idx,
                {
                    "role": "assistant",
                    "content": to_string(generation_dicts[candidate_idx][sample_id]["generation"]),
                },
            )
        )
    return candidate_options


def select_best_candidate(rm, context, candidate_options):
    if not candidate_options:
        raise ValueError("candidate_options cannot be empty.")

    best_candidate_idx, best_option = candidate_options[0]
    total_usage = init_usage()
    num_predictions = 0

    for challenger_idx, challenger_option in candidate_options[1:]:
        response, token_usage = predict_pairwise_compat(
            rm, context, best_option, challenger_option
        )
        add_usage(total_usage, token_usage)
        num_predictions += 1
        preference = getattr(response, "preference", response)

        if not prefers_first(preference):
            best_candidate_idx = challenger_idx
            best_option = challenger_option

    return best_candidate_idx, best_option, total_usage, num_predictions


def get_result_dir(best_of_n):
    chosen_suffix = "chosen" if best_of_n == 2 else f"chosen_n{best_of_n}"
    res_dir = (
        f"pa_back/output/generation/rag_{k}_llama3-8b_-1_False/"
        f"llama-3.1_{dataset}_20_raw_1.0_1.0_1.0_False_{chosen_suffix}"
    )
    if percent < 1:
        res_dir = (
            f"pa_back/output/generation/rag_{k}_llama3-8b_-1_False/"
            f"llama-3.1_{dataset}_{percent}_20_raw_1.0_1.0_1.0_False_{chosen_suffix}"
        )
    return res_dir


def format_elapsed_time(seconds):
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours >= 1:
        return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"
    if minutes >= 1:
        return f"{int(minutes):02d}:{secs:05.2f}"
    return f"{secs:.2f}s"


def main():
    overall_start_time = time.perf_counter()
    args = parse_args()
    configure_runtime(args)
    candidate_indices = resolve_candidate_indices(args)
    best_of_n = len(candidate_indices)
    synme_data = load_synme_data()

    with open(f"pa_back/data/{dataset}/processed/seen_test.pkl", "rb") as f:
        test_data = pickle.load(f)

    generation_dicts = load_generation_dicts(candidate_indices)
    chosen_gen = []
    token_usage_summary = {
        "overall": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "num_predictions": 0,
        },
        "per_user": {},
    }
    timing_summary = {
        "overall_elapsed_seconds": 0.0,
        "per_user": {},
    }

    print(
        f"Evaluating on dataset {dataset} with {len(test_data)} users and best_of_n={best_of_n}..."
    )
    for idx, (uid, test_samples) in enumerate(test_data.items()):
        if idx >= 30:
            break
        user_start_time = time.perf_counter()
        uid = str(uid)
        train_data = synme_data.get(uid, [])
        print(f"Processing user{idx} {uid} with {len(train_data)} train samples...")
        rm, rm_mode = prepare_user_rm(uid, train_data)
        token_usage_summary["per_user"][uid] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "num_predictions": 0,
            "rm_mode": rm_mode,
            "num_train_preferences": len(train_data),
        }

        for test_sample in tqdm(test_samples, desc=f"user {uid} [{rm_mode}]"):
            sample_id = str(test_sample["id"])
            context = [{"role": "user", "content": test_sample["input"]}]
            candidate_options = build_candidate_options(
                sample_id, generation_dicts, candidate_indices
            )
            selected_candidate_idx, chosen_option, token_usage, num_predictions = (
                select_best_candidate(rm, context, candidate_options)
            )

            add_usage(token_usage_summary["per_user"][uid], token_usage)
            add_usage(token_usage_summary["overall"], token_usage)
            token_usage_summary["per_user"][uid]["num_predictions"] += num_predictions
            token_usage_summary["overall"]["num_predictions"] += num_predictions

            chosen_gen.append(
                {
                    "id": sample_id,
                    "generation": chosen_option["content"],
                    "output": test_sample["output"],
                    "type": test_sample.get("type", None),
                    "uid": uid,
                    "rm_mode": rm_mode,
                    "num_train_preferences": len(train_data),
                    "selected_candidate_idx": selected_candidate_idx,
                    "predict_input_tokens": token_usage["input_tokens"],
                    "predict_output_tokens": token_usage["output_tokens"],
                    "predict_total_tokens": token_usage["total_tokens"],
                    "num_pairwise_predictions": num_predictions,
                }
            )

        user_elapsed_seconds = time.perf_counter() - user_start_time
        timing_summary["per_user"][uid] = {
            "elapsed_seconds": round(user_elapsed_seconds, 4),
            "num_test_samples": len(test_samples),
            "rm_mode": rm_mode,
        }
        print(
            f"Finished user{idx} {uid} in {format_elapsed_time(user_elapsed_seconds)}."
        )

    overall_elapsed_seconds = time.perf_counter() - overall_start_time
    timing_summary["overall_elapsed_seconds"] = round(overall_elapsed_seconds, 4)
    print(f"Finished all users in {format_elapsed_time(overall_elapsed_seconds)}.")

    result = {
        "dataset": dataset,
        "best_of_n": best_of_n,
        "candidate_indices": candidate_indices,
        "golds": chosen_gen,
        "predict_token_usage": token_usage_summary,
        "timing": timing_summary,
    }
    res_dir = get_result_dir(best_of_n)
    os.makedirs(res_dir, exist_ok=True)
    with open(f"{res_dir}/generation.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
