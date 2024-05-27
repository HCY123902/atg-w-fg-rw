import os
import json
import random

# seed = 42
# random.seed(seed)

# total = 1002

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--holistic", action="store_true", help="Whether to combine the holistic samples or not")

args = parser.parse_args()

with open("combined_train.json", "r") as combined_s_json:
    combined_s = json.load(combined_s_json)

behaviroal_cloning_samples = {}

suffix = "_rs_h" if args.holistic else "_rs_fg"

for dataset in ["asqa", "qampari", "eli5"]:
    path = "{}_train{}.json".format(dataset, suffix)
    with open(path, "r") as samples_json:
        samples = json.load(samples_json)
        questions = [s["question"] for s in combined_s if s["dataset"] == dataset]
        behaviroal_cloning_samples[dataset] = {s["question"]: s for s in samples if s["question"] in questions}
        if len(behaviroal_cloning_samples[dataset]) != len(questions):
            print("There are duplicate samples with the same questions in {}: Original count: {} | Count after deduplication: {}".format(dataset, len(questions), len(behaviroal_cloning_samples[dataset])))

for s in combined_s:
    s["output"] = behaviroal_cloning_samples[s["dataset"]][s["question"]]["output"]

with open("combined_train{}.json".format(suffix), "w") as new_combined_s_json:
    json.dump(combined_s, new_combined_s_json, indent=4)
