import os
import json


if __name__ == "__main__":
    label = json.load(open("label.json", "r"))
    mapping = json.load(open("mapping.json", "r"))
    # print(mapping.keys())
    vids = dict(map(lambda s: (s[:4], s), mapping.keys()))
    print(vids)
    for aug_vid in os.listdir("data/augmented/"):
        if aug_vid[:4] in vids:
            mapping[aug_vid] = mapping[vids[aug_vid[:4]]]
    json.dump(mapping, open("mapping_test.json", "w"))
