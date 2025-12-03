import json
import math
import matplotlib.pyplot as plt

# 1) Per class AP bar chart from eval_results.json
with open("./output/eval_results.json") as f:
    results = json.load(f)

bbox = results["bbox"]

overall_keys = {"AP", "AP50", "AP75", "APs", "APm", "APl"}
class_aps = {k.replace("AP-", ""): v
             for k, v in bbox.items()
             if k not in overall_keys and not (isinstance(v, float) and math.isnan(v))}

plt.figure()
plt.bar(list(class_aps.keys()), list(class_aps.values()))
plt.xticks(rotation=45, ha="right")
plt.ylabel("AP")
plt.title("Per-class AP (bbox)")
plt.tight_layout()
plt.savefig("per_class_ap.png", dpi=200)
print("Saved per_class_ap.png")

# 2) Confidence score histogram from coco_instances_results.json
with open("./output/coco_instances_results.json") as f:
    preds = json.load(f)

scores = [p["score"] for p in preds]

plt.figure()
plt.hist(scores, bins=20)
plt.xlabel("Confidence score")
plt.ylabel("Count")
plt.title("Detection confidence distribution")
plt.tight_layout()
plt.savefig("confidence_hist.png", dpi=200)
print("Saved confidence_hist.png")