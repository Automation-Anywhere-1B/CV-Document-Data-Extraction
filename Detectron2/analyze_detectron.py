import json
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt

GT_ANN_FILE = "./data/annotations/instances_val.json"
PRED_FILE = "./output/coco_instances_results.json"
IOU_THRESH_PR = 0.5
PR_CLASSES_TO_PLOT = None

def coco_xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)

def iou(box1, box2):
    """IoU between two [x,y,w,h] boxes in COCO format."""
    b1 = coco_xywh_to_xyxy(box1)
    b2 = coco_xywh_to_xyxy(box2)

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def load_data():
    print("Loading ground truth:", GT_ANN_FILE)
    with open(GT_ANN_FILE) as f:
        gt = json.load(f)

    print("Loading predictions:", PRED_FILE)
    with open(PRED_FILE) as f:
        preds = json.load(f)

    # Build mappings
    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    cat_ids = sorted(cat_id_to_name.keys())
    n_classes = len(cat_ids)
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

    gt_by_image = defaultdict(list)
    for ann in gt["annotations"]:
        gt_by_image[ann["image_id"]].append(
            {"bbox": ann["bbox"], "category_id": ann["category_id"]}
        )

    pred_by_image = defaultdict(list)
    for p in preds:
        pred_by_image[p["image_id"]].append(p)

    gt_count_per_class = Counter()
    for ann in gt["annotations"]:
        gt_count_per_class[ann["category_id"]] += 1

    return (
        gt_by_image,
        pred_by_image,
        cat_id_to_name,
        cat_id_to_idx,
        gt_count_per_class,
        n_classes,
    )


def build_confusion_and_iou(gt_by_image, pred_by_image, cat_id_to_name, cat_id_to_idx):
    n_classes = len(cat_id_to_idx)
    conf_mat = np.zeros((n_classes, n_classes), dtype=int)
    ious_correct_class = []

    for image_id, preds in pred_by_image.items():
        gts = gt_by_image[image_id]
        gt_used = np.zeros(len(gts), dtype=bool)

        for p in preds:
            p_box = p["bbox"]
            p_cat = p["category_id"]

            best_iou = 0.0
            best_gt_idx = -1
            best_gt_cat = None

            for j, g in enumerate(gts):
                i = iou(p_box, g["bbox"])
                if i > best_iou:
                    best_iou = i
                    best_gt_idx = j
                    best_gt_cat = g["category_id"]

            if best_iou >= IOU_THRESH_PR and best_gt_idx >= 0:
                r = cat_id_to_idx[best_gt_cat]
                c = cat_id_to_idx[p_cat]
                conf_mat[r, c] += 1

                if p_cat == best_gt_cat:
                    ious_correct_class.append(best_iou)

                gt_used[best_gt_idx] = True

    return conf_mat, np.array(ious_correct_class)


def plot_confusion(conf_mat, cat_id_to_name, cat_id_to_idx):
    labels = [cat_id_to_name[cid] for cid in sorted(cat_id_to_name.keys(), key=lambda x: cat_id_to_idx[x])]

    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, interpolation="nearest")
    plt.title(f"Confusion matrix (IoU ≥ {IOU_THRESH_PR})")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    print("Saved confusion_matrix.png")


def plot_iou_hist(ious_correct_class):
    plt.figure()
    plt.hist(ious_correct_class, bins=20)
    plt.xlabel("IoU (correct class matches)")
    plt.ylabel("Count")
    plt.title(f"IoU distribution (matches, IoU ≥ {IOU_THRESH_PR})")
    plt.tight_layout()
    plt.savefig("iou_distribution.png", dpi=200)
    print("Saved iou_distribution.png")


def compute_pr_for_class(class_id, gt_by_image, pred_by_image, gt_count_per_class):
    """Standard PR + AP at IOU_THRESH_PR for a single class."""
    n_gt = gt_count_per_class[class_id]
    if n_gt == 0:
        return None, None, None

    preds = []
    for img_id, plist in pred_by_image.items():
        for p in plist:
            if p["category_id"] == class_id:
                preds.append({"image_id": img_id, "bbox": p["bbox"], "score": p["score"]})

    if not preds:
        return None, None, None

    preds.sort(key=lambda x: x["score"], reverse=True)

    gt_used = {img_id: np.zeros(len(gts), dtype=bool) for img_id, gts in gt_by_image.items()}

    tps = []
    fps = []

    for p in preds:
        img_id = p["image_id"]
        p_box = p["bbox"]
        gts = gt_by_image[img_id]

        best_iou = 0.0
        best_gt_idx = -1

        for j, g in enumerate(gts):
            if g["category_id"] != class_id:
                continue
            i = iou(p_box, g["bbox"])
            if i > best_iou:
                best_iou = i
                best_gt_idx = j

        if best_iou >= IOU_THRESH_PR and best_gt_idx >= 0 and not gt_used[img_id][best_gt_idx]:
            tps.append(1)
            fps.append(0)
            gt_used[img_id][best_gt_idx] = True
        else:
            tps.append(0)
            fps.append(1)

    tps = np.array(tps)
    fps = np.array(fps)

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recall = cum_tp / max(n_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

    ap = compute_ap(recall, precision)
    return recall, precision, ap


def compute_ap(recall, precision):
    """COCO-style area under PR curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def plot_pr_curves(gt_by_image, pred_by_image, cat_id_to_name, cat_id_to_idx, gt_count_per_class):
    cat_ids_sorted = sorted(cat_id_to_name.keys(), key=lambda x: cat_id_to_idx[x])
    if PR_CLASSES_TO_PLOT is None:
        class_ids_to_plot = cat_ids_sorted
    else:
        name_to_id = {v: k for k, v in cat_id_to_name.items()}
        class_ids_to_plot = [name_to_id[n] for n in PR_CLASSES_TO_PLOT if n in name_to_id]

    plt.figure()
    aps = []

    for cid in class_ids_to_plot:
        rec, prec, ap = compute_pr_for_class(cid, gt_by_image, pred_by_image, gt_count_per_class)
        cname = cat_id_to_name[cid]
        if rec is None:
            print(f"No GT or predictions for class {cname}, skipping PR.")
            continue
        aps.append(ap)
        plt.plot(rec, prec, label=f"{cname} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curves (IoU ≥ {IOU_THRESH_PR})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pr_curves.png", dpi=200)
    print("Saved pr_curves.png")


def main():
    (
        gt_by_image,
        pred_by_image,
        cat_id_to_name,
        cat_id_to_idx,
        gt_count_per_class,
        n_classes,
    ) = load_data()

    # 1) Confusion matrix + IoU distribution
    conf_mat, ious_correct = build_confusion_and_iou(
        gt_by_image, pred_by_image, cat_id_to_name, cat_id_to_idx
    )
    plot_confusion(conf_mat, cat_id_to_name, cat_id_to_idx)
    plot_iou_hist(ious_correct)

    # 2) PR curves + AP for each plotted class
    plot_pr_curves(
        gt_by_image,
        pred_by_image,
        cat_id_to_name,
        cat_id_to_idx,
        gt_count_per_class,
    )

    print("Done.")


if __name__ == "__main__":
    main()