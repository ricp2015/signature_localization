from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from types import List, Dict, Tuple

def rect_inter(a, b):
    inter =  (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    if inter[0] > inter[2] or inter[1] > inter[3]:
        return ()
    return inter

def area(a):
    if len(a) == 4:
        return (a[2] - a[0]) * (a[3] - a[1])
    return 0

def overlap(a, b):
    return area(rect_inter(a, b)) / min(area(a), area(b))    

def intersect_over_union(a, b):
    return area(rect_inter(a, b)) / (area(a) + area(b) - area(rect_inter(a, b)))

def filter_result_dict(result, thresh=0.75, overlap_thres=0.5):
    return filter_result(result["boxes"], result["scores"], thresh, overlap_thres)

def filter_result(boxes: List[Tuple[float]], scores:List[float], thresh=0.75, overlap_thres=0.5) -> Tuple[List[Tuple[float]], List[float]]:
    """
    Similar to `torchvision.ops.nms`.

    The `filter_result` function filters boxes and scores based on a minimum score threshold and avoids
    overlap between boxes exceeding a specified threshold.
    
    :param boxes: The `boxes` parameter in the `filter_result` function represents a list of bounding
    boxes detected in an object detection task. Each bounding box is typically represented as a tuple or
    a list containing the coordinates of the box, such as (x_min, y_min, x_max, y_max), where
    :param scores: The `scores` parameter in the `filter_result` function represents the confidence
    scores associated with each box in the `boxes` parameter. These scores indicate the likelihood that
    the object detected in the corresponding box is correct. The function filters out boxes with scores
    below a certain threshold (`thresh`) and also ensures
    :param thresh: The `thresh` parameter in the `filter_result` function represents the minimum score
    threshold that a box must meet in order to be considered as a valid detection. Boxes with scores
    below this threshold will be filtered out from the final result
    :param overlap_thres: The `overlap_thres` parameter in the `filter_result` function is the overlap
    threshold that determines whether two boxes are considered to be overlapping. It is calculated as
    the overlap coefficient between the two boxes
    :return: `chosen_boxes` and `chosen_scores`: the filtered boxes and their corresponding scores that meet the minimum score
    threshold (`thresh`) and do not have overlap greater than the specified overlap threshold
    (`overlap_thres`).
    """
    chosen_boxes = []
    chosen_scores = []
    result = sorted(zip(boxes, scores), key=lambda x:-x[1])
    for box, score in result:
        # box = box.to(torch.device("cpu"))
        if score < thresh:
            break
        good = True
        for other in chosen_boxes:
            cost = overlap(box, other)
            if cost > overlap_thres:
                good = False
                break
        if good:
            chosen_boxes.append(box)
            chosen_scores.append(score)
    return chosen_boxes, chosen_scores

def final_loss(predicted, expected):
    """
    Calculates the loss after selecting the best assignment of prediction to ground truth bounding boxes.
    The loss is sum (1-IoU(ground truth box, predicted box)). If there's a mismatched box either in the ground truth 
    or in the prediction, that amounts to a loss of 1.

    :param predicted: the prediction of the model
    :param expected: the ground truth bounding boxes
    :return: the loss, as described above
    """
    mat = np.array([[1 - intersect_over_union(a, b) for b in expected] for a in predicted])
    if len(mat) == 0:
        return abs(len(predicted) - len(expected))
    row_ind, col_ind = linear_sum_assignment(mat, maximize=False)
    return mat[row_ind, col_ind].sum() + abs(len(predicted) - len(expected))

def plot_threshold(raw_results:List[Dict], expected:List[List[Tuple[float]]], step:int=2):
    """
    Plots the loss over the score threshold. Takes the result and ground truth for each entry in the dataset as lists

    :param raw_results: `raw_results[i]` = {"boxes":list of boxes detected, "scores":list of predicted scores of the boxes} 
    relative to dataset[i]
    :param expected: `expected[i]` = ground truth bounding boxes for train_dataset[i]
    :param step: The `step` parameter in the `plot_threshold` function determines the percentage at which
    the threshold values are incremented while iterating from 1 to 100.
    defaults to 2
    """
    xs = np.arange(1, 101, step=step)
    ys = []
    for thresh in tqdm(range(1, 101, step)):
        thresh /= 100.0
        sum = 0
        for result, exp in zip(raw_results, expected):
            filtered, _ = filter_result_dict(result, thresh=thresh)
            sum += final_loss(filtered, exp)
        ys.append(sum / len(expected))
        print(ys[-1])
    
    plt.plot(xs, ys)
    plt.show()

"""
Usage example:
the_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    results = []
    expected = []
    for i, x in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        results.append(the_model([x[0].to(device)])[0]) # the prediction based on the ith document
        expected.append(x[1]['boxes']) # ground truth boxes of the ith document
plot_threshold(results, expected, 1)
"""