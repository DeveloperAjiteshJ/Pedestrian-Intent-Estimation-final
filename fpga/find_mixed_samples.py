import numpy as np
from pathlib import Path

# Load features and labels
features = np.load('fpga_test_vectors/test_features_set02_set05.npy')
labels = np.load('fpga_test_vectors/test_labels_set02_set05.npy')

# Simplified tree classifier logic
def tree_classify(se, so, q0, q1, q2, q3, hi, lo):
    if lo <= 12374:
        if hi <= 23735:
            if lo <= 9186:
                return 1
            else:
                if lo <= 9320:
                    return 0
                else:
                    if hi <= 20038:
                        if lo <= 9816:
                            if so <= 2688871:
                                return 0
                            else:
                                return 1
                        else:
                            return 1
                    else:
                        if q0 <= 1450671:
                            return 0
                        else:
                            return 1
        else:
            if so <= 3060880:
                return 0
            else:
                return 1
    else:
        if lo <= 13331:
            if hi <= 18551:
                return 0
            else:
                return 1
        else:
            if lo <= 19477:
                if q1 <= 1239714:
                    return 0
                else:
                    if hi <= 16221:
                        return 1
                    else:
                        if hi <= 17000:
                            return 0
                        else:
                            if lo <= 18188:
                                return 1
                            else:
                                return 0
            else:
                if q0 <= 1198942:
                    if hi <= 10202:
                        if se <= 2141931:
                            if q1 <= 1053654:
                                return 1
                            else:
                                return 0
                        else:
                            return 1
                    else:
                        if q1 <= 1157503:
                            return 0
                        else:
                            if se <= 2407553:
                                return 1
                            else:
                                return 0
                else:
                    return 1

# Find misclassified samples
correct_indices = []
wrong_indices = []

for i in range(len(labels)):
    se, so, q0, q1, q2, q3, hi, lo = features[i]
    pred = tree_classify(int(se), int(so), int(q0), int(q1), int(q2), int(q3), int(hi), int(lo))
    label = labels[i]
    
    if pred == label:
        correct_indices.append(i)
    else:
        wrong_indices.append(i)

print(f"Total samples: {len(labels)}")
print(f"Correct: {len(correct_indices)}")
print(f"Wrong: {len(wrong_indices)}")
print(f"\nWrong sample indices: {wrong_indices}")
print(f"\nFirst 10 correct indices: {correct_indices[:10]}")
print(f"\nLet's pick 6 samples: 3 correct + 3 wrong")

# Pick 3 correct and 3 wrong (if available)
if len(wrong_indices) >= 3:
    selected_wrong = wrong_indices[:3]
else:
    selected_wrong = wrong_indices

selected_correct = correct_indices[:max(6-len(selected_wrong), 3)]

selected = sorted(selected_wrong + selected_correct)
print(f"\nSelected 6 samples: {selected}")
print(f"  Correct: {[i for i in selected if i in correct_indices]}")
print(f"  Wrong: {[i for i in selected if i in wrong_indices]}")

for idx in selected:
    print(f"Sample {idx}: Label={labels[idx]}, Pred={tree_classify(int(features[idx][0]), int(features[idx][1]), int(features[idx][2]), int(features[idx][3]), int(features[idx][4]), int(features[idx][5]), int(features[idx][6]), int(features[idx][7]))}")
