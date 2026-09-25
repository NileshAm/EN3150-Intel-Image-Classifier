import numpy as np
import os
import cv2
import pandas as pd

def loadData(setType:str, labels=None, dataframe=False):
    if setType not in ["train", "test"]:
        raise ValueError("Type must be either 'train' or 'test'")
    data = []
    path = os.path.join(os.curdir, "dataset", f"seg_{setType}")
    if labels is None:
        labels = np.array(os.listdir(path))

    for label in labels:
        labelPath = os.path.join(path, label)
        imgs = os.listdir(labelPath)
        for img in imgs:
            img = cv2.imread(os.path.join(labelPath, img))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (64,64))
            data.append({"img":img, "label":int(np.where(labels == label)[0][0])})

    data = pd.DataFrame(data)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    if dataframe:
        return data, np.array(labels)
    
    x = np.stack(data["img"].to_numpy()).astype(np.float32)
    y = data["label"].to_numpy(dtype=np.int32)

    return x, y, np.array(labels)

def reseed_grid_picker(tuner):
    """Restore GridSearch's missing in-memory state after reload."""
    oracle = tuner.oracle

    # A new search or a future fixed version may already have this state.
    if oracle._ordered_ids._memory:
        return

    previous_id = None
    for trial_id in oracle.start_order:
        oracle._ordered_ids.insert(trial_id, previous_id)
        previous_id = trial_id

    if oracle.end_order:
        oracle._populate_next.append(oracle.end_order[-1])