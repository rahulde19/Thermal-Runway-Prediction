import os, cv2, numpy as np
import shutil

ROOT = r"C:\Users\Harshit\Desktop\24MEM1R12 ML\24MEM1R12 ML\Anim"
SAVE = r"C:\Users\Harshit\Desktop\Battery_DL_Dataset"

if os.path.exists(SAVE): shutil.rmtree(SAVE)
os.makedirs(SAVE)

# ALIGNED WITH PAPER: 8 frames sequence, 96x96 resolution
SEQ_LEN, IMG_SIZE = 8, 96 
MAX_FRAMES = 704 # Paper uses 704s limit for early detection

def natural_sort(files):
    return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))

cases = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]

all_X, all_y, all_case_ids = [], [], []

for case_idx, case in enumerate(cases):
    case_path = os.path.join(ROOT, case)
    # Filter and sort, then limit to first 704 frames per paper
    frames = natural_sort([f for f in os.listdir(case_path) if f.lower().endswith('.png')])[:MAX_FRAMES]
    
    # Labeling: 0-352 (Safe), 353-704 (Risk)
    if case.lower() == "normal":
        labels = [0] * len(frames)
    else:
        labels = [0] * min(352, len(frames)) + [1] * max(0, len(frames) - 352)
    
    print(f"Processing {case}: {len(frames)} frames")
    
    for i in range(SEQ_LEN-1, len(frames)):
        seq_imgs = []
        for j in range(i-SEQ_LEN+1, i+1):
            img = cv2.imread(os.path.join(case_path, frames[j]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            seq_imgs.append(img.astype(np.float32))
        
        all_X.append(seq_imgs)
        all_y.append(labels[i])
        all_case_ids.append(case_idx)

X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.int32)
case_ids = np.array(all_case_ids)

# Split by Case (70/15/15)
u_cases = np.unique(case_ids)
np.random.shuffle(u_cases)
train_end = int(0.7 * len(u_cases))
val_end = int(0.85 * len(u_cases))

train_cases = u_cases[:train_end]
val_cases = u_cases[train_end:val_end]
test_cases = u_cases[val_end:]

for split, ids in [("train", train_cases), ("val", val_cases), ("test", test_cases)]:
    mask = np.isin(case_ids, ids)
    np.save(os.path.join(SAVE, f"X_{split}.npy"), X[mask])
    np.save(os.path.join(SAVE, f"y_{split}.npy"), y[mask])
    print(f"{split.capitalize()} set ready. Cases: {len(ids)}")
