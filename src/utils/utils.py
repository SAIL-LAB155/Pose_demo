

def process_kp(kps, idx):
    new_kp = []
    for bdp in range(len(kps)):
        for coord in range(2):
            new_kp.append(kps[bdp][coord])
    return {idx: new_kp}
