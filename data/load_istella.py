def load_istella(file_path, limit=None):
    X, y, qid = [], [], []

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            parts = line.strip().split()

            y.append(int(parts[0]))
            qid.append(int(parts[1].split(':')[1]))

            features = [float(x.split(':')[1]) for x in parts[2:]]
            X.append(features)

    return X, y, qid