import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.load_istella import load_istella
from data.grouping import get_group

X, y, qid = load_istella("G:\search-ranking-model\data\raw\train.txt", limit=10000)
group = get_group(qid)

print("Samples:", len(X))
print("Labels:", len(y))
print("Queries:", len(group))
print("Features per sample:", len(X[0]))