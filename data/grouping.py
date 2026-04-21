from collections import Counter

def get_group(qid_list):
    return list(Counter(qid_list).values())