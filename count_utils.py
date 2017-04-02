import pandas as pd

from candidates import candidates

cnames = [c[0] for c in candidates]

def all_ballots(store_file):
    with pd.HDFStore(store_file) as store:
        return pd.concat([store[t] for t in store.tables], ignore_index=True)

def assign_votes(votetable, lothreshold, hithreshold):
    darknessCols = []
    bubbles = votetable.loc[:, cnames]
    result = pd.Series(index=bubbles.index)
    boxfail = votetable.loc[:, "badBoxes"]
    result[boxfail] = "badBoxes"
    remaining = ~boxfail
    fuzzy = ((bubbles >= lothreshold) & (bubbles < hithreshold)).any(axis=1)
    result[remaining & fuzzy] = "fuzzy"
    remaining = remaining & ~fuzzy
    ismark = bubbles > hithreshold
    bubblecounts = ismark.sum(axis=1)
    result[remaining & (bubblecounts > 1)] = "multiple"
    result[remaining & (bubblecounts == 0)] = "blank"
    remaining = remaining & (bubblecounts == 1)
    for c in cnames:
        result[remaining & ismark.loc[:, c]] = c
    return result
