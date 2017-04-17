import collections
import pandas as pd
import re

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

def official_groups(official_csv):
    votes = pd.read_csv(official_csv).sort_values(by="Precinct").reset_index(drop=True)
    keys = votes.loc[:, "Precinct"]
    bykey = collections.defaultdict(list)
    for (i,key) in enumerate(keys):
        sndword = key.split(' ')[1]
        bykey[sndword].append(votes.loc[i, :])
    return {k : pd.DataFrame(v).set_index('Precinct').fillna(0)
            for (k, v) in bykey.items()}

def audit_groups(store_file, lothreshold, hithreshold):
    with pd.HDFStore(store_file) as store:
        bykey = collections.defaultdict(list)
        for t in store.tables:
            key = t[len("wards/"):-len(".zip")]
            sndword = key.split(" ")[0]
            vs = assign_votes(store[t], lothreshold, hithreshold).value_counts()
            vs["Precinct"] = key
            bykey[sndword].append(vs)
    return {k : pd.DataFrame(v).set_index('Precinct').fillna(0)
            for (k, v) in bykey.items()}

def parse_madison(name):
    if name.startswith("Madison City Wards"):
        rest = name[len("Madison City Wards"):]
        ttype = "City"
    elif name.startswith("Madison Town Wards"):
        rest = name[len("Madison Town Wards"):]
        ttype = "Town"
    else:
        raise Exception("Unknown name: " + name)
    numstrs = re.split(',|&', rest)
    nums = []
    for s in numstrs:
        ps = s.split('-')
        if len(ps) == 1:
            [x] = ps
            nums.append(int(x.strip()))
        elif len(ps) == 2:
            [x,y] = ps
            for i in range(int(x.strip()),int(y.strip())+1):
                nums.append(i)
        else:
            raise Exception("Unknown nums: " + name)
    return (nums, ttype)

def madison_city_groups(official_madison, counted_madison):
    sumofficial = []
    inds = []
    nums_unused = set(range(1,139))
    for name in counted_madison.index:
        (nums, ttype) = parse_madison(name)
        if ttype == "Town":
            continue
        assert ttype == "City"
        inds.append(name)
        precincts = []
        for num in nums:
            if num not in nums_unused:
                raise Exception(num + " already used")
            nums_unused.remove(num)
            precincts.append("C Madison Wd " + str(num))
        sumofficial.append(official_madison.loc[precincts, :].sum(axis=0))
    return (pd.DataFrame(sumofficial, index=inds), nums_unused)
