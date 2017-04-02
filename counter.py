import collections
import multiprocessing as mp
import numpy as np
import os
from os import path
import pandas as pd
from scipy import misc
import scipy.ndimage as im
import sys
import zipfile

candidates = [
    ("trump", (19, 1)),
    ("clinton", (21, 1)),
    ("castle", (23, 1)),
    ("johnson", (25, 1)),
    ("stein", (27, 1)),
    ("moorehead", (29, 1)),
    ("roque", (31, 1)),
    ("write-in", (33, 1))
]

def add(box, shift):
    (sr, sc) = box
    (rd, cd) = shift
    return (slice(sr.start+rd, sr.stop+rd), slice(sc.start+cd, sc.stop+cd))

def sub(lbox, rbox):
    (slr, slc) = lbox
    (srr, src) = rbox
    return (slr.start - srr.start, slc.start - src.start)

def expand(box):
    (sr, sc) = box
    return (slice(sr.start - 1, sr.stop + 1), slice(sc.start - 1, sc.stop + 1))

def showcvotes(cvotes):
    return '\n'.join("%s: %d" % (k, len(v))
                     for (k, v) in sorted(cvotes.items()))

class Counter(object):
    def __init__(self,
                 scalefactor = 0.125, # Scale ballot images
                 ignoreleftmargin = 5, # Ignore any items with pixels in this margin
                 ignoretopmargin = 5, # ""
                 tiltallowance = 8): # The left/top alignment boxes can vary by this many pixels
        self.scalefactor = scalefactor
        self.ignoreleftmargin = ignoreleftmargin
        self.ignoretopmargin = ignoretopmargin
        self.tiltallowance = tiltallowance
        self.bubbleDarkness = []

    def count(self, ballotname, imarr):
        arr = misc.imresize(imarr, self.scalefactor)
        (nrows, ncols) = arr.shape
        # After rescaling, look for blocks which are totally black
        blocksOnly = arr == 0
        (labeled, lcount) = im.label(blocksOnly)
        boxes = im.find_objects(labeled)
        minrow = min(
            b[0].start for b in boxes if b[0].start > self.ignoretopmargin)
        mincol = min(
            b[1].start for b in boxes if b[1].start > self.ignoreleftmargin)
        # There should be 4 alignment boxes along the top
        horizboxes = [b for b in boxes
                      if b[0].start - minrow < self.tiltallowance
                        and b[0].start > self.ignoretopmargin
                     ]
        # And 38 along the side
        vertboxes = [b for b in boxes
                     if b[1].start - mincol < self.tiltallowance
                        and b[1].start > self.ignoreleftmargin
                    ]
        horizboxes.sort(key=lambda b: b[1].start)
        vertboxes.sort(key=lambda b: b[0].start)
        bubbles = {"ballot":ballotname}
        # If we have the wrong number of alignment boxes, ignore this ballot
        if(len(vertboxes) != 38 or len(horizboxes) != 4):
            bubbles["badBoxes"] = True
        else:
            bubbles["badBoxes"] = False
            for (name, (r, c)) in candidates:
                (posr, posc) = expand(add(vertboxes[r],
                                          sub(horizboxes[c], horizboxes[0])))
                # How dark is this bubble?
                value = np.sum(255 - arr[posr, posc])
                bubbles[name] = value
        self.bubbleDarkness.append(bubbles)
    def toDF(self):
        return pd.DataFrame(self.bubbleDarkness)

def runctr(zfname):
    print("Starting " + zfname, file=sys.stderr)
    ctr = Counter()
    (unused_base, ext) = path.splitext(zfname)
    if ext == ".zip":
        with zipfile.ZipFile(zfname, 'r') as f:
            allnames = f.namelist()
            for filename in allnames:
                if filename.endswith("F.pbm"):
                    imarr = misc.imread(f.open(filename))
                    ctr.count(filename, imarr)
    elif ext == ".pbm":
        ctr.count(zfname, misc.imread(zfname))
    else:
        raise Exception("Unknown extension: %s" % ext)
    print("Finished " + zfname, file=sys.stderr)
    return (zfname, ctr.toDF())

def main(readdir, store_file, nprocs):
    files = os.listdir(readdir)
    fullfiles = [path.join(readdir, f) for f in files]
    with pd.HDFStore(store_file) as store:
        incomplete = [f for f in fullfiles if f not in store]
    with mp.Pool(nprocs) as pool:
        for (zfname, df) in pool.imap_unordered(runctr, incomplete):
            df.to_hdf(store_file, zfname)
    pd.Series(fullfiles).to_hdf(store_file, "tables")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
