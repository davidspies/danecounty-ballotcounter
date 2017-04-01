import collections
import numpy as np
import scipy.misc as misc
import scipy.ndimage as im
import zipfile
import sys

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
                 tiltallowance = 8, # The left/top alignment boxes can vary by this many pixels
                 emptybubblethreshold = 1200, # Less than this counts as no tick
                 filledbubblethreshold = 1800): # More than this counts as a tick
        self.unclear = [] # List of ballots which had difficulty being read
        self.votesFor = collections.defaultdict(list) # Normal ballots
        self.blank = [] # No bubble filled in
        self.multiple = [] # Multiple bubbles filled in
        self.scalefactor = scalefactor
        self.ignoreleftmargin = ignoreleftmargin
        self.ignoretopmargin = ignoretopmargin
        self.tiltallowance = tiltallowance
        self.emptybubblethreshold = emptybubblethreshold
        self.filledbubblethreshold = filledbubblethreshold

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
        isvote = []
        # If we have the wrong number of alignment boxes, ignore this ballot
        if(len(vertboxes) != 38 or len(horizboxes) != 4):
            self.unclear.append(ballotname)
            return
        for (name, (r, c)) in candidates:
            (posr, posc) = expand(add(vertboxes[r],
                                      sub(horizboxes[c], horizboxes[0])))
            # How dark is this bubble?
            value = np.sum(255 - arr[posr, posc])
            if value >= self.filledbubblethreshold:
                isvote.append(name)
            elif value >= self.emptybubblethreshold:
                self.unclear.append(ballotname)
                break
        else:
            if len(isvote) == 0:
                self.blank.append(ballotname)
            elif len(isvote) == 1:
                [k] = isvote
                self.votesFor[k].append(ballotname)
            else:
                self.multiple.append(ballotname)
    def __str__(self):
        return "fuzzy: %d\nmultiple: %d\nblank: %d\n%s" % (
            len(self.unclear),
            len(self.multiple),
            len(self.blank),
            showcvotes(self.votesFor)
        )

def main(zfname):
    ctr = Counter()
    if zfname.endswith(".zip"):
        with zipfile.ZipFile(zfname, 'r') as f:
            allnames = f.namelist()
            for filename in allnames:
                if filename.endswith("F.pbm"):
                    print(filename, file=sys.stderr)
                    imarr = misc.imread(f.open(filename))
                    ctr.count(filename, imarr)
    elif zfname.endswith(".pbm"):
        ctr.count(zfname, misc.imread(zfname))
    print(ctr)

if __name__ == '__main__':
    main(sys.argv[1])
