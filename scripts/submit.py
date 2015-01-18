import os
import time
import hashlib
from subprocess import check_call

bp = os.path.join(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(bp, "job.pbs"), "r") as f:
    txt = f.read()

idhash = str(time.time())
txt = txt.format(idhash=idhash)

fn = os.path.join(bp, "jobs", "{0}.pbs".format(idhash))
with open(fn, "w") as f:
    f.write(txt)

check_call(["qsub", fn])
