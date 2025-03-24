#!/usr/bin/env python3

import os

_, dirlist, _ = next(os.walk("./"))

dirlist = [dname for dname in dirlist if dname.startswith("mp-")]

dirlist.sort()

with open("dirlist.py", "w") as f:
    f.write("dirlist = [\n")
    for dname in dirlist:
        f.write(f'"{dname}",\n')
    f.write("]\n")
