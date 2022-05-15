#!/usr/bin/env python3

import glob

matches = []
for f in glob.iglob("checkpoints/checkpoint-svr-*"):
    parts = f.split("-")
    c = float(parts[2])
    coef0 = float(parts[4])
    epsilon = float(parts[6])
    gamma = parts[7]
    kernel = parts[8]
    match = (kernel, gamma, coef0, c, epsilon)
    matches.append(match)

with open("svr-job-file.txt") as fo:
    with open("svr-job-file.txt-new", "w") as fn:
        for line in fo:
            parts = line.split()
            kernel = parts[4]
            gamma = parts[6]
            coef0 = float(parts[8])
            c = float(parts[10])
            epsilon = float(parts[12])
            match = (kernel, gamma, coef0, c, epsilon)
            if match not in matches:
                fn.write(line)
