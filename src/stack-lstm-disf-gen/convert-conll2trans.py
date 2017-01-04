import sys
import os
action = []
buffer = []
for line in sys.stdin:
    if line.strip() != "":
        line = line.strip().split(" ")
        word = line[0] + "-" + line[1]
        for i in range(4,len(line)):
            word = word + "-" + line[i]
        buffer.append(word)
        if line[3] == "O":
            action.append("OUT")
        elif line[3] == "B" or line[3] == "I":
            action.append("SHIFT")
        elif line[3] == "E" or line[3] == "S":
            action.append("SHIFT")
            action.append("REDUCE")

    else:
        if len(action) != 0:
            print " ".join(buffer).strip() + " ||| " + " ".join(action).strip()
        action = []
        buffer = []


