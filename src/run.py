import sys
import data_extration as de
import train as model

if len(sys.argv) < 4:
    print("Usage: python3 run.py <positive training data> <negative training data> <positive testing data> <negative testing data>")
    exit(1)

TR_pos = sys.argv[1]
TR_neg = sys.argv[2]
TS_pos = sys.argv[3]
TS_neg = sys.argv[4]

de.run(TR_pos, TR_neg, TS_pos, TS_neg)
model.run()