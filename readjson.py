import json
import sys

with open(sys.argv[1], encoding="utf8") as jsonfh:
    everything = jsonfh.read()


dec = json.JSONDecoder()
pos = 0
while True:
    try:
        structure, pos = dec.raw_decode(everything, pos)
        print(structure)
        pos += 1
    except json.JSONDecodeError:
        print("eof")
        break
