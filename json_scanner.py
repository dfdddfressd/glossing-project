import json
import re
import sys

class JSONScanner:
    def __init__(self, string):
        self.string = string
        self.pos = 0
        self.dec = json.JSONDecoder()

    def scan(self):
        jsonLabels = ["```json",]

        #first, check if the lm has labeled the json for us
        for label in jsonLabels:
            if label in self.string[self.pos:]:
                self.pos += self.string[self.pos:].index(label) + len(label)
                self.consumeWhitespace()

        try:
            res = self.readJS()
            return res
        except json.JSONDecodeError:
            pass
        
        #just wildly search for something useful
        jsonLabels = ["[", "{"]
        posns = []
        
        for label in jsonLabels:
            if label in self.string[self.pos:]:
                pos = self.string[self.pos:].index(label)
                posns.append(pos)

        if posns:
            best = min(posns)
            self.pos += best
                
        try:
            res = self.readJS()
            return res
        except json.JSONDecodeError:
            raise StopIteration

    def readJS(self):
        (res, pos) = self.dec.raw_decode(self.string, self.pos)
        self.pos = pos
        return res

    def consumeWhitespace(self):
        while self.pos < len(self.string) and self.string[self.pos].isspace():
            self.pos += 1

if __name__ == "__main__":
    jsfile = sys.argv[1]
    with open(jsfile, "r", encoding="utf-8") as fh:
        everything = fh.read()

    scan = JSONScanner(everything)
    while True:
        try:
            print(scan.scan())
        except StopIteration:
            break
