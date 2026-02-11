import sys, base64
data = sys.stdin.read()
target = r"c:/Users/skazempour/Dropbox/Projects/42 - Machine learning from the crowd/Code/02 - prepare training dataset/check_merge.py"
with open(target, "w", encoding="utf-8") as out:
    out.write(base64.b64decode(data).decode("utf-8"))
print("Decoded and written to", target)
