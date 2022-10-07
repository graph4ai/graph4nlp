import sys
import chardet

fin = open(sys.argv[1], "rb")
fout = open(sys.argv[2], "w")

for line in fin:
    try:
        line = line.decode("utf-8")
    except Exception:
        line = line.decode(chardet.detect(line)["encoding"])

    data = line.strip().split()
    text = " ".join(data[1:])
    label = data[0].split(":")[0]
    fout.write("{}\t{}\n".format(text, label))
