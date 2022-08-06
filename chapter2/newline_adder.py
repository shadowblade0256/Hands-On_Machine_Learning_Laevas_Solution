line = []
with open("src.txt",mode="r") as f:
    line = f.readlines()

with open("result.txt",mode="w") as f2:
    for l in line:
        f2.write("<p>"+l.replace(" ","&nbsp; ").rstrip()+"</p>"+"\n")