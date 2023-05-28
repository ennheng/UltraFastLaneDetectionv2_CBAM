# !/usr/bin/python
import os


def ListFilesToTxt(dir, file, wildcard, recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname = os.path.join(dir, name)
        if (os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname, file, wildcard, recursion)
        else:
            for ext in exts:
                if (name.endswith(ext)):
                    file.write("/media/ennheng/software/culane-cust/" + name + "\n")  # 存放合集（原图和json等）文件夹的名字
                    break


def Test():
    dir = "/media/ennheng/software/customdata/custom/pic"  # 存放原图文件夹(只存放原图)路径
    outfile = "trainlist.txt"  # 写入的txt文件名
    wildcard = ".jpg"  # 要读取的文件类型；
    file = open(outfile, "w")
    if not file:
        print("cannot open the file %s for writing" % outfile)
    ListFilesToTxt(dir, file, wildcard, 1)
    file.close()


Test()