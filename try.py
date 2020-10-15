
import glob

path = 'data/custom/train.txt'
a = sorted(glob.glob("%s/*.*" % path))
print(a)