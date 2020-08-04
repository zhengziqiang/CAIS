import os
import glob
import shutil
img_path=""
cnt=0
for files in glob.glob(img_path+"/*.*"):
    p,n=os.path.split(files)
    if cnt<7000:
        shutil.copyfile(files,"datasets/flower/train_cover/"+n)
        shutil.copyfile(files, "datasets/flower/train_message/" + n)
    else:
        shutil.copyfile(files, "datasets/flower/test_cover/" + n)
        shutil.copyfile(files, "datasets/flower/test_message/" + n)
    cnt+=1