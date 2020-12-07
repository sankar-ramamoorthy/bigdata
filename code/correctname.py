import re, os
from pathlib import Path
MyDataImageDir=Path(r'/mnt/data/X-RayImageDataSet/test')
for dirname in os.listdir(MyDataImageDir):
  for filename in  os.listdir(MyDataImageDir/dirname):
    split_file_name,extenstion = os.path.splitext(filename)
    new_file_name = re.sub("\.","_",split_file_name)
    new_file_name = re.sub("  ","_",split_file_name)
    new_file_name = re.sub(" ","_",new_file_name)
    new_file_name = re.sub("\(","_",new_file_name)
    new_file_name = re.sub("\)","_",new_file_name)
    new_file_name = new_file_name.replace("_","")
    if new_file_name != split_file_name:
      new_file_name = new_file_name + extenstion
      print(new_file_name)
      os.rename(MyDataImageDir/dirname/filename,MyDataImageDir/dirname/new_file_name)

MyDataImageDir=Path(r'/mnt/data/X-RayImageDataSet/train')
for dirname in os.listdir(MyDataImageDir):
  for filename in  os.listdir(MyDataImageDir/dirname):
    split_file_name,extenstion = os.path.splitext(filename)
    new_file_name = re.sub("\.","_",split_file_name)
    new_file_name = re.sub("  ","_",split_file_name)
    new_file_name = re.sub(" ","_",new_file_name)
    new_file_name = re.sub("\(","_",new_file_name)
    new_file_name = re.sub("\)","_",new_file_name)
    new_file_name = new_file_name.replace("_","")
    if new_file_name != split_file_name:
      new_file_name = new_file_name + extenstion
      print(new_file_name)
      os.rename(MyDataImageDir/dirname/filename,MyDataImageDir/dirname/new_file_name)

