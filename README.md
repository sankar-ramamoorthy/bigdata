# cse6250team47
CSE6250 Team47 Covid-19 detection/classifications of x-rays using bigdata and CNNs
https://github.gatech.edu/sramamoorthy30/cse6250team47


#installation

<pre>Download and Install docker
Download and Install Nvidia container runtime
docker pull sramamoorthy30/team47
</pre>

```
mkdir -p /media/gatech/cse6250/hw5/homework5
```
<pre>
upload the files the code directory from github into /media/gatech/cse6250/hw5/homework5
</pre>

##Data
<pre>
following the instructions from https://github.com/lindawangg/COVID-Net/blob/master/create_COVIDx.ipynb
Download data from
https://github.com/ieee8023/covid-chestxray-dataset
https://github.com/agchung/Figure1-COVID-chestxray-dataset
https://github.com/agchung/Actualmed-COVID-chestxray-dataset
https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

Create the directories
</pre>
```
mkdir -p /media/data/gatech/COVID-19
mkdir -p /media/data/gatech/COVID-19/petacache
mkdir -p /media/data/gatech/COVID-19/tmp

mkdir -p /media/data/gatech/COVID-19/X-RayImageDataSet/train/pneumonia  
mkdir -p /media/data/gatech/COVID-19/X-RayImageDataSet/train/nofindings
mkdir -p /media/data/gatech/COVID-19/X-RayImageDataSet/train/covid  

mkdir -p /media/data/gatech/COVID-19/X-RayImageDataSet/test/pneumonia  
mkdir -p /media/data/gatech/COVID-19/X-RayImageDataSet/test/nofindings
mkdir -p /media/data/gatech/COVID-19/X-RayImageDataSet/test/covid  
```
<pre>
upload images into the train and test directories.
we divided the images in a 80:20 split.

start docker and load the data into hdfs
</pre>
```
docker pull sramamoorthy30/team47
cd cse6250team47-master
make start
make connect

hdfs dfs -mkdir -p /mnt/data/X-RayImageDataSet/train/pneumonia
hdfs dfs -mkdir -p /mnt/data/X-RayImageDataSet/train/nofindings
hdfs dfs -mkdir -p /mnt/data/X-RayImageDataSet/train/covid
hdfs dfs -mkdir -p /mnt/data/X-RayImageDataSet/test/pneumonia
hdfs dfs -mkdir -p /mnt/data/X-RayImageDataSet/test/nofindings
hdfs dfs -mkdir -p /mnt/data/X-RayImageDataSet/test/covid

hdfs dfs -put  /mnt/data/X-RayImageDataSet/train/pneumonia/* /mnt/data/X-RayImageDataSet/train/pneumonia   
hdfs dfs -put /mnt/data/X-RayImageDataSet/train/nofindings/* /mnt/data/X-RayImageDataSet/train/nofindings
hdfs dfs -put /mnt/data/X-RayImageDataSet/train/covid/* /mnt/data/X-RayImageDataSet/train/covid
hdfs dfs -put /mnt/data/X-RayImageDataSet/test/pneumonia/* /mnt/data/X-RayImageDataSet/test/pneumonia
hdfs dfs -put /mnt/data/X-RayImageDataSet/test/nofindings/* /mnt/data/X-RayImageDataSet/test/nofindings
hdfs dfs -put /mnt/data/X-RayImageDataSet/test/covid/* /mnt/data/X-RayImageDataSet/test/covid
```

<pre>
start jupyter
</pre>

```
cd /mnt/host
/workspace/run_jupyter.sh
```
<pre>
use the url provided to login to jupyter
open the notebook ProjectPyAsANotebook
and run all steps.
</pre>
