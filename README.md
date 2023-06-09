# cse6250team47
CSE6250 Team47 Covid-19 detection/classifications of x-rays using bigdata and CNNs
<pre>

COVID-19: Automatic radiographic classification by using convolutional neural network variants.  
Nilesh Domadia , Arnab K Giri, Sankaranarayanan Ramamoorthy and  Srimathy Thyagarajan. 
Georgia Institute of Technology, Atlanta, Georgia, United States 
CSE 6250 Bigdata For Health Fall 2020

https://github.gatech.edu/sramamoorthy30/cse6250team47

docker Image inspired by Pierre Kieffer . https://github.com/PierreKieffer/docker-spark-yarn-cluster
</pre>


#installation

<pre>Download and Install docker
Download and Install Nvidia container runtime
Next download the Docker image using
docker pull sramamoorthy30/team47
Create the following directory
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
use the provided code/correctname.py program to remove spaces etc in filenames of the images.

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

When completed
exit the container
and stop the containers using the following command.
</pre>
```
make stop
```
### software and hardware stack
<pre>
pyspark/spark    : 3.0.1 
open JDK 64-bit  : 11.0.9.1
hadoop           : 3.3.0 
scala            : 2.11.12 
conda            : 3.9.1 
python           : 3.8.3 
gcc              : 7.3.0 
torch            : 1.7.0

jupyter core     : 4.7.0
jupyter-notebook : 6.1.5
qtconsole        : 4.7.7
ipython          : 7.18.1
ipykernel        : 5.3.4
jupyter client   : 6.1.7
nbconvert        : 5.6.1
ipywidgets       : 7.5.1
nbformat         : 5.0.8
traitlets        : 5.0.5
matplotlib

Docker           : 19.03.13, build 4484c46d9d
GPU 0: GeForce GTX 1080 Ti (UUID: GPU-1b19620b-0b77-71f3-14fb-bb0ebba1468f)
NVIDIA-SMI        : 455.38       
Driver            : 455.38       
CUDA              : 11.1

The docker image has a spark master-cluster container and two slave containers

</pre>
