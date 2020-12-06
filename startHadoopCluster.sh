#!/bin/bash

# VARIABLES
IMG_NAME="spark-hadoop-cluster"
HOST_PREFIX="cluster"
NETWORK_NAME=$HOST_PREFIX

N=${1:-2}
NET_QUERY=$(docker network ls | grep -i $NETWORK_NAME)
if [ -z "$NET_QUERY" ]; then
	docker network create --driver=bridge $NETWORK_NAME
fi

# START HADOOP SLAVES 
i=1
while [ $i -le $N ]
do
	HADOOP_SLAVE="$HOST_PREFIX"-slave-$i
	docker run --name $HADOOP_SLAVE -h $HADOOP_SLAVE --net=$NETWORK_NAME -itd "$IMG_NAME"
	i=$(( $i + 1 ))
done

# START HADOOP MASTER

HADOOP_MASTER="$HOST_PREFIX"-master
docker run --gpus all --name $HADOOP_MASTER -h $HADOOP_MASTER --net=$NETWORK_NAME \
                -v /media/gatech/cse6250/hw5/homework5:/mnt/host \
                -v /media/data/gatech/COVID-19:/mnt/data \
		-p  8088:8088  -p 50070:50070 -p 50090:50090 \
		-p  8080:8080  -p 9998:8888 \
		-itd "$IMG_NAME"

docker cp -a config/hadoop-env.sh /opt/hadoop-3.3.0/etc/hadoop/hadoop-env.sh
# START MULTI-NODES CLUSTER
#docker exec -it $HADOOP_MASTER "/opt/hadoop-3.3.0/spark-services.sh"
docker exec -it $HADOOP_MASTER "/opt/hadoop-3.3.0/spark-services.sh"






