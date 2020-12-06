#!/bin/bash

$HADOOP_HOME/sbin/start-dfs.sh
$HADOOP_HOME/sbin/start-yarn.sh

jps -lm

hdfs dfsadmin -report

hdfs dfs -mkdir -p /apps/spark
zip $SPARK_HOME/spark/jars/spark-jars.zip $SPARK_HOME/spark/jars/*
hadoop fs -put $SPARK_HOME/spark/jars/spark-jars.zip  /apps/spark

$SPARK_HOME/sbin/start-all.sh
scala -version
jps -lm



