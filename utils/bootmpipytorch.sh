#!/bin/bash

add=172.19.61

#create network
docker network create --subnet=${add}.0/24 mpi

rm -rf ${2}/hosts

#start up nodes
for((i=0;i<${1};i++));
do
    echo ${add}'.'$(expr $i + 2)' node'$i >> ${2}/hosts
	NV_GPU=$(expr $i) nvidia-docker run -itd --name=node${i} --hostname=node${i} --network=mpi --ip ${add}.$(expr $i + 2) -v ${2}:/volume --ipc=host --user mpi shiyh/pytorchmpi:8.0-cudnn7-devel-ubuntu16.04;
done

#modify /etc/hosts
for((i=0;i<${1};i++));
do
	docker exec node${i} sh -c "sudo sh -c 'cat /volume/hosts >> /etc/hosts'";
done

#create ssh key
for((i=0;i<${1};i++));
do
	docker exec node${i} sh -c "sudo service ssh restart && ssh-keygen -t rsa -P '' -f /home/mpi/.ssh/id_rsa";
done

#ssh key-free
for((i=0;i<${1};i++));
do
	#copy to shared directory
    docker exec node${i} sh -c "sudo chown -R mpi:mpi /volume/mpi && cp /home/mpi/.ssh/id_rsa.pub /volume/mpi";
	#all nodes add ssh key and add all nodes to the known hosts
	for((j=0;j<${1};j++));
	do
		docker exec node${j} sh -c "cat /volume/mpi/id_rsa.pub >> /home/mpi/.ssh/authorized_keys";
		docker exec node${i} sh -c "ssh-keyscan -H node${j} >> /home/mpi/.ssh/known_hosts";
	done
done
