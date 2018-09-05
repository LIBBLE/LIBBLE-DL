#!/bin/bash
for((i=0;i<${1};i++));
do
	docker start node${i};
    docker exec node${i} sh -c "sudo service ssh restart";
done
