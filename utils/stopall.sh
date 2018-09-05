#!/bin/bash
for((i=0;i<${1};i++));
do
	docker stop node${i};
done
