# MDPA

## Introduction

This repository is the simulator part of MDPA, which trains Deep Reinforcement Learning based cluster scheduling algorithms by massively parallel methods. It's Spark simulator is a minor modified version of [Park](https://github.com/park-project/park).

Although MDPA is designed for cluster scheduling algorithms, but its capacity of parallel training is universal. As long as users plug a environment capatiable with `Gym`, users can fully utilize MDPA's capacity of parallel training.

## Requirements

`python==3.7, tensorflow==1.15, numpy==1.18, scipy==1.5, networkx==2.5, pyarrow==3.0, tensorboardX, crc32c, pyyaml, pyzmq, matplotlib`

Other versions of dependecies might also work, but we didn't test them.

## Examples

MDPA works in a Parameter Server manner. Users need to specify the ip address of master in a `.yaml` config file. Make sure the two ports in config file `pub_sub_port` and `push_pull_port` as not used. Other paramsters such as the number of actors, the number executors, the job arrival interval (Poisson Process) of straming jobs, learning rate, and so on are all in `params.py`.

Start worker
`python start_worker.py -f config.yaml`

Start master
`python start_master.py -f config.yaml`


The average job completion time through the training process should look like below:
![screenshot](/results/screenshot.png)
