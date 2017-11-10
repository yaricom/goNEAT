[![Build Status](https://travis-ci.org/yaricom/goNEAT.svg?branch=master)](https://travis-ci.org/yaricom/goNEAT) [![GoDoc](https://godoc.org/github.com/yaricom/goNEAT/neat?status.svg)](https://godoc.org/github.com/yaricom/goNEAT/neat)

## Overview
This repository provides implementation of [NeuroEvolution of Augmenting Topologies (NEAT)][1] method written in Go language.

The Neuroevolution (NE) is an artificial evolution of Neural Networks (NN) using genetic algorithms in order to find
optimal NN parameters and topology. Neuroevolution of NN may assume search for optimal weights of connections between
NN nodes as well as search for optimal topology of resulting NN. The NEAT method implemented in this work do search for
both: optimal connections weights and topology for given task (number of NN nodes per layer and their interconnections).

#### System Requirements
The source code written and compiled against GO 1.8.x.

## Installation
Make sure that you have at least GO 1.8.x. environment installed onto your system and execute following command:
```bash

go get github.com/yaricom/goNEAT
```

## Performance Evaluations
The basic system's performance is evaluated by two kind of experiments:
1. The XOR experiment which test whether topology augmenting actually happens by NEAT algorithm evaluation. To build XOR
solving network the NEAT algorithm should grow new hidden unit in the provided start genome.
2. The pool balancing experiment which is classic Reinforcement Learning experiment allowing us to estimate performance
of NEAT algorithm against proven results by many of other algorithms. I.e. we can benchmark NEAT performance against
other algorithms and find out if it performs better or worse.


### 1. The XOR Experiments
Because XOR is not linearly separable, a neural network requires hidden units to solve it. The two inputs must be
combined at some hidden unit, as opposed to only at the output node, because there is no function over a linear
combination of the inputs that can separate the inputs into the proper classes. These structural requirements make XOR
suitable for testing NEAT’s ability to evolve structure.

#### >>> The XOR Experiment with connected inputs in start genome
In this experiment we will use start (seed) genome with inputs connected to the output. Thus it will check mostly the
ability of NEAT to grow new hidden unit necessary for solving XOR problem.

To run this experiment execute following commands:
```bash

cd $GOPATH/src/github.com/yaricom/goNEAT
go run xor_runner.go -out ./out -context ./data/xor.neat -genome ./data/xorstartgenes

```
Where: ./data/xor.neat is the configuration of NEAT execution context and ./data/xorstartgenes is the start genome
configuration.

As result of execution into the ./out directory will be stored several 'gen_x' files with snapshots of population per 'print_every'
epoch or when winner solution found. Also in mentioned directory will be stored 'xor_winner' with winner genome and
'xor_optimal' with optimal XOR solution if any (has exactly 5 units).

By examining resulting 'xor_winner' from series of experiments you will find that at least one hidden unit was grown by NEAT
to solve XOR problem which is proof that it works as expected.

The XOR experiment for start genes with inputs connected will not fail almost always (at least 100 simulations)

The experiment results will be similar to the following:

```
Average
	Winner Nodes:	5.0
	Winner Genes:	6.0
	Winner Evals:	7753.0
Mean
	Complexity:	10.6
	Diversity:	19.8
	Age:		34.6
```

Where:
- **Winner nodes/genes** is number of units and links between in produced Neural Network which was able to solve XOR problem.
- **Winner evals** is the number of evaluations of intermediate organisms/genomes before winner was found.
- **Mean Complexity** is an average compexity (number of nodes + number of links) of best organisms per epoch for all epochs.
- **Mean Diversity** is an average diversity (number of species) per epoch for all epochs
- **Mean Age** is an average age of surviving species per epoch for all epochs

#### >>> The XOR experiment with disconnected inputs in start genome
This experiment will use start genome with disconnected inputs in order to check ability of algorithm to not only grow
need hidden nodes, but also to build missed connections between input nodes and rest of the network.

To run this experiment execute following commands:
```bash

cd $GOPATH/src/github.com/yaricom/goNEAT
go run xor_runner.go -out ./out -context ./data/xor.neat -genome ./data/xordisconnectedstartgenes

```

The results of experiment execution will be saved into the ./out directory as in previous experiment.

The experiment will fail sometimes to produce XOR solution over 100 epochs, but most of times solution will be found. This
confirms that algorithm is able not only grow needed hidden units, but also to restore input connections as needed.

The example output of the command as following:
```

Average
	Winner Nodes:	7.0
	Winner Genes:	14.0
	Winner Evals:	6967.0
Mean
	Complexity:	9.2
	Diversity:	11.4
	Age:		29.8

```

## Authors

This source code maintained and managed by Iaroslav Omelianenko

[1]:http://www.cs.ucf.edu/~kstanley/neat.html
