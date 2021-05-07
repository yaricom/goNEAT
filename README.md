# goNEAT
[![banner](contents/book_title.png)](https://www.oreilly.com/library/view/hands-on-neuroevolution-with/9781838824914/)

[![version](https://img.shields.io/github/v/tag/yaricom/goNEAT.svg?sort=semver)](https://github.com/yaricom/goNEAT/releases/latest)
[![Build Status](https://travis-ci.org/yaricom/goNEAT.svg?branch=master)](https://travis-ci.org/yaricom/goNEAT) [![GoDoc](https://godoc.org/github.com/yaricom/goNEAT/neat?status.svg)](https://godoc.org/github.com/yaricom/goNEAT/neat)
[![Go version](https://img.shields.io/badge/go-1.15-blue.svg)](https://github.com/moovweb/gvm)
[![license](https://img.shields.io/github/license/yaricom/goNEAT.svg)](https://github.com/yaricom/goNEAT/blob/master/LICENSE)
[![yaricom/goNEAT](https://tokei.rs/b1/github/yaricom/goNEAT?category=lines)](https://github.com/yaricom/goNEAT)
[![Sourcegraph](https://sourcegraph.com/github.com/yaricom/goNEAT/-/badge.svg)](https://sourcegraph.com/github.com/yaricom/goNEAT?badge)

| Branch | Tests                                                                                      | Coverage                                                                                                                             | Linting                                                                    |
|--------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| master | [![CI](https://github.com/yaricom/goNEAT/actions/workflows/ci.yml/badge.svg)](https://github.com/yaricom/goNEAT/actions/workflows/ci.yml) | [![codecov](https://codecov.io/gh/yaricom/goNEAT/branch/master/graph/badge.svg?token=as31613DnV)](https://codecov.io/gh/yaricom/goNEAT) | [![Lint](https://github.com/yaricom/goNEAT/actions/workflows/lint.yml/badge.svg)](https://github.com/yaricom/goNEAT/actions/workflows/lint.yml) |

## Overview
This repository provides implementation of [NeuroEvolution of Augmenting Topologies (NEAT)][1] method written in Go language.

The NeuroEvolution (NE) is an artificial evolution of Neural Networks (NN) using genetic algorithms in order to find
optimal NN parameters and topology. NeuroEvolution of NN may assume search for optimal weights of connections between
NN nodes as well as search for optimal topology of resulting NN. The NEAT method implemented in this work do search for
both: optimal connections weights and topology for given task (number of NN nodes per layer and their interconnections).

## Minimum requirements
| Requirement | Notes            |
|-------------|------------------|
| Go version  | Go1.15 or higher |

## Installation
Make sure that you have at least GO 1.15.x. environment installed onto your system and execute following command:
```bash

go get github.com/yaricom/goNEAT
```

## Performance Evaluations
The basic system's performance is evaluated by two kind of experiments:
1. The XOR experiment which test whether topology augmenting actually happens by NEAT algorithm evaluation. To build XOR
solving network the NEAT algorithm should grow new hidden unit in the provided start genome.
2. The pole balancing experiments which is classic Reinforcement Learning experiment allowing us to estimate performance
of NEAT algorithm against proven results by many of other algorithms. I.e. we can benchmark NEAT performance against
other algorithms and find out if it performs better or worse.


### 1. The XOR Experiments
Because XOR is not linearly separable, a neural network requires hidden units to solve it. The two inputs must be
combined at some hidden unit, as opposed to only at the output node, because there is no function over a linear
combination of the inputs that can separate the inputs into the proper classes. These structural requirements make XOR
suitable for testing NEAT’s ability to evolve structure.

#### 1.1. The XOR Experiment with connected inputs in start genome
In this experiment we will use start (seed) genome with inputs connected to the output. Thus, it will check mostly the
ability of NEAT to grow new hidden unit necessary for solving XOR problem.

To run this experiment execute following commands:
```bash

cd $GOPATH/src/github.com/yaricom/goNEAT
go run executor.go -out ./out/xor -context ./data/xor.neat -genome ./data/xorstartgenes -experiment XOR

```
Where: ./data/xor.neat is the configuration of NEAT execution context and ./data/xorstartgenes is the start genome
configuration.

This will execute 100 trials of XOR experiment within 100 generations. As result of execution into the ./out directory
will be stored several 'gen_x' files with snapshots of population per 'print_every'
generation or when winner solution found. Also in mentioned directory will be stored 'xor_winner' with winner genome and
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
- **Mean Complexity** is an average complexity (number of nodes + number of links) of the best organisms per epoch for all epochs.
- **Mean Diversity** is an average diversity (number of species) per epoch for all epochs
- **Mean Age** is an average age of surviving species per epoch for all epochs

#### 1.2. The XOR experiment with disconnected inputs in start genome
This experiment will use start genome with disconnected inputs in order to check ability of algorithm to not only grow
need hidden nodes, but also to build missed connections between input nodes and rest of the network.

To run this experiment execute following commands:
```bash

cd $GOPATH/src/github.com/yaricom/goNEAT
go run executor.go -out ./out/xor_disconnected -context ./data/xor.neat -genome ./data/xordisconnectedstartgenes -experiment XOR

```

This will execute 100 trials of XOR (disconnected) experiment within 100 generations. The results of experiment execution
will be saved into the ./out directory as in previous experiment.

The experiment sometimes fails to produce XOR solution over 100 generations, but most of times solution will be found. This
confirms that algorithm is able not only grow needed hidden units, but also to restore input connections as needed.

The example output of the command as following:
```

Average
	Winner Nodes:	5.7
	Winner Genes:	9.2
	Winner Evals:	9347.7
Mean
	Complexity:	7.8
	Diversity:	20.0
	Age:		46.7

```

### 2. The single pole-balancing experiment
The pole-balancing or inverted pendulum problem has long been established as a standard benchmark for artificial learning
systems. It is one of the best early examples of a reinforcement learning task under conditions of incomplete knowledge.

![alt text][single_pole-balancing_scheme]

Figure 1.

##### System Constraints
1. The pole must remain upright within ±r the pole failure angle.
2. The cart must remain within ±h of origin.
3. The controller must always exert a non-zero force F.

Where r is a pole failure angle (±12 ̊ from 0) and h is a track limit (±2.4 meters from the track centre).

The simulation of the cart ends when either the pole inclines over the failure angle, or the cart position is outside the track.
The objective is to devise a controller that can keep the pole balanced for a defined length of simulation time.
The controller must always output a force at full magnitude in either direction (bang-bang control).

In this experiment the Genome considered as a winner if it's able to simulate single pole balancing at least 500’000 time
steps (10’000 simulated seconds).

To run this experiment with 150 population size execute following commands:
```bash

cd $GOPATH/src/github.com/yaricom/goNEAT
go run executor.go -out ./out/pole1 -context ./data/pole1_150.neat -genome ./data/pole1startgenes -experiment cart_pole

```

This will execute 100 trials of single pole-balancing experiment within 100 generations and over population with 150
organisms. The results of experiment execution will be saved in ./out directory under specified folder.

To run this experiment with 1’000 population size execute following commands:
```bash

cd $GOPATH/src/github.com/yaricom/goNEAT
go run executor.go -out ./out/pole1 -context ./data/pole1_1000.neat -genome ./data/pole1startgenes -experiment cart_pole

```

The example output of the command for population of 1000 organisms as following:
```

Average
	Winner Nodes:	7.0
	Winner Genes:	10.2
	Winner Evals:	1880.0
Mean
	Complexity:	17.1
	Diversity:	25.3
	Age:		2.2

```

This will execute 100 trials of single pole-balancing experiment within 100 generations and over population with 1’000
organisms.

The results demonstrate that winning Genome can be found in average within 2 generations among population of 1’000 organisms (which
belongs to 17 species in average) and within 30 generations for population of 150 organisms.

It's interesting to note that for population with 1000 organisms the winning solution often found in the null generation,
i.e. within initial random population. Thus, the next experiment with double pole-balancing setup seems more interesting
for performance testing.

In both single pole-balancing configurations described above the optimal winner organism has number of nodes and genes - 7 and 10 correspondingly.

The seven network nodes has following meaning:
* node #1 is a bias
* nodes #2-5 are sensors receiving system state: X position, acceleration among X, pole angle, and pole angular velocity
* nodes #6, 7 are output nodes signaling what action should be applied to the system to balance pole at
each simulation step, i.e. force direction to be applied. The applied force direction depends on relative strength of
activations of both output neurons. If activation of first output neuron (6-th node) greater than activation of second
neuron (7-th node) then the positive force direction applied. Otherwise, the negative force direction applied.

The TEN genes is exactly number of links required to connect FIVE input sensor nodes with TWO output neuron nodes (5x2).

### 3. The double pole-balancing experiment

This is advanced version of pole-balancing which assumes that cart has two poles with different mass and length to be balanced.

![alt text][double_pole-balancing_scheme]

Figure 2.

We will consider for benchmarking the two types of this problem:
* the Markovian with full system state known (including velocities);
* the Non-Markovian without velocity information.

The former one is fairly simple and last one is a quite challenging.

##### System Constraints
1. The both poles must remain upright within ±r the pole failure angle.
2. The cart must remain within ±h of origin.
3. The controller must always exert a non-zero force F.

Where r is a pole failure angle (±36 ̊ from 0) and h is a track limit (±2.4 meters from the track centre).

#### 3.1. The double pole-balancing Markovian experiment (with known velocity)

In this experiment agent will receive at each time step full system state including velocity of cart and both poles. The
winner solution will be determined as the one which is able to perform double pole-balancing at least 100’000 time steps or
1’000 simulated seconds.

To run experiment execute following command:
```bash

cd $GOPATH/src/github.com/yaricom/goNEAT
go run executor.go -out ./out/pole2_markov -context ./data/pole2_markov.neat -genome ./data/pole2_markov_startgenes -experiment cart_2pole_markov

```

This will execute 10 trials of double pole-balancing experiment within 100 generations and over population with 1’000
organisms.

The example output of the command:

```

Average
	Winner Nodes:	16.6
	Winner Genes:	35.2
	Winner Evals:	35593.2
Mean
	Complexity:	29.4
	Diversity:	686.9
	Age:		14.7

```

The winner solution can be found approximately within 13 generation with nearly doubled complexity of resulting genome
compared to the seed genome. The seed genome has eight nodes where nodes #1-6 is sensors for x, x', θ1, θ1', θ2, and θ2'
correspondingly, node #7 is a bias, and node #8 is an output signaling what action should be applied at each time step.


#### 3.2. The double pole-balancing Non-Markovian experiment (without velocity information)

In this experiment agent will receive at each time step partial system state excluding velocity information about cart and both poles.
Only horizontal cart position X, and angles of both poles θ1 and θ2 will be provided to the agent.

The best individual (i.e. the one with the highest fitness value) of every generation is tested for
its ability to balance the system for a longer time period. If a potential solution passes this test
by keeping the system balanced for 100’000 time steps, the so-called generalization score(GS) of this
particular individual is calculated. This score measures the potential of a controller to balance the
system starting from different initial conditions. It's calculated with a series of experiments, running
over 1000 time steps, starting from 625 different initial conditions.

The initial conditions are chosen by assigning each value of the set Ω = \[0.05, 0.25, 0.5, 0.75, 0.95\] to
each of the states x, ∆x/∆t, θ1 and ∆θ1/∆t, scaled to the range of the corresponding variables. The short pole 
angle θ2 and its angular velocity ∆θ2/∆t are set to zero. The GS is
defined as the number of successful runs from the 625 initial conditions. The organism considered
as successful solver if it reaches a generalization score of 200 or more.

To run experiment execute following command:
```bash

cd $GOPATH/src/github.com/yaricom/goNEAT
go run executor.go -out ./out/pole2_non-markov -context ./data/pole2_non-markov.neat -genome ./data/pole2_non-markov_startgenes -experiment cart_2pole_non-markov

```

This will execute 10 trials of double pole-balancing Non-Markovian experiment within 100 generations and over population
with 1’000 organisms.

The example output of the command:

```
Average
	Winner Nodes:	5.5
	Winner Genes:	11.5
	Winner Evals:	44049.3
Mean
	Complexity:	13.9
	Diversity:	611.1
	Age:		18.9

```

The maximal generalization score achieved in this test run is about 347 for very simple genome which consists of five nodes
and five genes (links). It has the same number of nodes as seed genome and only grew one extra recurrent gene connecting
output node to itself. Furthermore, it happens to be the most useful configuration among other.

The most fit organism's genome based on test non-markov run with maximal generalization score of 347:

```
genomestart 46
trait 1 0.2138946217467174 0.06002354203083418 0.0028906105813590443 0.2713154623271218 0.35856547551861573 0.15170654613596346 0.14290235997205494 0.28328098202348406
node 1 1 1 1
node 2 1 1 1
node 3 1 1 1
node 4 1 1 3
node 5 1 0 2
gene 1 1 5 -0.8193796421450023 false 1 -0.8193796421450023 true
gene 1 2 5 -6.591284446514077 false 2 -6.591284446514077 true
gene 1 3 5 5.783520492164443 false 3 5.783520492164443 true
gene 1 4 5 0.6165420336465628 false 4 0.6165420336465628 true
gene 1 5 5 -1.240555929293543 true 50 -1.240555929293543 true
genomeend 46

```

## Jupyter notebook with experiment results

The goNEAT library saves results of the experiments using Numpy NPZ format, which allows analysis of collected experimental
data using available Python libraries.

We included [Jupyter notebook](contents/notebooks/experiments_results.ipynb) with analysis of collected experimental data,
which can be used as starter kit for your own experiments.

## Conclusion

The experiments described in this work confirm that implemented NEAT method is able to evolve new structures in ANNs (XOR
experiment) and is able to solve reinforcement learning tasks under conditions of incomplete knowledge (pole-balancing).

## Credits

* The original C++ NEAT implementation created by Kenneth Stanley, see: [NEAT][1]
* Other NEAT implementations may be found at [NEAT Software Catalog][2]

This source code maintained and managed by [Iaroslav Omelianenko][3]


[1]:http://www.cs.ucf.edu/~kstanley/neat.html
[2]:http://eplex.cs.ucf.edu/neat_software/
[3]:https://io42.space

[single_pole-balancing_scheme]: https://github.com/yaricom/goNEAT/blob/master/contents/single_pole-balancing.jpg "The single pole-balancing experimental setup"
[double_pole-balancing_scheme]: https://github.com/yaricom/goNEAT/blob/master/contents/double_pole-balancing.png "The double pole-balancing experimental setup"
