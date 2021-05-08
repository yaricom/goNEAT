# goNEAT
[![banner](contents/book_title.png)][6]

[NeuroEvolution — evolving Artificial Neural Networks topology from the scratch](https://becominghuman.ai/neuroevolution-evolving-artificial-neural-networks-topology-from-the-scratch-d1ebc5540d84)

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

The NeuroEvolution (NE) is an artificial evolution of Neural Networks (NN) using genetic algorithms to find optimal NN 
parameters and network topology. NeuroEvolution of NN may assume a search for optimal weights of connections between NN 
nodes and search for the optimal topology of the resulting network graph. The NEAT method implemented in this work 
searches for optimal connection weights and the network graph topology for a given task (number of NN nodes per layer 
and their interconnections).

## Minimum requirements
| Requirement | Notes            |
|-------------|------------------|
| Go version  | Go1.15 or higher |

## Releases

Please do not depend on master as your production branch. Use [releases](https://github.com/yaricom/goNEAT/releases) instead.

## Documentation

The algorithm performance evaluation and related documentation can be found in project's [wiki](https://github.com/yaricom/goNEAT/wiki)

The goNEAT library saves results of the experiments using Numpy NPZ format, which allows analysis of collected experimental
data using available Python libraries.  We included [Jupyter notebook](contents/notebooks/experiments_results.ipynb) with analysis of collected experimental data,
which can be used as starter kit to analyse data samples acquired from your experiments.

### Installation

Make sure you have at least GO 1.15.x installed onto your system and execute the following command:
```bash

go get github.com/yaricom/goNEAT
```

For new projects, consider using the v2 of the library with the following import:

```go
import "github.com/yaricom/goNEAT/v2"
```

[`genetics`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/genetics "API documentation") package
-------------------------------------------------------------------------------------------

The `genetics` package provides the genetic part of the NEAT algorithm describing all the machinery related to
genome mutations, mating, and speciation of the population of organisms.

It contains implementation of all important types related to the NEAT algorithm:
* [`Gene`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/genetics#Gene) type in this system specifies a "Connection Gene."
* [`MIMOControlGene`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/genetics#MIMOControlGene) type is the Multiple-Input Multiple-Output (MIMO) control Gene which allows creation of modular genomes  
* [`Genome`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/genetics#Genome) type is the primary source of genotype information used to create a phenotype.
* [`Organism`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/genetics#Organism) type is Genotypes (Genomes) and Phenotypes (Networks) combined with fitness information, i.e. the genotype and phenotype together.
* [`Population`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/genetics#Population) type is a group of Organisms including their Species
* [`Species`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/genetics#Species) type  is a group of similar Organisms. Reproduction takes place mostly within a single species, so that compatible organisms can mate.

Additionally, it contains variety of utility functions to serialise/deserialize specified above types using two
supported data formats:
* plain text
* YAML

The current implementation supports sequential and parallel execution of evolution epoch which controlled by
[related parameter](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat#EpochExecutorType) in the NEAT context options.

[`math`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/math "API documentation") package
-------------------------------------------------------------------------------------------

Package `math` defines standard mathematical primitives used by the NEAT algorithm as well as utility functions

[`network`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/network "API documentation") package
-------------------------------------------------------------------------------------------

Package `network` provides data structures and utilities to describe Artificial Neural Network and network solvers.

The most important types are:
* [`NNode`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/network#NNode) type defines the node of the network and is a part of organism's genotype as well as phenotype
* [`Link`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/network#Link) type is a connection from one node to another with an associated weight.
* [`Network`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/network#Network) type is a collection of all nodes within an organism's phenotype, which effectively defines Neural Network topology.
* [`Solver`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/network#Solver) type defines network solver interface, which allows propagation of the activation waves through the underlying network graph.

The current implementation supports two types of network solvers: 
* [`FastModularNetworkSolver`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/neat/network#FastModularNetworkSolver) is the network solver implementation to be used for large neural networks simulation.
* Standard Network Solver implemented by the `Network` type

[`experiment`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/experiment "API documentation") package
-------------------------------------------------------------------------------------------

Package `experiment` defines standard evolutionary epochs evaluators and experimental data samples collectors. It provides
standardised approach to define experiments using the NEAT algorithm implementation.

The most important type here is:
* [`GenerationEvaluator`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/experiment#GenerationEvaluator) is the interface to be implemented by custom experiments

You can find examples of `GenerationEvaluator` implementations at [experiments](https://github.com/yaricom/goNEAT/tree/master/experiments):
* [`pole`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/experiments/pole) - single-, double-pole balancing experiments
* [`xor`](https://pkg.go.dev/github.com/yaricom/goNEAT/v2/experiments/xor) - XOR solver experiment

Also, consider taking a look at [executor.go](https://github.com/yaricom/goNEAT/blob/master/executor.go) implementation
for the details about how to set up and run experiment executors.

## Conclusion

The experiments described in this work confirm that introduced NEAT algorithm implementation can evolve new structures in 
_Artificial Neural Networks_ ([XOR experiments](https://github.com/yaricom/goNEAT/wiki/XOR-Experiment)) and can solve reinforcement learning 
tasks under conditions of incomplete knowledge ([single-pole balancing](https://github.com/yaricom/goNEAT/wiki/The-single-pole-balancing-experiment) and
[double-pole balancing](https://github.com/yaricom/goNEAT/wiki/Double-Pole-Balancing-Experiments)).

We hope that you will find great applications in your research and work projects for the provided NEAT algorithm's 
implementation as well as utilities to run experiments while collecting relevant data samples.

## Support this work

If you found this library helpful, please consider supporting further work on the project by donating.

You can help to evolve this project either by pressing [`Sponsor`](https://www.patreon.com/io42) or by sending some funds 
to:
* _LTC:_ LPi2hvnMQLWy1BKbjtyPeEqVcfyPfQLErs
* _DOGE:_ D9u3YQJfpjYQT67ZQRub97jjgiiG7S3S6x

## Credits

* The original C++ NEAT implementation created by Kenneth Stanley, see: [NEAT][1]
* Other NEAT implementations may be found at [NEAT Software Catalog][2]
* Iaroslav Omelianenko, [NeuroEvolution — evolving Artificial Neural Networks topology from the scratch][4], Medium, 2018
* Kenneth O. Stanley, [Ph.D. Dissertation: EFFICIENT EVOLUTION OF NEURAL NETWORKS THROUGH COMPLEXIFICATION][5], Department of Computer Sciences, The University of Texas at Austin, Technical Report~AI-TR-04–39, August 2004
* [Hands-On NeuroEvolution with Python, Build high-performing artificial neural network architectures using neuroevolution-based algorithms][6], Iaroslav Omelianenko, Birmingham: Packt Publishing, 2019

This source code maintained and managed by [Iaroslav Omelianenko][3]

[1]:http://www.cs.ucf.edu/~kstanley/neat.html
[2]:http://eplex.cs.ucf.edu/neat_software/
[3]:https://io42.space
[4]:https://becominghuman.ai/neuroevolution-evolving-artificial-neural-networks-topology-from-the-scratch-d1ebc5540d84
[5]:http://nn.cs.utexas.edu/keyword?stanley:phd04
[6]:https://www.packtpub.com/product/hands-on-neuroevolution-with-python/9781838824914