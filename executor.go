package main

import (
	"os"
	"time"
	"fmt"
	"log"
	"flag"
	"math/rand"
	"github.com/yaricom/goNEAT/experiments"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/genetics"
	"github.com/yaricom/goNEAT/experiments/xor"
	"github.com/yaricom/goNEAT/experiments/pole"
)

// The experiment runner boilerplate code
func main() {
	var out_dir_path = flag.String("out", "./out", "The output directory to store results.")
	var context_path = flag.String("context", "./data/xor.neat", "The execution context configuration file.")
	var genome_path = flag.String("genome", "./data/xorstartgenes", "The seed genome to start with.")
	var experiment_name = flag.String("experiment", "XOR", "The name of experiment to run. [XOR, cart_pole, cart_2pole_markov, cart_2pole_non-markov]")
	var trials_count = flag.Int("trials", 0, "The numbar of trials for experiment. Overrides the one set in configuration.")
	var log_level = flag.Int("log_level", -1, "The logger level to be used. Overrides the one set in configuration.")

	flag.Parse()

	// Seed the random-number generator with current time so that
	// the numbers will be different every time we run.
	rand.Seed(time.Now().Unix())

	// Load context configuration
	configFile, err := os.Open(*context_path)
	if err != nil {
		log.Fatal("Failed to open context configuration file: ", err)
	}
	context := neat.LoadContext(configFile)

	// Load Genome
	log.Printf("Loading start genome for %s experiment\n", *experiment_name)
	genomeFile, err := os.Open(*genome_path)
	if err != nil {
		log.Fatal("Failed to open genome file: ", err)
	}
	start_genome, err := genetics.ReadGenome(genomeFile, 1)
	if err != nil {
		log.Fatal("Failed to read start genome: ", err)
	}
	fmt.Println(start_genome)

	// Check if output dir exists
	out_dir := *out_dir_path
	if _, err := os.Stat(out_dir); err == nil {
		// backup it
		back_up_dir := fmt.Sprintf("%s-%s", out_dir, time.Now().Format("2006-01-02T15_04_05"))
		// clear it
		err = os.Rename(out_dir, back_up_dir)
		if err != nil {
			log.Fatal("Failed to do previous results backup: ", err)
		}
	}
	// create output dir
	err = os.MkdirAll(out_dir, os.ModePerm)
	if err != nil {
		log.Fatal("Failed to create output directory: ", err)
	}

	// Override context configuration parameters with ones set from command line
	if *trials_count > 0 {
		context.NumRuns = *trials_count
	}
	if *log_level >= 0 {
		neat.LogLevel = neat.LoggerLevel(*log_level)
	}

	// The 100 generation XOR experiment
	experiment := experiments.Experiment{
		Id:0,
		Trials:make(experiments.Trials, context.NumRuns),
	}
	var generationEvaluator experiments.GenerationEvaluator
	if *experiment_name == "XOR" {
		experiment.MaxFintessScore = 16.0 // as given by fitness function definition
		generationEvaluator = xor.XORGenerationEvaluator{OutputPath:out_dir}
	} else if *experiment_name == "cart_pole" {
		experiment.MaxFintessScore = 1.0 // as given by fitness function definition
		generationEvaluator = pole.CartPoleGenerationEvaluator{
			OutputPath:out_dir,
			WinBalancingSteps:500000,
			RandomStart:true,
		}
	} else if *experiment_name == "cart_2pole_markov" {
		experiment.MaxFintessScore = 1.0 // as given by fitness function definition
		generationEvaluator = pole.CartDoublePoleGenerationEvaluator{
			OutputPath:out_dir,
			Markov:true,
			ActionType:experiments.ContinuousAction,
		}
	} else if *experiment_name == "cart_2pole_non-markov" {
		generationEvaluator = pole.CartDoublePoleGenerationEvaluator{
			OutputPath:out_dir,
			Markov:false,
			ActionType:experiments.ContinuousAction,
		}
	}

	err = experiment.Execute(context, start_genome, generationEvaluator)
	if err != nil {
		log.Fatal("Failed to perform XOR experiment: ", err)
	}

	// Print statistics
	experiment.PrintStatistics()

	fmt.Printf(">>> Start genome file:  %s\n", *genome_path)
	fmt.Printf(">>> Configuration file: %s\n", *context_path)

	// Save experiment data
	expResPath := fmt.Sprintf("%s/%s.dat", out_dir, *experiment_name)
	expResFile, err := os.Create(expResPath)
	if err == nil {
		err = experiment.Write(expResFile)
	}
	if err != nil {
		log.Fatal("Failed to save experiment results", err)
	}
}