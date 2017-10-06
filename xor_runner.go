package main

import (
	"os"
	"time"
	"fmt"
	"bytes"
	"math/rand"
	"github.com/yaricom/goNEAT/experiments"
)

// The XOR experiment runner
func main() {
	out_dir_path, context_path, genome_path := "./out", "./data/xor.neat", "./data/xorstartgenes"
	if len(os.Args) == 4 {
		out_dir_path = os.Args[1]
		context_path = os.Args[2]
		genome_path = os.Args[3]
	}

	// Seed the random-number generator with current time so that
	// the numbers will be different every time we run.
	rand.Seed(time.Now().Unix())

	// The 100 generation XOR experiment
	pop, err := experiments.XOR(context_path, genome_path, out_dir_path, 100)
	if err != nil {
		fmt.Println("Failed to perform XOR experiment:")
		fmt.Println(err)
		return
	} else if pop != nil {
		out_buf := bytes.NewBufferString("")
		pop.Write(out_buf)

		fmt.Println("The winning population:")
		fmt.Println(out_buf)
	}
}