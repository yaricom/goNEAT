package utils

import (
	"github.com/pkg/errors"
	"github.com/yaricom/goNEAT/v4/neat"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
	"os"
)

func CreateOutputDir(outDirPath string) error {
	// Check if output dir exists
	if _, err := os.Stat(outDirPath); err == nil {
		// clear it
		if err = os.RemoveAll(outDirPath); err != nil {
			return err
		}
	}
	// create output dir
	return os.MkdirAll(outDirPath, os.ModePerm)
}

func LoadOptionsAndGenome(contextPath, genomePath string) (*neat.Options, *genetics.Genome, error) {
	// Load context configuration
	configFile, err := os.Open(contextPath)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to open context file")
	}
	context, err := neat.LoadNeatOptions(configFile)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to load NEAT options")
	}

	// Load start Genome
	genomeFile, err := os.Open(genomePath)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to open genome file")
	}
	startGenome, err := genetics.ReadGenome(genomeFile, 1)
	if err != nil {
		return nil, nil, errors.Wrap(err, "failed to read start genome")
	}
	return context, startGenome, nil
}
