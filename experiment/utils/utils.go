// Package utils provides common utilities to be used by experiments.
package utils

import (
	"fmt"
	"github.com/yaricom/goNEAT/v3/experiment"
	"github.com/yaricom/goNEAT/v3/neat/genetics"
	"github.com/yaricom/goNEAT/v3/neat/network/formats"
	"log"
	"os"
)

// WriteGenomePlain is to write genome of the organism to the genomeFile in the outDir directory using plain encoding.
// The method return path to the file if successful or error if failed.
func WriteGenomePlain(genomeFile, outDir string, org *genetics.Organism, epoch *experiment.Generation) (string, error) {
	orgPath := fmt.Sprintf("%s/%s_%d-%d", CreateOutDirForTrial(outDir, epoch.TrialId),
		genomeFile, org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
	if file, err := os.Create(orgPath); err != nil {
		return "", err
	} else if err = org.Genotype.Write(file); err != nil {
		return "", err
	}
	return orgPath, nil
}

// WriteGenomeDOT is to write genome of the organism to the genomeFile in the outDir directory using DOT encoding.
// The method return path to the file if successful or error if failed.
func WriteGenomeDOT(genomeFile, outDir string, org *genetics.Organism, epoch *experiment.Generation) (string, error) {
	orgPath := fmt.Sprintf("%s/%s_%d-%d.dot", CreateOutDirForTrial(outDir, epoch.TrialId),
		genomeFile, org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
	if file, err := os.Create(orgPath); err != nil {
		return "", err
	} else if err = formats.WriteDOT(file, org.Phenotype); err != nil {
		return "", err
	}
	return orgPath, nil
}

// WriteGenomeCytoscapeJSON is to write genome of the organism to the genomeFile in the outDir directory using Cytoscape JSON encoding.
// The method return path to the file if successful or error if failed.
func WriteGenomeCytoscapeJSON(genomeFile, outDir string, org *genetics.Organism, epoch *experiment.Generation) (string, error) {
	orgPath := fmt.Sprintf("%s/%s_%d-%d.cyjs", CreateOutDirForTrial(outDir, epoch.TrialId),
		genomeFile, org.Phenotype.NodeCount(), org.Phenotype.LinkCount())
	if file, err := os.Create(orgPath); err != nil {
		return "", err
	} else if err = formats.WriteCytoscapeJSON(file, org.Phenotype); err != nil {
		return "", err
	}
	return orgPath, nil
}

// WritePopulationPlain is to write genomes of the entire population using plain encoding in the outDir directory.
// The methods return path to the file if successful or error if failed.
func WritePopulationPlain(outDir string, pop *genetics.Population, epoch *experiment.Generation) (string, error) {
	popPath := fmt.Sprintf("%s/gen_%d", CreateOutDirForTrial(outDir, epoch.TrialId), epoch.Id)
	if file, err := os.Create(popPath); err != nil {
		return "", err
	} else if err = pop.WriteBySpecies(file); err != nil {
		return "", err
	}
	return popPath, nil
}

// CreateOutDirForTrial allows creating the output directory for specific trial of the experiment using standard name.
func CreateOutDirForTrial(outDir string, trialID int) string {
	dir := fmt.Sprintf("%s/%d", outDir, trialID)
	if _, err := os.Stat(dir); err != nil {
		// create output dir
		if err = os.MkdirAll(dir, os.ModePerm); err != nil {
			log.Fatal("Failed to create output directory: ", err)
		}
	}
	return dir
}
