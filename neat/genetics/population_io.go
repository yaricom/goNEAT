package genetics

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"io"
	"strconv"
	"strings"
)

// ReadPopulation reads population from provided reader
func ReadPopulation(ir io.Reader, options *neat.Options) (pop *Population, err error) {
	pop = newPopulation()

	// Loop until file is finished, parsing each line
	scanner := bufio.NewScanner(ir)
	scanner.Split(bufio.ScanLines)
	var outBuff *bytes.Buffer
	var idCheck int
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, " ", 2)
		if len(parts) < 2 {
			return nil, fmt.Errorf("line: [%s] can not be split when reading Population", line)
		}
		switch parts[0] {
		case "genomestart":
			outBuff = bytes.NewBufferString(fmt.Sprintf("genomestart %s", parts[1]))
			idCheck, err = strconv.Atoi(parts[1])
			if err != nil {
				return nil, err
			}
		case "genomeend":
			if _, err = fmt.Fprintf(outBuff, "genomeend %d", idCheck); err != nil {
				return nil, err
			}
			newGenome, err := ReadGenome(bufio.NewReader(outBuff), idCheck)
			if err != nil {
				return nil, err
			}
			// add new organism for read genome
			if newOrganism, err := NewOrganism(0.0, newGenome, 1); err != nil {
				return nil, err
			} else {
				pop.Organisms = append(pop.Organisms, newOrganism)
			}

			if lastNodeId, err := newGenome.getLastNodeId(); err == nil {
				if pop.nextNodeId < int32(lastNodeId) {
					pop.nextNodeId = int32(lastNodeId + 1)
				}
			} else {
				return nil, err
			}

			if lastGeneInnovNum, err := newGenome.getNextGeneInnovNum(); err == nil {
				if pop.nextInnovNum < lastGeneInnovNum {
					pop.nextInnovNum = lastGeneInnovNum
				}
			} else {
				return nil, err
			}
			// clear buffer
			outBuff = nil
			idCheck = -1

		case "/*":
			// read all comments and print it
			neat.InfoLog(line)
		default:
			// write line to buffer
			if _, err = fmt.Fprintln(outBuff, line); err != nil {
				return nil, err
			}
		}

	}
	if err = scanner.Err(); err != nil {
		return nil, err
	}

	if err = pop.speciate(options.NeatContext(), pop.Organisms); err != nil {
		return nil, err
	}
	return pop, nil
}

// Writes given population to a writer
func (p *Population) Write(w io.Writer) error {
	// Prints all the Organisms' Genomes to the outFile
	for _, o := range p.Organisms {
		if err := o.Genotype.Write(w); err != nil {
			return err
		}
	}
	return nil
}

// WriteBySpecies Writes given population by species
func (p *Population) WriteBySpecies(w io.Writer) error {
	// Step through the Species and write them
	for _, sp := range p.Species {
		if err := sp.Write(w); err != nil {
			return err
		}
	}
	return nil
}
