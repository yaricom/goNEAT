package genetics

import (
	"io"
	"fmt"

	"github.com/yaricom/goNEAT/neat"
	"bufio"
	"errors"
	"github.com/yaricom/goNEAT/neat/network"
	"strings"
)


// The interface to define genome reader
type GenomeReader interface {
	// Reads one Genome record
	Read() (*Genome, error)
}

// Creates reader for Genome data with specified encoding format.
func NewGenomeReader(r io.Reader, encoding GenomeEncoding) (GenomeReader, error) {
	switch encoding {
	case PlainGenomeEncoding:
		return &PlainGenomeReader{r: bufio.NewReader(r)}, nil
	case YAMLGenomeEncoding:
		return &YAMLGenomeReader{r: bufio.NewReader(r)}, nil
	default:
		return nil, ErrUnsupportedGenomeEncoding
	}
}

// A PlainGenomeReader reads genome data from plain text file.
type PlainGenomeReader struct {
	r *bufio.Reader
}

func (pgr *PlainGenomeReader) Read() (*Genome, error) {
	gnome := Genome{
		Traits:make([]*neat.Trait, 0),
		Nodes:make([]*network.NNode, 0),
		Genes:make([]*Gene, 0),
	}

	var g_id int
	// Loop until file is finished, parsing each line
	scanner := bufio.NewScanner(pgr.r)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, " ", 2)
		if len(parts) < 2 {
			return nil, errors.New(fmt.Sprintf("Line: [%s] can not be split when reading Genome", line))
		}
		lr := strings.NewReader(parts[1])

		switch parts[0] {
		case "trait":
			// Read a Trait
			new_trait := readPlainTrait(lr)
			gnome.Traits = append(gnome.Traits, new_trait)

		case "node":
			// Read a Network Node
			new_node := readPlainNetworkNode(lr, gnome.Traits)
			gnome.Nodes = append(gnome.Nodes, new_node)

		case "gene":
			// Read a Gene
			new_gene := readPlainConnectionGene(lr, gnome.Traits, gnome.Nodes)
			gnome.Genes = append(gnome.Genes, new_gene)

		case "genomeend":
			// Read Genome ID
			fmt.Fscanf(lr, "%d", &g_id)
			// save genome ID
			gnome.Id = g_id

		case "/*":
			// read all comments and print it
			neat.InfoLog(line)
		}
	}
	return &gnome, nil
}

// The method to read Trait in plain text format
func readPlainTrait(r io.Reader) *neat.Trait {
	nt := neat.NewTrait()
	fmt.Fscanf(r, "%d ", &nt.Id)
	for i := 0; i < neat.Num_trait_params; i++ {
		fmt.Fscanf(r, "%g ", &nt.Params[i])
	}
	return nt
}

// Read a Network Node from specified Reader in plain text format
// and applies corresponding trait to it from a list of traits provided
func readPlainNetworkNode(r io.Reader, traits []*neat.Trait) *network.NNode {
	n := network.NewNetworkNode()
	var trait_id, node_type int
	fmt.Fscanf(r, "%d %d %d %d ", &n.Id, &trait_id, &node_type, &n.NeuronType)
	if trait_id != 0 && traits != nil {
		// find corresponding node trait from list
		for _, t := range traits {
			if trait_id == t.Id {
				n.Trait = t
				break
			}
		}
	}
	return n
}

// Reads Gene from reader in plain text format
func readPlainConnectionGene(r io.Reader, traits []*neat.Trait, nodes []*network.NNode) *Gene  {
	var traitId, inNodeId, outNodeId int
	var inov_num int64
	var weight, mut_num float64
	var recurrent, enabled bool
	fmt.Fscanf(r, "%d %d %d %g %t %d %g %t ",
		&traitId, &inNodeId, &outNodeId, &weight, &recurrent, &inov_num, &mut_num, &enabled)

	var trait *neat.Trait = nil
	if traitId != 0 && traits != nil {
		for _, tr := range traits {
			if tr.Id == traitId {
				trait = tr
			}
		}
	}
	var inNode, outNode *network.NNode
	for _, np := range nodes {
		if np.Id == inNodeId {
			inNode = np
		}
		if np.Id == outNodeId {
			outNode = np
		}
	}
	if trait != nil {
		return newGene(network.NewLinkWithTrait(trait, weight, inNode, outNode, recurrent), inov_num, mut_num, enabled)
	} else {
		return newGene(network.NewLink(weight, inNode, outNode, recurrent), inov_num, mut_num, enabled)
	}
}

// A YAMLGenomeReader reads genome data from YAML encoded text file
type YAMLGenomeReader struct {
	r *bufio.Reader
}

func (ygr *YAMLGenomeReader) Read()(*Genome, error) {
	return nil, nil
}


