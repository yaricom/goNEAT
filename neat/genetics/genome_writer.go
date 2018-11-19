package genetics

import (
	"io"
	"bufio"
	"fmt"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/network"
)

// The interface to define genome writer
type GenomeWriter interface {
	// Writes Genome record
	WriteGenome(genome *Genome) error
}

// Creates genome writer with specified data encoding format
func NewGenomeWriter(w io.Writer, encoding GenomeEncoding) (GenomeWriter, error) {
	switch encoding {
	case PlainGenomeEncoding:
		return &plainGenomeWriter{w:bufio.NewWriter(w)}, nil
	case YAMLGenomeEncoding:
		return &yamlGenomeWriter{w:bufio.NewWriter(w)}, nil
	default:
		return nil, ErrUnsupportedGenomeEncoding
	}
}

// The plain text encoded genome writer
type plainGenomeWriter struct {
	w *bufio.Writer
}

// Writes genome in Plain Text format
func (wr *plainGenomeWriter) WriteGenome(g *Genome) error {
	_, err:=fmt.Fprintf(wr.w, "genomestart %d\n", g.Id)
	if err != nil {
		return err
	}

	for _, tr := range g.Traits {
		fmt.Fprint(wr.w, "trait ")
		err := wr.writeTrait(tr)
		if err != nil {
			return err
		}
		fmt.Fprintln(wr.w, "")
	}

	for _, nd := range g.Nodes {
		fmt.Fprint(wr.w, "node ")
		err := wr.writeNetworkNode(nd)
		if err != nil {
			return err
		}
		fmt.Fprintln(wr.w, "")
	}

	for _, gn := range g.Genes {
		fmt.Fprint(wr.w, "gene ")
		err := wr.writeConnectionGene(gn)
		if err != nil {
			return err
		}
		fmt.Fprintln(wr.w, "")
	}
	_, err = fmt.Fprintf(wr.w, "genomeend %d\n", g.Id)

	// flush buffer
	err = wr.w.Flush()

	return err
}

// Dump trait in plain text format
func (wr *plainGenomeWriter) writeTrait(t *neat.Trait) error {
	_, err := fmt.Fprintf(wr.w, "%d ", t.Id)
	if err == nil {
		for _, p := range t.Params {
			_, err = fmt.Fprintf(wr.w, "%g ", p)
			if err != nil {
				return err
			}
		}
	}
	return err
}
// Dump network node in plain text format
func (wr *plainGenomeWriter) writeNetworkNode(n *network.NNode) error {
	trait_id := 0
	if n.Trait != nil {
		trait_id = n.Trait.Id
	}
	_, err := fmt.Fprintf(wr.w, "%d %d %d %d", n.Id, trait_id, n.NodeType(), n.NeuronType)
	return err
}
// Dump connection gene in plain text format
func (wr *plainGenomeWriter) writeConnectionGene(g *Gene) error {
	link := g.Link
	traitId := 0
	if link.Trait != nil {
		traitId = link.Trait.Id
	}
	inNodeId := link.InNode.Id
	outNodeId := link.OutNode.Id
	weight := link.Weight
	recurrent := link.IsRecurrent
	innov_num := g.InnovationNum
	mut_num := g.MutationNum
	enabled := g.IsEnabled

	_, err := fmt.Fprintf(wr.w, "%d %d %d %g %t %d %g %t",
		traitId, inNodeId, outNodeId, weight, recurrent, innov_num, mut_num, enabled)
	return err
}

// The YAML encoded genome writer
type yamlGenomeWriter struct {
	w *bufio.Writer
}

func (wr *yamlGenomeWriter) WriteGenome(g *Genome) error {
	return nil
}