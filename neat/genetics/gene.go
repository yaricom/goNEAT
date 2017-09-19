package genetics

import (
	"github.com/yaricom/goNEAT/neat/network"
	"io"
	"fmt"
	"github.com/yaricom/goNEAT/neat"
)

// The Gene class in this system specifies a "Connection Gene."
// Nodes are represented using the NNode class, which serves as both a genotypic and phenotypic representation of nodes.
// Genetic Representation of connections uses this special class because it calls for special operations better served
// by a specific genetic representation.
// A Gene object in this system specifies a link between two nodes along with an "innovation number" which tells when
// in the history of a population the gene first arose. This allows the system to track innovations and use those to
// determine which organisms are compatible (i.e. in the same species).
// A mutation_num gives a rough sense of how much mutation the gene has experienced since it originally appeared
// (Since it was first innovated). In the current implementation the mutation number is the same as the weight.
type Gene struct {
	// The link between nodes
	Link *network.Link
	// The current innovation number for this gene
	InnovationNum int64
	// Used to see how much mutation has changed the link
	MutationNum float64
	// If true the gene is enabled
	IsEnabled bool
}

// Creates new Gene
func NewGene(weight float64, in_node, out_node *network.NNode, recurrent bool, inov_num int64, mut_num float64) *Gene  {
	return newGene(network.NewLink(weight, in_node, out_node, recurrent), inov_num, mut_num, true)
}

// Creates new Gene with Trait
func NewGeneWithTrait(trait *neat.Trait, weight float64, in_node, out_node *network.NNode,
			recurrent bool, inov_num int64, mut_num float64) *Gene  {
	return newGene(network.NewLinkWithTrait(trait, weight, in_node, out_node, recurrent), inov_num, mut_num, true)
}

// Construct a gene off of another gene as a duplicate
func NewGeneGeneCopy(g *Gene, trait *neat.Trait, in_node, out_node *network.NNode) *Gene {
	return newGene(network.NewLinkWithTrait(trait, g.Link.Weight, in_node, out_node, g.Link.IsRecurrent),
		g.InnovationNum, g.MutationNum, true)
}

// Reads Gene from reader
func ReadGene(r io.Reader, traits []*neat.Trait, nodes []*network.NNode) *Gene  {
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

func newGene(link *network.Link, inov_num int64, mut_num float64, enabled bool) *Gene {
	return &Gene{
		Link:link,
		InnovationNum:inov_num,
		MutationNum:mut_num,
		IsEnabled:enabled,
	}
}

// Writes Gene to the provided writer
func (g *Gene) WriteGene(w io.Writer)  {
	link := g.Link
	traitId := 0
	if link.LinkTrait != nil {
		traitId = link.LinkTrait.Id
	}
	inNodeId := link.InNode.Id
	outNodeId := link.OutNode.Id
	weight := link.Weight
	recurrent := link.IsRecurrent
	innov_num := g.InnovationNum
	mut_num := g.MutationNum
	enabled := g.IsEnabled

	fmt.Fprintf(w, "%d %d %d %g %t %d %g %t",
		traitId, inNodeId, outNodeId, weight, recurrent, innov_num, mut_num, enabled)
}

func (g *Gene) String() string  {
	enabl_str := ""
	if !g.IsEnabled {
		enabl_str = " -DISABLED-"
	}
	recurr_str := ""
	if g.Link.IsRecurrent {
		recurr_str = " -RECUR-"
	}
	trait_str := ""
	if g.Link.LinkTrait != nil {
		trait_str = fmt.Sprintf("Link's trait_id %d", g.Link.LinkTrait.Id)
	}
	return fmt.Sprintf("[Link (%4d, %4d) INNOV (%4d, %.3f) Weight %.3f %s %s %s]",
		g.Link.InNode.Id, g.Link.OutNode.Id, g.InnovationNum, g.MutationNum, g.Link.Weight,
		trait_str, enabl_str, recurr_str)
}
