package genetics

import (
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/network"
)

// The Gene type in this system specifies a "Connection Gene."
// Nodes are represented using the NNode class, which serves as both a genotypic and phenotypic representation of nodes.
// Genetic Representation of connections uses this special class because it calls for special operations better served
// by a specific genetic representation.
// A Gene object in this system specifies a link between two nodes along with an InnovationNum which tells when
// in the history of a population the gene first arose. This allows the system to track innovations and use those to
// determine which organisms are compatible (i.e. in the same species).
// A MutationNum gives a rough sense of how much mutation the gene has experienced since it originally appeared
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

// NewGene Creates new Gene
func NewGene(weight float64, inNode, outNode *network.NNode, recurrent bool, innovationNum int64, mutationNum float64) *Gene {
	return NewConnectionGene(network.NewLink(weight, inNode, outNode, recurrent), innovationNum, mutationNum, true)
}

// NewGeneWithTrait Creates new Gene with Trait
func NewGeneWithTrait(trait *neat.Trait, weight float64, inNode, outNode *network.NNode,
	recurrent bool, innovationNum int64, mutationNum float64) *Gene {
	return NewConnectionGene(network.NewLinkWithTrait(trait, weight, inNode, outNode, recurrent), innovationNum, mutationNum, true)
}

// NewGeneCopy Construct a gene off of another gene as a duplicate
func NewGeneCopy(g *Gene, trait *neat.Trait, inNode, outNode *network.NNode) *Gene {
	return NewConnectionGene(network.NewLinkWithTrait(trait, g.Link.ConnectionWeight, inNode, outNode, g.Link.IsRecurrent),
		g.InnovationNum, g.MutationNum, true)
}

// NewConnectionGene is to create new connection gene with provided link
func NewConnectionGene(link *network.Link, innovationNum int64, mutationNum float64, enabled bool) *Gene {
	return &Gene{
		Link:          link,
		InnovationNum: innovationNum,
		MutationNum:   mutationNum,
		IsEnabled:     enabled,
	}
}

func (g *Gene) String() string {
	enabledStr := ""
	if !g.IsEnabled {
		enabledStr = " -DISABLED-"
	}
	recurrentStr := ""
	if g.Link.IsRecurrent {
		recurrentStr = " -RECUR-"
	}
	traitStr := ""
	if g.Link.Trait != nil {
		traitStr = fmt.Sprintf(" Link's trait_id: %d", g.Link.Trait.Id)
	}
	return fmt.Sprintf("[Link (%4d ->%4d) INNOV (%4d, %.3f) Weight: %.3f %s%s%s : %s->%s]",
		g.Link.InNode.Id, g.Link.OutNode.Id, g.InnovationNum, g.MutationNum, g.Link.ConnectionWeight,
		traitStr, enabledStr, recurrentStr, g.Link.InNode, g.Link.OutNode)
}
