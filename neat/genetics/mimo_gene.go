package genetics

import (
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat/network"
)

// MIMOControlGene The Multiple-Input Multiple-Output (MIMO) control Gene allows creating modular genomes, in which several groups of genes
// connected through single MIMO Gene and corresponding control function is applied to all inputs in order to produce
// outputs. This allows to build modular hierarchical genomes which can be considered as sum of constituent components
// and evolved as a whole and as a concrete parts simultaneously.
type MIMOControlGene struct {
	// The current innovation number for this gene
	InnovationNum int64
	// Used to see how much mutation has changed the link
	MutationNum float64
	// If true the gene is enabled
	IsEnabled bool

	// The control node with control/activation function
	ControlNode *network.NNode

	// The list of associated IO nodes for fast traversal
	ioNodes []*network.NNode
}

// NewMIMOGene Creates new MIMO gene
func NewMIMOGene(controlNode *network.NNode, innovNum int64, mutNum float64, enabled bool) *MIMOControlGene {
	gene := &MIMOControlGene{
		ControlNode:   controlNode,
		InnovationNum: innovNum,
		MutationNum:   mutNum,
		IsEnabled:     enabled,
	}
	// collect IO nodes list
	gene.ioNodes = make([]*network.NNode, 0)
	for _, l := range controlNode.Incoming {
		gene.ioNodes = append(gene.ioNodes, l.InNode)
	}
	for _, l := range controlNode.Outgoing {
		gene.ioNodes = append(gene.ioNodes, l.OutNode)
	}

	return gene
}

// NewMIMOGeneCopy The copy constructor taking parameters from provided control gene for given control node
func NewMIMOGeneCopy(g *MIMOControlGene, controlNode *network.NNode) *MIMOControlGene {
	cg := NewMIMOGene(controlNode, g.InnovationNum, g.MutationNum, g.IsEnabled)
	return cg
}

// Tests whether this gene has intersection with provided map of nodes, i.e. any of it's IO nodes included into list
func (g *MIMOControlGene) hasIntersection(nodes map[int]*network.NNode) bool {
	for _, nd := range g.ioNodes {
		if _, Ok := nodes[nd.Id]; Ok {
			// found
			return true
		}
	}
	return false
}

// The stringer
func (g *MIMOControlGene) String() string {
	enabledStr := ""
	if !g.IsEnabled {
		enabledStr = " -DISABLED-"
	}
	return fmt.Sprintf("[MIMO Gene INNOV (%5d, %2.3f) %s control node: %s]",
		g.InnovationNum, g.MutationNum, enabledStr, g.ControlNode.String())
}
