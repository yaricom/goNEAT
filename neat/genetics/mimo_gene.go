package genetics

import (
	"github.com/yaricom/goNEAT/neat/network"
	"fmt"
)

// The Multiple-Input Multiple-Output (MIMO) control Gene allows to create modular genomes, in which several groups of genes
// connected through single MIMO Gene and corresponding control function is applied to all inputs in order to produce
// outputs. This allows to build modular hierarchical genomes which can be considered as sum of constituent components
// and evolved as a whole and as a concrete parts simultaneously.
type MIMOControlGene struct {
	// The current innovation number for this gene
	InnovationNum int64
	// Used to see how much mutation has changed the link
	MutationNum   float64
	// If true the gene is enabled
	IsEnabled     bool

	// The control node with control/activation function
	ControlNode   *network.NNode

	// The list of associated IO nodes for fast traversal
	ioNodes       []*network.NNode
}

// Creates new MIMO gene
func NewMIMOGene(control_node *network.NNode, innov_num int64, mut_num float64, enabled bool) *MIMOControlGene {
	gene := &MIMOControlGene{
		ControlNode:control_node,
		InnovationNum:innov_num,
		MutationNum:mut_num,
		IsEnabled:enabled,
	}
	// collect IO nodes list
	gene.ioNodes = make([]*network.NNode, 0)
	for _, l := range control_node.Incoming {
		gene.ioNodes = append(gene.ioNodes, l.InNode)
	}
	for _, l := range control_node.Outgoing {
		gene.ioNodes = append(gene.ioNodes, l.OutNode)
	}

	return gene
}

// The copy constructor taking parameters from provided control gene for given control node
func NewMIMOGeneCopy(g *MIMOControlGene, control_node *network.NNode) *MIMOControlGene {
	cg := NewMIMOGene(control_node, g.InnovationNum, g.MutationNum, g.IsEnabled)
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
	enabl_str := ""
	if !g.IsEnabled {
		enabl_str = " -DISABLED-"
	}
	return fmt.Sprintf("[MIMO Gene INNOV (%4d, % .3f) %s control node: %s]",
		g.InnovationNum, g.MutationNum, enabl_str, g.ControlNode.String())
}