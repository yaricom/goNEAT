package genetics

import "github.com/yaricom/goNEAT/neat/network"

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
}

// Creates new MIMO gene
func NewMIMOGene(control_node *network.NNode, innov_num int64, mut_num float64, enabled bool) *MIMOControlGene {
	gene := &MIMOControlGene{
		ControlNode:control_node,
		InnovationNum:innov_num,
		MutationNum:mut_num,
		IsEnabled:enabled,
	}
	return gene
}

// The copy constructor taking parameters from provided control gene for given control node
func NewMIMOGeneCopy(g *MIMOControlGene, control_node *network.NNode) *MIMOControlGene {
	cg := NewMIMOGene(control_node, g.InnovationNum, g.MutationNum, g.IsEnabled)
	return cg
}