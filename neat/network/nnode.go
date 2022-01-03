package network

import (
	"bytes"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/math"
)

// NNode is either a NEURON or a SENSOR.
//   - If it's a sensor, it can be loaded with a value for output
//   - If it's a neuron, it has a list of its incoming input signals ([]*Link is used)
// Use an activation count to avoid flushing
type NNode struct {
	// The ID of the node
	Id int

	// The type of node activation function (SIGMOID, ...)
	ActivationType math.NodeActivationType
	// The neuron type for this node (HIDDEN, INPUT, OUTPUT, BIAS)
	NeuronType NodeNeuronType

	// The node's activation value
	Activation float64
	// The number of activations for current node
	ActivationsCount int32
	// The activation sum
	ActivationSum float64

	// The list of all incoming connections
	Incoming []*Link
	// The list of all outgoing connections
	Outgoing []*Link
	// The trait linked to the node
	Trait *neat.Trait

	// Used for Gene decoding by referencing analogue to this node in organism phenotype
	PhenotypeAnalogue *NNode

	// the flag to use for loop detection
	visited bool

	/* ************ LEARNING PARAMETERS *********** */
	// The following parameters are for use in neurons that learn through habituation,
	// sensitization, or Hebbian-type processes  */
	Params []float64

	// Activation value of node at time t-1; Holds the previous step's activation for recurrency
	lastActivation float64
	// Activation value of node at time t-2 Holds the activation before  the previous step's
	// This is necessary for a special recurrent case when the innode of a recurrent link is one time step ahead of the outnode.
	// The innode then needs to send from TWO time steps ago
	lastActivation2 float64

	// If true the node is active - used during node activation
	isActive bool
}

// NewNNode Creates new node with specified ID and neuron type associated (INPUT, HIDDEN, OUTPUT, BIAS)
func NewNNode(nodeId int, neuronType NodeNeuronType) *NNode {
	n := NewNetworkNode()
	n.Id = nodeId
	n.NeuronType = neuronType
	return n
}

// NewNNodeCopy Construct a NNode off another NNode with given trait for genome purposes
func NewNNodeCopy(n *NNode, t *neat.Trait) *NNode {
	node := NewNetworkNode()
	node.Id = n.Id
	node.NeuronType = n.NeuronType
	node.ActivationType = n.ActivationType
	node.Trait = t
	return node
}

// NewNetworkNode The default constructor
func NewNetworkNode() *NNode {
	return &NNode{
		NeuronType:     HiddenNeuron,
		ActivationType: math.SigmoidSteepenedActivation,
		Incoming:       make([]*Link, 0),
		Outgoing:       make([]*Link, 0),
	}
}

// Set new activation value to this node
func (n *NNode) setActivation(input float64) {
	// Keep a memory of activations for potential time delayed connections
	n.saveActivations()
	// Set new activation value
	n.Activation = input
	// Increment the activation_count
	n.ActivationsCount++
}

// Saves current node's activations for potential time delayed connections
func (n *NNode) saveActivations() {
	n.lastActivation2 = n.lastActivation
	n.lastActivation = n.Activation
}

// GetActiveOut Returns activation for a current step
func (n *NNode) GetActiveOut() float64 {
	if n.ActivationsCount > 0 {
		return n.Activation
	} else {
		return 0.0
	}
}

// GetActiveOutTd Returns activation from PREVIOUS time step
func (n *NNode) GetActiveOutTd() float64 {
	if n.ActivationsCount > 1 {
		return n.lastActivation
	} else {
		return 0.0
	}
}

// IsSensor Returns true if this node is SENSOR
func (n *NNode) IsSensor() bool {
	return n.NeuronType == InputNeuron || n.NeuronType == BiasNeuron
}

// IsNeuron returns true if this node is NEURON
func (n *NNode) IsNeuron() bool {
	return n.NeuronType == HiddenNeuron || n.NeuronType == OutputNeuron
}

// SensorLoad If the node is a SENSOR, returns TRUE and loads the value
func (n *NNode) SensorLoad(load float64) bool {
	if n.IsSensor() {
		// Keep a memory of activations for potential time delayed connections
		n.saveActivations()
		// Puts sensor into next time-step
		n.ActivationsCount++
		n.Activation = load
		return true
	} else {
		return false
	}
}

// AddOutgoing adds a non-recurrent outgoing link to this node. You should use this with caution because this doesn't
// create full duplex link needed for proper network activation.
// This method only intended for linking the control nodes. For all other needs use ConnectFrom which properly creates
// all needed links.
func (n *NNode) AddOutgoing(out *NNode, weight float64) *Link {
	newLink := NewLink(weight, n, out, false)
	n.Outgoing = append(n.Outgoing, newLink)
	return newLink
}

// AddIncoming adds a non-recurrent incoming link to this node. You should use this with caution because this doesn't
// create full duplex link needed for proper network activation.
// This method only intended for linking the control nodes. For all other needs use ConnectFrom which properly creates
// all needed links.
func (n *NNode) AddIncoming(in *NNode, weight float64) *Link {
	newLink := NewLink(weight, in, n, false)
	n.Incoming = append(n.Incoming, newLink)
	return newLink
}

// ConnectFrom is to create link between two nodes. The incoming links of current node and outgoing links of in node
// would be updated to have reference to the new link.
func (n *NNode) ConnectFrom(in *NNode, weight float64) *Link {
	newLink := NewLink(weight, in, n, false)
	n.Incoming = append(n.Incoming, newLink)
	in.Outgoing = append(in.Outgoing, newLink)
	return newLink
}

// Flushback Recursively deactivate backwards through the network
func (n *NNode) Flushback() {
	n.ActivationsCount = 0
	n.Activation = 0
	n.lastActivation = 0
	n.lastActivation2 = 0
	n.isActive = false
	n.visited = false
}

// FlushbackCheck is to verify flushing for debugging
func (n *NNode) FlushbackCheck() error {
	if n.ActivationsCount > 0 {
		return fmt.Errorf("NNODE: %s has activation count %d", n, n.ActivationsCount)
	}
	if n.Activation > 0 {
		return fmt.Errorf("NNODE: %s has activation %f", n, n.Activation)
	}
	if n.lastActivation > 0 {
		return fmt.Errorf("NNODE: %s has last_activation %f", n, n.lastActivation)
	}
	if n.lastActivation2 > 0 {
		return fmt.Errorf("NNODE: %s has last_activation2 %f", n, n.lastActivation2)
	}
	return nil
}

// Depth Find the greatest depth starting from this neuron at depth d. If maxDepth > 0 it can be used to stop early in
// case if very deep network detected
func (n *NNode) Depth(d int, maxDepth int) (int, error) {
	if maxDepth > 0 && d > maxDepth {
		// to avoid very deep network traversing
		return maxDepth, ErrMaximalNetDepthExceeded
	}
	n.visited = true
	// Base Case
	if n.IsSensor() {
		return d, nil
	} else {
		// recursion
		max := d // The max depth
		for _, l := range n.Incoming {
			if l.InNode.visited {
				// was already visited (loop detected) - skipping
				continue
			}
			curDepth, err := l.InNode.Depth(d+1, maxDepth)
			if err != nil {
				return curDepth, err
			}
			if curDepth > max {
				max = curDepth
			}
		}
		return max, nil
	}

}

// NodeType Convenient method to check network's node type (SENSOR, NEURON)
func (n *NNode) NodeType() NodeType {
	if n.IsSensor() {
		return SensorNode
	}
	return NeuronNode
}

func (n *NNode) String() string {
	activation, _ := math.NodeActivators.ActivationNameFromType(n.ActivationType)
	active := "active"
	if !n.isActive {
		active = "inactive"
	}
	return fmt.Sprintf("(%s id:%03d, %s, %s,\t%s -> step: %d = %.3f %.3f)",
		NodeTypeName(n.NodeType()), n.Id, NeuronTypeName(n.NeuronType), activation, active,
		n.ActivationsCount, n.Activation, n.Params)
}

// PrintDebug is to print all fields of the node to the string
func (n *NNode) PrintDebug() string {
	str := "NNode fields\n"
	b := bytes.NewBufferString(str)
	_, _ = fmt.Fprintf(b, "\tId: %d\n", n.Id)
	_, _ = fmt.Fprintf(b, "\tIsActive: %t\n", n.isActive)
	_, _ = fmt.Fprintf(b, "\tActivation: %f\n", n.Activation)
	activation, _ := math.NodeActivators.ActivationNameFromType(n.ActivationType)
	_, _ = fmt.Fprintf(b, "\tActivation Type: %s\n", activation)
	_, _ = fmt.Fprintf(b, "\tNeuronType: %d\n", n.NeuronType)
	_, _ = fmt.Fprintf(b, "\tActivationsCount: %d\n", n.ActivationsCount)
	_, _ = fmt.Fprintf(b, "\tActivationSum: %f\n", n.ActivationSum)
	_, _ = fmt.Fprintf(b, "\tIncoming: %s\n", n.Incoming)
	_, _ = fmt.Fprintf(b, "\tOutgoing: %s\n", n.Outgoing)
	_, _ = fmt.Fprintf(b, "\tTrait: %s\n", n.Trait)
	_, _ = fmt.Fprintf(b, "\tPhenotypeAnalogue: %s\n", n.PhenotypeAnalogue)
	_, _ = fmt.Fprintf(b, "\tParams: %f\n", n.Params)
	_, _ = fmt.Fprintf(b, "\tlastActivation: %f\n", n.lastActivation)
	_, _ = fmt.Fprintf(b, "\tlastActivation2: %f\n", n.lastActivation2)

	return b.String()
}
