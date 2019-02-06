package network

import (
	"fmt"
	"errors"
	"bytes"
	"github.com/yaricom/goNEAT/neat"
	"github.com/yaricom/goNEAT/neat/utils"
)

// A NODE is either a NEURON or a SENSOR.
//   - If it's a sensor, it can be loaded with a value for output
//   - If it's a neuron, it has a list of its incoming input signals ([]*Link is used)
// Use an activation count to avoid flushing
type NNode struct {
	// The ID of the node
	Id                int

	// The type of node activation function (SIGMOID, ...)
	ActivationType    utils.NodeActivationType
	// The neuron type for this node (HIDDEN, INPUT, OUTPUT, BIAS)
	NeuronType        NodeNeuronType

	// The node's activation value
	Activation        float64
	// The number of activations for current node
	ActivationsCount  int32
	// The activation sum
	ActivationSum     float64

	// The list of all incoming connections
	Incoming          []*Link
	// The list of all outgoing connections
	Outgoing          []*Link
	// The trait linked to the node
	Trait             *neat.Trait

	// Used for Gene decoding by referencing analogue to this node in organism phenotype
	PhenotypeAnalogue *NNode

	// the flag to use for loop detection
	visited           bool

	/* ************ LEARNING PARAMETERS *********** */
	// The following parameters are for use in neurons that learn through habituation,
	// sensitization, or Hebbian-type processes  */
	Params            []float64

	// Activation value of node at time t-1; Holds the previous step's activation for recurrency
	lastActivation    float64
	// Activation value of node at time t-2 Holds the activation before  the previous step's
	// This is necessary for a special recurrent case when the innode of a recurrent link is one time step ahead of the outnode.
	// The innode then needs to send from TWO time steps ago
	lastActivation2   float64

	// If true the node is active - used during node activation
	isActive          bool
}

// Creates new node with specified ID and neuron type associated (INPUT, HIDDEN, OUTPUT, BIAS)
func NewNNode(nodeid int, neuronType NodeNeuronType) *NNode {
	n := NewNetworkNode()
	n.Id = nodeid
	n.NeuronType = neuronType
	return n
}

// Construct a NNode off another NNode with given trait for genome purposes
func NewNNodeCopy(n *NNode, t *neat.Trait) *NNode {
	node := NewNetworkNode()
	node.Id = n.Id
	node.NeuronType = n.NeuronType
	node.ActivationType = n.ActivationType
	node.Trait = t
	return node
}

// The default constructor
func NewNetworkNode() *NNode {
	return &NNode{
		NeuronType:HiddenNeuron,
		ActivationType:utils.SigmoidSteepenedActivation,
		Incoming:make([]*Link, 0),
		Outgoing:make([]*Link, 0),
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

// Returns activation for a current step
func (n *NNode) GetActiveOut() float64 {
	if n.ActivationsCount > 0 {
		return n.Activation
	} else {
		return 0.0
	}
}

// Returns activation from PREVIOUS time step
func (n *NNode) GetActiveOutTd() float64 {
	if n.ActivationsCount > 1 {
		return n.lastActivation
	} else {
		return 0.0
	}
}

// Returns true if this node is SENSOR
func (n *NNode) IsSensor() bool {
	return n.NeuronType == InputNeuron || n.NeuronType == BiasNeuron
}

// returns true if this node is NEURON
func (n *NNode) IsNeuron() bool {
	return n.NeuronType == HiddenNeuron || n.NeuronType == OutputNeuron
}

// If the node is a SENSOR, returns TRUE and loads the value
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

// Adds a non recurrent outgoing link to this node
func (n *NNode) addOutgoing(out *NNode, weight float64) {
	newLink := NewLink(weight, n, out, false)
	n.Outgoing = append(n.Outgoing, newLink)
}

// Adds a NONRECURRENT Link to an incoming NNode in the incoming List
func (n *NNode) addIncoming(in *NNode, weight float64) {
	newLink := NewLink(weight, in, n, false)
	n.Incoming = append(n.Incoming, newLink)
}

// Recursively deactivate backwards through the network
func (n *NNode) Flushback() {
	n.ActivationsCount = 0
	n.Activation = 0
	n.lastActivation = 0
	n.lastActivation2 = 0
	n.isActive = false
	n.visited = false
}

// Verify flushing for debuging
func (n *NNode) FlushbackCheck() error {
	if n.ActivationsCount > 0 {
		return errors.New(fmt.Sprintf("NNODE: %s has activation count %d", n, n.ActivationsCount))
	}
	if n.Activation > 0 {
		return errors.New(fmt.Sprintf("NNODE: %s has activation %f", n, n.Activation))
	}
	if n.lastActivation > 0 {
		return errors.New(fmt.Sprintf("NNODE: %s has last_activation %f", n, n.lastActivation))
	}
	if n.lastActivation2 > 0 {
		return errors.New(fmt.Sprintf("NNODE: %s has last_activation2 %f", n, n.lastActivation2))
	}
	return nil
}

// Find the greatest depth starting from this neuron at depth d
func (n *NNode) Depth(d int) (int, error) {
	if d > 1000 {
		// to avoid infinite recursion
		return 10, NetErrDepthCalculationFailedLoopDetected
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
			cur_depth, err := l.InNode.Depth(d + 1)
			if err != nil {
				return cur_depth, err
			}
			if cur_depth > max {
				max = cur_depth
			}
		}
		return max, nil
	}

}

// Convenient method to check network's node type (SENSOR, NEURON)
func (n *NNode) NodeType() NodeType {
	if n.IsSensor() {
		return SensorNode
	}
	return NeuronNode
}

func (n *NNode) String() string {
	activation, _ := utils.NodeActivators.ActivationNameFromType(n.ActivationType)
	active := "active"
	if !n.isActive {
		active = "inactive"
	}
	return fmt.Sprintf("(%s id:%03d, %s, %s,\t%s -> step: %d = %.3f %.3f)",
		NodeTypeName(n.NodeType()), n.Id, NeuronTypeName(n.NeuronType), activation, active,
		n.ActivationsCount, n.Activation, n.Params)
}

// Prints all node's fields to the string
func (n *NNode) Print() string {
	str := "NNode fields\n"
	b := bytes.NewBufferString(str)
	fmt.Fprintf(b, "\tId: %d\n", n.Id)
	fmt.Fprintf(b, "\tIsActive: %t\n", n.isActive)
	fmt.Fprintf(b, "\tActivation: %f\n", n.Activation)
	activation, _ := utils.NodeActivators.ActivationNameFromType(n.ActivationType)
	fmt.Fprintf(b, "\tActivation Type: %s\n", activation)
	fmt.Fprintf(b, "\tNeuronType: %d\n", n.NeuronType)
	fmt.Fprintf(b, "\tActivationsCount: %d\n", n.ActivationsCount)
	fmt.Fprintf(b, "\tActivationSum: %f\n", n.ActivationSum)
	fmt.Fprintf(b, "\tIncoming: %s\n", n.Incoming)
	fmt.Fprintf(b, "\tOutgoing: %s\n", n.Outgoing)
	fmt.Fprintf(b, "\tTrait: %s\n", n.Trait)
	fmt.Fprintf(b, "\tPhenotypeAnalogue: %s\n", n.PhenotypeAnalogue)
	fmt.Fprintf(b, "\tParams: %f\n", n.Params)
	fmt.Fprintf(b, "\tlastActivation: %f\n", n.lastActivation)
	fmt.Fprintf(b, "\tlastActivation2: %f\n", n.lastActivation2)

	return b.String()
}


