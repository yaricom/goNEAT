package network

import (
	"io"
	"fmt"
	"errors"
	"github.com/yaricom/goNEAT/neat"
)

// A NODE is either a NEURON or a SENSOR.
//   - If it's a sensor, it can be loaded with a value for output
//   - If it's a neuron, it has a list of its incoming input signals ([]*Link is used)
// Use an activation count to avoid flushing
type NNode struct {
	// The ID of the node
	Id               int

	// If true the node is active
	IsActive         bool

	// The the type of the node (NEURON or SENSOR)
	NodeType         NodeType
	// The type of node activation function (SIGMOID, ...)
	ActivationType   ActivationType
	// The neuron type for this node (HIDDEN, INPUT, OUTPUT, BIAS)
	NeuronType       NeuronType

	// The activation for current step
	ActiveOut        float64
	// The activation from PREVIOUS (time-delayed) time step, if there is one
	ActiveOutTd      float64
	// The node's activation value
	Activation       float64
	// The number of activations for current node
	ActivationsCount int32
	// The activation sum
	ActivationSum    float64

	// The list of all incoming connections
	Incoming         []*Link
	// The list of all outgoing connections
	Outgoing         []*Link
	// The trait linked to the node
	Trait            *neat.Trait

	// Used for Gene decoding
	Analogue         *NNode
	// Used for Genome duplication
	Duplicate        *NNode

	/* ************ LEARNING PARAMETERS *********** */
	// The following parameters are for use in neurons that learn through habituation,
	// sensitization, or Hebbian-type processes  */
	Params           []float64

	// Activation value of node at time t-1; Holds the previous step's activation for recurrency
	lastActivation   float64
	// Activation value of node at time t-2 Holds the activation before  the previous step's
	// This is necessary for a special recurrent case when the innode of a recurrent link is one time step ahead of the outnode.
	// The innode then needs to send from TWO time steps ago
	lastActivation2  float64
}

// Creates new node with specified type (NEURON or SENSOR) and ID
func NewNNode(ntype NodeType, nodeid int) *NNode {
	n := newNode(ntype)
	n.Id = nodeid
	return n
}

// Creates new node with specified type (NEURON or SENSOR), ID and in the specified
// layer (INPUT, HIDDEN, OUTPUT)
func NewNNodeInPlace(ntype NodeType, nodeid int, neuronType NeuronType) *NNode {
	n := newNode(ntype)
	n.Id = nodeid
	n.NeuronType = neuronType
	return n
}

// Construct a NNode off another NNode with given trait for genome purposes
func NewNNodeCopy(n *NNode, t *neat.Trait) *NNode {
	node := newNode(n.NodeType)
	node.Id = n.Id
	node.NeuronType = n.NeuronType
	node.Trait = t
	node.deriveTrait(t)
	return node
}

// Read a NNode from specified Reader and applies corresponding trait to it from a list of traits provided
func ReadNNode(r io.Reader, traits []*neat.Trait) *NNode {
	n := newNode(NEURON)
	var trait_id int
	fmt.Fscanf(r, "%d %d %d %d ", &n.Id, &trait_id, &n.NodeType, &n.NeuronType)
	if trait_id != 0 && traits != nil {
		// find corresponding node trait from list
		for _, t := range traits {
			if trait_id == t.Id {
				n.Trait = t
				n.deriveTrait(t)
				break
			}
		}
	} else {
		// just create empty params
		n.deriveTrait(nil)
	}
	return n
}

// The private default constructor
func newNode(ntype NodeType) *NNode {
	return &NNode{
		NodeType:ntype,
		NeuronType:HIDDEN,
		ActivationType:SigmoidSteepened,
		Incoming:make([]*Link, 0),
		Outgoing:make([]*Link, 0),
	}
}

// Copy trait parameters into this node's parameters
func (n *NNode) deriveTrait(t *neat.Trait) {
	n.Params = make([]float64, neat.Num_trait_params)
	if t != nil {
		for i, p := range t.Params {
			n.Params[i] = p
		}
	}
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
	return n.NodeType == SENSOR
}

// returns true if this node is NEURON
func (n *NNode) IsNeuron() bool {
	return n.NodeType == NEURON
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

// Adds a NONRECURRENT Link to an incoming NNode in the incoming List
func (n *NNode) AddIncoming(in *NNode, weight float64) {
	newLink := NewLink(weight, in, n, false)
	n.Incoming = append(n.Incoming, newLink)
}

// Adds a Link to a new NNode in the incoming List
func (n *NNode) AddIncomingRecurrent(in *NNode, weight float64, recur bool) {
	newLink := NewLink(weight, in, n, recur)
	n.Incoming = append(n.Incoming, newLink)
}

// Recursively deactivate backwards through the network
func (n *NNode) Flushback() {
	n.ActivationsCount = 0
	n.Activation = 0
	n.lastActivation = 0
	n.lastActivation2 = 0
}

// Verify flushing for debuginh
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

// Dump node to a writer
func (n *NNode) Write(w io.Writer) {
	trait_id := 0
	if n.Trait != nil {
		trait_id = n.Trait.Id
	}
	fmt.Fprintf(w, "%d %d %d %d", n.Id, trait_id, n.NodeType, n.NeuronType)
}

// Find the greatest depth starting from this neuron at depth d
func (n *NNode) Depth(d int) (int, error) {
	if d > 100 {
		return 10, nil//errors.New("NNODE: DEPTH NOT DETERMINED FOR NETWORK WITH LOOP");
	}
	// Base Case
	if n.IsSensor() {
		return d, nil
	} else {
		// recursion
		max := d // The max depth
		for _, l := range n.Incoming {
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

func (n *NNode) String() string {
	return fmt.Sprintf("(%s %3d, layer: %s, step: %d = %.3f %.3f)",
		NodeTypeName(n.NodeType), n.Id, NeuronTypeName(n.NeuronType), n.ActivationsCount, n.Activation, n.Params)
}




