package network

import (
	"github.com/yaricom/goNEAT/neat"
	"io"
	"fmt"
	"errors"
)

// A NODE is either a NEURON or a SENSOR.
//   - If it's a sensor, it can be loaded with a value for output
//   - If it's a neuron, it has a list of its incoming input signals ([]*Link is used)
// Use an activation count to avoid flushing
type NNode interface {

	// Return the ID of the node
	NodeId() int32
	// Returns the placement of the node in the network layers (INPUT, HIDDEN, OUTPUT)
	GenNodeLabel() int32
	// Sets new trait to the node
	SetTrait(t *Trait)
	// Returns number of activations for current node
	ActivationCount() int32


	// Return activation for current step
	GetActiveOut() float64
	// Return activation from PREVIOUS (time-delayed) time step, if there is one
	GetActiveOutTd() float64

	// Returns the type of the node (NEURON or SENSOR)
	GetType() int32
	// Allows alteration between NEURON and SENSOR.  Returns its argument
	SetType(ntype int32)

	// If the node is a SENSOR, returns TRUE and loads the value
	SensorLoad(load float64) bool

	// Adds a NONRECURRENT Link to a new NNode with specified weight in the incoming List
	AddIncoming(in *NNode, weight float64)
	// Adds a Link to a new NNode in the incoming List
	AddIncomingRecurrent(in *NNode, weight float64, recur bool);

	// Recursively deactivate backwards through the network including this NNode and reccurencies
	Flushback()

	// Write this node into writer
	WriteNode(w *io.Writer)

	// Find the greatest depth starting from this neuron at depth d
	Depth(d int32, mynet *Network) int32

	// Verify flushing for debug
	FlushbackCheck() error

}

// Creates new node with specified type (NEURON or SENSOR) and ID
func NewNNode(ntype, nodeid int) NNode {
	n := newNode()
	n.ntype = ntype
	n.node_id = nodeid
	return n
}

// Creates new node with specified type (NEURON or SENSOR), ID and in the specified
// layer (INPUT, HIDDEN, OUTPUT)
func NewNNodeInPlace(ntype, nodeid, placement int) NNode {
	n := newNode()
	n.ntype = ntype
	n.node_id = nodeid
	n.gen_node_label = placement
	return n
}

// Construct a NNode off another NNode with given trait for genome purposes
func NewNNodeCopy(n *NNode, t *Trait) NNode {
	node := newNode()
	node.ntype = (*n).GetType()
	node.node_id = (*n).NodeId()
	node.gen_node_label = (*n).GenNodeLabel()
	node.SetTrait(t)
	return node
}

// Read a NNode from specified Reader (r) and applies corresponding trait to it from a list of traits provided
func ReadNNode(r *io.Reader, traits []*Trait) {
	n := newNode()
	var trait_id int32
	fmt.Fscanf(r, "%d %d %d %d", &n.node_id, &trait_id, &n.ntype, &n.gen_node_label)
	if trait_id != 0 && traits != nil {
		// find corresponding node trait from list
		for _, t := range traits {
			if trait_id == (*t).TraitId() {
				n.nodetrait = t
				break
			}
		}
	}
	return n
}


// private structure to hold values
type nnode struct {
	// The activation function type is either SIGMOID ..or others that can be added
	ftype int32
	// The NN node type is either NEURON or SENSOR
	ntype int32
	// The incoming activity before being processed
	activesum float64
	// The total activation entering the NNode
	activation float64
	// To make sure outputs are active (allows to disable this node)
	active_flag bool

	// The following parameters are for use in neurons that learn through habituation,
	// sensitization, or Hebbian-type processes
	params []float64

	// Keeps track of which activation the node is currently in
	activation_count int32
	// Activation value of node at time t-1; Holds the previous step's activation for recurrency
	last_activation float64
	// Activation value of node at time t-2 Holds the activation before  the previous step's
	// This is necessary for a special recurrent case when the innode of a recurrent link is one time step ahead of the outnode.
	// The innode then needs to send from TWO time steps ago
	last_activation2 float64

	//Points to a trait of parameters
	nodetrait *Trait

	// Is a reference to a Node; It's used to generate and point from a genetic node (genotype)
	// to a real node (fenotype) during 'genesis' process (Gene decoding)
	analogue *NNode
	// Is a  temporary reference to a Node; It's used to generate a new genome during duplicate phase of genotype.
	dup *NNode

	// A list of pointers to incoming weighted signals from other nodes
	incoming []*Link
	// A list of pointers to links carrying this node's signal
	outgoing []*Link

	// A node can be given an identification number for saving in files
	node_id int32
	// Used for genetic marking of nodes
	gen_node_label int32
}

// The private default constructor
func newNode() nnode {
	return nnode{
		ftype:SIGMOID,
		params:make([]float64, neat.Num_trait_params),
		incoming:make([]*Link, 0),
		outgoing:make([]*Link, 0),
		gen_node_label:HIDDEN,
	}
}

// The NNode interface implementation
func (n *nnode) ActivationCount() {
	return n.activation_count
}
func (n *nnode) NodeType() int32 {
	return n.ntype
}
func (n *nnode) NodeId() int32  {
	return n.node_id
}
func (n *nnode) GenNodeLabel() int32  {
	return n.gen_node_label
}
func (n *nnode) SetTrait(t *Trait) {
	n.nodetrait = t
}
func (n *nnode) GetActiveOut() float64 {
	if n.activation_count > 0 {
		return n.activation
	} else {
		return 0.0
	}
}
func (n *nnode) GetActiveOutTd() float64 {
	if n.activation_count > 1 {
		return n.last_activation
	} else {
		return 0.0
	}
}
func (n *nnode) GetType() int32 {
	return n.ntype
}
func (n *nnode) SetType(ntype int32) {
	n.ntype = ntype
}
func (n *nnode) SensorLoad(load float64) bool {
	if n.ntype == SENSOR {
		// Time delay memory
		n.last_activation2 = n.last_activation
		n.last_activation = n.activation
		// Puts sensor into next time-step
		n.activation_count += 1
		n.activation = load
		return true
	} else {
		return false
	}
}
func (n *nnode) AddIncoming(in *NNode, weight float64) {
	newLink := NewLink(weight, in, n, false)
	n.incoming = append(n.incoming, newLink)
}
func (n *nnode) AddIncomingRecurrent(in *NNode, weight float64, recur bool) {
	newLink := NewLink(weight, in, n, recur)
	n.incoming = append(n.incoming, newLink)
}
func (n *nnode) Flushback() {
	n.activation_count = 0
	n.activation = 0
	n.last_activation = 0
	n.last_activation2 = 0

	//if n.ntype == NEURON {
	//	// Flush back recursively
	//	for _, l := range n.incoming {
	//		(*l).SetAddedWeight(0)
	//		if (*l).InNode().ActivationCount() > 0 {
	//			(*l).InNode().Flushback()
	//		}
	//	}
	//}
}
func (n *nnode) FlushbackCheck() error {
	if n.activation_count > 0 {
		return errors.New(fmt.Sprintf("ALERT: %s has activation count %d", n, n.activation_count))
	}
	if n.activation > 0 {
		return errors.New(fmt.Sprintf("ALERT: %s has activation %f", n, n.activation))
	}
	if n.last_activation > 0 {
		return errors.New(fmt.Sprintf("ALERT: %s has last_activation %f", n, n.last_activation))
	}
	if n.last_activation2 > 0 {
		return errors.New(fmt.Sprintf("ALERT: %s has last_activation2 %f", n, n.last_activation2))
	}

	//if n.ntype == NEURON {
	//	// Flush back check recursively
	//	for _, l := range n.incoming {
	//		err := (*l).InNode().FlushbackCheck()
	//		if err != nil {
	//			return err
	//		}
	//	}
	//
	//}
	return nil
}
func (n *nnode) WriteNode(w *io.Writer) {
	trait_id := 0
	if n.nodetrait != nil {
		trait_id = (*n.nodetrait).TraitId()
	}
	fmt.Fprintf(w, "%d %d %d %d", n.node_id, trait_id, n.ntype, n.gen_node_label)
}
func (n *nnode) Depth(d int32, mynet *Network) int32 {
	cur_depth := 0 //The depth of the current node
	max := d //The max depth

	if d > 100 {
		fmt.Println("** DEPTH NOT DETERMINED FOR NETWORK WITH LOOP")
		return 10;
	}
	// Base Case
	if n.ntype == SENSOR {
		return d
	} else {
		// recursion
		for _, l := range n.incoming {
			cur_depth = (*l).InNode().Depth(d + 1, mynet)
			if cur_depth > max {
				max = cur_depth
			}
		}
		return max
	}

}

func (n nnode) String() string {
	if n.ntype == SENSOR {
		return fmt.Sprintf("(S %d, step %d : %f)", n.node_id, n.activation_count, n.activation)
	} else {
		return fmt.Sprintf("(N %d, step %d : %f)", n.node_id, n.activation_count, n.activation)
	}
}




