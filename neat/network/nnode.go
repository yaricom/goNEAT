package network

import (
	"github.com/yaricom/goNEAT/neat"
	"io"
	"fmt"
	"errors"
	"math"
)

// A NODE is either a NEURON or a SENSOR.
//   - If it's a sensor, it can be loaded with a value for output
//   - If it's a neuron, it has a list of its incoming input signals ([]*Link is used)
// Use an activation count to avoid flushing
type NNode interface {

	// Return the ID of the node
	NodeId() int
	// Returns the placement of the node in the network layers (INPUT, HIDDEN, OUTPUT)
	GenNodeLabel() int
	// Sets new trait to the node
	SetTrait(t Trait)
	// Returns trait associated with node
	GetTrait() Trait
	// Returns number of activations for current node
	ActivationCount() int32
	// Increments activation count
	IncrementActivationCount()

	// Sets activation sum
	SetActiveSum(sum float64)
	// Returns activation sum
	GetActiveSum() float64
	// Adds activation value to the total activation sum of this node
	AddToActiveSum(value float64)
	// Set node active flag
	SetActiveFlag(flag bool)
	// Returns true if this link is active
	IsActive() bool

	// Save current node's activations for potential time delayed connections
	SaveActivations()
	// Set node's activation value
	SetActivation(val float64)
	// Returns type of node activation function (SIGMOID, ...)
	GetFtype() int


	// Return activation for current step
	GetActiveOut() float64
	// Return activation from PREVIOUS (time-delayed) time step, if there is one
	GetActiveOutTd() float64

	// Returns the type of the node (NEURON or SENSOR)
	GetType() int
	// Allows alteration between NEURON and SENSOR.  Returns its argument
	SetType(ntype int)
	// Returns true if this node is SENSOR
	IsSensor() bool
	// Returns true if this node is NEURON
	IsNeuron() bool

	// If the node is a SENSOR, returns TRUE and loads the value
	SensorLoad(load float64) bool

	// Adds a NONRECURRENT Link to a new NNode with specified weight in the incoming List
	AddIncoming(in NNode, weight float64)
	// Adds a Link to a new NNode in the incoming List
	AddIncomingRecurrent(in NNode, weight float64, recur bool);
	// Returns list of all incoming connections
	GetIncoming() []Link

	// Recursively deactivate backwards through the network including this NNode and reccurencies
	Flushback()

	// Write this node into writer
	WriteNode(w io.Writer)

	// Find the greatest depth starting from this neuron at depth d
	Depth(d int32) (int32, error)

	// Verify flushing for debug
	FlushbackCheck() error

}

// SIGMOID FUNCTION ********************************
// This is a signmoidal activation function, which is an S-shaped squashing function.
// It smoothly limits the amplitude of the output of a neuron to between 0 and 1.
// It is a helper to the neural-activation function get_active_out.
// It is made inline so it can execute quickly since it is at every non-sensor node in a network.
// NOTE:  In order to make node insertion in the middle of a link possible,
// the signmoid can be shifted to the right and more steeply sloped:
// slope=4.924273
// constant= 2.4621365
// These parameters optimize mean squared error between the old output,
// and an output of a node inserted in the middle of a link between
// the old output and some other node.
// When not right-shifted, the steepened slope is closest to a linear
// ascent as possible between -0.5 and 0.5
func fsigmoid(activesum, slope, constant float64) float64 {
	//RIGHT SHIFTED ---------------------------------------------------------
	//return (1/(1+(exp(-(slope*activesum-constant))))); //ave 3213 clean on 40 runs of p2m and 3468 on another 40
	//41394 with 1 failure on 8 runs

	//LEFT SHIFTED ----------------------------------------------------------
	//return (1/(1+(exp(-(slope*activesum+constant))))); //original setting ave 3423 on 40 runs of p2m, 3729 and 1 failure also

	//PLAIN SIGMOID ---------------------------------------------------------
	//return (1/(1+(exp(-activesum)))); //3511 and 1 failure

	//LEFT SHIFTED NON-STEEPENED---------------------------------------------
	//return (1/(1+(exp(-activesum-constant)))); //simple left shifted

	//NON-SHIFTED STEEPENED
	return 1.0 / (1.0 + (math.Exp(-(slope * activesum)))) //Compressed
}

// Creates new node with specified type (NEURON or SENSOR) and ID
func NewNNode(ntype, nodeid int) NNode {
	n := newNode()
	n.ntype = ntype
	n.node_id = nodeid
	return &n
}

// Creates new node with specified type (NEURON or SENSOR), ID and in the specified
// layer (INPUT, HIDDEN, OUTPUT)
func NewNNodeInPlace(ntype, nodeid, placement int) NNode {
	n := newNode()
	n.ntype = ntype
	n.node_id = nodeid
	n.gen_node_label = placement
	return &n
}

// Construct a NNode off another NNode with given trait for genome purposes
func NewNNodeCopy(n NNode, t Trait) NNode {
	node := newNode()
	node.ntype = n.GetType()
	node.node_id = n.NodeId()
	node.gen_node_label = n.GenNodeLabel()
	node.SetTrait(t)
	return &node
}

// Read a NNode from specified Reader (r) and applies corresponding trait to it from a list of traits provided
func ReadNNode(r io.Reader, traits []Trait) NNode {
	n := newNode()
	var trait_id int
	fmt.Fscanf(r, "%d %d %d %d", &n.node_id, &trait_id, &n.ntype, &n.gen_node_label)
	if trait_id != 0 && traits != nil {
		// find corresponding node trait from list
		for _, t := range traits {
			if trait_id == t.TraitId() {
				n.nodetrait = t
				break
			}
		}
	}
	return &n
}


// private structure to hold values
type nnode struct {
	// The activation function type is either SIGMOID ..or others that can be added
	ftype int
	// The NN node type is either NEURON or SENSOR
	ntype int
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
	nodetrait Trait

	// Is a reference to a Node; It's used to generate and point from a genetic node (genotype)
	// to a real node (fenotype) during 'genesis' process (Gene decoding)
	analogue NNode
	// Is a  temporary reference to a Node; It's used to generate a new genome during duplicate phase of genotype.
	dup NNode

	// A list of pointers to incoming weighted signals from other nodes
	incoming []Link
	// A list of pointers to links carrying this node's signal
	outgoing []Link

	// A node can be given an identification number for saving in files
	node_id int
	// Used for genetic marking of nodes
	gen_node_label int
}

// The private default constructor
func newNode() nnode {
	return nnode{
		ftype:SIGMOID,
		params:make([]float64, neat.Num_trait_params),
		incoming:make([]Link, 0),
		outgoing:make([]Link, 0),
		gen_node_label:HIDDEN,
	}
}

// The NNode interface implementation
func (n *nnode) ActivationCount() int32 {
	return n.activation_count
}
func (n *nnode) IncrementActivationCount() {
	n.activation_count += 1
}
func (n *nnode) SetActiveSum(sum float64) {
	n.activesum = sum
}
func (n *nnode) GetActiveSum() float64 {
	return n.activesum
}
func (n *nnode) SaveActivations() {
	n.last_activation2 = n.last_activation
	n.last_activation = n.activation
}
func (n *nnode) SetActivation(val float64) {
	n.activation = val
}
func (n nnode) GetFtype() int {
	return n.ftype
}
func (n *nnode) AddToActiveSum(value float64) {
	n.activesum += value
}
func (n *nnode) SetActiveFlag(flag bool) {
	n.active_flag = flag
}
func (n *nnode) IsActive() bool {
	return n.active_flag
}
func (n *nnode) NodeId() int  {
	return n.node_id
}
func (n *nnode) GenNodeLabel() int  {
	return n.gen_node_label
}
func (n *nnode) SetTrait(t Trait) {
	n.nodetrait = t
}
func (n *nnode) GetTrait() Trait {
	return n.nodetrait
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
func (n *nnode) GetType() int {
	return n.ntype
}
func (n *nnode) SetType(ntype int) {
	n.ntype = ntype
}
func (n *nnode) IsSensor() bool {
	return n.ntype == SENSOR
}
func (n *nnode) IsNeuron() bool {
	return n.ntype == NEURON
}
func (n *nnode) SensorLoad(load float64) bool {
	if n.IsSensor() {
		// Keep a memory of activations for potential time delayed connections
		n.SaveActivations()
		// Puts sensor into next time-step
		n.IncrementActivationCount()
		n.activation = load
		return true
	} else {
		return false
	}
}
func (n *nnode) AddIncoming(in NNode, weight float64) {
	newLink := NewLink(weight, in, n, false)
	n.incoming = append(n.incoming, newLink)
}
func (n *nnode) AddIncomingRecurrent(in NNode, weight float64, recur bool) {
	newLink := NewLink(weight, in, n, recur)
	n.incoming = append(n.incoming, newLink)
}
func (n *nnode) GetIncoming() []Link {
	return n.incoming
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
func (n *nnode) WriteNode(w io.Writer) {
	trait_id := 0
	if n.nodetrait != nil {
		trait_id = n.nodetrait.TraitId()
	}
	fmt.Fprintf(w, "%d %d %d %d", n.node_id, trait_id, n.ntype, n.gen_node_label)
}
func (n *nnode) Depth(d int32) (int32, error) {
	if d > 100 {
		return 10, errors.New("** DEPTH NOT DETERMINED FOR NETWORK WITH LOOP");
	}
	// Base Case
	if n.ntype == SENSOR {
		return d, nil
	} else {
		// recursion
		max := d // The max depth
		for _, l := range n.incoming {
			cur_depth, err := l.InNode().Depth(d + 1)
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

func (n *nnode) String() string {
	if n.ntype == SENSOR {
		return fmt.Sprintf("(S %d, step %d : %f)", n.node_id, n.activation_count, n.activation)
	} else {
		return fmt.Sprintf("(N %d, step %d : %f)", n.node_id, n.activation_count, n.activation)
	}
}




