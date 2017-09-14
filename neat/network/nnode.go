package network

import (
	"io"
	"fmt"
	"errors"
	"math"
)

// A NODE is either a NEURON or a SENSOR.
//   - If it's a sensor, it can be loaded with a value for output
//   - If it's a neuron, it has a list of its incoming input signals ([]*Link is used)
// Use an activation count to avoid flushing
type NNode struct {
	// The ID of the node
	NodeId int

	// If true the node is active
	IsActive bool

	// The the type of the node (NEURON or SENSOR)
	NType int
	// The type of node activation function (SIGMOID, ...)
	FType int
	// The placement of the node in the network layers (INPUT, HIDDEN, OUTPUT)
	GenNodeLabel int

	// The activation for current step
	ActiveOut float64
	// The activation from PREVIOUS (time-delayed) time step, if there is one
	ActiveOutTd float64
	// The node's activation value
	Activation float64
	// The number of activations for current node
	ActivationsCount int32
	// The activation sum
	ActivationSum float64

	// The list of all incoming connections
	Incoming []*Link
	// The trait linked to the node
	Trait *Trait

	// Activation value of node at time t-1; Holds the previous step's activation for recurrency
	lastActivation float64
	// Activation value of node at time t-2 Holds the activation before  the previous step's
	// This is necessary for a special recurrent case when the innode of a recurrent link is one time step ahead of the outnode.
	// The innode then needs to send from TWO time steps ago
	lastActivation2 float64
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
func NewNNode(ntype, nodeid int) *NNode {
	n := newNode()
	n.NType = ntype
	n.NodeId = nodeid
	return n
}

// Creates new node with specified type (NEURON or SENSOR), ID and in the specified
// layer (INPUT, HIDDEN, OUTPUT)
func NewNNodeInPlace(ntype, nodeid, placement int) *NNode {
	n := newNode()
	n.NType = ntype
	n.NodeId = nodeid
	n.GenNodeLabel = placement
	return n
}

// Construct a NNode off another NNode with given trait for genome purposes
func NewNNodeCopy(n NNode, t *Trait) *NNode {
	node := newNode()
	node.NType = n.NType
	node.NodeId = n.NodeId
	node.GenNodeLabel = n.GenNodeLabel
	node.Trait = t
	return node
}

// Read a NNode from specified Reader (r) and applies corresponding trait to it from a list of traits provided
func ReadNNode(r io.Reader, traits []*Trait) *NNode {
	n := newNode()
	var trait_id int
	fmt.Fscanf(r, "node %d %d %d %d", &n.NodeId, &trait_id, &n.NType, &n.GenNodeLabel)
	if trait_id != 0 && traits != nil {
		// find corresponding node trait from list
		for _, t := range traits {
			if trait_id == t.TraitId {
				n.Trait = t
				break
			}
		}
	}
	return n
}

// The private default constructor
func newNode() *NNode {
	return &NNode{
		FType:SIGMOID,
		Incoming:make([]*Link, 0),
		GenNodeLabel:HIDDEN,
	}
}

// The NNode methods implementation
func (n *NNode) SaveActivations() {
	n.lastActivation2 = n.lastActivation
	n.lastActivation = n.Activation
}

func (n *NNode) GetActiveOut() float64 {
	if n.ActivationsCount > 0 {
		return n.Activation
	} else {
		return 0.0
	}
}
func (n *NNode) GetActiveOutTd() float64 {
	if n.ActivationsCount > 1 {
		return n.lastActivation
	} else {
		return 0.0
	}
}
func (n *NNode) IsSensor() bool {
	return n.NType == SENSOR
}
func (n *NNode) IsNeuron() bool {
	return n.NType == NEURON
}
func (n *NNode) SensorLoad(load float64) bool {
	if n.IsSensor() {
		// Keep a memory of activations for potential time delayed connections
		n.SaveActivations()
		// Puts sensor into next time-step
		n.ActivationsCount++
		n.Activation = load
		return true
	} else {
		return false
	}
}
func (n *NNode) AddIncoming(in *NNode, weight float64) {
	newLink := NewLink(weight, in, n, false)
	n.Incoming = append(n.Incoming, newLink)
}
func (n *NNode) AddIncomingRecurrent(in *NNode, weight float64, recur bool) {
	newLink := NewLink(weight, in, n, recur)
	n.Incoming = append(n.Incoming, newLink)
}
func (n *NNode) Flushback() {
	n.ActivationsCount = 0
	n.Activation = 0
	n.lastActivation = 0
	n.lastActivation2 = 0

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
func (n *NNode) FlushbackCheck() error {
	if n.ActivationsCount > 0 {
		return errors.New(fmt.Sprintf("ALERT: %s has activation count %d", n, n.ActivationsCount))
	}
	if n.Activation > 0 {
		return errors.New(fmt.Sprintf("ALERT: %s has activation %f", n, n.Activation))
	}
	if n.lastActivation > 0 {
		return errors.New(fmt.Sprintf("ALERT: %s has last_activation %f", n, n.lastActivation))
	}
	if n.lastActivation2 > 0 {
		return errors.New(fmt.Sprintf("ALERT: %s has last_activation2 %f", n, n.lastActivation2))
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
func (n *NNode) WriteNode(w io.Writer) {
	trait_id := 0
	if n.Trait != nil {
		trait_id = n.Trait.TraitId
	}
	fmt.Fprintf(w, "node %d %d %d %d", n.NodeId, trait_id, n.NType, n.GenNodeLabel)
}
func (n *NNode) Depth(d int32) (int32, error) {
	if d > 100 {
		return 10, errors.New("** DEPTH NOT DETERMINED FOR NETWORK WITH LOOP");
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
	if n.IsSensor() {
		return fmt.Sprintf("(S %d, step %d : %f)", n.NodeId, n.ActivationsCount, n.Activation)
	} else {
		return fmt.Sprintf("(N %d, step %d : %f)", n.NodeId, n.ActivationsCount, n.Activation)
	}
}




