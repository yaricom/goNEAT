package network

import (
	"github.com/yaricom/goNEAT/neat/genetics"
	"fmt"
	"errors"
)

// A NETWORK is a LIST of input NODEs and a LIST of output NODEs.
// The point of the network is to define a single entity which can evolve
// or learn on its own, even though it may be part of a larger framework.
type Network interface {
	// Puts the network back into an initial state
	Flush()
	// Activates the net such that all outputs are active
	Activate() (bool, error)
	// If at least one output is not active then return true
	OutputIsOff() bool

	// Prints the values of network outputs to the console
	PrintActivation()
	// Print the values of network inputs to the console
	PrintInput()
	// Verify that network was successfully flushed for debugging
	FlushCheck() error

	// Adds a new input node
	AddInputNode(node *NNode)
	// Adds a new output node
	AddOutputNode(node *NNode)

	// Takes an array of sensor values and loads it into SENSOR inputs ONLY
	LoadSensors(sensors []float64)
	// Set network name
	SetName(name string)

	// This checks a POTENTIAL link between a potential in_node
     	// and potential out_node to see if it must be recurrent.
	// Use count and thresh to jump out in the case of an infinite loop.
	IsRecurrent(potin_node, potout_node *NNode, count *int32, thresh int32) bool
	// Find the maximum number of neurons between an output and an input
	MaxDepth() (int32, error)

	// Counts the number of nodes in the net
	NodeCount() int
	// Counts the number of links in the net
	LinkCount() int
}

// Creates new network
func NewNetwork(in, out, all []*NNode, netid int32) Network {
	n := newNetwork(netid)
	n.inputs = in
	n.outputs = out
	n.all_nodes = all
	return &n
}

// The default private constructor
func newNetwork(netId int32) network {
	return network {
		numlinks:-1,
		net_id:netId,
	}
}

// The private network data holder
type network struct {
	//The number of links in the net (-1 means not yet counted)
	numlinks int

	// A list of all the nodes in the network
	all_nodes []*NNode
	// NNodes that input into the network
	inputs []*NNode
	// NNodes that output from the network
	outputs []*NNode

	// A network id
	net_id int32

	// Allows Network to be matched with its Genome
	genotype *genetics.Genome

	// Is a name of this network */
	name string
}

// The Network interface implementation
func (n *network) Flush() {
	// Flush back recursively
	for _, node := range n.all_nodes {
		node.Flushback()
	}
}
func (n *network) FlushCheck() error {
	for _, node := range n.all_nodes {
		err := node.FlushbackCheck()
		if err != nil {
			return err
		}
	}
	return nil
}
func (n *network) PrintActivation() {
	fmt.Printf("Network %s with id %d outputs: (", n.name, n.net_id)
	for i, node := range n.outputs {
		fmt.Printf("[Output #%d: %s] ", i, node)
	}
	fmt.Println(")")
}
func (n *network) PrintInput() {
	fmt.Printf("Network %s with id %d inputs: (", n.name, n.net_id)
	for i, node := range n.inputs {
		fmt.Printf("[Input #%d: %s] ", i, node)
	}
	fmt.Println(")")
}
func (n *network) OutputIsOff() bool {
	for _, node := range n.outputs {
		if node.ActivationsCount == 0 {
			return true
		}
	}
	return false
}
func (n *network) Activate() (bool, error) {
	//For adding to the activesum
	add_amount := 0.0
	//Make sure we at least activate once
	one_time := false
	//Used in case the output is somehow truncated from the network
	abort_count := 0

	// The sigmoid activator function
	sigmoid := ActivationFunc(SigmoidActivation)

	// Keep activating until all the outputs have become active
	// (This only happens on the first activation, because after that they are always active)
	for n.OutputIsOff() && !one_time {
		abort_count += 1

		if abort_count >= 20 {
			return false, errors.New("Inputs disconnected from outputa!")
		}

		// For each neuron node, compute the sum of its incoming activation
		for _, node := range n.all_nodes {
			if node.IsNeuron() {
				node.ActivationSum = 0.0 // reset activation value
				node.IsActive = false // flag node disabled

				// For each node's incoming connection, add the activity from the connection to the activesum
				for _, link := range node.Incoming {
					// Handle possible time delays
					if !link.IsTimeDelayed {
						add_amount = link.Weight * link.InNode.GetActiveOut()
						if link.InNode.IsActive && link.InNode.IsSensor() {
							link.InNode.IsActive = true
						}
					} else {
						add_amount = link.Weight * link.InNode.GetActiveOutTd()
					}
					node.ActivationSum += add_amount
				} // End {for} over incoming links
			} // End if != SENSOR
		}  // End {for} over all nodes

		// Now activate all the neuron nodes off their incoming activation
		for _, node := range n.all_nodes {
			if node.IsNeuron() {
				// Only activate if some active input came in
				if node.IsActive {
					// Keep a memory of activations for potential time delayed connections
					node.SaveActivations()
					// Now run the net activation through an activation function
					if node.FType == SIGMOID {
						node.Activation = sigmoid.Activation(node, 4.924273, 2.4621365)
					} else {
						return false, errors.New(
							fmt.Sprintf("Unknown activation function type: %d", node.FType))
					}
					// Increment the activation_count
					// First activation cannot be from nothing!!
					node.ActivationsCount++
				}
			}
		}
		one_time = true
	}
	return true, nil
}
func (n *network) AddInputNode(node *NNode) {
	n.inputs = append(n.inputs, node)
}
func (n *network) AddOutputNode(node *NNode) {
	n.outputs = append(n.outputs, node)
}
func (n *network) LoadSensors(sensors []float64) {
	counter := 0
	for _, node := range n.inputs{
		if node.IsSensor() {
			node.SensorLoad(sensors[counter])
			counter += 1
		}
	}
}
func (n *network) SetName(name string) {
	n.name = name
}
func (n network) NodeCount() int {
	return len(n.all_nodes)
}
func (n network) LinkCount() int {
	n.numlinks = 0
	for _, node := range n.all_nodes {
		n.numlinks += len(node.Incoming)
	}
	return n.numlinks
}

func (n *network) IsRecurrent(potin_node, potout_node *NNode, count *int32, thresh int32) bool {
	// Count the node as visited
	*count++

	if *count > thresh {
		return false // Short out the whole thing - loop detected
	}

	if potin_node == potout_node {
		return true
	} else {
		// Check back on all links ...
		for _, link := range potin_node.Incoming {
			// But skip links that are already recurrent -
			// We want to check back through the forward flow of signals only
			if link.IsRecurrent != true {
				if n.IsRecurrent(link.InNode, potout_node, count, thresh) {
					return true
				}
			}
		}
	}
	return false
}

func (n *network) MaxDepth() (int32, error) {
	max := int32(0) // The max depth
	for _, node := range n.outputs {
		curr_depth, err := node.Depth(0)
		if err != nil {
			return curr_depth, err
		}
		if curr_depth > max {
			max = curr_depth
		}
	}
	return max, nil
}
