package network

import (
	"fmt"
	"bytes"
)

// A NETWORK is a LIST of input NODEs and a LIST of output NODEs.
// The point of the network is to define a single entity which can evolve
// or learn on its own, even though it may be part of a larger framework.
type Network struct {
	// A network id
	Id            int
	// Is a name of this network */
	Name          string

	// The number of links in the net (-1 means not yet counted)
	numlinks      int

	// A list of all the nodes in the network
	all_nodes     []*NNode
	// NNodes that input into the network
	inputs        []*NNode
	// NNodes that output from the network
	Outputs       []*NNode

	// NNodes that connect network modules
	control_nodes []*NNode
}

// Creates new network
func NewNetwork(in, out, all []*NNode, net_id int) *Network {
	n := Network{
		Id:net_id,
		inputs:in,
		Outputs:out,
		all_nodes:all,
		numlinks:-1,
	}
	return &n
}

// Creates new modular network with control nodes
func NewModularNetwork(in, out, all, control []*NNode, net_id int) *Network {
	n := Network{
		Id:net_id,
		inputs:in,
		Outputs:out,
		all_nodes:all,
		control_nodes:control,
		numlinks:-1,
	}
	return &n
}

// Puts the network back into an initial state
func (n *Network) Flush() {
	// Flush back recursively
	for _, node := range n.all_nodes {
		node.Flushback()
	}
}

// Verify that network was successfully flushed for debugging
func (n *Network) FlushCheck() error {
	for _, node := range n.all_nodes {
		err := node.FlushbackCheck()
		if err != nil {
			return err
		}
	}
	return nil
}

// Prints the values of network outputs to the console
func (n *Network) PrintActivation() string {
	out := bytes.NewBufferString(fmt.Sprintf("Network %s with id %d outputs: (", n.Name, n.Id))
	for i, node := range n.Outputs {
		fmt.Fprintf(out, "[Output #%d: %s] ", i, node)
	}
	fmt.Fprint(out, ")")
	return out.String()
}

// Print the values of network inputs to the console
func (n *Network) PrintInput() string {
	out := bytes.NewBufferString(fmt.Sprintf("Network %s with id %d inputs: (", n.Name, n.Id))
	for i, node := range n.inputs {
		fmt.Fprintf(out, "[Input #%d: %s] ", i, node)
	}
	fmt.Fprint(out, ")")
	return out.String()
}

// If at least one output is not active then return true
func (n *Network) OutputIsOff() bool {
	for _, node := range n.Outputs {
		if node.ActivationsCount == 0 {
			return true
		}
	}
	return false
}

// Activates the net such that all outputs are active
func (n *Network) Activate() (bool, error) {
	//For adding to the activesum
	add_amount := 0.0
	//Make sure we at least activate once
	one_time := false
	//Used in case the output is somehow truncated from the network
	abort_count := 0

	// Keep activating until all the outputs have become active
	// (This only happens on the first activation, because after that they are always active)
	for n.OutputIsOff() || !one_time {
		abort_count += 1

		if abort_count >= 20 {
			return false, nil//errors.New("Inputs disconnected from outputs!")
		}

		// For each neuron node, compute the sum of its incoming activation
		for _, np := range n.all_nodes {
			if np.IsNeuron() {
				np.ActivationSum = 0.0 // reset activation value
				np.isActive = false // flag node disabled

				// For each node's incoming connection, add the activity from the connection to the activesum
				for _, link := range np.Incoming {
					// Handle possible time delays
					if !link.IsTimeDelayed {
						add_amount = link.Weight * link.InNode.GetActiveOut()
						//fmt.Printf("%f -> %f\n", link.Weight, (*link.InNode).GetActiveOut())
						if link.InNode.isActive || link.InNode.IsSensor() {
							np.isActive = true
						}
					} else {
						add_amount = link.Weight * link.InNode.GetActiveOutTd()
					}
					np.ActivationSum += add_amount
				} // End {for} over incoming links
			} // End if != SENSOR
		}  // End {for} over all nodes

		// Now activate all the neuron nodes off their incoming activation
		for _, np := range n.all_nodes {
			if np.IsNeuron() {
				// Only activate if some active input came in
				if np.isActive {
					// Now run the net activation through an activation function
					err := NodeActivators.ActivateNode(np)
					if err != nil {
						return false, err
					}

					// Increment the activation_count
					// First activation cannot be from nothing!!
					np.ActivationsCount++
				}
				//fmt.Printf("Node: %s, activation sum: %f, active: %t\n", np, np.ActivationSum, np.IsActive)
			}
		}

		// Now activate MIMO control genes to propagate activation through genome modules
		//for _,

		one_time = true
	}
	return true, nil
}

// Takes an array of sensor values and loads it into SENSOR inputs ONLY
func (n *Network) LoadSensors(sensors []float64) {
	counter := 0
	for _, node := range n.inputs {
		if node.IsSensor() {
			node.SensorLoad(sensors[counter])
			counter += 1
		}
	}
}

// Counts the number of nodes in the net
func (n *Network) NodeCount() int {
	if len(n.control_nodes) == 0 {
		return len(n.all_nodes)
	} else {
		return len(n.all_nodes) + len(n.control_nodes)
	}
}

// Counts the number of links in the net
func (n *Network) LinkCount() int {
	n.numlinks = 0
	for _, node := range n.all_nodes {
		n.numlinks += len(node.Incoming)
	}
	if len(n.control_nodes) != 0 {
		for _, node := range n.control_nodes {
			n.numlinks += len(node.Incoming)
		}
	}
	return n.numlinks
}

// Returns complexity of this network which is sum of nodes count and links count
func (n *Network) Complexity() int {
	return n.NodeCount() + n.LinkCount()
}

// This checks a POTENTIAL link between a potential in_node
// and potential out_node to see if it must be recurrent.
// Use count and thresh to jump out in the case of an infinite loop.
func (n *Network) IsRecurrent(in_node, out_node *NNode, count *int, thresh int) bool {
	// Count the node as visited
	*count++

	if *count > thresh {
		return false // Short out the whole thing - loop detected
	}

	if in_node == out_node {
		return true
	} else {
		// Check back on all links ...
		for _, link := range in_node.Incoming {
			// But skip links that are already recurrent -
			// We want to check back through the forward flow of signals only
			if !link.IsRecurrent {
				if n.IsRecurrent(link.InNode, out_node, count, thresh) {
					return true
				}
			}
		}
	}
	return false
}

// Find the maximum number of neurons between an output and an input
func (n *Network) MaxDepth() (int, error) {
	// The quick case when there are no hidden nodes
	if len(n.all_nodes) == len(n.inputs) + len(n.Outputs) && len(n.control_nodes) == 0 {
		return 1, nil // just one layer depth
	}

	max := 0 // The max depth
	for _, node := range n.Outputs {
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

// Returns all nodes in the network
func (n *Network) AllNodes() []*NNode {
	return n.all_nodes
}
