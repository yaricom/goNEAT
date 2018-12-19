package network

import (
	"fmt"
	"bytes"
	"errors"
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

	// A list of all the nodes in the network except MIMO control ones
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
	n := NewNetwork(in, out, all, net_id)
	n.control_nodes = control
	return n
}

// Creates fast network solver based on the architecture of this network. It's primarily aimed for big networks to improve
// processing speed.
func (n *Network) FastNetworkSolver() (NetworkSolver, error) {
	// calculate neurons per layer
	outputNeuronCount := len(n.Outputs)
	// build bias, input and hidden neurons lists
	biasNeuronCount := 0
	in_list := make([]*NNode, 0)
	bias_list := make([]*NNode, 0)
	hidn_list := make([]*NNode, 0)
	for _, ne := range n.all_nodes {
		switch ne.NeuronType {
		case BiasNeuron:
			biasNeuronCount += 1
			bias_list = append(bias_list, ne)
		case InputNeuron:
			in_list = append(in_list, ne)
		case HiddenNeuron:
			hidn_list = append(hidn_list, ne)
		}
	}
	inputNeuronCount := len(in_list)
	totalNeuronCount := len(n.all_nodes)

	// create activation functions array

	activations := make([]NodeActivationType, totalNeuronCount)
	neuronLookup := make(map[int]int)// id:index
	neuronIndex := 0
	// walk through neuron nodes in order: bias, input, output, hidden
	neuronIndex = processList(neuronIndex, bias_list, activations, neuronLookup)
	neuronIndex = processList(neuronIndex, in_list, activations, neuronLookup)
	neuronIndex = processList(neuronIndex, n.Outputs, activations, neuronLookup)
	neuronIndex = processList(neuronIndex, hidn_list, activations, neuronLookup)

	// walk through neurons in order: input, output, hidden and create bias and connections lists
	biases := make([]float64, totalNeuronCount)
	connections := make([]*FastNetworkLink, 0)

	if in_connects, err := processIncomingConnections(in_list, biases, neuronLookup); err == nil {
		connections = append(connections, in_connects...)
	} else {
		return nil, err
	}
	if in_connects, err := processIncomingConnections(hidn_list, biases, neuronLookup); err == nil {
		connections = append(connections, in_connects...)
	} else {
		return nil, err
	}
	if in_connects, err := processIncomingConnections(n.Outputs, biases, neuronLookup); err == nil {
		connections = append(connections, in_connects...)
	} else {
		return nil, err
	}

	// walk through control neurons
	modules := make([]*FastControlNode, len(n.control_nodes))
	for i, cn := range n.control_nodes {
		// collect inputs
		inputs := make([]int, len(cn.Incoming))
		for j, in := range cn.Incoming {
			if in_index, ok := neuronLookup[in.InNode.Id]; ok {
				inputs[j] = in_index
			} else {
				return nil, errors.New(
					fmt.Sprintf("Failed to lookup for input neuron with id: %d at control neuron: %d",
						in.InNode.Id, cn.Id))
			}
		}
		// collect outputs
		outputs := make([]int, len(cn.Outgoing))
		for j, out := range cn.Outgoing {
			if out_index, ok := neuronLookup[out.OutNode.Id]; ok {
				outputs[j] = out_index
			} else {
				return nil, errors.New(
					fmt.Sprintf("Failed to lookup for output neuron with id: %d at control neuron: %d",
						out.InNode.Id, cn.Id))
			}
		}
		// build control node
		modules[i] = &FastControlNode{InputIndxs:inputs, OutputIndxs:outputs, ActivationType:cn.ActivationType}
	}

	return NewFastModularNetworkSolver(biasNeuronCount, inputNeuronCount, outputNeuronCount, totalNeuronCount,
		activations, connections, biases, modules), nil
}

func processList(startIndex int, nList []*NNode, activations[]NodeActivationType, neuronLookup map[int]int) int {
	for _, ne := range nList {
		activations[startIndex] = ne.ActivationType
		neuronLookup[ne.Id] = startIndex
		startIndex += 1
	}
	return startIndex
}

func processIncomingConnections(nList []*NNode, biases []float64, neuronLookup map[int]int) (connections []*FastNetworkLink, err error) {
	connections = make([]*FastNetworkLink, 0)
	for _, ne := range nList {
		if targetIndex, ok := neuronLookup[ne.Id]; ok {
			for _, in := range ne.Incoming {
				if sourceIndex, ok := neuronLookup[in.InNode.Id]; ok {
					if in.InNode.NeuronType == BiasNeuron {
						// store bias for target neuron
						biases[targetIndex] += in.Weight
					} else {
						// save connection
						conn := FastNetworkLink{
							SourceIndx:sourceIndex,
							TargetIndx:targetIndex,
							Weight:in.Weight,
						}
						connections = append(connections, &conn)
					}
				} else {
					err = errors.New(
						fmt.Sprintf("Failed to lookup for source neuron with id: %d", in.InNode.Id))
					break
				}
			}
		} else {
			err = errors.New(fmt.Sprintf("Failed to lookup for target neuron with id: %d", ne.Id))
			break
		}
	}
	if err != nil {
		return nil, err
	}
	return connections, err
}

// Puts the network back into an initial state
func (n *Network) Flush() (res bool, err error) {
	res = true
	// Flush back recursively
	for _, node := range n.all_nodes {
		node.Flushback()
		err = node.FlushbackCheck()
		if err != nil {
			// failed - no need to continue
			res = false
			break
		}
	}
	return res, err
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

// Attempts to activate the network given number of steps before returning error.
func (n *Network) ActivateSteps(max_steps int) (bool, error) {
	// For adding to the activesum
	add_amount := 0.0
	// Make sure we at least activate once
	one_time := false
	// Used in case the output is somehow truncated from the network
	abort_count := 0

	// Keep activating until all the outputs have become active
	// (This only happens on the first activation, because after that they are always active)
	for n.OutputIsOff() || !one_time {

		if abort_count >= max_steps {
			return false, NetErrExceededMaxActivationAttempts
		}

		// For each neuron node, compute the sum of its incoming activation
		for _, np := range n.all_nodes {
			if np.IsNeuron() {
				np.ActivationSum = 0.0 // reset activation value

				// For each node's incoming connection, add the activity from the connection to the activesum
				for _, link := range np.Incoming {
					// Handle possible time delays
					if !link.IsTimeDelayed {
						add_amount = link.Weight * link.InNode.GetActiveOut()
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
				}
			}
		}

		// Now activate all MIMO control genes to propagate activation through genome modules
		for _, cn := range n.control_nodes {
			cn.isActive = false
			// Activate control MIMO node as control module
			err := NodeActivators.ActivateModule(cn)
			if err != nil {
				return false, err
			}
			// mark control node as active
			cn.isActive = true
		}

		one_time = true
		abort_count += 1
	}
	return true, nil
}

// Activates the net such that all outputs are active
func (n *Network) Activate() (bool, error) {
	return n.ActivateSteps(20)
}

// Propagates activation wave through all network nodes provided number of steps in forward direction.
// Returns true if activation wave passed from all inputs to outputs.
func (n *Network) ForwardSteps(steps int) (res bool, err error) {
	for i := 0; i < steps; i++ {
		res, err = n.Activate()
		if err != nil {
			// failure - no need to continue
			break
		}
	}
	return res, err
}

// Propagates activation wave through all network nodes provided number of steps by recursion from output nodes
// Returns true if activation wave passed from all inputs to outputs.
func (n *Network) RecursiveSteps() (bool, error) {
	return false, errors.New("RecursiveSteps Not Implemented")
}

// Attempts to relax network given amount of steps until giving up. The network considered relaxed when absolute
// value of the change at any given point is less than maxAllowedSignalDelta during activation waves propagation.
// If maxAllowedSignalDelta value is less than or equal to 0, the method will return true without checking for relaxation.
func (n *Network) Relax(maxSteps int, maxAllowedSignalDelta float64) (bool, error) {
	return false, errors.New("Relax Not Implemented")
}

// Takes an array of sensor values and loads it into SENSOR inputs ONLY
func (n *Network) LoadSensors(sensors []float64) error {
	if len(sensors) != len(n.inputs) {
		return NetErrUnsupportedSensorsArraySize
	}

	counter := 0
	for _, node := range n.inputs {
		if node.IsSensor() {
			node.SensorLoad(sensors[counter])
			counter += 1
		}
	}
	return nil
}

// Read output values from the output nodes of the network
func (n *Network) ReadOutputs() []float64 {
	outs := make([]float64, len(n.Outputs))
	for i, o := range n.Outputs {
		outs[i] = o.Activation
	}
	return outs
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
			n.numlinks += len(node.Outgoing)
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
	if len(n.control_nodes) > 0 {
		return -1, errors.New("unsupported for modular networks")
	}
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
