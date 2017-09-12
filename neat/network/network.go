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
}

func NewNetwork(in, out, all []*NNode, netid int32) Network {
	n := newNetwork(netid)
	n.inputs = in
	n.outputs = out
	n.all_nodes = all
	return n
}

// The default private constructor
func newNetwork(netId int32) network {
	return network{
		numnodes:-1,
		numlinks:-1,
		net_id:netId,
	}
}

// The private network data holder
type network struct {
	//The number of nodes in the net (-1 means not yet counted)
	numnodes int32
	//The number of links in the net (-1 means not yet counted)
	numlinks int32

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
	for _, n := range n.all_nodes {
		(*n).Flushback()
	}
}
func (n *network) FlushCheck() error {
	for _, n := range n.all_nodes {
		err := (*n).FlushbackCheck()
		if err != nil {
			return err
		}
	}
	return nil
}
func (n *network) PrintActivation() {
	if n != nil {
		fmt.Printf("Network %s with id %d outputs: (", n.name, n.net_id)
	} else {
		fmt.Printf("Network id %d outputs: (", n.net_id)
	}
	for i, n := range n.outputs {
		fmt.Printf("[Output #%d: %s] ", i, (*n))
	}
	fmt.Println(")")
}
func (n *network) PrintInput() {
	if n != nil {
		fmt.Printf("Network %s with id %d inputs: (", n.name, n.net_id)
	} else {
		fmt.Printf("Network id %d inputs: (", n.net_id)
	}
	for i, n := range n.inputs {
		fmt.Printf("[Input #%d: %s] ", i, (*n))
	}
	fmt.Println(")")
}

func (n *network) OutputIsOff() bool {
	for _, on := range n.outputs {
		if (*on).ActivationCount() == 0 {
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

	// Keep activating until all the outputs have become active
	// (This only happens on the first activation, because after that they are always active)
	for n.OutputIsOff() && !one_time {
		abort_count += 1

		if abort_count >= 20 {
			return false, errors.New("Inputs disconnected from output!")
		}

		// For each node, compute the sum of its incoming activation
		for _, np := range n.all_nodes {
			if (*np).GetType() != SENSOR {
				(*np).SetActiveSum(0.0) // reset activation value
				(*np).SetActiveFlag(false) // flag node disabled

				// For each node's incoming connection, add the activity from the connection to the activesum
				for _, lp := range (*np).GetIncoming() {
					// Handle possible time delays
					if !(*lp).IsTimeDelayed() {
						add_amount = (*lp).GetWeight() * (*(*lp).InNode()).GetActiveOut()
						if (*(*lp).InNode()).IsActive() && (*(*lp).InNode()).GetType() == SENSOR {
							(*(*lp).InNode()).SetActiveFlag(true)
						}
					} else {
						add_amount = (*lp).GetWeight() * (*(*lp).InNode()).GetActiveOutTd()
					}
					(*np).AddToActiveSum(add_amount)
				} // End {for} over incoming links
			} // End if != SENSOR
		}  // End {for} over all nodes

		// Now activate all the non-sensor nodes off their incoming activation
		for _, np := range n.all_nodes {
			if (*np).GetType() != SENSOR {
				// Only activate if some active input came in
				if (*np).IsActive() {
					// Keep a memory of activations for potential time delayed connections
					(*np).SaveActivations()
					// Now run the net activation through an activation function
					if (*np).GetFtype() == SIGMOID {
						activation := fsigmoid((*np).GetActiveSum(), 4.924273, 2.4621365)
						(*np).SetActivation(activation)
					} else {
						return false, errors.New(
							fmt.Sprintf("Unknown activation function type: %d", (*np).GetFtype()))
					}
					// Increment the activation_count
					// First activation cannot be from nothing!!
					(*np).IncrementActivationCount()
				}
			}
		}
		one_time = true
	}
	return true, nil
}
