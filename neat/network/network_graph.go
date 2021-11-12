package network

import (
	"encoding/json"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/encoding/dot"
	"gonum.org/v1/gonum/graph/formats/cytoscapejs"
	"gonum.org/v1/gonum/graph/iterator"
	"io"
)

// the Gonum graph.Graph
//

// Node returns the node with the given ID if it exists
// in the graph, and nil otherwise.
func (n *Network) Node(id int64) graph.Node {
	return n.nodeWithID(id)
}

// Nodes returns all the nodes in the graph.
//
// Nodes must not return nil.
func (n *Network) Nodes() graph.Nodes {
	nodes := make([]graph.Node, len(n.allNodesMIMO))
	for i, n := range n.allNodesMIMO {
		nodes[i] = n
	}
	return iterator.NewOrderedNodes(nodes)
}

// From returns all nodes that can be reached directly
// from the node with the given ID.
//
// From must not return nil.
func (n *Network) From(id int64) graph.Nodes {
	node := n.nodeWithID(id)
	if node == nil {
		return graph.Empty
	}
	var nodes []graph.Node
	for _, l := range node.Outgoing {
		nodes = append(nodes, l.OutNode)
	}
	// check control nodes - the control nodes can have this node as incoming.
	// The control node is not in the list of outgoing nodes of the ordinary node by design.
	// This is done to have clear demarcation between modules and to avoid any intersections.
	for _, cn := range n.controlNodes {
		for _, incoming := range cn.Incoming {
			if incoming.InNode.ID() == id {
				nodes = append(nodes, cn)
				break
			}
		}
	}

	return iterator.NewOrderedNodes(nodes)
}

// HasEdgeBetween returns whether an edge exists between
// nodes with IDs xid and yid without considering direction.
func (n *Network) HasEdgeBetween(xid, yid int64) bool {
	edge := n.edgeBetween(xid, yid, false)
	return edge != nil
}

// Edge returns the edge from u to v, with IDs uid and vid,
// if such an edge exists and nil otherwise. The node v
// must be directly reachable from u as defined by the
// From method.
func (n *Network) Edge(uid, vid int64) graph.Edge {
	return n.edgeBetween(uid, vid, true)
}

// the Gonum graph.Weighted
//

// WeightedEdge returns the weighted edge from u to v
// with IDs uid and vid if such an edge exists and
// nil otherwise. The node v must be directly
// reachable from u as defined by the From method.
func (n *Network) WeightedEdge(uid, vid int64) graph.WeightedEdge {
	return n.edgeBetween(uid, vid, true)
}

// Weight returns the weight for the edge between
// x and y with IDs xid and yid if Edge(xid, yid)
// returns a non-nil Edge.
// If x and y are the same node or there is no
// joining edge between the two nodes the weight
// value returned is implementation dependent.
// Weight returns true if an edge exists between
// x and y or if x and y have the same ID, false
// otherwise.
func (n *Network) Weight(xid, yid int64) (w float64, ok bool) {
	edge := n.edgeBetween(xid, yid, true)
	if edge == nil {
		return 0, false
	}
	return edge.Weight(), true
}

// the Gonum graph.Directed
//

// HasEdgeFromTo returns whether an edge exists
// in the graph from u to v with IDs uid and vid.
func (n *Network) HasEdgeFromTo(uid, vid int64) bool {
	edge := n.edgeBetween(uid, vid, true)
	return edge != nil
}

// To returns all nodes that can reach directly
// to the node with the given ID.
//
// To must not return nil.
func (n *Network) To(id int64) graph.Nodes {
	node := n.nodeWithID(id)
	if node == nil {
		return graph.Empty
	}
	var nodes []graph.Node
	for _, l := range node.Incoming {
		nodes = append(nodes, l.InNode)
	}

	// check control nodes - the control nodes can have this node as incoming.
	// The control node is not in the list of incoming nodes of the ordinary node by design.
	// This is done to have clear demarcation between modules and to avoid any intersections.
	for _, cn := range n.controlNodes {
		for _, outgoing := range cn.Outgoing {
			if outgoing.OutNode.ID() == id {
				nodes = append(nodes, cn)
				break
			}
		}
	}

	return iterator.NewOrderedNodes(nodes)
}

func (n *Network) edgeBetween(uid, vid int64, directed bool) *Link {
	var uNode, vNode *NNode
	for _, np := range n.allNodes {
		if np.ID() == uid {
			uNode = np
		}
		if np.ID() == vid {
			vNode = np
		}
		// check if already found - recursive link
		if uNode != nil && vNode != nil {
			// no need to iterate further
			break
		}
	}
	// nothing found - return immediately
	if uNode == nil && vNode == nil {
		return nil
	}

	// there are possibility of the control node on either side of the edge - exploring it
	if uNode == nil || vNode == nil {
		// check control nodes for possible edge. The control nodes is not double linked with ordinary nodes
		var cid, oid int64
		if uNode == nil {
			// possibility of the control node on the incoming side
			cid = uid
			oid = vid
		} else {
			// possibility of the control node on the outgoing side
			cid = vid
			oid = uid
		}
		// iterate over control nodes and check that it has edge to the ordinary node
		for _, cn := range n.controlNodes {
			if cn.ID() != cid {
				continue
			}
			// check connections
			for _, incoming := range cn.Incoming {
				if incoming.InNode.ID() == oid {
					if !directed {
						return incoming
					} else {
						// make sure that control node is on the outgoing side
						if uNode != nil {
							return incoming
						} else {
							return nil
						}
					}
				}
			}
			for _, outgoing := range cn.Outgoing {
				if outgoing.OutNode.ID() == oid {
					if !directed {
						return outgoing
					} else {
						// make sure that control node if on the incoming side
						if vNode != nil {
							return outgoing
						} else {
							return nil
						}
					}
				}
			}
		}

		// nothing was found
		return nil
	}

	//
	// process ordinary nodes found on both sides of the edge
	//

	// check if nodes linked
	if !directed {
		// for undirected check that incoming link of the source node point to the target node
		for _, l := range uNode.Incoming {
			if l.InNode.ID() == vid {
				return l
			}
		}
	} else {
		// for directed check that incoming link of the target node points to the source node
		for _, l := range vNode.Incoming {
			if l.InNode.ID() == uid {
				return l
			}
		}
	}
	// check that outgoing link of source node points to the target node
	for _, l := range uNode.Outgoing {
		if l.OutNode.ID() == vid {
			return l
		}
	}
	return nil
}

// WriteDOT is to write this network graph using the GraphViz DOT encoding.
// See DOT Guide: https://www.graphviz.org/pdf/dotguide.pdf
func (n *Network) WriteDOT(w io.Writer) error {
	data, err := dot.Marshal(n, n.Name, "", "")
	if err != nil {
		return err
	}
	if _, err = w.Write(data); err != nil {
		return err
	}
	return nil
}

// WriteCytoscapeJSON is to write this network graph using Cytoscape JSON encoding. Generated JSON file can be used
// for visualization with Cytoscape application https://cytoscape.org or as input to the Cytoscape JavaScript library
// https://js.cytoscape.org
func (n *Network) WriteCytoscapeJSON(w io.Writer) error {
	elements := cytoscapejs.Elements{
		Nodes: make([]cytoscapejs.Node, 0),
		Edges: make([]cytoscapejs.Edge, 0),
	}

	// add all ordinary nodes
	for _, node := range n.allNodes {
		// populate Nodes data
		elements.Nodes = append(elements.Nodes, nodeToCyJsNode(node, false))
		// populate edges data from incoming side
		for _, e := range node.Incoming {
			elements.Edges = append(elements.Edges, linkToCyJsEdge(e))
		}
	}

	// add all control nodes
	for _, node := range n.controlNodes {
		// populate Nodes data
		elements.Nodes = append(elements.Nodes, nodeToCyJsNode(node, true))

		// populate edges data from the incoming side
		for _, e := range node.Incoming {
			elements.Edges = append(elements.Edges, linkToCyJsEdge(e))
		}
		// populate edges data to the outgoing side
		for _, e := range node.Outgoing {
			elements.Edges = append(elements.Edges, linkToCyJsEdge(e))
		}
	}

	// create Cytoscape graph
	graphNodeEdge := cytoscapejs.GraphNodeEdge{
		Elements: elements,
	}
	// marshal to Cytoscape JSON
	if data, err := json.Marshal(graphNodeEdge); err != nil {
		return err
	} else if _, err = w.Write(data); err != nil {
		return err
	}
	return nil
}

func (n *Network) nodeWithID(id int64) *NNode {
	for _, np := range n.allNodesMIMO {
		if np.ID() == id {
			return np
		}
	}
	return nil
}

func nodeToCyJsNode(node *NNode, control bool) cytoscapejs.Node {
	actName, err := math.NodeActivators.ActivationNameFromType(node.ActivationType)
	if err != nil {
		actName = "unknown"
	}
	return cytoscapejs.Node{
		Data: cytoscapejs.NodeData{
			ID: fmt.Sprintf("%d", node.Id),
			Attributes: map[string]interface{}{
				"activation_value":      node.Activation,
				"activation_function":   actName,
				"neuron_type":           NeuronTypeName(node.NeuronType),
				"node_type":             NodeTypeName(node.NodeType()),
				"in_connections_count":  len(node.Incoming),
				"out_connections_count": len(node.Outgoing),
				"trait":                 node.Trait.String(),
				"control_node":          control,
			},
		},
		Selectable: true,
	}
}

func linkToCyJsEdge(link *Link) cytoscapejs.Edge {
	return cytoscapejs.Edge{
		Data: cytoscapejs.EdgeData{
			ID:     link.IDString(),
			Source: fmt.Sprintf("%d", link.InNode.Id),
			Target: fmt.Sprintf("%d", link.OutNode.Id),
			Attributes: map[string]interface{}{
				"weight":       link.ConnectionWeight,
				"recurrent":    link.IsRecurrent,
				"time_delayed": link.IsTimeDelayed,
				"trait":        link.Trait.String(),
			},
		},
		Selectable: true,
	}
}
