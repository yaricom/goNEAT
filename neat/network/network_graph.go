package network

import "gonum.org/v1/gonum/graph"

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
	return newNodesIterator(n.allNodesMIMO)
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
	var nodes []*NNode
	for _, l := range node.Outgoing {
		nodes = append(nodes, l.OutNode)
	}
	return newNodesIterator(nodes)
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
	var nodes []*NNode
	for _, l := range node.Incoming {
		nodes = append(nodes, l.InNode)
	}
	return newNodesIterator(nodes)
}

func (n *Network) edgeBetween(uid, vid int64, directed bool) *Link {
	var uNode, vNode *NNode
	for _, np := range n.allNodesMIMO {
		if np.ID() == uid {
			uNode = np
		}
		if np.ID() == vid {
			vNode = np
		}
		// check if already found
		if uNode != nil && vNode != nil {
			// no need to iterate further
			break
		}
	}
	if uNode == nil || vNode == nil {
		return nil
	}
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

func (n *Network) nodeWithID(id int64) *NNode {
	for _, np := range n.allNodesMIMO {
		if np.ID() == id {
			return np
		}
	}
	return nil
}

// nodesIterator is the definition of iterator for a list of nodes.
type nodesIterator struct {
	nodes []*NNode
	index int
	curr  *NNode
}

func newNodesIterator(nodes []*NNode) graph.Nodes {
	return &nodesIterator{nodes: nodes}
}

// Next advances the iterator and returns whether
// the next call to the item method will return a
// non-nil item.
//
// Next should be called prior to any call to the
// iterator's item retrieval method after the
// iterator has been obtained or reset.
//
// The order of iteration is implementation
// dependent.
func (i *nodesIterator) Next() bool {
	if i.index < len(i.nodes) {
		i.curr = i.nodes[i.index]
		i.index++
		return true
	}
	i.curr = nil
	return false
}

// Len returns the number of items remaining in the
// iterator.
//
// If the number of items in the iterator is unknown,
// too large to materialize or too costly to calculate
// then Len may return a negative value.
// In this case the consuming function must be able
// to operate on the items of the iterator directly
// without materializing the items into a slice.
// The magnitude of a negative length has
// implementation-dependent semantics.
func (i *nodesIterator) Len() int {
	return len(i.nodes) - i.index
}

// Node returns the current Node from the iterator.
func (i *nodesIterator) Node() graph.Node {
	return i.curr
}

// Reset returns the iterator to its start position.
func (i *nodesIterator) Reset() {
	i.index = 0
	i.curr = nil
}
