package network

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/graph"
	"testing"
)

func TestNetwork_Edge(t *testing.T) {
	net := buildNetwork()

	testCases := []struct {
		uid    int64
		vid    int64
		exists bool
	}{
		// existing
		{uid: 1, vid: 4, exists: true},
		{uid: 2, vid: 4, exists: true},
		{uid: 2, vid: 5, exists: true},
		{uid: 3, vid: 5, exists: true},
		{uid: 5, vid: 6, exists: true},
		{uid: 4, vid: 7, exists: true},
		{uid: 6, vid: 7, exists: true},
		{uid: 6, vid: 8, exists: true},
		// not existing, reverse
		{uid: 4, vid: 1, exists: false},
		{uid: 4, vid: 2, exists: false},
		{uid: 5, vid: 2, exists: false},
		{uid: 5, vid: 3, exists: false},
		{uid: 6, vid: 5, exists: false},
		{uid: 7, vid: 4, exists: false},
		{uid: 7, vid: 6, exists: false},
		{uid: 8, vid: 6, exists: false},
		// not existing, dummy
		{uid: 1, vid: 3, exists: false},
		{uid: 2, vid: 3, exists: false},
		{uid: 3, vid: 3, exists: false},
		{uid: 5, vid: 7, exists: false},
		{uid: 4, vid: 6, exists: false},
		{uid: 4, vid: 8, exists: false},
	}
	for _, tc := range testCases {
		edge := net.Edge(tc.uid, tc.vid)
		if tc.exists {
			require.NotNilf(t, edge, "edge expected from: %d to: %d", tc.uid, tc.vid)
		} else {
			require.Nilf(t, edge, "edge not expected from: %d to: %d", tc.uid, tc.vid)
		}
	}
}

func TestNetwork_Node(t *testing.T) {
	net := buildNetwork()
	for _, n := range net.allNodesMIMO {
		node := net.Node(n.ID())
		require.NotNilf(t, node, "node expected, id: %d", n.ID())
	}
}

func TestNetwork_From(t *testing.T) {
	net := buildNetwork()
	testCases := []struct {
		id  int64
		ids []int64
	}{
		// existing
		{id: 1, ids: []int64{4}},
		{id: 2, ids: []int64{4, 5}},
		{id: 3, ids: []int64{5}},
		{id: 4, ids: []int64{7}},
		{id: 5, ids: []int64{6}},
		{id: 6, ids: []int64{7, 8}},
		// not existing
		{id: 7, ids: []int64{}},
		{id: 8, ids: []int64{}},
	}
	for _, tc := range testCases {
		nodes := net.From(tc.id)
		require.NotNilf(t, nodes, "must not be nil")
		ids := idsFromNodes(nodes)
		assert.EqualValues(t, tc.ids, ids, "wrong IDs of nodes from: %d", tc.id)
	}
}

func TestNetwork_Nodes(t *testing.T) {
	net := buildNetwork()
	ids := idsFromNodes(net.Nodes())
	for _, n := range net.AllNodes() {
		assert.Contains(t, ids, n.ID(), "node ID: %d expected", n.ID())
	}
}

func TestNetwork_HasEdgeBetween(t *testing.T) {
	net := buildNetwork()
	testCases := []struct {
		uid    int64
		vid    int64
		exists bool
	}{
		// existing
		{uid: 1, vid: 4, exists: true},
		{uid: 2, vid: 4, exists: true},
		{uid: 2, vid: 5, exists: true},
		{uid: 3, vid: 5, exists: true},
		{uid: 5, vid: 6, exists: true},
		{uid: 4, vid: 7, exists: true},
		{uid: 6, vid: 7, exists: true},
		{uid: 6, vid: 8, exists: true},
		// existing, reverse
		{uid: 4, vid: 1, exists: true},
		{uid: 4, vid: 2, exists: true},
		{uid: 5, vid: 2, exists: true},
		{uid: 5, vid: 3, exists: true},
		{uid: 6, vid: 5, exists: true},
		{uid: 7, vid: 4, exists: true},
		{uid: 7, vid: 6, exists: true},
		{uid: 8, vid: 6, exists: true},
		// not existing, dummy
		{uid: 1, vid: 3, exists: false},
		{uid: 2, vid: 3, exists: false},
		{uid: 3, vid: 3, exists: false},
		{uid: 5, vid: 7, exists: false},
		{uid: 4, vid: 6, exists: false},
		{uid: 4, vid: 8, exists: false},
	}
	for _, tc := range testCases {
		exists := net.HasEdgeBetween(tc.uid, tc.vid)
		require.Equal(t, tc.exists, exists, "edge expectation failed between: %d and %d", tc.uid, tc.vid)
	}
}

func TestNetwork_WeightedEdge(t *testing.T) {
	net := buildNetwork()

	testCases := []struct {
		uid    int64
		vid    int64
		weight float64
		exists bool
	}{
		// existing
		{uid: 1, vid: 4, weight: 15.0, exists: true},
		{uid: 2, vid: 4, weight: 10.0, exists: true},
		{uid: 2, vid: 5, weight: 5.0, exists: true},
		{uid: 3, vid: 5, weight: 1.0, exists: true},
		{uid: 5, vid: 6, weight: 17.0, exists: true},
		{uid: 4, vid: 7, weight: 7.0, exists: true},
		{uid: 6, vid: 7, weight: 4.5, exists: true},
		{uid: 6, vid: 8, weight: 13.0, exists: true},
		// not existing, reverse
		{uid: 4, vid: 1, exists: false},
		{uid: 4, vid: 2, exists: false},
		{uid: 5, vid: 2, exists: false},
		{uid: 5, vid: 3, exists: false},
		{uid: 6, vid: 5, exists: false},
		{uid: 7, vid: 4, exists: false},
		{uid: 7, vid: 6, exists: false},
		{uid: 8, vid: 6, exists: false},
		// not existing, dummy
		{uid: 1, vid: 3, exists: false},
		{uid: 2, vid: 3, exists: false},
		{uid: 3, vid: 3, exists: false},
		{uid: 5, vid: 7, exists: false},
		{uid: 4, vid: 6, exists: false},
		{uid: 4, vid: 8, exists: false},
	}
	for _, tc := range testCases {
		edge := net.WeightedEdge(tc.uid, tc.vid)
		if tc.exists {
			require.NotNilf(t, edge, "edge expected from: %d to: %d", tc.uid, tc.vid)
			assert.Equal(t, tc.weight, edge.Weight(), "wrong edge weight between: %d and: %d", tc.uid, tc.vid)
		} else {
			require.Nilf(t, edge, "edge not expected from: %d to: %d", tc.uid, tc.vid)
		}
	}
}

func TestNetwork_Weight(t *testing.T) {
	net := buildNetwork()

	testCases := []struct {
		uid    int64
		vid    int64
		weight float64
		exists bool
	}{
		// existing
		{uid: 1, vid: 4, weight: 15.0, exists: true},
		{uid: 2, vid: 4, weight: 10.0, exists: true},
		{uid: 2, vid: 5, weight: 5.0, exists: true},
		{uid: 3, vid: 5, weight: 1.0, exists: true},
		{uid: 5, vid: 6, weight: 17.0, exists: true},
		{uid: 4, vid: 7, weight: 7.0, exists: true},
		{uid: 6, vid: 7, weight: 4.5, exists: true},
		{uid: 6, vid: 8, weight: 13.0, exists: true},
		// not existing, reverse
		{uid: 4, vid: 1, exists: false},
		{uid: 4, vid: 2, exists: false},
		{uid: 5, vid: 2, exists: false},
		{uid: 5, vid: 3, exists: false},
		{uid: 6, vid: 5, exists: false},
		{uid: 7, vid: 4, exists: false},
		{uid: 7, vid: 6, exists: false},
		{uid: 8, vid: 6, exists: false},
		// not existing, dummy
		{uid: 1, vid: 3, exists: false},
		{uid: 2, vid: 3, exists: false},
		{uid: 3, vid: 3, exists: false},
		{uid: 5, vid: 7, exists: false},
		{uid: 4, vid: 6, exists: false},
		{uid: 4, vid: 8, exists: false},
	}
	for _, tc := range testCases {
		weight, ok := net.Weight(tc.uid, tc.vid)
		if tc.exists {
			require.True(t, ok, "edge expected from: %d to: %d", tc.uid, tc.vid)
			assert.Equal(t, tc.weight, weight, "wrong edge weight between: %d and: %d", tc.uid, tc.vid)
		} else {
			require.False(t, ok, "edge not expected from: %d to: %d", tc.uid, tc.vid)
		}
	}
}

func TestNetwork_HasEdgeFromTo(t *testing.T) {
	net := buildNetwork()

	testCases := []struct {
		uid    int64
		vid    int64
		exists bool
	}{
		// existing
		{uid: 1, vid: 4, exists: true},
		{uid: 2, vid: 4, exists: true},
		{uid: 2, vid: 5, exists: true},
		{uid: 3, vid: 5, exists: true},
		{uid: 5, vid: 6, exists: true},
		{uid: 4, vid: 7, exists: true},
		{uid: 6, vid: 7, exists: true},
		{uid: 6, vid: 8, exists: true},
		// not existing, reverse
		{uid: 4, vid: 1, exists: false},
		{uid: 4, vid: 2, exists: false},
		{uid: 5, vid: 2, exists: false},
		{uid: 5, vid: 3, exists: false},
		{uid: 6, vid: 5, exists: false},
		{uid: 7, vid: 4, exists: false},
		{uid: 7, vid: 6, exists: false},
		{uid: 8, vid: 6, exists: false},
		// not existing, dummy
		{uid: 1, vid: 3, exists: false},
		{uid: 2, vid: 3, exists: false},
		{uid: 3, vid: 3, exists: false},
		{uid: 5, vid: 7, exists: false},
		{uid: 4, vid: 6, exists: false},
		{uid: 4, vid: 8, exists: false},
	}
	for _, tc := range testCases {
		ok := net.HasEdgeFromTo(tc.uid, tc.vid)
		if tc.exists {
			require.True(t, ok, "edge expected from: %d to: %d", tc.uid, tc.vid)
		} else {
			require.False(t, ok, "edge not expected from: %d to: %d", tc.uid, tc.vid)
		}
	}
}

func TestNetwork_To(t *testing.T) {
	net := buildNetwork()
	testCases := []struct {
		id  int64
		ids []int64
	}{
		// existing
		{id: 4, ids: []int64{1, 2}},
		{id: 5, ids: []int64{2, 3}},
		{id: 6, ids: []int64{5}},
		{id: 7, ids: []int64{4, 6}},
		{id: 8, ids: []int64{6}},
		// not existing
		{id: 1, ids: []int64{}},
		{id: 2, ids: []int64{}},
	}
	for _, tc := range testCases {
		nodes := net.To(tc.id)
		require.NotNilf(t, nodes, "must not be nil")
		ids := idsFromNodes(nodes)
		assert.EqualValues(t, tc.ids, ids, "wrong IDs of nodes to: %d", tc.id)
	}
}

func idsFromNodes(nodes graph.Nodes) []int64 {
	res := make([]int64, 0)
	for nodes.Next() {
		res = append(res, nodes.Node().ID())
	}
	return res
}
