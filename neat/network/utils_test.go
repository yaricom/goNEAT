package network

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/graph/path"
	"strings"
	"testing"
)

func TestPrintAllActivationDepthPaths_Simple(t *testing.T) {
	net := buildNetwork()

	b := bytes.NewBufferString("")
	err := PrintAllActivationDepthPaths(net, b)
	require.NoError(t, err, "failed to print")
	t.Log(b.String())
	expected := "1 -> 4 -> 7\n---------------\n2 -> 4 -> 7\n2 -> 5 -> 6 -> 8\n---------------\n3 -> 5 -> 6 -> 7\n3 -> 5 -> 6 -> 8\n---------------\n"
	assert.Equal(t, expected, b.String())
}

func TestPrintAllActivationDepthPaths_Modular(t *testing.T) {
	net := buildModularNetwork()

	b := bytes.NewBufferString("")
	err := PrintAllActivationDepthPaths(net, b)
	require.NoError(t, err, "failed to print")
	t.Log(b.String())
	expected := "1 -> 4 -> 6 -> 7 -> 8\n1 -> 4 -> 6 -> 7 -> 9\n---------------\n2 -> 5 -> 6 -> 7 -> 8\n2 -> 5 -> 6 -> 7 -> 9\n---------------\n3 -> 5 -> 6 -> 7 -> 8\n3 -> 5 -> 6 -> 7 -> 9\n---------------\n"
	assert.Equal(t, expected, b.String())
}

func TestPrintPath(t *testing.T) {
	net := buildNetwork()

	allPaths, ok := path.JohnsonAllPaths(net)
	require.True(t, ok, "failed to get all paths")
	paths, _ := allPaths.AllBetween(net.inputs[2].ID(), net.Outputs[1].ID())
	require.NotNilf(t, paths, "failed to get specific paths")

	b := bytes.NewBufferString("")
	err := PrintPath(b, paths)
	require.NoError(t, err, "failed to print path")
	t.Log(b)
	expected := "3 -> 5 -> 6 -> 8"
	assert.Equal(t, expected, strings.TrimSpace(b.String()), "wrong path")
}
