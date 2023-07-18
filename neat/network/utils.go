package network

import (
	"errors"
	"fmt"
	"gonum.org/v1/gonum/graph"
	"io"
)

// PrintAllActivationDepthPaths is to print all paths used to find the maximal activation depth of the network
func PrintAllActivationDepthPaths(n *Network, w io.Writer) error {
	if len(n.controlNodes) > 0 {
		_, err := n.maxActivationDepthModular(w)
		return err
	}

	for _, node := range n.Outputs {
		path := make([]int, n.NodeCount())
		pathIndex := 0
		if err := node.printDepthPaths(path, &pathIndex, w); err != nil {
			return err
		}
	}
	return nil
}

// PrintPath is to print the given paths into specified writer
func PrintPath(w io.Writer, paths [][]graph.Node) (err error) {
	if paths == nil {
		return errors.New("the paths are empty")
	}
	for _, p := range paths {
		l := len(p)
		for i, n := range p {
			if i < l-1 {
				_, err = fmt.Fprintf(w, "%d -> ", n.ID())
			} else {
				_, err = fmt.Fprintln(w, n.ID())
			}
		}
	}
	return err
}
