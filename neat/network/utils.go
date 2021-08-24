package network

import (
	"errors"
	"fmt"
	"gonum.org/v1/gonum/graph"
	"io"
)

// PrintAllActivationDepthPaths is to print all paths used to find the maximal activation depth of the network
func PrintAllActivationDepthPaths(n *Network, w io.Writer) error {
	_, err := n.maxActivationDepth(w)
	return err
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
