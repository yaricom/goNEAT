package genetics

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"strings"
	"testing"
)

func TestPlainGenomeWriter_WriteTrait(t *testing.T) {
	params := []float64{
		0.40227575878298616, 0.0, 0.0, 0.0, 0.0, 0.3245553261200018, 0.0, 0.12248956525856575,
	}
	traitId := 2
	trait := neat.NewTrait()
	trait.Id = traitId
	trait.Params = params

	traitStr := fmt.Sprintf("%d %g %g %g %g %g %g %g %g",
		traitId, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])

	outBuffer := bytes.NewBufferString("")
	wr := plainGenomeWriter{w: bufio.NewWriter(outBuffer)}
	err := wr.writeTrait(trait)
	require.NoError(t, err, "failed to write trait")
	err = wr.w.Flush()
	require.NoError(t, err)

	outStr := strings.TrimSpace(outBuffer.String())
	assert.Equal(t, traitStr, outStr)
}

func TestPlainGenomeWriter_WriteTrait_writeError(t *testing.T) {
	errorWriter := ErrorWriter(1)
	wr := plainGenomeWriter{w: bufio.NewWriterSize(&errorWriter, 1)}
	err := wr.writeTrait(neat.NewTrait())
	assert.EqualError(t, err, alwaysErrorText)
}

// Tests NNode serialization
func TestPlainGenomeWriter_WriteNetworkNode(t *testing.T) {
	nodeId, traitId, nodeType, neuronType := 1, 10, network.SensorNode, network.InputNeuron
	nodeStr := fmt.Sprintf("%d %d %d %d SigmoidSteepenedActivation", nodeId, traitId, nodeType, neuronType)
	trait := neat.NewTrait()
	trait.Id = 10

	node := network.NewNNode(nodeId, neuronType)
	node.Trait = trait
	outBuffer := bytes.NewBufferString("")

	wr := plainGenomeWriter{w: bufio.NewWriter(outBuffer)}
	err := wr.writeNetworkNode(node)
	require.NoError(t, err, "failed to write network node")
	err = wr.w.Flush()
	require.NoError(t, err)

	outStr := outBuffer.String()
	assert.Equal(t, nodeStr, outStr, "Node serialization failed")
}

func TestPlainGenomeWriter_WriteNetworkNode_writeError(t *testing.T) {
	errorWriter := ErrorWriter(1)
	wr := plainGenomeWriter{w: bufio.NewWriterSize(&errorWriter, 1)}
	err := wr.writeNetworkNode(network.NewNNode(1, network.InputNeuron))
	assert.EqualError(t, err, alwaysErrorText)
}

func TestPlainGenomeWriter_WriteConnectionGene(t *testing.T) {
	// gene  1 1 4 1.1983046913458986 0 1.0 1.1983046913458986 0
	traitId, inNodeId, outNodeId, innovNum := 1, 1, 4, int64(1)
	weight, mutNum := 1.1983046913458986, 1.1983046913458986
	recurrent, enabled := false, false
	geneStr := fmt.Sprintf("%d %d %d %g %t %d %g %t",
		traitId, inNodeId, outNodeId, weight, recurrent, innovNum, mutNum, enabled)

	trait := neat.NewTrait()
	trait.Id = traitId
	gene := NewGeneWithTrait(trait, weight, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(4, network.HiddenNeuron), false, innovNum, mutNum)
	gene.IsEnabled = false

	outBuf := bytes.NewBufferString("")

	wr := plainGenomeWriter{w: bufio.NewWriter(outBuf)}
	err := wr.writeConnectionGene(gene)
	require.NoError(t, err, "failed to write connection gene")
	err = wr.w.Flush()
	require.NoError(t, err)

	outStr := outBuf.String()
	assert.Equal(t, geneStr, outStr, "Wrong Gene serialization")
}

func TestPlainGenomeWriter_WriteConnectionGene_writeError(t *testing.T) {
	errorWriter := ErrorWriter(1)
	wr := plainGenomeWriter{w: bufio.NewWriterSize(&errorWriter, 1)}
	err := wr.writeConnectionGene(NewGeneWithTrait(neat.NewTrait(), 1.0, network.NewNNode(1, network.InputNeuron),
		network.NewNNode(4, network.HiddenNeuron), false, 1.0, 1.0))
	assert.EqualError(t, err, alwaysErrorText)
}

func TestPlainGenomeWriter_WriteGenome(t *testing.T) {
	gnome := buildTestGenome(1)
	outBuf := bytes.NewBufferString("")
	wr, err := NewGenomeWriter(bufio.NewWriter(outBuf), PlainGenomeEncoding)
	require.NoError(t, err, "failed to create genome writer")
	err = wr.WriteGenome(gnome)
	require.NoError(t, err, "failed to write genome")

	inputScanner := bufio.NewScanner(strings.NewReader(gnomeStr))
	inputScanner.Split(bufio.ScanLines)

	outScanner := bufio.NewScanner(outBuf)
	outScanner.Split(bufio.ScanLines)

	for inputScanner.Scan() {
		if !outScanner.Scan() {
			t.Error("Unexpected end of genome data")
		}
		inText := inputScanner.Text()
		outText := outScanner.Text()
		require.Equal(t, inText, outText, "lines mismatch at")
	}
}

func TestPlainGenomeWriter_WriteGenome_writeError(t *testing.T) {
	errorWriter := ErrorWriter(1)
	wr, err := NewGenomeWriter(bufio.NewWriter(&errorWriter), PlainGenomeEncoding)
	require.NoError(t, err)
	require.NotNil(t, wr)

	gnome := buildTestGenome(1)
	err = wr.WriteGenome(gnome)
	assert.EqualError(t, err, alwaysErrorText)
}

func TestYamlGenomeWriter_WriteGenome(t *testing.T) {
	gnome := buildTestModularGenome(1)

	// encode genome
	outBuf := bytes.NewBufferString("")
	wr, err := NewGenomeWriter(bufio.NewWriter(outBuf), YAMLGenomeEncoding)
	require.NoError(t, err)
	err = wr.WriteGenome(gnome)
	require.NoError(t, err, "failed to write genome")
	//t.Log(outBuf.String())

	// decode genome and compare
	enc := yamlGenomeReader{r: bufio.NewReader(bytes.NewBuffer(outBuf.Bytes()))}
	gnomeEnc, err := enc.Read()
	require.NoError(t, err, "failed to read genome")

	assert.Equal(t, gnome.Id, gnomeEnc.Id, "wrong genome ID")

	// check encoded genes
	//
	assert.Len(t, gnomeEnc.Genes, len(gnome.Genes), "wrong genes number")
	for i, g := range gnome.Genes {
		og := gnomeEnc.Genes[i]
		assert.True(t, g.Link.IsEqualGenetically(og.Link), "genes not equal genetically at: %d", i)
		assert.Equal(t, g.IsEnabled, og.IsEnabled, "at: %d", i)
		assert.Equal(t, g.MutationNum, og.MutationNum, "at: %d", i)
		assert.Equal(t, g.InnovationNum, og.InnovationNum, "at: %d", i)
	}

	// check encoded nodes
	//
	assert.Len(t, gnomeEnc.Nodes, len(gnome.Nodes), "wrong number of nodes encoded")
	for i, n := range gnome.Nodes {
		nd := gnomeEnc.Nodes[i]
		assert.Equal(t, n.Id, nd.Id, "wrong node ID at: %d", i)
		assert.Equal(t, n.ActivationType, nd.ActivationType, "wrong node activation at: %d", i)
		assert.Equal(t, n.NeuronType, nd.NeuronType, "wrong node neuron type at: %d", i)
	}

	// check encoded traits
	//
	assert.Len(t, gnomeEnc.Traits, len(gnome.Traits), "wrong number of traits encoded")
	for i, tr := range gnome.Traits {
		etr := gnomeEnc.Traits[i]
		assert.Equal(t, tr.Id, etr.Id, "wrong trait ID at: %d", i)
		assert.ElementsMatch(t, tr.Params, etr.Params, "wrong trait params at: %d", i)
	}

	// check control genes
	//
	assert.Len(t, gnomeEnc.ControlGenes, len(gnome.ControlGenes), "wrong numbre of control genes encoded")
	for i, cg := range gnome.ControlGenes {
		ocg := gnomeEnc.ControlGenes[i]
		assert.Equal(t, cg.IsEnabled, ocg.IsEnabled, "wrong enabled at: %d", i)
		assert.Equal(t, cg.MutationNum, ocg.MutationNum, "wrong mutation number at: %d", i)
		assert.Equal(t, cg.InnovationNum, ocg.InnovationNum, "wrong innovation at: %d", i)
		assert.Equal(t, cg.ControlNode.Id, ocg.ControlNode.Id, "wrong node ID at: %d", i)
		checkLinks(cg.ControlNode.Incoming, ocg.ControlNode.Incoming, t)
		checkLinks(cg.ControlNode.Outgoing, ocg.ControlNode.Outgoing, t)
	}
}

func TestYamlGenomeWriter_WriteGenome_writeError(t *testing.T) {
	errorWriter := ErrorWriter(1)
	wr, err := NewGenomeWriter(bufio.NewWriter(&errorWriter), YAMLGenomeEncoding)
	require.NoError(t, err)
	require.NotNil(t, wr)

	gnome := buildTestGenome(1)
	err = wr.WriteGenome(gnome)
	assert.EqualError(t, err, alwaysErrorText)
}

func checkLinks(left, right []*network.Link, t *testing.T) {
	require.Equal(t, len(left), len(right), "Links length mismatch")

	for i, l := range left {
		r := right[i]
		assert.Equal(t, l.InNode.Id, r.InNode.Id, "wrong link InNode ID at: %d", i)
		assert.Equal(t, l.OutNode.Id, r.OutNode.Id, "wrong link OutNode ID at: %d", i)
		assert.Equal(t, l.ConnectionWeight, r.ConnectionWeight, "wrong link Weight at: %d", i)
	}
}
