package genetics

import (
	"bufio"
	"errors"
	"fmt"
	"github.com/spf13/cast"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"gopkg.in/yaml.v3"
	"io"
	"strconv"
	"strings"
)

// GenomeReader The interface to define genome reader
type GenomeReader interface {
	// Read is tp read one Genome record
	Read() (*Genome, error)
}

// NewGenomeReader Creates reader for Genome data with specified encoding format.
func NewGenomeReader(r io.Reader, encoding GenomeEncoding) (GenomeReader, error) {
	switch encoding {
	case PlainGenomeEncoding:
		return &plainGenomeReader{r: bufio.NewReader(r)}, nil
	case YAMLGenomeEncoding:
		return &yamlGenomeReader{r: bufio.NewReader(r)}, nil
	default:
		return nil, ErrUnsupportedGenomeEncoding
	}
}

// A PlainGenomeReader reads genome data from plain text file.
type plainGenomeReader struct {
	r *bufio.Reader
}

func (r *plainGenomeReader) Read() (*Genome, error) {
	gnome := Genome{
		Traits: make([]*neat.Trait, 0),
		Nodes:  make([]*network.NNode, 0),
		Genes:  make([]*Gene, 0),
	}

	var gId int
	// Loop until file is finished, parsing each line
	scanner := bufio.NewScanner(r.r)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, " ", 2)
		if len(parts) < 2 {
			return nil, fmt.Errorf("line: [%s] can not be split when reading Genome", line)
		}
		lr := strings.NewReader(parts[1])

		switch parts[0] {
		case "trait":
			// Read a Trait
			newTrait, err := readPlainTrait(lr)
			if err != nil {
				return nil, err
			}
			// check that trait ID is unique
			if prevTrait := TraitWithId(newTrait.Id, gnome.Traits); prevTrait != nil {
				return nil, fmt.Errorf("trait ID: %d is not unique", newTrait.Id)
			}
			gnome.Traits = append(gnome.Traits, newTrait)

		case "node":
			// Read a Network Node
			newNode, err := readPlainNetworkNode(lr, gnome.Traits)
			if err != nil {
				return nil, err
			}
			// check that node ID is unique
			if prevNode := NodeWithId(newNode.Id, gnome.Nodes); prevNode != nil {
				return nil, fmt.Errorf("node ID: %d is not unique", newNode.Id)
			}
			gnome.Nodes = append(gnome.Nodes, newNode)

		case "gene":
			// Read a Gene
			gene, err := readPlainConnectionGene(lr, gnome.Traits, gnome.Nodes)
			if err != nil {
				return nil, err
			}
			gnome.Genes = append(gnome.Genes, gene)

		case "genomeend":
			// Read Genome ID
			_, err := fmt.Fscanf(lr, "%d", &gId)
			if err != nil {
				return nil, err
			}
			// save genome ID
			gnome.Id = gId

		case "/*":
			// read all comments and print it
			neat.InfoLog(line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return &gnome, nil
}

// The method to read Trait in plain text format
func readPlainTrait(r io.Reader) (*neat.Trait, error) {
	nt := neat.NewTrait()
	if _, err := fmt.Fscanf(r, "%d ", &nt.Id); err != nil {
		return nil, err
	}
	for i := 0; i < neat.NumTraitParams; i++ {
		if _, err := fmt.Fscanf(r, "%g ", &nt.Params[i]); err != nil {
			return nil, err
		}
	}

	return nt, nil
}

// Read a Network Node from specified Reader in plain text format
// and applies corresponding trait to it from a list of traits provided
func readPlainNetworkNode(r io.Reader, traits []*neat.Trait) (*network.NNode, error) {
	n := network.NewNetworkNode()
	buff := bufio.NewReader(r)
	line, _, err := buff.ReadLine()
	if err != nil {
		return nil, err
	}
	parts := strings.Split(string(line), " ")
	if len(parts) < 4 {
		return nil, fmt.Errorf("node line is too short: %d (%s)", len(parts), parts)
	}
	if nId, err := strconv.ParseInt(parts[0], 10, 32); err != nil {
		return nil, err
	} else {
		n.Id = int(nId)
	}
	if traitId, err := strconv.ParseInt(parts[1], 10, 32); err != nil {
		return nil, err
	} else {
		n.Trait = TraitWithId(int(traitId), traits)
	}

	if neuronType, err := strconv.ParseInt(parts[3], 10, 8); err != nil {
		return nil, err
	} else {
		n.NeuronType = network.NodeNeuronType(neuronType)
	}

	if len(parts) == 5 {
		n.ActivationType, err = math.NodeActivators.ActivationTypeFromName(parts[4])
	}

	return n, err
}

// Reads Gene from reader in plain text format
func readPlainConnectionGene(r io.Reader, traits []*neat.Trait, nodes []*network.NNode) (*Gene, error) {
	var traitId, inNodeId, outNodeId int
	var innovationNum int64
	var weight, mutNum float64
	var recurrent, enabled bool
	_, err := fmt.Fscanf(r, "%d %d %d %g %t %d %g %t ",
		&traitId, &inNodeId, &outNodeId, &weight, &recurrent, &innovationNum, &mutNum, &enabled)
	if err != nil {
		return nil, err
	}

	trait := TraitWithId(traitId, traits)
	var inNode, outNode *network.NNode
	for _, np := range nodes {
		if np.Id == inNodeId {
			inNode = np
		}
		if np.Id == outNodeId {
			outNode = np
		}
	}
	if trait != nil {
		return NewConnectionGene(network.NewLinkWithTrait(trait, weight, inNode, outNode, recurrent), innovationNum, mutNum, enabled), nil
	} else {
		return NewConnectionGene(network.NewLink(weight, inNode, outNode, recurrent), innovationNum, mutNum, enabled), nil
	}
}

// A YAMLGenomeReader reads genome data from YAML encoded text file
type yamlGenomeReader struct {
	r *bufio.Reader
}

func (r *yamlGenomeReader) Read() (*Genome, error) {
	m := make(map[string]interface{})
	dec := yaml.NewDecoder(r.r)
	err := dec.Decode(&m)
	if err != nil {
		return nil, err
	}

	gm, ok := m["genome"].(map[string]interface{})
	if !ok {
		return nil, errors.New("failed to parse YAML configuration")
	}

	// read Genome
	genId, err := cast.ToIntE(gm["id"])
	if err != nil {
		return nil, err
	}
	gnome := &Genome{
		Id:           genId,
		Traits:       make([]*neat.Trait, 0),
		Nodes:        make([]*network.NNode, 0),
		Genes:        make([]*Gene, 0),
		ControlGenes: make([]*MIMOControlGene, 0),
	}

	// read traits
	traits := gm["traits"].([]interface{})
	for _, tr := range traits {
		trait, err := readTrait(tr.(map[string]interface{}))
		if err != nil {
			return nil, err
		}
		// check that trait ID is unique
		if prevTrait := TraitWithId(trait.Id, gnome.Traits); prevTrait != nil {
			return nil, fmt.Errorf("trait ID: %d is not unique", trait.Id)
		}
		gnome.Traits = append(gnome.Traits, trait)
	}

	// read nodes
	nodes := gm["nodes"].([]interface{})
	for _, nd := range nodes {
		node, err := readNNode(nd.(map[string]interface{}), gnome.Traits)
		if err != nil {
			return nil, err
		}
		// check that node ID is unique
		if prevNode := NodeWithId(node.Id, gnome.Nodes); prevNode != nil {
			return nil, fmt.Errorf("node ID: %d is not unique", node.Id)
		}
		gnome.Nodes = append(gnome.Nodes, node)
	}

	// read Genes
	genes := gm["genes"].([]interface{})
	for _, g := range genes {
		gene, err := readGene(g.(map[string]interface{}), gnome.Traits, gnome.Nodes)
		if err != nil {
			return nil, err
		}
		gnome.Genes = append(gnome.Genes, gene)
	}

	// read MIMO control genes
	mimoGenes := gm["modules"]
	if mimoGenes != nil {
		for _, mg := range mimoGenes.([]interface{}) {
			mGene, err := readMIMOControlGene(mg.(map[string]interface{}), gnome.Traits, gnome.Nodes)
			if err != nil {
				return nil, err
			}
			// check that control node ID is unique
			if prevNode := NodeWithId(mGene.ControlNode.Id, gnome.Nodes); prevNode != nil {
				return nil, fmt.Errorf("control node ID: %d is not unique", mGene.ControlNode.Id)
			}
			gnome.ControlGenes = append(gnome.ControlGenes, mGene)
		}
	}

	return gnome, nil
}

// Reads gene configuration
func readGene(conf map[string]interface{}, traits []*neat.Trait, nodes []*network.NNode) (*Gene, error) {
	traitId := conf["trait_id"].(int)
	inNodeId := conf["src_id"].(int)
	outNodeId := conf["tgt_id"].(int)
	innovationNum, err := cast.ToInt64E(conf["innov_num"])
	if err != nil {
		return nil, err
	}
	weight, err := cast.ToFloat64E(conf["weight"])
	if err != nil {
		return nil, err
	}
	mutNum, err := cast.ToFloat64E(conf["mut_num"])
	if err != nil {
		return nil, err
	}
	recurrent, err := cast.ToBoolE(conf["recurrent"])
	if err != nil {
		return nil, err
	}
	enabled, err := cast.ToBoolE(conf["enabled"])
	if err != nil {
		return nil, err
	}

	trait := TraitWithId(traitId, traits)
	var inNode, outNode *network.NNode
	for _, np := range nodes {
		if np.Id == inNodeId {
			inNode = np
		}
		if np.Id == outNodeId {
			outNode = np
		}
	}
	if trait != nil {
		return NewConnectionGene(network.NewLinkWithTrait(trait, weight, inNode, outNode, recurrent), innovationNum, mutNum, enabled), nil
	} else {
		return NewConnectionGene(network.NewLink(weight, inNode, outNode, recurrent), innovationNum, mutNum, enabled), nil
	}
}

// Reads MIMOControlGene configuration
func readMIMOControlGene(conf map[string]interface{}, traits []*neat.Trait, nodes []*network.NNode) (gene *MIMOControlGene, err error) {
	// read control node parameters
	controlNode := network.NewNetworkNode()
	controlNode.Id = conf["id"].(int)
	controlNode.NeuronType = network.HiddenNeuron
	// set activation function
	activation := conf["activation"].(string)
	controlNode.ActivationType, err = math.NodeActivators.ActivationTypeFromName(activation)
	if err != nil {
		return nil, err
	}
	// set associated Trait
	traitId := conf["trait_id"].(int)
	trait := TraitWithId(traitId, traits)
	controlNode.Trait = trait

	// read MIMO gene parameters
	innovationNum, err := cast.ToInt64E(conf["innov_num"])
	if err != nil {
		return nil, err
	}
	mutNum, err := cast.ToFloat64E(conf["mut_num"])
	if err != nil {
		return nil, err
	}
	enabled, err := cast.ToBoolE(conf["enabled"])
	if err != nil {
		return nil, err
	}

	// read input links
	inNodes, err := cast.ToSliceE(conf["inputs"])
	if err != nil {
		return nil, err
	}
	controlNode.Incoming = make([]*network.Link, len(inNodes))
	for i, mn := range inNodes {
		n := mn.(map[string]interface{})
		nodeId, err := cast.ToIntE(n["id"])
		if err != nil {
			return nil, err
		}
		node := NodeWithId(nodeId, nodes)
		if node != nil {
			controlNode.Incoming[i] = network.NewLink(1.0, node, controlNode, false)
		} else {
			return nil, fmt.Errorf("no MIMO input node with id: %d can be found for module: %d",
				nodeId, controlNode.Id)
		}
	}

	// read output links
	outNodes, err := cast.ToSliceE(conf["outputs"])
	if err != nil {
		return nil, err
	}
	controlNode.Outgoing = make([]*network.Link, len(outNodes))
	for i, mn := range outNodes {
		n := mn.(map[string]interface{})
		nodeId, err := cast.ToIntE(n["id"])
		if err != nil {
			return nil, err
		}
		node := NodeWithId(nodeId, nodes)
		if node != nil {
			controlNode.Outgoing[i] = network.NewLink(1.0, controlNode, node, false)
		} else {
			return nil, fmt.Errorf("no MIMO output node with id: %d can be found for module: %d",
				nodeId, controlNode.Id)
		}
	}

	// build gene
	gene = NewMIMOGene(controlNode, innovationNum, mutNum, enabled)

	return gene, nil
}

// Reads NNode configuration
func readNNode(conf map[string]interface{}, traits []*neat.Trait) (*network.NNode, error) {
	nd := network.NewNetworkNode()
	nd.Id = conf["id"].(int)
	traitId := conf["trait_id"].(int)
	nd.Trait = TraitWithId(traitId, traits)
	typeName := conf["type"].(string)
	var err error
	nd.NeuronType, err = network.NeuronTypeByName(typeName)
	if err != nil {
		return nil, err
	}
	activation := conf["activation"].(string)
	nd.ActivationType, err = math.NodeActivators.ActivationTypeFromName(activation)
	return nd, err
}

// Reads Trait configuration
func readTrait(conf map[string]interface{}) (*neat.Trait, error) {
	nt := neat.NewTrait()
	nt.Id = conf["id"].(int)
	params := cast.ToSlice(conf["params"])
	var err error
	for i, p := range params {
		nt.Params[i], err = cast.ToFloat64E(p)
		if err != nil {
			return nil, err
		}
	}
	return nt, nil
}
