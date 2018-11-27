package genetics

import (
	"io"
	"fmt"

	"github.com/yaricom/goNEAT/neat"
	"bufio"
	"errors"
	"github.com/yaricom/goNEAT/neat/network"
	"strings"
	"github.com/spf13/cast"
	"gopkg.in/yaml.v2"
	"strconv"
)


// The interface to define genome reader
type GenomeReader interface {
	// Reads one Genome record
	Read() (*Genome, error)
}

// Creates reader for Genome data with specified encoding format.
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

func (pgr *plainGenomeReader) Read() (*Genome, error) {
	gnome := Genome{
		Traits:make([]*neat.Trait, 0),
		Nodes:make([]*network.NNode, 0),
		Genes:make([]*Gene, 0),
	}

	var g_id int
	// Loop until file is finished, parsing each line
	scanner := bufio.NewScanner(pgr.r)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, " ", 2)
		if len(parts) < 2 {
			return nil, errors.New(fmt.Sprintf("Line: [%s] can not be split when reading Genome", line))
		}
		lr := strings.NewReader(parts[1])

		switch parts[0] {
		case "trait":
			// Read a Trait
			new_trait, err := readPlainTrait(lr)
			if err != nil {
				return nil, err
			}
			gnome.Traits = append(gnome.Traits, new_trait)

		case "node":
			// Read a Network Node
			new_node, err := readPlainNetworkNode(lr, gnome.Traits)
			if err != nil {
				return nil, err
			}
			gnome.Nodes = append(gnome.Nodes, new_node)

		case "gene":
			// Read a Gene
			new_gene, err := readPlainConnectionGene(lr, gnome.Traits, gnome.Nodes)
			if err != nil {
				return nil, err
			}
			gnome.Genes = append(gnome.Genes, new_gene)

		case "genomeend":
			// Read Genome ID
			_, err := fmt.Fscanf(lr, "%d", &g_id)
			if err != nil {
				return nil, err
			}
			// save genome ID
			gnome.Id = g_id

		case "/*":
			// read all comments and print it
			neat.InfoLog(line)
		}
	}
	return &gnome, nil
}

// The method to read Trait in plain text format
func readPlainTrait(r io.Reader) (*neat.Trait, error) {
	nt := neat.NewTrait()
	_, err := fmt.Fscanf(r, "%d ", &nt.Id)
	if err == nil {
		for i := 0; i < neat.Num_trait_params; i++ {
			if _, err = fmt.Fscanf(r, "%g ", &nt.Params[i]); err != nil {
				return nil, err
			}
		}
	}

	return nt, err
}

// Read a Network Node from specified Reader in plain text format
// and applies corresponding trait to it from a list of traits provided
func readPlainNetworkNode(r io.Reader, traits []*neat.Trait) (*network.NNode, error) {
	n := network.NewNetworkNode()
	buf_r := bufio.NewReader(r)
	line, _, err := buf_r.ReadLine()
	if err != nil {
		return nil, err
	}
	parts := strings.Split(string(line), " ")
	if len(parts) < 4 {
		return nil, errors.New(fmt.Sprintf("node line is too short: %d (%s)", len(parts), parts))
	}
	if n_Id, err := strconv.ParseInt(parts[0], 10, 32); err != nil {
		return nil, err
	} else {
		n.Id = int(n_Id)
	}
	if trait_id, err := strconv.ParseInt(parts[1], 10, 32); err != nil {
		return nil, err
	} else {
		n.Trait = traitWithId(int(trait_id), traits)
	}

	if n_NeuronType, err := strconv.ParseInt(parts[3], 10, 32); err != nil {
		return nil, err
	} else {
		n.NeuronType = network.NodeNeuronType(n_NeuronType)
	}

	if len(parts) == 5 {
		n.ActivationType, err = network.NodeActivators.ActivationTypeFromName(parts[4])
	}

	return n, err
}

// Reads Gene from reader in plain text format
func readPlainConnectionGene(r io.Reader, traits []*neat.Trait, nodes []*network.NNode) (*Gene, error) {
	var traitId, inNodeId, outNodeId int
	var inov_num int64
	var weight, mut_num float64
	var recurrent, enabled bool
	_, err := fmt.Fscanf(r, "%d %d %d %g %t %d %g %t ",
		&traitId, &inNodeId, &outNodeId, &weight, &recurrent, &inov_num, &mut_num, &enabled)
	if err != nil {
		return nil, err
	}

	trait := traitWithId(traitId, traits)
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
		return newGene(network.NewLinkWithTrait(trait, weight, inNode, outNode, recurrent), inov_num, mut_num, enabled), nil
	} else {
		return newGene(network.NewLink(weight, inNode, outNode, recurrent), inov_num, mut_num, enabled), nil
	}
}

// A YAMLGenomeReader reads genome data from YAML encoded text file
type yamlGenomeReader struct {
	r *bufio.Reader
}

func (ygr *yamlGenomeReader) Read() (*Genome, error) {
	m := make(map[interface{}]interface{})
	dec := yaml.NewDecoder(ygr.r)
	err := dec.Decode(&m)
	if err != nil {
		return nil, err
	}

	gm, ok := m["genome"].(map[interface{}]interface{})
	if ok == false {
		return nil, errors.New("failed to parse YAML configuration")
	}

	// read Genome
	gen_id, err := cast.ToIntE(gm["id"])
	if err != nil {
		return nil, err
	}
	gnome := &Genome{
		Id:gen_id,
		Traits:make([]*neat.Trait, 0),
		Nodes:make([]*network.NNode, 0),
		Genes:make([]*Gene, 0),
		ControlGenes:make([]*MIMOControlGene, 0),
	}

	// read traits
	traits := gm["traits"].([]interface{})
	for _, tr := range traits {
		trait, err := readTrait(tr.(map[interface{}]interface{}))
		if err != nil {
			return nil, err
		}
		gnome.Traits = append(gnome.Traits, trait)
	}

	// read nodes
	nodes := gm["nodes"].([]interface{})
	for _, nd := range nodes {
		node, err := readNNode(nd.(map[interface{}]interface{}), gnome.Traits)
		if err != nil {
			return nil, err
		}
		gnome.Nodes = append(gnome.Nodes, node)
	}

	// read Genes
	genes := gm["genes"].([]interface{})
	for _, g := range genes {
		gene, err := readGene(g.(map[interface{}]interface{}), gnome.Traits, gnome.Nodes)
		if err != nil {
			return nil, err
		}
		gnome.Genes = append(gnome.Genes, gene)
	}

	// read MIMO control genes
	mimoGenes := gm["modules"]
	if mimoGenes != nil {
		for _, mg := range mimoGenes.([]interface{}) {
			mGene, err := readMIMOControlGene(mg.(map[interface{}]interface{}), gnome.Traits, gnome.Nodes)
			if err != nil {
				return nil, err
			}
			gnome.ControlGenes = append(gnome.ControlGenes, mGene)
		}
	}

	return gnome, nil
}

// Reads gene configuration
func readGene(conf map[interface{}]interface{}, traits []*neat.Trait, nodes []*network.NNode) (*Gene, error) {
	traitId := conf["trait_id"].(int)
	inNodeId := conf["src_id"].(int)
	outNodeId := conf["tgt_id"].(int)
	inov_num, err := cast.ToInt64E(conf["innov_num"])
	if err != nil {
		return nil, err
	}
	weight, err := cast.ToFloat64E(conf["weight"])
	if err != nil {
		return nil, err
	}
	mut_num, err := cast.ToFloat64E(conf["mut_num"])
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

	trait := traitWithId(traitId, traits)
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
		return newGene(network.NewLinkWithTrait(trait, weight, inNode, outNode, recurrent), inov_num, mut_num, enabled), nil
	} else {
		return newGene(network.NewLink(weight, inNode, outNode, recurrent), inov_num, mut_num, enabled), nil
	}
}

// Reads MIMOControlGene configuration
func readMIMOControlGene(conf map[interface{}]interface{}, traits []*neat.Trait, nodes []*network.NNode) (gene *MIMOControlGene, err error) {
	// read control node parameters
	control_node := network.NewNetworkNode()
	control_node.Id = conf["id"].(int)
	control_node.NeuronType = network.HiddenNeuron
	// set activation function
	activation := conf["activation"].(string)
	control_node.ActivationType, err = network.NodeActivators.ActivationTypeFromName(activation)
	if err != nil {
		return nil, err
	}
	// set associated Trait
	traitId := conf["trait_id"].(int)
	trait := traitWithId(traitId, traits)
	control_node.Trait = trait

	// read MIMO gene parameters
	inov_num, err := cast.ToInt64E(conf["innov_num"])
	if err != nil {
		return nil, err
	}
	mut_num, err := cast.ToFloat64E(conf["mut_num"])
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
	control_node.Incoming = make([]*network.Link, len(inNodes))
	for i, mn := range inNodes {
		n := mn.(map[interface{}]interface{})
		node_id, err := cast.ToIntE(n["id"])
		if err != nil {
			return nil, err
		}
		node := nodeWithId(node_id, nodes)
		if node != nil {
			control_node.Incoming[i] = network.NewLink(1.0, node, control_node, false)
		} else {
			return nil, errors.New(fmt.Sprintf("no MIMO input node with id: %d can be found for module: %d",
				node_id, control_node.Id))
		}
	}

	// read output links
	outNodes, err := cast.ToSliceE(conf["outputs"])
	if err != nil {
		return nil, err
	}
	control_node.Outgoing = make([]*network.Link, len(outNodes))
	for i, mn := range outNodes {
		n := mn.(map[interface{}]interface{})
		node_id, err := cast.ToIntE(n["id"])
		if err != nil {
			return nil, err
		}
		node := nodeWithId(node_id, nodes)
		if node != nil {
			control_node.Outgoing[i] = network.NewLink(1.0, control_node, node, false)
		} else {
			return nil, errors.New(fmt.Sprintf("no MIMO output node with id: %d can be found for module: %d",
				node_id, control_node.Id))
		}
	}

	// build gene
	gene = NewMIMOGene(control_node, inov_num, mut_num, enabled)

	return gene, nil
}

// Reads NNode configuration
func readNNode(conf map[interface{}]interface{}, traits []*neat.Trait) (*network.NNode, error) {
	nd := network.NewNetworkNode()
	nd.Id = conf["id"].(int)
	trait_id := conf["trait_id"].(int)
	nd.Trait = traitWithId(trait_id, traits)
	type_name := conf["type"].(string)
	var err error
	nd.NeuronType, err = network.NeuronTypeByName(type_name)
	if err != nil {
		return nil, err
	}
	activation := conf["activation"].(string)
	nd.ActivationType, err = network.NodeActivators.ActivationTypeFromName(activation)
	return nd, err
}

// Reads Trait configuration
func readTrait(conf map[interface{}]interface{}) (*neat.Trait, error) {
	nt := neat.NewTrait()
	nt.Id = conf["id"].(int)
	params_c := cast.ToSlice(conf["params"])
	var err error
	for i, p := range params_c {
		nt.Params[i], err = cast.ToFloat64E(p)
		if err != nil {
			return nil, err
		}
	}
	return nt, nil
}

