package genetics

import (
	"io"
	"fmt"

	"github.com/yaricom/goNEAT/neat"
	"bufio"
	"errors"
	"github.com/yaricom/goNEAT/neat/network"
	"strings"
	"github.com/spf13/viper"
	"github.com/spf13/cast"
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
			new_trait := readPlainTrait(lr)
			gnome.Traits = append(gnome.Traits, new_trait)

		case "node":
			// Read a Network Node
			new_node := readPlainNetworkNode(lr, gnome.Traits)
			gnome.Nodes = append(gnome.Nodes, new_node)

		case "gene":
			// Read a Gene
			new_gene := readPlainConnectionGene(lr, gnome.Traits, gnome.Nodes)
			gnome.Genes = append(gnome.Genes, new_gene)

		case "genomeend":
			// Read Genome ID
			fmt.Fscanf(lr, "%d", &g_id)
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
func readPlainTrait(r io.Reader) *neat.Trait {
	nt := neat.NewTrait()
	fmt.Fscanf(r, "%d ", &nt.Id)
	for i := 0; i < neat.Num_trait_params; i++ {
		fmt.Fscanf(r, "%g ", &nt.Params[i])
	}
	return nt
}

// Read a Network Node from specified Reader in plain text format
// and applies corresponding trait to it from a list of traits provided
func readPlainNetworkNode(r io.Reader, traits []*neat.Trait) *network.NNode {
	n := network.NewNetworkNode()
	var trait_id, node_type int
	fmt.Fscanf(r, "%d %d %d %d ", &n.Id, &trait_id, &node_type, &n.NeuronType)
	n.Trait = traitWithId(trait_id, traits)
	return n
}

// Reads Gene from reader in plain text format
func readPlainConnectionGene(r io.Reader, traits []*neat.Trait, nodes []*network.NNode) *Gene {
	var traitId, inNodeId, outNodeId int
	var inov_num int64
	var weight, mut_num float64
	var recurrent, enabled bool
	fmt.Fscanf(r, "%d %d %d %g %t %d %g %t ",
		&traitId, &inNodeId, &outNodeId, &weight, &recurrent, &inov_num, &mut_num, &enabled)

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
		return newGene(network.NewLinkWithTrait(trait, weight, inNode, outNode, recurrent), inov_num, mut_num, enabled)
	} else {
		return newGene(network.NewLink(weight, inNode, outNode, recurrent), inov_num, mut_num, enabled)
	}
}

// A YAMLGenomeReader reads genome data from YAML encoded text file
type yamlGenomeReader struct {
	r *bufio.Reader
}

func (ygr *yamlGenomeReader) Read() (*Genome, error) {
	v := viper.New()
	v.SetConfigType("YAML")
	err := v.ReadConfig(ygr.r)
	if err != nil {
		return nil, err
	}
	// read Genome
	gnome := &Genome{
		Id:v.GetInt("genome.id"),
		Traits:make([]*neat.Trait, 0),
		Nodes:make([]*network.NNode, 0),
		Genes:make([]*Gene, 0),
	}

	// read traits
	traits := v.Get("genome.traits").([]interface{})
	for _, tr := range traits {
		trait, err := readTrait(tr.(map[interface{}]interface{}))
		if err != nil {
			return nil, err
		}
		gnome.Traits = append(gnome.Traits, trait)
	}

	// read nodes
	nodes := v.Get("genome.nodes").([]interface{})
	for _, nd := range nodes {
		node, err := readNNode(nd.(map[interface{}]interface{}), gnome.Traits)
		if err != nil {
			return nil, err
		}
		gnome.Nodes = append(gnome.Nodes, node)
	}

	// read Genes
	genes := v.Get("genome.genes").([]interface{})
	for _, g := range genes {
		gene, err := readGene(g.(map[interface{}]interface{}), gnome.Traits, gnome.Nodes)
		if err != nil {
			return nil, err
		}
		gnome.Genes = append(gnome.Genes, gene)
	}


	//t_type := reflect.TypeOf(traits )
	//println(t_type.String())
	println(len(gnome.Traits))

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

func traitWithId(trait_id int, traits []*neat.Trait) *neat.Trait {
	var trait *neat.Trait = nil
	if trait_id != 0 && traits != nil {
		for _, tr := range traits {
			if tr.Id == trait_id {
				trait = tr
			}
		}
	}
	return trait
}


