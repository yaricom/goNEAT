package genetics

import (
	"bufio"
	"fmt"
	"github.com/yaricom/goNEAT/v2/neat"
	"github.com/yaricom/goNEAT/v2/neat/math"
	"github.com/yaricom/goNEAT/v2/neat/network"
	"gopkg.in/yaml.v3"
	"io"
)

// GenomeWriter is the interface to define genome writer
type GenomeWriter interface {
	// WriteGenome writes Genome record into underlying writer
	WriteGenome(genome *Genome) error
}

// NewGenomeWriter creates genome writer with specified data encoding format
func NewGenomeWriter(w io.Writer, encoding GenomeEncoding) (GenomeWriter, error) {
	switch encoding {
	case PlainGenomeEncoding:
		return &plainGenomeWriter{w: bufio.NewWriter(w)}, nil
	case YAMLGenomeEncoding:
		return &yamlGenomeWriter{w: bufio.NewWriter(w)}, nil
	default:
		return nil, ErrUnsupportedGenomeEncoding
	}
}

// The plain text encoded genome writer
type plainGenomeWriter struct {
	w *bufio.Writer
}

func (wr *plainGenomeWriter) WriteGenome(g *Genome) error {
	if _, err := fmt.Fprintf(wr.w, "genomestart %d\n", g.Id); err != nil {
		return err
	}

	for _, tr := range g.Traits {
		if _, err := fmt.Fprint(wr.w, "trait "); err != nil {
			return err
		}
		if err := wr.writeTrait(tr); err != nil {
			return err
		}
		if _, err := fmt.Fprintln(wr.w, ""); err != nil {
			return err
		}
	}

	for _, nd := range g.Nodes {
		if _, err := fmt.Fprint(wr.w, "node "); err != nil {
			return err
		}
		if err := wr.writeNetworkNode(nd); err != nil {
			return err
		}
		if _, err := fmt.Fprintln(wr.w, ""); err != nil {
			return err
		}
	}

	for _, gn := range g.Genes {
		if _, err := fmt.Fprint(wr.w, "gene "); err != nil {
			return err
		}
		if err := wr.writeConnectionGene(gn); err != nil {
			return err
		}
		if _, err := fmt.Fprintln(wr.w, ""); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(wr.w, "genomeend %d\n", g.Id); err != nil {
		return err
	}

	// flush buffer
	return wr.w.Flush()
}

// Dump trait in plain text format
func (wr *plainGenomeWriter) writeTrait(t *neat.Trait) error {
	if _, err := fmt.Fprintf(wr.w, "%d ", t.Id); err != nil {
		return err
	}
	for i, p := range t.Params {
		if i < len(t.Params)-1 {
			if _, err := fmt.Fprintf(wr.w, "%g ", p); err != nil {
				return err
			}
		} else {
			if _, err := fmt.Fprintf(wr.w, "%g", p); err != nil {
				return err
			}
		}
	}
	return nil
}

// Dump network node in plain text format
func (wr *plainGenomeWriter) writeNetworkNode(n *network.NNode) error {
	traitId := 0
	if n.Trait != nil {
		traitId = n.Trait.Id
	}
	actStr, err := math.NodeActivators.ActivationNameFromType(n.ActivationType)
	if err == nil {
		_, err = fmt.Fprintf(wr.w, "%d %d %d %d %s", n.Id, traitId, n.NodeType(),
			n.NeuronType, actStr)
	}
	return err
}

// Dump connection gene in plain text format
func (wr *plainGenomeWriter) writeConnectionGene(g *Gene) error {
	link := g.Link
	traitId := 0
	if link.Trait != nil {
		traitId = link.Trait.Id
	}
	inNodeId := link.InNode.Id
	outNodeId := link.OutNode.Id
	weight := link.ConnectionWeight
	recurrent := link.IsRecurrent
	innovNum := g.InnovationNum
	mutNum := g.MutationNum
	enabled := g.IsEnabled

	_, err := fmt.Fprintf(wr.w, "%d %d %d %g %t %d %g %t",
		traitId, inNodeId, outNodeId, weight, recurrent, innovNum, mutNum, enabled)
	return err
}

// The YAML encoded genome writer
type yamlGenomeWriter struct {
	w *bufio.Writer
}

func (wr *yamlGenomeWriter) WriteGenome(g *Genome) (err error) {
	gMap := make(map[string]interface{})
	gMap["id"] = g.Id

	// encode traits
	traits := make([]map[string]interface{}, len(g.Traits))
	for i, t := range g.Traits {
		traits[i] = wr.encodeGenomeTrait(t)
	}
	gMap["traits"] = traits

	// encode network nodes
	nodes := make([]map[string]interface{}, len(g.Nodes))
	for i, n := range g.Nodes {
		nodes[i], err = wr.encodeNetworkNode(n)
		if err != nil {
			return err
		}
	}
	gMap["nodes"] = nodes

	// encode connection genes
	genes := make([]map[string]interface{}, len(g.Genes))
	for i, gn := range g.Genes {
		genes[i] = wr.encodeConnectionGene(gn)
	}
	gMap["genes"] = genes

	// encode control genes if any
	if len(g.ControlGenes) > 0 {
		modules := make([]map[string]interface{}, len(g.ControlGenes))
		for i, cg := range g.ControlGenes {
			modules[i], err = wr.encodeControlGene(cg)
			if err != nil {
				return err
			}
		}
		gMap["modules"] = modules
	}

	// store genome map
	rMap := make(map[string]interface{})
	rMap["genome"] = gMap

	// encode everything as YAML
	enc := yaml.NewEncoder(wr.w)
	err = enc.Encode(rMap)
	if err == nil {
		// flush stream
		err = wr.w.Flush()
	}

	return err
}

func (wr *yamlGenomeWriter) encodeControlGene(gene *MIMOControlGene) (gMap map[string]interface{}, err error) {
	gMap = make(map[string]interface{})
	gMap["id"] = gene.ControlNode.Id
	if gene.ControlNode.Trait != nil {
		gMap["trait_id"] = gene.ControlNode.Trait.Id
	} else {
		gMap["trait_id"] = 0
	}
	gMap["innov_num"] = gene.InnovationNum
	gMap["mut_num"] = gene.MutationNum
	gMap["enabled"] = gene.IsEnabled
	gMap["activation"], err = math.NodeActivators.ActivationNameFromType(gene.ControlNode.ActivationType)
	if err != nil {
		return nil, err
	}
	// store inputs
	inputs := make([]map[string]interface{}, len(gene.ControlNode.Incoming))
	for i, in := range gene.ControlNode.Incoming {
		inputs[i] = wr.encodeModuleLink(in.InNode.Id, i)
	}
	gMap["inputs"] = inputs

	// store outputs
	outputs := make([]map[string]interface{}, len(gene.ControlNode.Outgoing))
	for i, out := range gene.ControlNode.Outgoing {
		outputs[i] = wr.encodeModuleLink(out.OutNode.Id, i)
	}
	gMap["outputs"] = outputs

	return gMap, err
}

func (wr *yamlGenomeWriter) encodeModuleLink(id, order int) map[string]interface{} {
	lMap := make(map[string]interface{})
	lMap["id"] = id
	lMap["order"] = order
	return lMap
}

func (wr *yamlGenomeWriter) encodeConnectionGene(gene *Gene) map[string]interface{} {
	gMap := make(map[string]interface{})
	if gene.Link.Trait != nil {
		gMap["trait_id"] = gene.Link.Trait.Id
	} else {
		gMap["trait_id"] = 0
	}
	gMap["src_id"] = gene.Link.InNode.Id
	gMap["tgt_id"] = gene.Link.OutNode.Id
	gMap["innov_num"] = gene.InnovationNum
	gMap["weight"] = gene.Link.ConnectionWeight
	gMap["mut_num"] = gene.MutationNum
	gMap["recurrent"] = gene.Link.IsRecurrent
	gMap["enabled"] = gene.IsEnabled
	return gMap
}

func (wr *yamlGenomeWriter) encodeNetworkNode(node *network.NNode) (nMap map[string]interface{}, err error) {
	nMap = make(map[string]interface{})
	nMap["id"] = node.Id
	if node.Trait != nil {
		nMap["trait_id"] = node.Trait.Id
	} else {
		nMap["trait_id"] = 0
	}
	nMap["type"] = network.NeuronTypeName(node.NeuronType)
	nMap["activation"], err = math.NodeActivators.ActivationNameFromType(node.ActivationType)
	return nMap, err
}

func (wr *yamlGenomeWriter) encodeGenomeTrait(trait *neat.Trait) map[string]interface{} {
	trMap := make(map[string]interface{})
	trMap["id"] = trait.Id
	trMap["params"] = trait.Params
	return trMap
}
