package network

import (
	"encoding/json"
	"github.com/yaricom/goNEAT/v4/neat/math"
	"io"
)

// WriteModel is to write this FastModularNetworkSolver as a model to be used later.
func (s *FastModularNetworkSolver) WriteModel(w io.Writer) error {
	dataHolder := newFastModularNetworkSolverData(s)
	enc := json.NewEncoder(w)
	return enc.Encode(dataHolder)
}

type NodeActivator struct {
	NodeActivation math.NodeActivationType
}

type fastControlNodeData struct {
	ActivationType NodeActivator `json:"activation_type"`
	InputIndexes   []int         `json:"input_indexes"`
	OutputIndexes  []int         `json:"output_indexes"`
}

type fastModularNetworkSolverData struct {
	Id                  int                   `json:"id"`
	Name                string                `json:"name"`
	InputNeuronCount    int                   `json:"input_neuron_count"`
	SensorNeuronCount   int                   `json:"sensor_neuron_count"`
	OutputNeuronCount   int                   `json:"output_neuron_count"`
	BiasNeuronCount     int                   `json:"bias_neuron_count"`
	TotalNeuronCount    int                   `json:"total_neuron_count"`
	ActivationFunctions []NodeActivator       `json:"activation_functions"`
	BiasList            []float64             `json:"bias_list"`
	Connections         []*FastNetworkLink    `json:"connections"`
	Modules             []fastControlNodeData `json:"modules"`
}

func newFastModularNetworkSolverData(n *FastModularNetworkSolver) *fastModularNetworkSolverData {
	data := &fastModularNetworkSolverData{
		Id:                  n.Id,
		Name:                n.Name,
		InputNeuronCount:    n.inputNeuronCount,
		SensorNeuronCount:   n.sensorNeuronCount,
		OutputNeuronCount:   n.outputNeuronCount,
		BiasNeuronCount:     n.biasNeuronCount,
		TotalNeuronCount:    n.totalNeuronCount,
		ActivationFunctions: make([]NodeActivator, 0, len(n.activationFunctions)),
		BiasList:            n.biasList,
		Connections:         n.connections,
		Modules:             make([]fastControlNodeData, 0),
	}
	for _, v := range n.activationFunctions {
		data.ActivationFunctions = append(data.ActivationFunctions, NodeActivator{
			NodeActivation: v,
		})
	}
	if n.modules != nil {
		for _, v := range n.modules {
			data.Modules = append(data.Modules, fastControlNodeData{
				ActivationType: NodeActivator{NodeActivation: v.ActivationType},
				InputIndexes:   v.InputIndexes,
				OutputIndexes:  v.OutputIndexes,
			})
		}
	}
	return data
}

func (n *NodeActivator) MarshalText() ([]byte, error) {
	if activationName, err := math.NodeActivators.ActivationNameFromType(n.NodeActivation); err != nil {
		return nil, err
	} else {
		return []byte(activationName), nil
	}
}
