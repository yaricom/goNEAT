package performance

import (
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v4/neat/genetics"
	"github.com/yaricom/goNEAT/v4/neat/network"
	"testing"
)

const genomeStr = `genomestart 0
trait 1 0.1 0 0 0 0 0 0 0
trait 2 0.2 0 0 0 0 0 0 0
trait 3 0.3 0 0 0 0 0 0 0
node 1 1 1 1 SigmoidSteepenedActivation
node 2 2 1 1 SigmoidSteepenedActivation
node 3 1 1 1 SigmoidSteepenedActivation
node 4 1 1 1 SigmoidSteepenedActivation
node 5 1 1 1 SigmoidSteepenedActivation
node 6 3 1 1 SigmoidSteepenedActivation
node 7 1 1 3 SigmoidSteepenedActivation
node 8 1 0 2 SigmoidSteepenedActivation
node 30 3 0 0 SigmoidSteepenedActivation
node 71 1 0 0 SigmoidSteepenedActivation
node 266 1 0 0 SigmoidSteepenedActivation
node 1685 1 0 0 SigmoidSteepenedActivation
node 1811 1 0 0 SigmoidSteepenedActivation
node 3436 1 0 0 SigmoidSteepenedActivation
node 3627 1 0 0 SigmoidSteepenedActivation
node 4077 1 0 0 SigmoidSteepenedActivation
node 6640 2 0 0 SigmoidSteepenedActivation
gene 1 1 8 -4.08386891981225 false 1 -4.08386891981225 true
gene 2 2 8 -2.1690052056140163 false 2 -2.1690052056140163 true
gene 3 3 8 -10.699778103607123 false 3 -10.699778103607123 true
gene 1 4 8 -9.342647981011918 false 4 -9.342647981011918 true
gene 2 5 8 6.996472488106343 false 5 6.996472488106343 true
gene 2 6 8 2.2286271196270824 false 6 2.2286271196270824 true
gene 2 7 8 5.305934250174099 false 7 5.305934250174099 true
gene 1 1 30 5.327875888469306 false 121 5.327875888469306 true
gene 1 30 8 5.684209149585454 false 122 5.684209149585454 true
gene 3 3 30 0.9475646340769113 false 271 0.9475646340769113 true
gene 3 5 30 1.2529446872848236 false 439 1.2529446872848236 true
gene 1 1 71 2.0097007140389365 false 583 2.0097007140389365 true
gene 1 71 8 -2.6692540269378506 false 584 -2.6692540269378506 true
gene 1 3 71 -4.073066558696308 false 834 -4.073066558696308 true
gene 3 2 71 2.0095928118285507 false 1223 2.0095928118285507 true
gene 3 2 266 3.4187907757610367 false 1627 3.4187907757610367 true
gene 3 266 71 -3.2830938391621856 false 1628 -3.2830938391621856 true
gene 1 6 71 -3.463234141766563 false 2528 -3.463234141766563 true
gene 3 3 266 -2.281968137235494 false 3056 -2.281968137235494 true
gene 2 4 266 3.497559408686352 false 4212 3.497559408686352 true
gene 3 30 71 -2.8430156534468214 false 4678 -2.8430156534468214 true
gene 1 7 71 -3.45518023982917 false 6526 -3.45518023982917 true
gene 1 71 1685 2.338576667094656 false 7008 2.338576667094656 true
gene 1 1685 8 -0.2433441365216774 false 7009 -0.2433441365216774 true
gene 1 30 1811 -1.1814519616174946 false 7485 -1.1814519616174946 true
gene 1 1811 8 -0.9537417818325291 false 7486 -0.9537417818325291 true
gene 2 7 266 2.748821688559035 false 9914 2.748821688559035 true
gene 2 1811 266 3.908454872776794 false 10435 3.908454872776794 true
gene 3 6 266 7.145812334508661 false 12872 7.145812334508661 true
gene 1 71 3436 -0.09282935033826101 false 13446 -0.09282935033826101 true
gene 1 3436 1685 2.021217174697969 false 13447 2.021217174697969 true
gene 2 1811 3627 -3.0538854223916507 false 14092 -3.0538854223916507 true
gene 2 3627 266 -3.0405574668478694 false 14093 -3.0405574668478694 true
gene 1 71 4077 6.692035802696705 false 15777 6.692035802696705 true
gene 1 4077 1685 -3.413636512977078 false 15778 -3.413636512977078 true
gene 1 3627 30 1.11052014349801 true 16364 1.11052014349801 true
gene 2 1811 4077 5.8503253679873435 false 17006 5.8503253679873435 true
gene 2 2 30 1.6963498266122063 false 17650 1.6963498266122063 true
gene 1 2 3627 -6.386739930810184 false 18335 -6.386739930810184 true
gene 1 266 4077 -3.4520706013634195 false 22326 -3.4520706013634195 true
gene 1 4077 3436 4.013733453814073 false 24107 4.013733453814073 true
gene 1 3 6640 -0.4098657935148693 false 24713 -0.4098657935148693 true
gene 1 6640 71 -1.753803731332276 false 24714 -1.753803731332276 true
gene 2 3627 4077 -0.28877942085982367 false 25611 -0.28877942085982367 true
genomeend 0`

const genomeStrSimple = `genomestart 131
trait 1 0.1 0 0 0 0 0 0 0
trait 2 0.2 0 0 0 0 0 0 0
trait 3 0.3 0 0 0 0 0 0 0
node 1 1 1 3 SigmoidSteepenedActivation
node 2 1 1 1 SigmoidSteepenedActivation
node 3 1 1 1 SigmoidSteepenedActivation
node 4 1 1 1 SigmoidSteepenedActivation
node 5 1 1 1 SigmoidSteepenedActivation
node 6 3 0 2 SigmoidSteepenedActivation
node 7 1 0 2 SigmoidSteepenedActivation
node 9 1 0 0 SigmoidSteepenedActivation
gene 1 1 6 2.2319059895867506 false 1 2.2319059895867506 true
gene 2 2 6 -1.2321454922952315 false 2 -1.2321454922952315 true
gene 3 3 6 -0.49043752407567043 false 3 -0.49043752407567043 true
gene 1 4 6 0.46891774508325124 false 4 0.46891774508325124 true
gene 2 5 6 -2.4300089957414337 false 5 -2.4300089957414337 true
gene 3 1 7 -1.6431550383146705 false 6 -1.6431550383146705 true
gene 1 2 7 0.5398285078832726 false 7 0.5398285078832726 true
gene 1 3 7 0.13384471791198438 false 8 0.13384471791198438 true
gene 3 4 7 2.6321796987645034 false 9 2.6321796987645034 true
gene 1 5 7 -0.09060533978387858 false 10 -0.09060533978387858 true
gene 3 3 9 -3.089340847206661 false 13 -3.089340847206661 true
gene 3 9 6 0.5616266601637229 false 14 0.5616266601637229 true
genomeend 131`

func buildNetworkFromGenome(str string) (*network.Network, error) {
	buf := bytes.NewBufferString(str)
	reader, err := genetics.NewGenomeReader(buf, genetics.PlainGenomeEncoding)
	if err != nil {
		return nil, err
	}
	if genome, err := reader.Read(); err != nil {
		return nil, err
	} else {
		return genome.Genesis(0)
	}
}

func TestNetwork_MaxActivationDepth_FromGenome(t *testing.T) {
	net, err := buildNetworkFromGenome(genomeStr)
	require.NoError(t, err)

	str := logNetworkActivationPath(net, t)
	t.Log(str)

	depth, err := net.MaxActivationDepth()
	assert.NoError(t, err, "failed to calculate max depth")
	assert.Equal(t, 9, depth)
}

func TestNetwork_MaxActivationDepthCap_FromGenome(t *testing.T) {
	net, err := buildNetworkFromGenome(genomeStr)
	require.NoError(t, err)

	limit := 5
	depth, err := net.MaxActivationDepthWithCap(limit)
	require.EqualError(t, err, network.ErrMaximalNetDepthExceeded.Error())
	assert.Equal(t, limit, depth)
}

func Test_PrintAllActivationDepthPaths_simple(t *testing.T) {
	net, err := buildNetworkFromGenome(genomeStrSimple)
	require.NoError(t, err)

	str := logNetworkActivationPath(net, t)
	t.Log(str)
	expected := "1 -> 6\n2 -> 6\n3 -> 6\n4 -> 6\n5 -> 6\n3 -> 9 -> 6\n1 -> 7\n2 -> 7\n3 -> 7\n4 -> 7\n5 -> 7\n"
	assert.Equal(t, expected, str)
}

func logNetworkActivationPath(net *network.Network, t *testing.T) string {
	buf := bytes.NewBufferString("")
	err := network.PrintAllActivationDepthPaths(net, buf)
	require.NoError(t, err)
	return buf.String()
}
