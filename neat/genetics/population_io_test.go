package genetics

import (
	"bufio"
	"bytes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yaricom/goNEAT/v4/neat"
	"strings"
	"testing"
)

const popStr = "genomestart 1\n" +
	"trait 1 0.1 0 0 0 0 0 0 0\n" +
	"trait 2 0.2 0 0 0 0 0 0 0\n" +
	"trait 3 0.3 0 0 0 0 0 0 0\n" +
	"node 1 0 1 1\n" +
	"node 2 0 1 1\n" +
	"node 3 0 1 3\n" +
	"node 4 0 0 2\n" +
	"gene 1 1 4 1.5 false 1 0 true\n" +
	"gene 2 2 4 2.5 false 2 0 true\n" +
	"gene 3 3 4 3.5 false 3 0 true\n" +
	"genomeend 1\n" +
	"genomestart 2\n" +
	"trait 1 0.1 0 0 0 0 0 0 0\n" +
	"trait 2 0.2 0 0 0 0 0 0 0\n" +
	"trait 3 0.3 0 0 0 0 0 0 0\n" +
	"node 1 0 1 1\n" +
	"node 2 0 1 1\n" +
	"node 3 0 1 3\n" +
	"node 4 0 0 2\n" +
	"gene 1 1 4 1.5 false 1 0 true\n" +
	"gene 2 2 4 2.5 false 2 0 true\n" +
	"gene 3 3 4 3.5 false 3 0 true\n" +
	"genomeend 2\n"

func TestReadPopulation(t *testing.T) {
	conf := neat.Options{
		CompatThreshold: 0.5,
	}
	pop, err := ReadPopulation(strings.NewReader(popStr), &conf)
	require.NoError(t, err, "failed to create population")
	require.NotNil(t, pop, "population expected")
	require.Len(t, pop.Organisms, 2, "wrong population size")
	require.Len(t, pop.Species, 1, "wrong species number")
}

func TestReadPopulation_readError(t *testing.T) {
	errorReader := ErrorReader(1)

	conf := neat.Options{
		CompatThreshold: 0.5,
	}
	pop, err := ReadPopulation(&errorReader, &conf)
	assert.EqualError(t, err, alwaysErrorText)
	assert.Nil(t, pop)
}

func TestPopulation_Write(t *testing.T) {
	// first create population
	conf := neat.Options{
		CompatThreshold: 0.5,
	}
	pop, err := ReadPopulation(strings.NewReader(popStr), &conf)
	require.NoError(t, err, "failed to create population")
	require.NotNil(t, pop, "population expected")

	// write it again and test
	outBuf := bytes.NewBufferString("")
	err = pop.Write(outBuf)
	require.NoError(t, err, "failed to write population")

	_, inputTokens, err := bufio.ScanLines([]byte(popStr), true)
	require.NoError(t, err, "failed to parse input string")
	_, outputTokens, err := bufio.ScanLines(outBuf.Bytes(), true)
	require.NoError(t, err, "failed to parse output string")

	for i, gsr := range inputTokens {
		assert.Equal(t, gsr, outputTokens[i], "lines mismatch at: %d", i)
	}
}

func TestPopulation_Write_writeError(t *testing.T) { // first create population
	conf := neat.Options{
		CompatThreshold: 0.5,
	}
	pop, err := ReadPopulation(strings.NewReader(popStr), &conf)
	require.NoError(t, err, "failed to create population")
	require.NotNil(t, pop, "population expected")

	errorWriter := ErrorWriter(1)
	err = pop.Write(&errorWriter)
	assert.EqualError(t, err, alwaysErrorText)
}
