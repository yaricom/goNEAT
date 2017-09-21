package genetics

type Population struct {
	// Species in the Population. Note that the species should comprise all the genomes
	Species            []*Species
	// The highest species number
	LastSpecies        int
	// For holding the genetic innovations of the newest generation
	Innovations        []*Innovation
	// An integer that when above zero tells when the first winner appeared
	WinnerGen          int


	// Stagnation detector
	HighestFitness     float64
	// If too high, leads to delta coding
	HighestLastChanged int


	// The current innovation number for population
	currInnovNum       int64
	// The current ID for new node in population
	currNodeId         int
}

/* Construct off of a single spawning Genome */
func NewPopulation(g *Genome, size int) *Population {
	pop := Population{
		WinnerGen:0,
		HighestFitness:0.0,
		HighestLastChanged:0,
	}
	pop.spawn(g, size)
	return &pop
}

// Returns current innovation number and increment innovations number counter after that
func (p *Population) getInnovationNumberAndIncrement() int64 {
	inn_num := p.currInnovNum
	p.currInnovNum++
	return inn_num
}
// Returns the current node ID which can be used to create new node in population and increment it after
func (p *Population) getCurrentNodeIdAndIncrement() int {
	node_id := p.currNodeId
	p.currNodeId++
	return node_id
}

// A Population can be spawned off of a single Genome. There will be size Genomes added to the Population.
// The Population does not have to be empty to add Genomes.
func (p *Population) spawn(g *Genome, size int) bool {
	// TODO implement this
	return false
}
