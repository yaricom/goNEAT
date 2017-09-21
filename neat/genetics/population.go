package genetics

type Population struct {
	// Species in the Population. Note that the species should comprise all the genomes
	Species     []*Species
	// The highest species number
	LastSpecies int
	// For holding the genetic innovations of the newest generation
	Innovations []*Innovation


	// The current innovation number for population
	currInnovNum int64
}

// Returns current innovation number and increment innovations number counter after that
func (p *Population) getInnovationNumberAndIncrement() int64 {
	inn_num := p.currInnovNum
	p.currInnovNum++
	return inn_num
}
