package genetics

type Population struct {
	// Species in the Population. Note that the species should comprise all the genomes
	species []*Species
	// The highest species number
	lastSpecies int

}
