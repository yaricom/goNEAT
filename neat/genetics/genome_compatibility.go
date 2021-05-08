package genetics

import (
	"github.com/yaricom/goNEAT/v2/neat"
	"math"
)

/* ******** COMPATIBILITY CHECKING METHODS * ********/

// This function gives a measure of compatibility between two Genomes by computing a linear combination of three
// characterizing variables of their compatibility. The three variables represent PERCENT DISJOINT GENES,
// PERCENT EXCESS GENES, MUTATIONAL DIFFERENCE WITHIN MATCHING GENES. So the formula for compatibility
// is:  disjoint_coeff * pdg + excess_coeff * peg + mutdiff_coeff * mdmg
// The three coefficients are global system parameters.
// The bigger returned value the less compatible the genomes.
//
// Fully compatible genomes has 0.0 returned.
func (g *Genome) compatibility(og *Genome, opts *neat.Options) float64 {
	if opts.GenCompatMethod == neat.GenomeCompatibilityMethodLinear {
		return g.compatLinear(og, opts)
	} else {
		return g.compatFast(og, opts)
	}
}

// The compatibility checking method with linear performance depending on the size of the lognest genome in comparison.
// When genomes are small this method is compatible in performance with Genome#compatFast method.
// The compatibility formula remains the same: disjoint_coeff * pdg + excess_coeff * peg + mutdiff_coeff * mdmg
// where: pdg - PERCENT DISJOINT GENES, peg - PERCENT EXCESS GENES, and mdmg - MUTATIONAL DIFFERENCE WITHIN MATCHING GENES
//
// Fully compatible genomes has 0.0 returned.
func (g *Genome) compatLinear(og *Genome, opts *neat.Options) float64 {
	numDisjoint, numExcess, mutDiffTotal, numMatching := 0.0, 0.0, 0.0, 0.0
	size1, size2 := len(g.Genes), len(og.Genes)
	maxGenomeSize := size2
	if size1 > size2 {
		maxGenomeSize = size1
	}
	var gene1, gene2 *Gene
	for i, i1, i2 := 0, 0, 0; i < maxGenomeSize; i++ {
		if i1 >= size1 {
			numExcess += 1.0
			i2++
		} else if i2 >= size2 {
			numExcess += 1.0
			i1++
		} else {
			gene1 = g.Genes[i1]
			gene2 = og.Genes[i2]
			p1innov := gene1.InnovationNum
			p2innov := gene2.InnovationNum

			if p1innov == p2innov {
				numMatching += 1.0
				mutDiff := math.Abs(gene1.MutationNum - gene2.MutationNum)
				mutDiffTotal += mutDiff
				i1++
				i2++
			} else if p1innov < p2innov {
				i1++
				numDisjoint += 1.0
			} else if p2innov < p1innov {
				i2++
				numDisjoint += 1.0
			}
		}
	}

	//fmt.Printf("num_disjoint: %.f num_excess: %.f mut_diff_total: %.f num_matching: %.f\n", num_disjoint, num_excess, mut_diff_total, num_matching)

	// Return the compatibility number using compatibility formula
	// Note that mut_diff_total/num_matching gives the AVERAGE difference between mutation_nums for any two matching
	// Genes in the Genome. Look at disjointedness and excess in the absolute (ignoring size)
	comp := opts.DisjointCoeff*numDisjoint + opts.ExcessCoeff*numExcess +
		opts.MutdiffCoeff*(mutDiffTotal/numMatching)

	return comp
}

// The faster version of genome compatibility checking. The compatibility check will start from the end of genome where
// the most of disparities are located - the novel genes with greater innovation ID are always attached at the end (see geneInsert).
// This has the result of complicating the routine because we must now invoke additional logic to determine which genes
// are excess and when the first disjoint gene is found. This is done with an extra integer:
// * excessGenesSwitch=0 // indicates to the loop that it is handling the first gene.
// * excessGenesSwitch=1 // Indicates that the first gene was excess and on genome 1.
// * excessGenesSwitch=2 // Indicates that the first gene was excess and on genome 2.
// * excessGenesSwitch=3 // Indicates that there are no more excess genes.
//
// The compatibility formula remains the same: disjoint_coeff * pdg + excess_coeff * peg + mutdiff_coeff * mdmg
// where: pdg - PERCENT DISJOINT GENES, peg - PERCENT EXCESS GENES, and mdmg - MUTATIONAL DIFFERENCE WITHIN MATCHING GENES
//
// Fully compatible genomes has 0.0 returned.
func (g *Genome) compatFast(og *Genome, opts *neat.Options) float64 {
	list1Count, list2Count := len(g.Genes), len(og.Genes)
	// First test edge cases
	if list1Count == 0 && list2Count == 0 {
		// Both lists are empty! No disparities, therefore the genomes are compatible!
		return 0.0
	}
	if list1Count == 0 {
		// All list2 genes are excess.
		return float64(list2Count) * opts.ExcessCoeff
	}

	if list2Count == 0 {
		// All list1 genes are excess.
		return float64(list1Count) * opts.ExcessCoeff
	}

	excessGenesSwitch, numMatching := 0, 0
	compatibility, mutDiff := 0.0, 0.0
	list1Idx, list2Idx := list1Count-1, list2Count-1
	gene1, gene2 := g.Genes[list1Idx], og.Genes[list2Idx]

	for {
		if gene2.InnovationNum > gene1.InnovationNum {
			// Most common test case(s) at top for efficiency.
			if excessGenesSwitch == 3 {
				// No more excess genes. Therefore this mismatch is disjoint.
				compatibility += opts.DisjointCoeff
			} else if excessGenesSwitch == 2 {
				// Another excess gene on genome 2.
				compatibility += opts.ExcessCoeff
			} else if excessGenesSwitch == 1 {
				// We have found the first non-excess gene.
				excessGenesSwitch = 3
				compatibility += opts.DisjointCoeff
			} else {
				// First gene is excess, and is on genome 2.
				excessGenesSwitch = 2
				compatibility += opts.ExcessCoeff
			}

			// Move to the next gene in list2.
			list2Idx--
		} else if gene1.InnovationNum == gene2.InnovationNum {
			// No more excess genes. It's quicker to set this every time than to test if is not yet 3.
			excessGenesSwitch = 3

			// Matching genes. Increase compatibility by MutationNum difference * coeff.
			mutDiff += math.Abs(gene1.MutationNum - gene2.MutationNum)
			numMatching++

			// Move to the next gene in both lists.
			list1Idx--
			list2Idx--
		} else {
			// Most common test case(s) at top for efficiency.
			if excessGenesSwitch == 3 {
				// No more excess genes. Therefore this mismatch is disjoint.
				compatibility += opts.DisjointCoeff
			} else if excessGenesSwitch == 1 {
				// Another excess gene on genome 1.
				compatibility += opts.ExcessCoeff
			} else if excessGenesSwitch == 2 {
				// We have found the first non-excess gene.
				excessGenesSwitch = 3
				compatibility += opts.DisjointCoeff
			} else {
				// First gene is excess, and is on genome 1.
				excessGenesSwitch = 1
				compatibility += opts.ExcessCoeff
			}

			// Move to the next gene in list1.
			list1Idx--
		}

		// Check if we have reached the end of one (or both) of the lists. If we have reached the end of both then
		// we execute the first 'if' block - but it doesn't matter since the loop is not entered if both lists have
		// been exhausted.
		if list1Idx < 0 {
			// All remaining list2 genes are disjoint.
			compatibility += float64(list2Idx+1) * opts.DisjointCoeff
			break

		}

		if list2Idx < 0 {
			// All remaining list1 genes are disjoint.
			compatibility += float64(list1Idx+1) * opts.DisjointCoeff
			break
		}

		gene1, gene2 = g.Genes[list1Idx], og.Genes[list2Idx]
	}
	if numMatching > 0 {
		compatibility += mutDiff * opts.MutdiffCoeff / float64(numMatching)
	}
	return compatibility
}
