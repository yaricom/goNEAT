package genetics

// This Innovation class serves as a way to record innovations specifically, so that an innovation in one genome can be
// compared with other innovations in the same epoch, and if they are the same innovation, they can both be assigned the
// same innovation number.
//
// This class can encode innovations that represent a new link forming, or a new node being added.  In each case, two
// nodes fully specify the innovation and where it must have occurred (between them).
type Innovation struct {

	// Two nodes specify where the innovation took place
	InNodeId       int
	OutNodeId      int
	// The number assigned to the innovation
	InnovationNum  int64
	// If this is a new node innovation, then there are 2 innovations (links) added for the new node
	InnovationNum2 int64

	// If a link is added, this is its weight
	NewWeight      float64
	// If a link is added, this is its connected trait index
	NewTraitNum    int
	// If a new node was created, this is its node_id
	NewNodeId      int

	// If a new node was created, this is the innovation number of the gene's link it is being stuck inside
	OldInnovNum    int64

	// Flag to indicate whether its innovation for recurrent link
	IsRecurrent    bool

	// Either NEWNODE or NEWLINK
	innovationType innovationType
}

// Constructor for the new node case
func NewInnovationForNode(node_in_id, node_out_id int, innovation_num1, innovation_num2 int64,
				newnode_id int, old_innov_num int64) *Innovation {
	return &Innovation {
		innovationType:newNodeInnType,
		InNodeId:node_in_id,
		OutNodeId:node_out_id,
		InnovationNum:innovation_num1,
		InnovationNum2:innovation_num2,
		NewNodeId:newnode_id,
		OldInnovNum:old_innov_num,
	}
}

// Constructor for new link case
func NewInnovationForLink(node_in_id, node_out_id int, innovation_num int64, weight float64, trait_id int) *Innovation {
	return &Innovation {
		innovationType:newLinkInnType,
		InNodeId:node_in_id,
		OutNodeId:node_out_id,
		InnovationNum:innovation_num,
		NewWeight:weight,
		NewTraitNum:trait_id,
	}
}

//Constructor for a recur link
func NewInnovationForRecurrentLink(node_in_id, node_out_id int, innovation_num int64, weight float64,
					trait_id int, recur bool) *Innovation {
	return &Innovation{
		innovationType:newLinkInnType,
		InNodeId:node_in_id,
		OutNodeId:node_out_id,
		InnovationNum:innovation_num,
		NewWeight:weight,
		NewTraitNum:trait_id,
		IsRecurrent:recur,
	}
}
