package genetics

// InnovationsObserver the definition of component able to manage records of innovations
type InnovationsObserver interface {
	// StoreInnovation is to store specific innovation
	StoreInnovation(innovation Innovation)
	// Innovations is to get list of known innovations
	Innovations() []Innovation
	// NextInnovationNumber is to get next unique global innovation number
	NextInnovationNumber() int64
}

// Innovation serves as a way to record innovations specifically, so that an innovation in one genome can be
// compared with other innovations in the same epoch, and if they are the same innovation, they can both be assigned the
// same innovation number.
//
// This class can encode innovations that represent a new link forming, or a new node being added.  In each case, two
// nodes fully specify the innovation and where it must have occurred (between them).
type Innovation struct {

	// Two nodes specify where the innovation took place
	InNodeId  int
	OutNodeId int
	// The number assigned to the innovation
	InnovationNum int64
	// If this is a new node innovation, then there are 2 innovations (links) added for the new node
	InnovationNum2 int64

	// If a link is added, this is its weight
	NewWeight float64
	// If a link is added, this is its connected trait index
	NewTraitNum int
	// If a new node was created, this is its node_id
	NewNodeId int

	// If a new node was created, this is the innovation number of the gene's link it is being stuck inside
	OldInnovNum int64

	// Flag to indicate whether its innovation for recurrent link
	IsRecurrent bool

	// Either NEWNODE or NEWLINK
	innovationType innovationType
}

// NewInnovationForNode is a constructor for the new node case
func NewInnovationForNode(nodeInId, nodeOutId int, innovationNum1, innovationNum2 int64, newNodeId int, oldInnovNum int64) *Innovation {
	return &Innovation{
		innovationType: newNodeInnType,
		InNodeId:       nodeInId,
		OutNodeId:      nodeOutId,
		InnovationNum:  innovationNum1,
		InnovationNum2: innovationNum2,
		NewNodeId:      newNodeId,
		OldInnovNum:    oldInnovNum,
	}
}

// NewInnovationForLink is a constructor for new link case
func NewInnovationForLink(nodeInId, nodeOutId int, innovationNum int64, weight float64, traitId int) *Innovation {
	return &Innovation{
		innovationType: newLinkInnType,
		InNodeId:       nodeInId,
		OutNodeId:      nodeOutId,
		InnovationNum:  innovationNum,
		NewWeight:      weight,
		NewTraitNum:    traitId,
	}
}

// NewInnovationForRecurrentLink is a constructor for a recurrent link
func NewInnovationForRecurrentLink(nodeInId, nodeOutId int, innovationNum int64, weight float64, traitId int, recur bool) *Innovation {
	return &Innovation{
		innovationType: newLinkInnType,
		InNodeId:       nodeInId,
		OutNodeId:      nodeOutId,
		InnovationNum:  innovationNum,
		NewWeight:      weight,
		NewTraitNum:    traitId,
		IsRecurrent:    recur,
	}
}
