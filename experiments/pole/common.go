// The pole balancing experiments is classic Reinforced Learning task proposed by Richard Sutton and Charles Anderson.
// In this experiment we will try to teach RF model of balancing pole placed on the moving cart.
package pole

// The type of action to be applied to environment
type ActionType byte

// The supported action types
const (
	// The continuous action type meaning continuous values to be applied to environment
	ContinuousAction ActionType = iota
	// The discrete action assumes that there are only discrete values of action (e.g. 0, 1)
	DiscreteAction
)
