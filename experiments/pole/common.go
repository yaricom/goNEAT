// The pole balancing experiments is classic Reinforced Learning task proposed by Richard Sutton and Charles Anderson.
// In this experiment we will try to teach RF model of balancing pole placed on the moving cart.
package pole

// The cart pole configuration values
const GRAVITY = 9.8
const MASSCART = 1.0
const MASSPOLE = 0.1
const TOTAL_MASS = (MASSPOLE + MASSCART)
const LENGTH = 0.5      /* actually half the pole's length */
const POLEMASS_LENGTH = (MASSPOLE * LENGTH)
const FORCE_MAG = 10.0
const TAU = 0.02      /* seconds between state updates */
const FOURTHIRDS = 1.3333333333333