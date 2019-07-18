# Unsymmetrical mixing functions

*The casual user has no need to explicitly call this module.*

`.unsymmetrical` provides different ways to evaluate the *J* function that appears in the [unsymmetrical mixing terms of the Pitzer model](../../modules/model/#etheta).

One of these functions must be contained within the [coefficient library](../cflibs) in order to execute the Pitzer model functions:

  * `.P75_eq46` - [P75](../../references/#P75) Eq. (46);
  * `.P75_eq47` - [P75](../../references/#P75) Eq. (47);
  * `.Harvie` - "Harvie's method", as described by [P91](../../references/#P91), pages 124 to 125;
  * `.numint` - using numerical integration method to evaluate J "exactly" (note: not currently Autograd-able, so cannot be used to calculate any properties other than the excess Gibbs energy);
  * `.none` - ignore the unsymmetrical mixing terms.
