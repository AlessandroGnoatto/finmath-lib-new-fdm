package net.finmath.finitedifference.solvers;

/**
 * Provides outer boundary values for the active regime in the two-state knock-in solver.
 *
 * <p>
 * The active regime represents the post-activation value, i.e. the corresponding
 * vanilla option after the barrier has been hit.
 * </p>
 */
public interface TwoStateActiveBoundaryProvider {

	double getLowerBoundaryValue(double time, double stateVariable);

	double getUpperBoundaryValue(double time, double stateVariable);
}