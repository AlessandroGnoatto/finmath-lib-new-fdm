package net.finmath.finitedifference.solvers;

/**
 * Provides outer boundary values for the active regime in the two-state 2D knock-in solver.
 *
 * <p>
 * The active regime represents the post-activation value, i.e. the corresponding
 * vanilla option after the barrier has been hit.
 * </p>
 *
 * <p>
 * The returned arrays correspond to the two state variables in order:
 * </p>
 * <ul>
 *   <li>index 0: first state variable boundary (typically spot),</li>
 *   <li>index 1: second state variable boundary (typically variance).</li>
 * </ul>
 */
public interface TwoStateActiveBoundaryProvider2D {

	double[] getLowerBoundaryValues(double time, double... stateVariables);

	double[] getUpperBoundaryValues(double time, double... stateVariables);
}