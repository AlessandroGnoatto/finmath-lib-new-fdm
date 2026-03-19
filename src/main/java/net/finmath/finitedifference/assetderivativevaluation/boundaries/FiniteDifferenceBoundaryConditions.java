package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;

/**
 * Extended boundary interface returning explicit boundary-condition objects.
 *
 * <p>
 * This interface makes boundary semantics explicit:
 * a boundary component may either impose a Dirichlet value or specify that the
 * PDE row should be left untouched.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public interface FiniteDifferenceBoundaryConditions {

	/**
	 * Returns the boundary conditions at the lower boundary.
	 *
	 * <p>
	 * The returned array is indexed by state-variable dimension.
	 * </p>
	 *
	 * @param product The product being valued.
	 * @param time The running time.
	 * @param stateVariables The state variables specifying the boundary location.
	 * @return The lower-boundary conditions by dimension.
	 */
	BoundaryCondition[] getBoundaryConditionsAtLowerBoundary(
			FiniteDifferenceProduct product,
			double time,
			double... stateVariables);

	/**
	 * Returns the boundary conditions at the upper boundary.
	 *
	 * <p>
	 * The returned array is indexed by state-variable dimension.
	 * </p>
	 *
	 * @param product The product being valued.
	 * @param time The running time.
	 * @param stateVariables The state variables specifying the boundary location.
	 * @return The upper-boundary conditions by dimension.
	 */
	BoundaryCondition[] getBoundaryConditionsAtUpperBoundary(
			FiniteDifferenceProduct product,
			double time,
			double... stateVariables);
}