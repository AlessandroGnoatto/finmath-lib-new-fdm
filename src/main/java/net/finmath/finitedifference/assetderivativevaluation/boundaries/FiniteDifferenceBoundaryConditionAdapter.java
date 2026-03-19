package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.boundaries.StandardBoundaryCondition;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;

/**
 * Utility methods converting legacy double[] boundary values into explicit
 * {@link BoundaryCondition} arrays.
 *
 * <p>
 * Legacy convention:
 * finite value -> Dirichlet,
 * NaN or missing entry -> NONE.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public final class FiniteDifferenceBoundaryConditionAdapter {

	private FiniteDifferenceBoundaryConditionAdapter() {
	}

	/**
	 * Converts a legacy boundary array into explicit boundary conditions.
	 *
	 * @param values Legacy boundary values.
	 * @param dimension Number of state-variable dimensions.
	 * @return Boundary conditions.
	 */
	public static BoundaryCondition[] fromLegacyArray(final double[] values, final int dimension) {
		final BoundaryCondition[] conditions = new BoundaryCondition[dimension];

		for(int i = 0; i < dimension; i++) {
			if(values != null && i < values.length && Double.isFinite(values[i])) {
				conditions[i] = StandardBoundaryCondition.dirichlet(values[i]);
			}
			else {
				conditions[i] = StandardBoundaryCondition.none();
			}
		}

		return conditions;
	}

	/**
	 * Converts the result of the legacy lower-boundary method into explicit boundary conditions.
	 *
	 * @param boundary Legacy boundary provider.
	 * @param product The product.
	 * @param time The running time.
	 * @param dimension Number of state variables.
	 * @param stateVariables The state variables.
	 * @return Boundary conditions.
	 */
	public static BoundaryCondition[] getLowerBoundaryConditions(
			final FiniteDifferenceBoundary boundary,
			final FiniteDifferenceProduct product,
			final double time,
			final int dimension,
			final double... stateVariables) {

		return fromLegacyArray(boundary.getValueAtLowerBoundary(product, time, stateVariables), dimension);
	}

	/**
	 * Converts the result of the legacy upper-boundary method into explicit boundary conditions.
	 *
	 * @param boundary Legacy boundary provider.
	 * @param product The product.
	 * @param time The running time.
	 * @param dimension Number of state variables.
	 * @param stateVariables The state variables.
	 * @return Boundary conditions.
	 */
	public static BoundaryCondition[] getUpperBoundaryConditions(
			final FiniteDifferenceBoundary boundary,
			final FiniteDifferenceProduct product,
			final double time,
			final int dimension,
			final double... stateVariables) {

		return fromLegacyArray(boundary.getValueAtUpperBoundary(product, time, stateVariables), dimension);
	}
}