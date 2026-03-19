package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.boundaries.StandardBoundaryCondition;

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
	 * @param boundaryProvider Boundary provider.
	 * @param product The product.
	 * @param time The running time.
	 * @param dimension Number of state variables.
	 * @param stateVariables The state variables.
	 * @return Boundary conditions.
	 */
	public static BoundaryCondition[] getLowerBoundaryConditions(
			final FiniteDifferenceEquityModel boundaryProvider,
			final FiniteDifferenceProduct product,
			final double time,
			final int dimension,
			final double... stateVariables) {

		return fromLegacyArray(boundaryProvider.getValueAtLowerBoundary(product, time, stateVariables), dimension);
	}

	/**
	 * Converts the result of the legacy upper-boundary method into explicit boundary conditions.
	 *
	 * @param boundaryProvider Boundary provider.
	 * @param product The product.
	 * @param time The running time.
	 * @param dimension Number of state variables.
	 * @param stateVariables The state variables.
	 * @return Boundary conditions.
	 */
	public static BoundaryCondition[] getUpperBoundaryConditions(
			final FiniteDifferenceEquityModel boundaryProvider,
			final FiniteDifferenceProduct product,
			final double time,
			final int dimension,
			final double... stateVariables) {

		return fromLegacyArray(boundaryProvider.getValueAtUpperBoundary(product, time, stateVariables), dimension);
	}
}