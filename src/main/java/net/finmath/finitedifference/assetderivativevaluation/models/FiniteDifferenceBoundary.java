package net.finmath.finitedifference.assetderivativevaluation.models;

import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;

/**
 * Interface for boundary conditions provided to finite-difference solvers.
 *
 * <p>
 * Boundary values are given in the coordinates of the state variables used
 * by the PDE discretization.
 * </p>
 * 
 * @author Christian Fries
 * @version 1.0
 */
public interface FiniteDifferenceBoundary {

	/**
	 * Return the value of the value process at the lower boundary for a given time and asset value.
	 * @param product The product which uses the boundary condition
	 * @param time The time at which the boundary is observed.
	 * @param riskFactors The value of the assets or risk factors specifying the location of the boundary.
	 *
	 * @return the value process at the lower boundary
	 */
	double[] getValueAtLowerBoundary(FiniteDifferenceProduct product, double time, double... riskFactors);

	/**
	 * Return the value of the value process at the upper boundary for a given time and asset value.
	 * @param product The product which uses the boundary condition
	 * @param time The time at which the boundary is observed.
	 * @param riskFactors The value of the assets or risk factors specifying the location of the boundary.
	 *
	 * @return the value process at the upper boundary
	 */
	double[] getValueAtUpperBoundary(FiniteDifferenceProduct product, double time, double... riskFactors);

}
