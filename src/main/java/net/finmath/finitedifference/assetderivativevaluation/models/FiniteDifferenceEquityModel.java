package net.finmath.finitedifference.assetderivativevaluation.models;

import net.finmath.finitedifference.FiniteDifferenceModel;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.marketdata.model.curves.DiscountCurve;

/**
 * Interface for a (possibly multi-dimensional) finite difference equity model for option pricing.
 *
 * <p>
 * Implementations provide the ingredients required by finite difference schemes, in particular
 * access to discounting curves and model coefficients (drift and factor loadings) used to
 * assemble the PDE operator.
 * </p>
 *
 * <p>
 * The finite difference framework does not assume discretization of the spot price {@code S}
 * or {@code log(S)}. It is agnostic. It is the task of the user to provide coherent drifts and factor loadings.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public interface FiniteDifferenceEquityModel extends FiniteDifferenceModel, FiniteDifferenceBoundary {

	/**
	 * Returns the risk-free discount curve used for pricing.
	 *
	 * @return The risk-free discount curve.
	 */
	DiscountCurve getRiskFreeCurve();

	/**
	 * Returns the dividend yield discount curve.
	 *
	 * @return The dividend yield discount curve.
	 */
	DiscountCurve getDividendYieldCurve();

	/**
	 * Returns the drift coefficients for the state variables.
	 *
	 * <p>
	 * The framework assumes discretization in the spot coordinate {@code S}. Therefore, the drift
	 * is returned as a percentage coefficient. When assembling the finite difference operator, it
	 * is typically combined with the current state value (e.g., multiplied by the spot grid).
	 * </p>
	 *
	 * @param time          The evaluation time.
	 * @param stateVariables The state variables (varargs to support multi-dimensional models).
	 * @return The drift vector.
	 */
	double[] getDrift(double time, double... stateVariables);

	/**
	 * Returns the matrix of factor loadings for the state variables.
	 *
	 * <p>
	 * The framework assumes discretization in the spot coordinate {@code S}. Therefore, the factor
	 * loadings are returned as percentage coefficients. When assembling the finite difference
	 * operator, they are typically combined with the current state value (e.g., multiplied by the
	 * spot grid).
	 * </p>
	 *
	 * @param time          The evaluation time.
	 * @param stateVariables The state variables (varargs to support multi-dimensional models).
	 * @return The factor loading matrix.
	 */
	double[][] getFactorLoading(double time, double... stateVariables);

	/**
	 * Method that returns a clone of this model for a different choice
	 * of the space-time discretization.
	 * 
	 * @param newSpaceTimeDiscretization
	 * @return the same model with a new SpaceTimeDiscretization
	 */
	FiniteDifferenceEquityModel getCloneWithModifiedSpaceTimeDiscretization(
			SpaceTimeDiscretization newSpaceTimeDiscretization);

	/**
	 * Returns the initial value of the system of SDE.
	 * @return the initial value of the system of SDEs.
	 */
	public double[] getInitialValue();

}