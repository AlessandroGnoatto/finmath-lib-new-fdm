package net.finmath.finitedifference.assetderivativevaluation.models;

import net.finmath.finitedifference.FiniteDifferenceModel;
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
 * The current finite difference framework assumes discretization of the spot price {@code S}
 * (not {@code log(S)}). Hence, drift and factor loadings are interpreted as percentage
 * coefficients that are typically combined with the diagonal matrix of spot values when
 * constructing the discrete operator.
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

}