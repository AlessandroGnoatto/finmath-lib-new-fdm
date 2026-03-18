package net.finmath.finitedifference.assetderivativevaluation.models;

import net.finmath.finitedifference.FiniteDifferenceModel;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.marketdata.model.curves.DiscountCurve;

/**
 * Interface for a finite-difference equity model.
 *
 * <p>
 * Implementations provide the ingredients required by the finite-difference solvers,
 * in particular discount curves, drift coefficients, factor loadings, boundary values,
 * and the initial state.
 * </p>
 *
 * <p>
 * The framework is formulated in terms of the <i>chosen state variables</i> of the PDE.
 * It does <b>not</b> assume that the first spatial coordinate is necessarily the spot
 * price {@code S}. It may instead be {@code S}, {@code log(S)}, or any other state variable
 * for which the model provides consistent coefficients.
 * </p>
 *
 * <p>
 * Therefore, {@link #getDrift(double, double...)} and
 * {@link #getFactorLoading(double, double...)} must return coefficients that are
 * consistent with the very same state variables on which the PDE is discretized.
 * </p>
 *
 * <p>
 * Example:
 * </p>
 * <ul>
 *   <li>if the grid variable is {@code S}, then the drift and factor loadings should
 *       correspond to the SDE for {@code S},</li>
 *   <li>if the grid variable is {@code log(S)}, then the drift and factor loadings should
 *       correspond to the SDE for {@code log(S)}.</li>
 * </ul>
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
	 * Returns the drift vector of the model state variables.
	 *
	 * <p>
	 * The returned coefficients must be expressed in the same coordinates as the PDE
	 * state variables used by the finite-difference solver.
	 * </p>
	 *
	 * <p>
	 * For example, if the first state variable is {@code X}, then the first component
	 * of the returned vector is the drift of {@code X} in the SDE
	 * </p>
	 *
	 * <p>
	 * {@code dX_t = mu_X(t, X_t, ...) dt + ...}
	 * </p>
	 *
	 * @param time The evaluation time.
	 * @param stateVariables The current values of the model state variables.
	 * @return The drift vector.
	 */
	double[] getDrift(double time, double... stateVariables);

	/**
	 * Returns the factor loading matrix of the model state variables.
	 *
	 * <p>
	 * The returned coefficients must be expressed in the same coordinates as the PDE
	 * state variables used by the finite-difference solver.
	 * </p>
	 *
	 * <p>
	 * If the state vector is {@code X = (X1, ..., Xn)}, then this method returns the
	 * matrix {@code b(t, X)} in
	 * </p>
	 *
	 * <p>
	 * {@code dX_i(t) = mu_i(t, X_t) dt + sum_j b_{i,j}(t, X_t) dW_j(t)}.
	 * </p>
	 *
	 * @param time The evaluation time.
	 * @param stateVariables The current values of the model state variables.
	 * @return The factor loading matrix.
	 */
	double[][] getFactorLoading(double time, double... stateVariables);

	/**
	 * Returns a clone of this model with a modified space-time discretization.
	 *
	 * <p>
	 * The returned model should represent the same stochastic dynamics and market data
	 * as the original one, but on the provided discretization.
	 * </p>
	 *
	 * @param newSpaceTimeDiscretization The new space-time discretization.
	 * @return A clone of this model with the modified discretization.
	 */
	FiniteDifferenceEquityModel getCloneWithModifiedSpaceTimeDiscretization(
			SpaceTimeDiscretization newSpaceTimeDiscretization);

	/**
	 * Returns the initial state vector of the model.
	 *
	 * <p>
	 * The returned values must be consistent with the state variables used by the PDE.
	 * For example, if the first spatial coordinate is {@code log(S)}, then the first
	 * component should be {@code log(S0)} rather than {@code S0}.
	 * </p>
	 *
	 * @return The initial state vector.
	 */
	double[] getInitialValue();
}