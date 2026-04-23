package net.finmath.finitedifference.assetderivativevaluation.products;

import java.util.Arrays;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMSolverFactory;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;

/**
 * Finite-difference valuation of a European arithmetic basket option.
 *
 * <p>
 * Let {@code S_1, ..., S_n} denote the underlying assets, let
 * {@code w_1, ..., w_n} be the basket weights, let {@code K} be the strike,
 * and let {@code T} be the maturity. The arithmetic basket at maturity is
 * </p>
 *
 * <p>
 * <i>
 * B(T) = \sum_{i=1}^{n} w_i S_i(T).
 * </i>
 * </p>
 *
 * <p>
 * The payoff is
 * </p>
 *
 * <p>
 * <i>
 * max(B(T) - K, 0)
 * </i>
 * </p>
 *
 * <p>
 * for a call, and
 * </p>
 *
 * <p>
 * <i>
 * max(K - B(T), 0)
 * </i>
 * </p>
 *
 * <p>
 * for a put.
 * </p>
 *
 * <p>
 * The class is written in a dimension-agnostic style through the basket-weight
 * vector, but the current finite-difference implementation supports only the
 * two-asset case, that is, a two-dimensional model state and exactly two
 * basket weights.
 * </p>
 *
 * <p>
 * In the present first version, the product is restricted to European exercise.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BasketOption implements FiniteDifferenceProduct {

	private final String[] underlyingNames;
	private final double maturity;
	private final double[] weights;
	private final double strike;
	private final CallOrPut callOrPut;
	private final Exercise exercise;

	/**
	 * Creates a European arithmetic basket option.
	 *
	 * @param underlyingNames Names of the underlyings. May be {@code null}. If
	 *        provided, the array length must match the basket dimension.
	 * @param maturity Maturity {@code T}.
	 * @param weights Basket weights.
	 * @param strike Strike {@code K}.
	 * @param callOrPut Option type.
	 */
	public BasketOption(
			final String[] underlyingNames,
			final double maturity,
			final double[] weights,
			final double strike,
			final CallOrPut callOrPut) {

		if(weights == null || weights.length == 0) {
			throw new IllegalArgumentException("weights must contain at least one entry.");
		}
		if(underlyingNames != null && underlyingNames.length != weights.length) {
			throw new IllegalArgumentException(
					"underlyingNames must have the same length as weights.");
		}
		if(callOrPut == null) {
			throw new IllegalArgumentException("callOrPut must not be null.");
		}
		if(maturity < 0.0) {
			throw new IllegalArgumentException("maturity must be non-negative.");
		}
		if(strike < 0.0) {
			throw new IllegalArgumentException("strike must be non-negative.");
		}

		boolean hasStrictlyPositiveWeight = false;
		for(final double weight : weights) {
			if(weight < 0.0) {
				throw new IllegalArgumentException(
						"weights must be non-negative in the current BasketOption implementation.");
			}
			if(weight > 0.0) {
				hasStrictlyPositiveWeight = true;
			}
		}
		if(!hasStrictlyPositiveWeight) {
			throw new IllegalArgumentException("At least one basket weight must be strictly positive.");
		}

		this.underlyingNames = underlyingNames != null ? underlyingNames.clone() : null;
		this.maturity = maturity;
		this.weights = weights.clone();
		this.strike = strike;
		this.callOrPut = callOrPut;
		this.exercise = new EuropeanExercise(maturity);
	}

	/**
	 * Creates a European arithmetic basket option.
	 *
	 * @param maturity Maturity {@code T}.
	 * @param weights Basket weights.
	 * @param strike Strike {@code K}.
	 * @param callOrPut Option type.
	 */
	public BasketOption(
			final double maturity,
			final double[] weights,
			final double strike,
			final CallOrPut callOrPut) {
		this(null, maturity, weights, strike, callOrPut);
	}

	/**
	 * Creates a European arithmetic basket option.
	 *
	 * @param underlyingNames Names of the underlyings. May be {@code null}. If
	 *        provided, the array length must match the basket dimension.
	 * @param maturity Maturity {@code T}.
	 * @param weights Basket weights.
	 * @param strike Strike {@code K}.
	 * @param callOrPutSign Payoff sign, where {@code 1.0} corresponds to a call
	 *        and {@code -1.0} corresponds to a put.
	 */
	public BasketOption(
			final String[] underlyingNames,
			final double maturity,
			final double[] weights,
			final double strike,
			final double callOrPutSign) {
		this(
				underlyingNames,
				maturity,
				weights,
				strike,
				mapCallOrPut(callOrPutSign)
		);
	}

	/**
	 * Creates a European arithmetic basket option.
	 *
	 * @param maturity Maturity {@code T}.
	 * @param weights Basket weights.
	 * @param strike Strike {@code K}.
	 * @param callOrPutSign Payoff sign, where {@code 1.0} corresponds to a call
	 *        and {@code -1.0} corresponds to a put.
	 */
	public BasketOption(
			final double maturity,
			final double[] weights,
			final double strike,
			final double callOrPutSign) {
		this(null, maturity, weights, strike, mapCallOrPut(callOrPutSign));
	}

	@Override
	public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {
		validateModel(model);

		final FDMSolver solver = FDMSolverFactory.createSolver(model, this, exercise);
		return solver.getValue(
				evaluationTime,
				maturity,
				this::terminalPayoff
		);
	}

	@Override
	public double[][] getValues(final FiniteDifferenceEquityModel model) {
		validateModel(model);

		final FDMSolver solver = FDMSolverFactory.createSolver(model, this, exercise);
		return solver.getValues(
				maturity,
				this::terminalPayoff
		);
	}

	private double terminalPayoff(final double firstAssetValue, final double secondAssetValue) {
		final double basketValue = weights[0] * firstAssetValue + weights[1] * secondAssetValue;

		if(callOrPut == CallOrPut.CALL) {
			return Math.max(basketValue - strike, 0.0);
		}
		else {
			return Math.max(strike - basketValue, 0.0);
		}
	}

	private void validateModel(final FiniteDifferenceEquityModel model) {
		if(model == null) {
			throw new IllegalArgumentException("model must not be null.");
		}

		if(weights.length != 2) {
			throw new IllegalArgumentException(
					"BasketOption currently supports only the two-asset case.");
		}

		if(model.getSpaceTimeDiscretization().getNumberOfSpaceGrids() != 2) {
			throw new IllegalArgumentException(
					"BasketOption currently requires a two-dimensional space discretization.");
		}

		if(model.getInitialValue() == null || model.getInitialValue().length != 2) {
			throw new IllegalArgumentException(
					"BasketOption currently requires a two-dimensional model state.");
		}
	}

	private static CallOrPut mapCallOrPut(final double sign) {
		if(sign == 1.0) {
			return CallOrPut.CALL;
		}
		if(sign == -1.0) {
			return CallOrPut.PUT;
		}
		throw new IllegalArgumentException("Unknown option type.");
	}

	/**
	 * Returns the underlying names.
	 *
	 * @return The underlying names, or {@code null} if unspecified.
	 */
	public String[] getUnderlyingNames() {
		return underlyingNames != null ? underlyingNames.clone() : null;
	}

	/**
	 * Returns the maturity.
	 *
	 * @return The maturity.
	 */
	public double getMaturity() {
		return maturity;
	}

	/**
	 * Returns the basket weights.
	 *
	 * @return The basket weights.
	 */
	public double[] getWeights() {
		return weights.clone();
	}

	/**
	 * Returns the strike.
	 *
	 * @return The strike.
	 */
	public double getStrike() {
		return strike;
	}

	/**
	 * Returns the option type.
	 *
	 * @return The option type.
	 */
	public CallOrPut getCallOrPut() {
		return callOrPut;
	}

	/**
	 * Returns the exercise specification.
	 *
	 * @return The exercise specification.
	 */
	public Exercise getExercise() {
		return exercise;
	}

	@Override
	public String toString() {
		return "BasketOption [maturity=" + maturity
				+ ", strike=" + strike
				+ ", callOrPut=" + callOrPut
				+ ", weights=" + Arrays.toString(weights)
				+ "]";
	}
}