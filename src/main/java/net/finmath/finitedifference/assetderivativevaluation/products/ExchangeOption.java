package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMSolverFactory;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;

/**
 * Finite-difference valuation of a European exchange option on two assets.
 *
 * <p>
 * Let {@code S1} and {@code S2} denote the two underlying assets and let
 * {@code T} be the maturity. The payoff is
 * </p>
 *
 * <p>
 * <i>max(S_1(T) - S_2(T), 0)</i>
 * </p>
 *
 * <p>
 * for a call, and
 * </p>
 *
 * <p>
 * <i>max(S_2(T) - S_1(T), 0)</i>
 * </p>
 *
 * <p>
 * for a put.
 * </p>
 *
 * <p>
 * This product is the canonical two-asset equity payoff and provides a natural
 * first validation target for a two-dimensional PDE engine, since in the
 * Black-Scholes setting it admits the Margrabe closed-form formula. In the
 * finite-difference framework, the terminal condition is applied pointwise on
 * the two-dimensional state grid.
 * </p>
 *
 * <p>
 * The current implementation requires a two-dimensional model state and
 * European exercise.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class ExchangeOption implements FiniteDifferenceProduct {

	private final String firstUnderlyingName;
	private final String secondUnderlyingName;
	private final double maturity;
	private final CallOrPut callOrPut;
	private final Exercise exercise;

	/**
	 * Creates a European exchange option for two named underlyings.
	 *
	 * @param firstUnderlyingName Name of the first underlying.
	 * @param secondUnderlyingName Name of the second underlying.
	 * @param maturity Maturity {@code T}.
	 * @param callOrPut Option type. A call pays
	 *        {@code max(S1(T) - S2(T), 0)} and a put pays
	 *        {@code max(S2(T) - S1(T), 0)}.
	 */
	public ExchangeOption(
			final String firstUnderlyingName,
			final String secondUnderlyingName,
			final double maturity,
			final CallOrPut callOrPut) {

		if(callOrPut == null) {
			throw new IllegalArgumentException("callOrPut must not be null.");
		}
		if(maturity < 0.0) {
			throw new IllegalArgumentException("maturity must be non-negative.");
		}

		this.firstUnderlyingName = firstUnderlyingName;
		this.secondUnderlyingName = secondUnderlyingName;
		this.maturity = maturity;
		this.callOrPut = callOrPut;
		this.exercise = new EuropeanExercise(maturity);
	}

	/**
	 * Creates a European exchange option for two named underlyings.
	 *
	 * @param firstUnderlyingName Name of the first underlying.
	 * @param secondUnderlyingName Name of the second underlying.
	 * @param maturity Maturity {@code T}.
	 * @param callOrPutSign Payoff sign, where {@code 1.0} corresponds to a call
	 *        and {@code -1.0} corresponds to a put.
	 */
	public ExchangeOption(
			final String firstUnderlyingName,
			final String secondUnderlyingName,
			final double maturity,
			final double callOrPutSign) {
		this(
				firstUnderlyingName,
				secondUnderlyingName,
				maturity,
				mapCallOrPut(callOrPutSign)
		);
	}

	/**
	 * Creates a European exchange option with unnamed underlyings.
	 *
	 * @param maturity Maturity {@code T}.
	 * @param callOrPut Option type.
	 */
	public ExchangeOption(
			final double maturity,
			final CallOrPut callOrPut) {
		this(null, null, maturity, callOrPut);
	}

	/**
	 * Creates a European exchange option with unnamed underlyings.
	 *
	 * @param maturity Maturity {@code T}.
	 * @param callOrPutSign Payoff sign, where {@code 1.0} corresponds to a call
	 *        and {@code -1.0} corresponds to a put.
	 */
	public ExchangeOption(
			final double maturity,
			final double callOrPutSign) {
		this(null, null, maturity, mapCallOrPut(callOrPutSign));
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
		if(callOrPut == CallOrPut.CALL) {
			return Math.max(firstAssetValue - secondAssetValue, 0.0);
		}
		else {
			return Math.max(secondAssetValue - firstAssetValue, 0.0);
		}
	}

	private void validateModel(final FiniteDifferenceEquityModel model) {
		if(model == null) {
			throw new IllegalArgumentException("model must not be null.");
		}

		if(model.getSpaceTimeDiscretization().getNumberOfSpaceGrids() != 2) {
			throw new IllegalArgumentException(
					"ExchangeOption currently requires a two-dimensional space discretization.");
		}

		if(model.getInitialValue() == null || model.getInitialValue().length != 2) {
			throw new IllegalArgumentException(
					"ExchangeOption currently requires a two-dimensional model state.");
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
	 * Returns the name of the first underlying.
	 *
	 * @return The name of the first underlying, or {@code null} if unspecified.
	 */
	public String getFirstUnderlyingName() {
		return firstUnderlyingName;
	}

	/**
	 * Returns the name of the second underlying.
	 *
	 * @return The name of the second underlying, or {@code null} if unspecified.
	 */
	public String getSecondUnderlyingName() {
		return secondUnderlyingName;
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
}