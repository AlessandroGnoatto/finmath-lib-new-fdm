package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.solvers.FDMSolver;
import net.finmath.finitedifference.solvers.FDMThetaMethod1DStateVariableForm;
import net.finmath.finitedifference.solvers.FDMThetaMethod2DStateVariableForm;
import net.finmath.modelling.products.CallOrPut;

/**
 * Finite difference valuation of an American option on a single asset.
 *
 * <p>
 * Given an underlying price {@code S} with strike {@code K} and maturity {@code T}, the payoff is
 * </p>
 * <p>
 * {@code max(sign * (S(T) - K), 0)},
 * </p>
 * <p>
 * where {@code sign} corresponds to {@link CallOrPut#CALL} or {@link CallOrPut#PUT}.
 * </p>
 *
 * @author Alessandro Gnoatto
 * @version 1.0
 */
public class AmericanOption implements FiniteDifferenceProduct {

	private final String underlyingName;
	private final double maturity;
	private final double strike;
	private final CallOrPut callOrPutSign;
	private final ExerciseType exercise;

	/**
	 * Creates an American option for a named underlying.
	 *
	 * @param underlyingName Name of the underlying.
	 * @param maturity       Maturity {@code T}.
	 * @param strike         Strike {@code K}.
	 * @param callOrPutSign  Payoff sign, where {@code 1.0} corresponds to a call and
	 *                       {@code -1.0} corresponds to a put.
	 */
	public AmericanOption(
			final String underlyingName,
			final double maturity,
			final double strike,
			final double callOrPutSign) {

		super();
		this.underlyingName = underlyingName;
		this.maturity = maturity;
		this.strike = strike;

		if(callOrPutSign == 1.0) {
			this.callOrPutSign = CallOrPut.CALL;
		}
		else if(callOrPutSign == -1.0) {
			this.callOrPutSign = CallOrPut.PUT;
		}
		else {
			throw new IllegalArgumentException("Unknown option type");
		}

		this.exercise = ExerciseType.AMERICAN;
	}

	/**
	 * Creates an American option for a named underlying.
	 *
	 * @param underlyingName Name of the underlying.
	 * @param maturity       Maturity {@code T}.
	 * @param strike         Strike {@code K}.
	 * @param callOrPutSign  Option type.
	 */
	public AmericanOption(
			final String underlyingName,
			final double maturity,
			final double strike,
			final CallOrPut callOrPutSign) {

		super();
		this.underlyingName = underlyingName;
		this.maturity = maturity;
		this.strike = strike;
		this.callOrPutSign = callOrPutSign;
		this.exercise = ExerciseType.AMERICAN;
	}

	/**
	 * Creates an American option (single-asset case, unnamed underlying).
	 *
	 * @param maturity      Maturity {@code T}.
	 * @param strike        Strike {@code K}.
	 * @param callOrPutSign Payoff sign, where {@code 1.0} corresponds to a call and
	 *                      {@code -1.0} corresponds to a put.
	 */
	public AmericanOption(final double maturity, final double strike, final double callOrPutSign) {

		super();
		this.maturity = maturity;
		this.strike = strike;

		if(callOrPutSign == 1.0) {
			this.callOrPutSign = CallOrPut.CALL;
		}
		else if(callOrPutSign == -1.0) {
			this.callOrPutSign = CallOrPut.PUT;
		}
		else {
			throw new IllegalArgumentException("Unknown option type");
		}

		this.underlyingName = null;
		this.exercise = ExerciseType.AMERICAN;
	}

	/**
	 * Creates an American option (single-asset case, unnamed underlying).
	 *
	 * @param maturity      Maturity {@code T}.
	 * @param strike        Strike {@code K}.
	 * @param callOrPutSign Option type.
	 */
	public AmericanOption(final double maturity, final double strike, final CallOrPut callOrPutSign) {

		super();
		this.maturity = maturity;
		this.strike = strike;
		this.callOrPutSign = callOrPutSign;
		this.underlyingName = null;
		this.exercise = ExerciseType.AMERICAN;
	}

	/**
	 * Creates an American call option for a named underlying.
	 *
	 * @param underlyingName Name of the underlying.
	 * @param maturity       Maturity {@code T}.
	 * @param strike         Strike {@code K}.
	 */
	public AmericanOption(final String underlyingName, final double maturity, final double strike) {
		this(underlyingName, maturity, strike, 1.0);
	}

	/**
	 * Creates an American call option (single-asset case, unnamed underlying).
	 *
	 * @param maturity Maturity {@code T}.
	 * @param strike   Strike {@code K}.
	 */
	public AmericanOption(final double maturity, final double strike) {
		this(maturity, strike, 1.0);
	}

	@Override
	public double[] getValue(final double evaluationTime, final FiniteDifferenceEquityModel model) {
		final FDMSolver solver;

		if(model instanceof FDMBlackScholesModel) {
			solver = new FDMThetaMethod1DStateVariableForm(model, this, model.getSpaceTimeDiscretization(), exercise);
		}
		else if(model instanceof FDMHestonModel) {
			solver = new FDMThetaMethod2DStateVariableForm(model, this, model.getSpaceTimeDiscretization(), exercise);
		}
		else {
			throw new IllegalArgumentException(
					"Unsupported model type: " + model.getClass().getName());
		}

		if(callOrPutSign == CallOrPut.CALL) {
			return solver.getValue(evaluationTime, maturity, assetValue -> Math.max(assetValue - strike, 0));
		}
		else {
			return solver.getValue(evaluationTime, maturity, assetValue -> Math.max(strike - assetValue, 0));
		}
	}

	@Override
	public double[][] getValues(final FiniteDifferenceEquityModel model) {
		final FDMSolver solver;

		if(model instanceof FDMBlackScholesModel) {
			solver = new FDMThetaMethod1DStateVariableForm(model, this, model.getSpaceTimeDiscretization(), exercise);
		}
		else if(model instanceof FDMHestonModel) {
			solver = new FDMThetaMethod2DStateVariableForm(model, this, model.getSpaceTimeDiscretization(), exercise);
		}
		else {
			throw new IllegalArgumentException(
					"Unsupported model type: " + model.getClass().getName());
		}

		if(callOrPutSign == CallOrPut.CALL) {
			return solver.getValues(maturity, assetValue -> Math.max(assetValue - strike, 0));
		}
		else {
			return solver.getValues(maturity, assetValue -> Math.max(strike - assetValue, 0));
		}
	}

	/**
	 * Returns the name of the underlying.
	 *
	 * @return The underlying name, or {@code null} if unspecified.
	 */
	public String getUnderlyingName() {
		return underlyingName;
	}

	/**
	 * Returns the option maturity.
	 *
	 * @return The maturity.
	 */
	public double getMaturity() {
		return maturity;
	}

	/**
	 * Returns the option strike.
	 *
	 * @return The strike.
	 */
	public double getStrike() {
		return strike;
	}

	/**
	 * Returns whether the option is a call or put.
	 *
	 * @return The option type.
	 */
	public CallOrPut getCallOrPut() {
		return callOrPutSign;
	}

	/**
	 * Returns the exercise type.
	 *
	 * @return The exercise type.
	 */
	public ExerciseType getExercise() {
		return exercise;
	}
}