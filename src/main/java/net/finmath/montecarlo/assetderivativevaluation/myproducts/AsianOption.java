package net.finmath.montecarlo.assetderivativevaluation.myproducts;

import net.finmath.exception.CalculationException;
import net.finmath.modelling.products.AsianStrike;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.montecarlo.assetderivativevaluation.AssetModelMonteCarloSimulationModel;
import net.finmath.montecarlo.assetderivativevaluation.products.AbstractAssetMonteCarloProduct;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.TimeDiscretization;

/**
 * Implements the valuation of arithmetic Asian options.
 *
 * <p>
 * Given a model for an asset <i>S</i>, the Asian option with maturity <i>T</i>
 * and averaging points <i>T_i</i> for <i>i = 1,...,n</i> pays one of:
 * </p>
 *
 * <p>
 * Fixed-strike call: <i>max(A(T) - K, 0)</i><br>
 * Fixed-strike put: <i>max(K - A(T), 0)</i><br>
 * Floating-strike call: <i>max(S(T) - A(T), 0)</i><br>
 * Floating-strike put: <i>max(A(T) - S(T), 0)</i>
 * </p>
 *
 * <p>
 * where
 * <br>
 * <i>A(T) = 1/n (S(T_1)+ ... + S(T_n))</i>
 * </p>
 *
 * <p>
 * Backward compatibility:
 * The legacy constructors keep their original meaning and create a fixed-strike call.
 * </p>
 */
public class AsianOption extends AbstractAssetMonteCarloProduct {

	private final double maturity;
	private final double strike;
	private final TimeDiscretization timesForAveraging;
	private final Integer underlyingIndex;
	private final CallOrPut callOrPut;
	private final AsianStrike asianStrike;

	/**
	 * Legacy constructor, fully backward compatible.
	 * Creates a fixed-strike Asian call.
	 *
	 * @param maturity The maturity T.
	 * @param strike The strike K.
	 * @param timesForAveraging The times t_i used in the calculation of A(T).
	 * @param underlyingIndex The index of the asset S to be fetched from the model.
	 */
	public AsianOption(
			final double maturity,
			final double strike,
			final TimeDiscretization timesForAveraging,
			final Integer underlyingIndex) {
		super();
		this.maturity = maturity;
		this.strike = strike;
		this.timesForAveraging = timesForAveraging;
		this.underlyingIndex = underlyingIndex;
		this.callOrPut = CallOrPut.CALL;
		this.asianStrike = AsianStrike.FIXED_STRIKE;
	}

	/**
	 * Legacy constructor, fully backward compatible.
	 * Creates a fixed-strike Asian call.
	 *
	 * @param maturity The maturity T.
	 * @param strike The strike K.
	 * @param timesForAveraging The times t_i used in the calculation of A(T).
	 */
	public AsianOption(
			final double maturity,
			final double strike,
			final TimeDiscretization timesForAveraging) {
		this(maturity, strike, timesForAveraging, 0);
	}

	/**
	 * Constructor for fixed-strike Asian options.
	 *
	 * @param maturity The maturity T.
	 * @param strike The strike K.
	 * @param timesForAveraging The averaging times.
	 * @param underlyingIndex The underlying index.
	 * @param callOrPut Call or put.
	 */
	public AsianOption(
			final double maturity,
			final double strike,
			final TimeDiscretization timesForAveraging,
			final Integer underlyingIndex,
			final CallOrPut callOrPut) {
		this(maturity, strike, timesForAveraging, underlyingIndex, callOrPut, AsianStrike.FIXED_STRIKE);
	}

	/**
	 * Constructor for fixed-strike Asian options.
	 *
	 * @param maturity The maturity T.
	 * @param strike The strike K.
	 * @param timesForAveraging The averaging times.
	 * @param callOrPut Call or put.
	 */
	public AsianOption(
			final double maturity,
			final double strike,
			final TimeDiscretization timesForAveraging,
			final CallOrPut callOrPut) {
		this(maturity, strike, timesForAveraging, 0, callOrPut, AsianStrike.FIXED_STRIKE);
	}

	/**
	 * General constructor for Asian options.
	 *
	 * For floating-strike options, strike is ignored and may be set to Double.NaN.
	 *
	 * @param maturity The maturity T.
	 * @param strike The strike K for fixed-strike options. Ignored for floating-strike options.
	 * @param timesForAveraging The averaging times.
	 * @param underlyingIndex The underlying index.
	 * @param callOrPut Call or put.
	 * @param asianStrike Fixed or floating strike.
	 */
	public AsianOption(
			final double maturity,
			final double strike,
			final TimeDiscretization timesForAveraging,
			final Integer underlyingIndex,
			final CallOrPut callOrPut,
			final AsianStrike asianStrike) {
		super();

		if(callOrPut == null) {
			throw new IllegalArgumentException("callOrPut must not be null.");
		}
		if(asianStrike == null) {
			throw new IllegalArgumentException("asianStrike must not be null.");
		}
		if(timesForAveraging == null) {
			throw new IllegalArgumentException("timesForAveraging must not be null.");
		}
		if(asianStrike == AsianStrike.FIXED_STRIKE && Double.isNaN(strike)) {
			throw new IllegalArgumentException("Strike must be specified for fixed-strike Asian options.");
		}

		this.maturity = maturity;
		this.strike = strike;
		this.timesForAveraging = timesForAveraging;
		this.underlyingIndex = underlyingIndex;
		this.callOrPut = callOrPut;
		this.asianStrike = asianStrike;
	}

	/**
	 * General constructor for Asian options.
	 *
	 * For floating-strike options, strike is ignored and may be set to Double.NaN.
	 *
	 * @param maturity The maturity T.
	 * @param strike The strike K for fixed-strike options. Ignored for floating-strike options.
	 * @param timesForAveraging The averaging times.
	 * @param callOrPut Call or put.
	 * @param asianStrike Fixed or floating strike.
	 */
	public AsianOption(
			final double maturity,
			final double strike,
			final TimeDiscretization timesForAveraging,
			final CallOrPut callOrPut,
			final AsianStrike asianStrike) {
		this(maturity, strike, timesForAveraging, 0, callOrPut, asianStrike);
	}

	/**
	 * Convenience constructor for floating-strike Asian options.
	 *
	 * @param maturity The maturity T.
	 * @param timesForAveraging The averaging times.
	 * @param underlyingIndex The underlying index.
	 * @param callOrPut Call or put.
	 */
	public AsianOption(
			final double maturity,
			final TimeDiscretization timesForAveraging,
			final Integer underlyingIndex,
			final CallOrPut callOrPut) {
		this(maturity, Double.NaN, timesForAveraging, underlyingIndex, callOrPut, AsianStrike.FLOATING_STRIKE);
	}

	/**
	 * Convenience constructor for floating-strike Asian options.
	 *
	 * @param maturity The maturity T.
	 * @param timesForAveraging The averaging times.
	 * @param callOrPut Call or put.
	 */
	public AsianOption(
			final double maturity,
			final TimeDiscretization timesForAveraging,
			final CallOrPut callOrPut) {
		this(maturity, Double.NaN, timesForAveraging, 0, callOrPut, AsianStrike.FLOATING_STRIKE);
	}

	@Override
	public RandomVariable getValue(final double evaluationTime, final AssetModelMonteCarloSimulationModel model)
			throws CalculationException {

		/*RandomVariable average = model.getRandomVariableForConstant(0.0);
		for(final double time : timesForAveraging) {
			final RandomVariable underlying = model.getAssetValue(time, underlyingIndex);
			average = average.add(underlying);
		}
		average = average.div(timesForAveraging.getNumberOfTimes());*/
		
		RandomVariable integral = model.getRandomVariableForConstant(0.0);

		for(int timeIndex = 0; timeIndex < timesForAveraging.getNumberOfTimes() - 1; timeIndex++) {
			final double time = timesForAveraging.getTime(timeIndex);
			final double nextTime = timesForAveraging.getTime(timeIndex + 1);
			final double dt = nextTime - time;

			final RandomVariable sLeft = model.getAssetValue(time, underlyingIndex);
			final RandomVariable sRight = model.getAssetValue(nextTime, underlyingIndex);

			integral = integral.add(sLeft.add(sRight).mult(0.5 * dt));
		}

		final RandomVariable average = integral.div(maturity);

		final RandomVariable underlyingAtMaturity = model.getAssetValue(maturity, underlyingIndex);

		RandomVariable values;
		if(asianStrike == AsianStrike.FIXED_STRIKE) {
			if(callOrPut == CallOrPut.CALL) {
				values = average.sub(strike).floor(0.0);
			}
			else {
				values = average.mult(-1.0).add(strike).floor(0.0);
			}
		}
		else if(asianStrike == AsianStrike.FLOATING_STRIKE) {
			if(callOrPut == CallOrPut.CALL) {
				values = underlyingAtMaturity.sub(average).floor(0.0);
			}
			else {
				values = average.sub(underlyingAtMaturity).floor(0.0);
			}
		}
		else {
			throw new IllegalArgumentException("Unsupported asian strike type.");
		}

		final RandomVariable numeraireAtMaturity = model.getNumeraire(maturity);
		final RandomVariable monteCarloWeights = model.getMonteCarloWeights(maturity);
		values = values.div(numeraireAtMaturity).mult(monteCarloWeights);

		final RandomVariable numeraireAtEvalTime = model.getNumeraire(evaluationTime);
		final RandomVariable monteCarloWeightsAtEvalTime = model.getMonteCarloWeights(evaluationTime);
		values = values.mult(numeraireAtEvalTime).div(monteCarloWeightsAtEvalTime);

		return values;
	}

	public double getMaturity() {
		return maturity;
	}

	public double getStrike() {
		return strike;
	}

	public TimeDiscretization getTimesForAveraging() {
		return timesForAveraging;
	}

	public Integer getUnderlyingIndex() {
		return underlyingIndex;
	}

	public CallOrPut getCallOrPut() {
		return callOrPut;
	}

	public AsianStrike getAsianStrike() {
		return asianStrike;
	}
}