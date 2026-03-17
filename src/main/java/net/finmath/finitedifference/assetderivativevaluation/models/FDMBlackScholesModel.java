package net.finmath.finitedifference.assetderivativevaluation.models;

import java.time.LocalDate;

import net.finmath.finitedifference.assetderivativevaluation.boundaries.FDBoundaryFactory;
import net.finmath.finitedifference.assetderivativevaluation.boundaries.FiniteDifferenceBoundary;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;

/**
 * Finite difference model for option pricing under the Black–Scholes framework
 * for European and American options.
 *
 * @author Alessandro Gnoatto (this version)
 * @author Christian Fries, Ralph Rudd, Jörg Kienitz (original version)
 */
public class FDMBlackScholesModel implements FiniteDifferenceEquityModel, FiniteDifferenceBoundary {

	private final double initialValue;
	private final DiscountCurve riskFreeCurve;
	private final double volatility;
	private final DiscountCurve dividendYieldCurve;
	private final SpaceTimeDiscretization spaceTimeDiscretization;

	/**
	 * Constructs a Black–Scholes finite difference model for option pricing.
	 *
	 * @param initialValue            Initial spot price.
	 * @param riskFreeCurve           Risk-free discount curve.
	 * @param dividendYieldCurve      Dividend yield discount curve.
	 * @param volatility              Constant volatility of the underlying asset.
	 * @param spaceTimeDiscretization Grid object defining the spatial discretization.
	 */
	public FDMBlackScholesModel(
			final double initialValue,
			final DiscountCurve riskFreeCurve,
			final DiscountCurve dividendYieldCurve,
			final double volatility,
			final SpaceTimeDiscretization spaceTimeDiscretization) {

		this.initialValue = initialValue;
		this.riskFreeCurve = riskFreeCurve;
		this.dividendYieldCurve = dividendYieldCurve;
		this.volatility = volatility;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
	}

	/**
	 * Constructs a Black–Scholes finite difference model for option pricing without
	 * dividend yield.
	 *
	 * @param initialValue            Initial spot price.
	 * @param riskFreeCurve           Risk-free discount curve.
	 * @param volatility              Constant volatility of the underlying asset.
	 * @param spaceTimeDiscretization Grid object defining the spatial discretization.
	 */
	public FDMBlackScholesModel(
			final double initialValue,
			final DiscountCurve riskFreeCurve,
			final double volatility,
			final SpaceTimeDiscretization spaceTimeDiscretization) {

		this.initialValue = initialValue;
		this.riskFreeCurve = riskFreeCurve;

		final double[] times = new double[] {0.0, 1.0};
		final double[] givenAnnualizedZeroRates = new double[] {0.0, 0.0};
		final InterpolationMethod interpolationMethod = InterpolationMethod.LINEAR;
		final InterpolationEntity interpolationEntity = InterpolationEntity.LOG_OF_VALUE_PER_TIME;
		final ExtrapolationMethod extrapolationMethod = ExtrapolationMethod.CONSTANT;

		final DiscountCurve dividendYieldCurve =
				DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
						"dividendCurve",
						LocalDate.of(2010, 8, 1),
						times,
						givenAnnualizedZeroRates,
						interpolationMethod,
						extrapolationMethod,
						interpolationEntity);

		this.dividendYieldCurve = dividendYieldCurve;

		this.volatility = volatility;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
	}

	/**
	 * Constructs a Black–Scholes finite difference model for option pricing.
	 *
	 * @param initialValue            Initial spot price.
	 * @param riskFreeRate            Constant risk-free rate.
	 * @param dividendYieldRate       Constant dividend yield rate.
	 * @param volatility              Constant volatility of the underlying asset.
	 * @param spaceTimeDiscretization Grid object defining the spatial discretization.
	 */
	public FDMBlackScholesModel(
			final double initialValue,
			final double riskFreeRate,
			final double dividendYieldRate,
			final double volatility,
			final SpaceTimeDiscretization spaceTimeDiscretization) {

		this.initialValue = initialValue;

		final double[] times = new double[] {0.0, 1.0};
		final double[] givenAnnualizedZeroRates = new double[] {riskFreeRate, riskFreeRate};
		final InterpolationMethod interpolationMethod = InterpolationMethod.LINEAR;
		final InterpolationEntity interpolationEntity = InterpolationEntity.VALUE;
		final ExtrapolationMethod extrapolationMethod = ExtrapolationMethod.CONSTANT;

		final DiscountCurve riskFreeCurve =
				DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
						"dividendCurve",
						null,
						times,
						givenAnnualizedZeroRates,
						interpolationMethod,
						extrapolationMethod,
						interpolationEntity);

		this.riskFreeCurve = riskFreeCurve;

		final double[] times1 = new double[] {0.0, 1.0};
		final double[] givenAnnualizedZeroRates1 = new double[] {dividendYieldRate, dividendYieldRate};
		final InterpolationMethod interpolationMethod1 = InterpolationMethod.LINEAR;
		final InterpolationEntity interpolationEntity1 = InterpolationEntity.VALUE;
		final ExtrapolationMethod extrapolationMethod1 = ExtrapolationMethod.CONSTANT;

		final DiscountCurve dividendYieldCurve =
				DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
						"dividendCurve",
						null,
						times1,
						givenAnnualizedZeroRates1,
						interpolationMethod1,
						extrapolationMethod1,
						interpolationEntity1);

		this.dividendYieldCurve = dividendYieldCurve;

		this.volatility = volatility;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
	}

	/**
	 * Constructs a Black–Scholes finite difference model for option pricing without
	 * dividend yield.
	 *
	 * @param initialValue            Initial spot price.
	 * @param riskFreeRate            Constant risk-free rate.
	 * @param volatility              Constant volatility of the underlying asset.
	 * @param spaceTimeDiscretization Grid object defining the spatial discretization.
	 */
	public FDMBlackScholesModel(
			final double initialValue,
			final double riskFreeRate,
			final double volatility,
			final SpaceTimeDiscretization spaceTimeDiscretization) {

		this.initialValue = initialValue;

		final double[] times = new double[] {0.0, 1.0};
		final double[] givenAnnualizedZeroRates = new double[] {riskFreeRate, riskFreeRate};
		final InterpolationMethod interpolationMethod = InterpolationMethod.LINEAR;
		final InterpolationEntity interpolationEntity = InterpolationEntity.VALUE;
		final ExtrapolationMethod extrapolationMethod = ExtrapolationMethod.CONSTANT;

		final DiscountCurve riskFreeCurve =
				DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
						"dividendCurve",
						LocalDate.of(2010, 8, 1),
						times,
						givenAnnualizedZeroRates,
						interpolationMethod,
						extrapolationMethod,
						interpolationEntity);

		this.riskFreeCurve = riskFreeCurve;

		final double[] times1 = new double[] {0.0, 1.0};
		final double[] givenAnnualizedZeroRates1 = new double[] {0.0, 0.0};
		final InterpolationMethod interpolationMethod1 = InterpolationMethod.LINEAR;
		final InterpolationEntity interpolationEntity1 = InterpolationEntity.VALUE;
		final ExtrapolationMethod extrapolationMethod1 = ExtrapolationMethod.CONSTANT;

		final DiscountCurve dividendYieldCurve =
				DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
						"dividendCurve",
						LocalDate.of(2010, 8, 1),
						times1,
						givenAnnualizedZeroRates1,
						interpolationMethod1,
						extrapolationMethod1,
						interpolationEntity1);

		this.dividendYieldCurve = dividendYieldCurve;

		this.volatility = volatility;
		this.spaceTimeDiscretization = spaceTimeDiscretization;
	}

	@Override
	public DiscountCurve getRiskFreeCurve() {
		return riskFreeCurve;
	}

	/**
	 * Returns the initial value (spot) of the underlying.
	 *
	 * @return The initial value.
	 */
	public double[] getInitialValue() {
		return new double[] {initialValue};
	}

	@Override
	public double[] getDrift(double time, double... stateVariables) {
		if(time == 0) {
			time = 0.000001;
		}
		final double[] result = new double[1];

		final double rF = getRiskFreeCurve().getDiscountFactor(time);
		final double riskFreeRate = -Math.log(rF) / time;

		final double dY = getDividendYieldCurve().getDiscountFactor(time);
		final double dividendYieldRate = -Math.log(dY) / time;

		result[0] = (riskFreeRate - dividendYieldRate) * stateVariables[0];
		return result;
	}

	@Override
	public double[][] getFactorLoading(double time, double... stateVariables) {
		final double[][] result = new double[1][1];
		result[0][0] = volatility * stateVariables[0];
		return result;
	}

	@Override
	public DiscountCurve getDividendYieldCurve() {
		return dividendYieldCurve;
	}

	@Override
	public SpaceTimeDiscretization getSpaceTimeDiscretization() {
		return spaceTimeDiscretization;
	}

	@Override
	public double[] getValueAtLowerBoundary(
			final FiniteDifferenceProduct product,
			final double time,
			final double... riskFactors) {

		final FiniteDifferenceBoundary boundary =
				FDBoundaryFactory.createBoundary(this, product);

		return boundary.getValueAtLowerBoundary(product, time, riskFactors);
	}

	@Override
	public double[] getValueAtUpperBoundary(
			final FiniteDifferenceProduct product,
			final double time,
			final double... riskFactors) {

		final FiniteDifferenceBoundary boundary =
				FDBoundaryFactory.createBoundary(this, product);

		return boundary.getValueAtUpperBoundary(product, time, riskFactors);
	}

	@Override
	public FiniteDifferenceEquityModel getCloneWithModifiedSpaceTimeDiscretization(
			final SpaceTimeDiscretization newSpaceTimeDiscretization) {
		return new FDMBlackScholesModel(
				initialValue,
				riskFreeCurve,
				dividendYieldCurve,
				volatility,
				newSpaceTimeDiscretization
		);
	}
}