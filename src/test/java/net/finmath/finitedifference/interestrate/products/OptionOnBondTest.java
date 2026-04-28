package net.finmath.finitedifference.interestrate.products;

import static org.junit.Assert.assertEquals;

import java.time.LocalDate;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Test;

import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.GridWithMandatoryPoint;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.interestrate.models.FDMHullWhiteModel;
import net.finmath.marketdata.model.AnalyticModel;
import net.finmath.marketdata.model.AnalyticModelFromCurvesAndVols;
import net.finmath.marketdata.model.curves.Curve;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.montecarlo.interestrate.models.covariance.ShortRateVolatilityModel;
import net.finmath.montecarlo.interestrate.models.covariance.ShortRateVolatilityModelAsGiven;
import net.finmath.time.Period;
import net.finmath.time.Schedule;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;
import net.finmath.time.daycount.DayCountConvention;

/**
 * Tests for {@link OptionOnBond} under {@link FDMHullWhiteModel}.
 *
 * @author Alessandro Gnoatto
 */
public class OptionOnBondTest {

	private static final double THETA = 0.5;
	private static final double INITIAL_STATE = 0.0;

	private static final double FLAT_ZERO_RATE = 0.03;
	private static final double MEAN_REVERSION = 0.10;
	private static final double VOLATILITY = 0.01;

	private static final int NUMBER_OF_TIME_STEPS = 400;
	private static final int NUMBER_OF_SPACE_STEPS = 241;

	private static final double ZERO_COUPON_OPTION_TOLERANCE = 2.0E-3;
	private static final double COUPON_BOND_OPTION_TOLERANCE = 3.0E-3;

	private static final double TIME_TOLERANCE = 1E-12;
	private static final double ROOT_TOLERANCE = 1E-12;
	private static final int MAX_BRACKETING_STEPS = 100;
	private static final int MAX_BISECTION_STEPS = 200;

	private static final NormalDistribution NORMAL_DISTRIBUTION = new NormalDistribution();

	@Test
	public void testOptionOnZeroCouponBondAgainstAnalyticHullWhiteFormula() {
		final Bond underlyingBond = Bond.ofZeroCouponBond(2.0, 1.0);
		final double exerciseDate = 1.0;
		final double strike = 0.95;

		final OptionOnBond option = new OptionOnBond(
				underlyingBond,
				exerciseDate,
				strike,
				CallOrPut.CALL
		);

		final FDMHullWhiteModel model = createHullWhiteModel(
				exerciseDate,
				-0.20,
				0.20,
				NUMBER_OF_SPACE_STEPS
		);

		final double[] values = option.getValue(0.0, model);
		final double pdeValue = extractValueAtInitialState(values, model);

		final double analyticValue = getAnalyticOptionValue(option, model);

		assertEquals(analyticValue, pdeValue, ZERO_COUPON_OPTION_TOLERANCE);
	}

	@Test
	public void testOptionOnCouponBondAgainstAnalyticHullWhiteFormula() {
		final double[] periodStarts = new double[] { 1.0, 1.5, 2.0 };
		final double[] periodEnds = new double[] { 1.5, 2.0, 2.5 };
		final double[] paymentTimes = new double[] { 1.5, 2.0, 2.5 };

		final Schedule schedule = new TestSchedule(periodStarts, periodEnds, paymentTimes);

		final Bond underlyingBond = new Bond(schedule, 0.04, 1.0, 1.0);
		final double exerciseDate = 1.0;
		final double strike = 1.00;

		final OptionOnBond option = new OptionOnBond(
				underlyingBond,
				exerciseDate,
				strike,
				CallOrPut.CALL
		);

		final FDMHullWhiteModel model = createHullWhiteModel(
				exerciseDate,
				-0.20,
				0.20,
				NUMBER_OF_SPACE_STEPS
		);

		final double[] values = option.getValue(0.0, model);
		final double pdeValue = extractValueAtInitialState(values, model);

		final double analyticValue = getAnalyticOptionValue(option, model);

		assertEquals(analyticValue, pdeValue, COUPON_BOND_OPTION_TOLERANCE);
	}

	private static FDMHullWhiteModel createHullWhiteModel(
			final double solverHorizon,
			final double stateLowerBound,
			final double stateUpperBound,
			final int numberOfSpaceSteps) {

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				NUMBER_OF_TIME_STEPS,
				solverHorizon / NUMBER_OF_TIME_STEPS
		);

		final Grid stateGrid = new GridWithMandatoryPoint(
				numberOfSpaceSteps,
				stateLowerBound,
				stateUpperBound,
				INITIAL_STATE
		);

		final SpaceTimeDiscretization spaceTimeDiscretization = new SpaceTimeDiscretization(
				stateGrid,
				timeDiscretization,
				THETA,
				new double[] { INITIAL_STATE }
		);

		final DiscountCurve discountCurve = createFlatDiscountCurve("discountCurve", FLAT_ZERO_RATE);
		final AnalyticModel analyticModel = new AnalyticModelFromCurvesAndVols(new Curve[] { discountCurve });

		final double[] volatilities = new double[timeDiscretization.getNumberOfTimeSteps()];
		final double[] meanReversions = new double[timeDiscretization.getNumberOfTimeSteps()];
		Arrays.fill(volatilities, VOLATILITY);
		Arrays.fill(meanReversions, MEAN_REVERSION);

		final ShortRateVolatilityModel volatilityModel = new ShortRateVolatilityModelAsGiven(
				timeDiscretization,
				volatilities,
				meanReversions
		);

		return new FDMHullWhiteModel(
				analyticModel,
				discountCurve,
				volatilityModel,
				spaceTimeDiscretization
		);
	}

	private static double extractValueAtInitialState(
			final double[] values,
			final FDMHullWhiteModel model) {

		final double[] grid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - INITIAL_STATE) < 1E-12) {
				return values[i];
			}
		}

		throw new IllegalArgumentException("Initial state is not a grid node.");
	}

	private static DiscountCurve createFlatDiscountCurve(final String name, final double rate) {
		final double[] times = new double[] { 0.0, 1.0, 2.0, 5.0, 10.0 };
		final double[] zeroRates = new double[] { rate, rate, rate, rate, rate };

		return DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
				name,
				null,
				times,
				zeroRates,
				InterpolationMethod.LINEAR,
				ExtrapolationMethod.CONSTANT,
				InterpolationEntity.VALUE
		);
	}

	private static double getAnalyticOptionValue(
			final OptionOnBond option,
			final FDMHullWhiteModel model) {

		final Bond underlyingBond = option.getUnderlyingBond();
		final double exerciseDate = option.getExerciseDate();
		final double strike = option.getStrike();
		final CallOrPut callOrPut = option.getCallOrPut();

		if(Math.abs(strike) <= TIME_TOLERANCE) {
			if(callOrPut == CallOrPut.CALL) {
				return getUnderlyingBondValueAtTime(underlyingBond, model, 0.0, INITIAL_STATE);
			}
			return 0.0;
		}

		final double exerciseBoundaryState =
				findExerciseBoundaryState(underlyingBond, model, exerciseDate, strike);

		double value = 0.0;

		for(int periodIndex = 0; periodIndex < underlyingBond.getSchedule().getNumberOfPeriods(); periodIndex++) {
			final double paymentTime = underlyingBond.getSchedule().getPayment(periodIndex);

			if(paymentTime < exerciseDate - TIME_TOLERANCE) {
				continue;
			}

			final double cashflow = underlyingBond.getCashflow(periodIndex);
			final double zeroCouponStrike = model.getDiscountBond(
					exerciseDate,
					paymentTime,
					exerciseBoundaryState
			);

			value += cashflow * getZeroCouponBondOptionValue(
					model,
					0.0,
					INITIAL_STATE,
					exerciseDate,
					paymentTime,
					zeroCouponStrike,
					callOrPut
			);
		}

		return value;
	}

	private static double getUnderlyingBondValueAtTime(
			final Bond bond,
			final FDMHullWhiteModel model,
			final double time,
			final double stateVariable) {

		double value = 0.0;

		for(int periodIndex = 0; periodIndex < bond.getSchedule().getNumberOfPeriods(); periodIndex++) {
			final double paymentTime = bond.getSchedule().getPayment(periodIndex);

			if(paymentTime < time - TIME_TOLERANCE) {
				continue;
			}

			value += bond.getCashflow(periodIndex) * model.getDiscountBond(
					time,
					paymentTime,
					stateVariable
			);
		}

		return value;
	}

	private static double findExerciseBoundaryState(
			final Bond bond,
			final FDMHullWhiteModel model,
			final double exerciseDate,
			final double strike) {

		double lower = -1.0;
		double upper = 1.0;

		double functionValueAtLower =
				getBondValueMinusStrikeAtExercise(bond, model, exerciseDate, lower, strike);
		double functionValueAtUpper =
				getBondValueMinusStrikeAtExercise(bond, model, exerciseDate, upper, strike);

		int bracketingStep = 0;
		while(functionValueAtLower < 0.0 && bracketingStep < MAX_BRACKETING_STEPS) {
			lower *= 2.0;
			functionValueAtLower =
					getBondValueMinusStrikeAtExercise(bond, model, exerciseDate, lower, strike);
			bracketingStep++;
		}

		bracketingStep = 0;
		while(functionValueAtUpper > 0.0 && bracketingStep < MAX_BRACKETING_STEPS) {
			upper *= 2.0;
			functionValueAtUpper =
					getBondValueMinusStrikeAtExercise(bond, model, exerciseDate, upper, strike);
			bracketingStep++;
		}

		if(functionValueAtLower < 0.0 || functionValueAtUpper > 0.0) {
			throw new IllegalArgumentException("Could not bracket the Jamshidian root.");
		}

		double left = lower;
		double right = upper;

		for(int iteration = 0; iteration < MAX_BISECTION_STEPS; iteration++) {
			final double midpoint = 0.5 * (left + right);
			final double functionValueAtMidpoint =
					getBondValueMinusStrikeAtExercise(bond, model, exerciseDate, midpoint, strike);

			if(Math.abs(functionValueAtMidpoint) < ROOT_TOLERANCE
					|| Math.abs(right - left) < ROOT_TOLERANCE) {
				return midpoint;
			}

			if(functionValueAtMidpoint > 0.0) {
				left = midpoint;
			}
			else {
				right = midpoint;
			}
		}

		return 0.5 * (left + right);
	}

	private static double getBondValueMinusStrikeAtExercise(
			final Bond bond,
			final FDMHullWhiteModel model,
			final double exerciseDate,
			final double stateVariable,
			final double strike) {
		return getUnderlyingBondValueAtTime(bond, model, exerciseDate, stateVariable) - strike;
	}

	private static double getZeroCouponBondOptionValue(
			final FDMHullWhiteModel model,
			final double currentTime,
			final double currentStateVariable,
			final double exerciseDate,
			final double bondMaturity,
			final double strike,
			final CallOrPut callOrPut) {

		final int sign = callOrPut.toInteger();

		final double discountBondToExercise = model.getDiscountBond(
				currentTime,
				exerciseDate,
				currentStateVariable
		);

		if(Math.abs(bondMaturity - exerciseDate) <= TIME_TOLERANCE) {
			return discountBondToExercise * Math.max(sign * (1.0 - strike), 0.0);
		}

		final double discountBondToMaturity = model.getDiscountBond(
				currentTime,
				bondMaturity,
				currentStateVariable
		);

		final double bondVolatilitySquared =
				model.getShortRateConditionalVariance(currentTime, exerciseDate)
				* model.getB(exerciseDate, bondMaturity)
				* model.getB(exerciseDate, bondMaturity);

		if(bondVolatilitySquared <= TIME_TOLERANCE) {
			final double forwardBondPrice = discountBondToMaturity / discountBondToExercise;
			return discountBondToExercise * Math.max(sign * (forwardBondPrice - strike), 0.0);
		}

		final double bondVolatility = Math.sqrt(bondVolatilitySquared);

		final double dPlus =
				(Math.log(discountBondToMaturity / (strike * discountBondToExercise))
						+ 0.5 * bondVolatilitySquared)
				/ bondVolatility;

		final double dMinus = dPlus - bondVolatility;

		if(callOrPut == CallOrPut.CALL) {
			return discountBondToMaturity * NORMAL_DISTRIBUTION.cumulativeProbability(dPlus)
					- strike * discountBondToExercise * NORMAL_DISTRIBUTION.cumulativeProbability(dMinus);
		}
		else {
			return strike * discountBondToExercise * NORMAL_DISTRIBUTION.cumulativeProbability(-dMinus)
					- discountBondToMaturity * NORMAL_DISTRIBUTION.cumulativeProbability(-dPlus);
		}
	}

	/**
	 * Minimal deterministic schedule for tests.
	 *
	 * @author Alessandro Gnoatto
	 */
	private static final class TestSchedule implements Schedule {

		private final double[] periodStarts;
		private final double[] periodEnds;
		private final double[] payments;

		private TestSchedule(
				final double[] periodStarts,
				final double[] periodEnds,
				final double[] payments) {

			if(periodStarts == null || periodEnds == null || payments == null) {
				throw new IllegalArgumentException("Schedule arrays must not be null.");
			}
			if(periodStarts.length == 0 || periodStarts.length != periodEnds.length || periodStarts.length != payments.length) {
				throw new IllegalArgumentException("Schedule arrays must be non-empty and have equal length.");
			}

			this.periodStarts = periodStarts.clone();
			this.periodEnds = periodEnds.clone();
			this.payments = payments.clone();

			for(int i = 0; i < this.periodStarts.length; i++) {
				if(this.periodStarts[i] < 0.0 || this.periodEnds[i] < this.periodStarts[i] || this.payments[i] < this.periodEnds[i]) {
					throw new IllegalArgumentException("Invalid schedule period.");
				}
				if(i > 0) {
					if(this.periodStarts[i] < this.periodEnds[i - 1] - 1E-12) {
						throw new IllegalArgumentException("Schedule periods must be ordered.");
					}
					if(this.payments[i] <= this.payments[i - 1]) {
						throw new IllegalArgumentException("Payments must be strictly increasing.");
					}
				}
			}
		}

		@Override
		public LocalDate getReferenceDate() {
			return null;
		}

		@Override
		public List<Period> getPeriods() {
			return Collections.nCopies(getNumberOfPeriods(), (Period) null);
		}

		@Override
		public DayCountConvention getDaycountconvention() {
			return null;
		}

		@Override
		public int getNumberOfPeriods() {
			return payments.length;
		}

		@Override
		public Period getPeriod(final int periodIndex) {
			validateIndex(periodIndex);
			return null;
		}

		@Override
		public double getFixing(final int periodIndex) {
			validateIndex(periodIndex);
			return periodStarts[periodIndex];
		}

		@Override
		public double getPayment(final int periodIndex) {
			validateIndex(periodIndex);
			return payments[periodIndex];
		}

		@Override
		public double getPeriodStart(final int periodIndex) {
			validateIndex(periodIndex);
			return periodStarts[periodIndex];
		}

		@Override
		public double getPeriodEnd(final int periodIndex) {
			validateIndex(periodIndex);
			return periodEnds[periodIndex];
		}

		@Override
		public double getPeriodLength(final int periodIndex) {
			validateIndex(periodIndex);
			return periodEnds[periodIndex] - periodStarts[periodIndex];
		}

		@Override
		public int getPeriodIndex(final double time) {
			for(int i = 0; i < getNumberOfPeriods(); i++) {
				if(time >= periodStarts[i] - 1E-12 && time <= periodEnds[i] + 1E-12) {
					return i;
				}
			}
			return -1;
		}

		@Override
		public int getPeriodIndex(final LocalDate date) {
			return -1;
		}

		@Override
		public Iterator<Period> iterator() {
			return getPeriods().iterator();
		}

		private void validateIndex(final int periodIndex) {
			if(periodIndex < 0 || periodIndex >= getNumberOfPeriods()) {
				throw new IllegalArgumentException("periodIndex out of bounds.");
			}
		}
	}
}