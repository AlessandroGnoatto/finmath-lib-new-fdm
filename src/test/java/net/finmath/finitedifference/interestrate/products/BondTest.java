package net.finmath.finitedifference.interestrate.products;

import static org.junit.Assert.assertEquals;

import java.time.LocalDate;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

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
import net.finmath.montecarlo.interestrate.models.covariance.ShortRateVolatilityModel;
import net.finmath.montecarlo.interestrate.models.covariance.ShortRateVolatilityModelAsGiven;
import net.finmath.time.Period;
import net.finmath.time.Schedule;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;
import net.finmath.time.daycount.DayCountConvention;

/**
 * Tests for {@link Bond} under {@link FDMHullWhiteModel}.
 *
 * @author Alessandro Gnoatto
 */
public class BondTest {

	private static final double THETA = 0.5;
	private static final double INITIAL_STATE = 0.0;

	private static final double FLAT_ZERO_RATE = 0.03;
	private static final double MEAN_REVERSION = 0.10;
	private static final double VOLATILITY = 0.01;

	private static final double ZERO_COUPON_TOLERANCE = 2.0E-3;
	private static final double FIXED_COUPON_TOLERANCE = 3.0E-3;

	@Test
	public void testZeroCouponBondAgainstHullWhiteDiscountBond() {
		final double maturity = 2.0;
		final double notional = 1.0;

		final Bond zeroCouponBond = Bond.ofZeroCouponBond(maturity, notional);
		final FDMHullWhiteModel model = createHullWhiteModel(
				maturity,
				new double[] { maturity },
				-0.15,
				0.15,
				161
		);

		final double[] values = zeroCouponBond.getValue(0.0, model);
		final double pdeValue = extractValueAtInitialState(values, model);

		final double analyticValue =
				notional * model.getDiscountBond(0.0, maturity, INITIAL_STATE);

		assertEquals(analyticValue, pdeValue, ZERO_COUPON_TOLERANCE);
	}

	@Test
	public void testFixedCouponBondAgainstDiscountedCashflowSum() {
		final double[] paymentTimes = new double[] { 1.0, 1.5, 2.0 };
		final double[] periodStarts = new double[] { 0.5, 1.0, 1.5 };
		final double[] periodEnds = new double[] { 1.0, 1.5, 2.0 };

		final Schedule schedule = new TestSchedule(periodStarts, periodEnds, paymentTimes);

		final double fixedCoupon = 0.04;
		final double notional = 1.0;
		final double redemption = 1.0;

		final Bond fixedCouponBond = new Bond(schedule, fixedCoupon, notional, redemption);
		final FDMHullWhiteModel model = createHullWhiteModel(
				2.0,
				paymentTimes,
				-0.15,
				0.15,
				161
		);

		final double[] values = fixedCouponBond.getValue(0.0, model);
		final double pdeValue = extractValueAtInitialState(values, model);

		double analyticValue = 0.0;
		for(int periodIndex = 0; periodIndex < schedule.getNumberOfPeriods(); periodIndex++) {
			final double paymentTime = schedule.getPayment(periodIndex);
			analyticValue += fixedCouponBond.getCashflow(periodIndex)
					* model.getDiscountBond(0.0, paymentTime, INITIAL_STATE);
		}

		assertEquals(analyticValue, pdeValue, FIXED_COUPON_TOLERANCE);
	}

	private static FDMHullWhiteModel createHullWhiteModel(
			final double maturity,
			final double[] mandatoryTimes,
			final double stateLowerBound,
			final double stateUpperBound,
			final int numberOfSpaceSteps) {

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, 100, 0.02);

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