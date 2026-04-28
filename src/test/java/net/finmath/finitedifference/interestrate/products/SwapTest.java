package net.finmath.finitedifference.interestrate.products;

import static org.junit.Assert.assertEquals;

import java.time.LocalDate;
import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import net.finmath.exception.CalculationException;
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
import net.finmath.marketdata.model.curves.ForwardCurve;
import net.finmath.marketdata.model.curves.ForwardCurveFromDiscountCurve;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.BrownianMotionLazyInit;
import net.finmath.montecarlo.interestrate.LIBORMonteCarloSimulationFromLIBORModel;
import net.finmath.montecarlo.interestrate.LIBORModelMonteCarloSimulationModel;
import net.finmath.montecarlo.interestrate.models.HullWhiteModel;
import net.finmath.montecarlo.interestrate.models.covariance.ShortRateVolatilityModel;
import net.finmath.montecarlo.interestrate.models.covariance.ShortRateVolatilityModelAsGiven;
import net.finmath.montecarlo.interestrate.products.SimpleSwap;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.time.FloatingpointDate;
import net.finmath.time.Period;
import net.finmath.time.Schedule;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;
import net.finmath.time.daycount.DayCountConvention;

/**
 * Tests for {@link Swap} under {@link FDMHullWhiteModel}.
 *
 * <p>
 * The benchmark is the Monte Carlo Hull-White implementation from finmath,
 * using {@link SimpleSwap}.
 * </p>
 *
 * <p>
 * The PDE swap is interpreted in the current reduced-scope Markovian sense as a
 * remaining forward-looking swap. Therefore the benchmark is performed at
 * evaluation time 0, where this interpretation coincides with the standard
 * forward swap representation used by {@link SimpleSwap}.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class SwapTest {

	private static final LocalDate REFERENCE_DATE = LocalDate.of(2021, 1, 1);

	private static final double THETA = 0.5;
	private static final double INITIAL_STATE = 0.0;

	private static final double FLAT_ZERO_RATE = 0.03;
	private static final double MEAN_REVERSION = 0.10;
	private static final double VOLATILITY = 0.01;

	private static final int NUMBER_OF_SPACE_STEPS = 241;
	private static final int NUMBER_OF_MONTE_CARLO_PATHS = 50000;

	private static final double PAR_SWAP_TOLERANCE = 1.0E-2;
	private static final double OFF_MARKET_SWAP_TOLERANCE = 1.2E-2;

	@Test
	public void testPayerParSwapAgainstMonteCarloHullWhite() throws CalculationException {
		final LocalDate swapStartDate = REFERENCE_DATE.plusYears(1);
		final LocalDate swapEndDate = REFERENCE_DATE.plusYears(6);

		final Schedule fixSchedule = createAnnualSchedule(REFERENCE_DATE, swapStartDate, swapEndDate);
		final Schedule floatSchedule = createAnnualSchedule(REFERENCE_DATE, swapStartDate, swapEndDate);

		final double horizon = extractPaymentDates(floatSchedule)[floatSchedule.getNumberOfPeriods() - 1] + 0.25;
		final ModelSetup modelSetup = createModelSetup(horizon);

		final double parSwapRate = net.finmath.marketdata.products.Swap.getForwardSwapRate(
				fixSchedule,
				floatSchedule,
				modelSetup.forwardCurve,
				modelSetup.analyticModel
		);

		final SwapLeg receiverFloatLeg = new SwapLeg(
				floatSchedule,
				modelSetup.forwardCurve.getName(),
				0.0,
				false
		);

		final SwapLeg payerFixedLeg = new SwapLeg(
				fixSchedule,
				null,
				parSwapRate,
				false
		);

		final Swap pdeSwap = new Swap(receiverFloatLeg, payerFixedLeg);

		final double[] pdeValues = pdeSwap.getValue(0.0, modelSetup.pdeModel);
		final double pdeValue = extractValueAtInitialState(pdeValues, modelSetup.pdeModel);

		final SimpleSwap monteCarloSwap = new SimpleSwap(
				extractFixingDates(floatSchedule),
				extractPaymentDates(floatSchedule),
				createConstantArray(floatSchedule.getNumberOfPeriods(), parSwapRate),
				true,
				1.0
		);

		final double monteCarloValue = monteCarloSwap.getValue(0.0, modelSetup.monteCarloModel).getAverage();

		assertEquals(monteCarloValue, pdeValue, PAR_SWAP_TOLERANCE);
	}

	@Test
	public void testReceiverOffMarketSwapWithVaryingNotionalsAgainstMonteCarloHullWhite() throws CalculationException {
		final LocalDate swapStartDate = REFERENCE_DATE.plusYears(1);
		final LocalDate swapEndDate = REFERENCE_DATE.plusYears(6);

		final Schedule fixSchedule = createAnnualSchedule(REFERENCE_DATE, swapStartDate, swapEndDate);
		final Schedule floatSchedule = createAnnualSchedule(REFERENCE_DATE, swapStartDate, swapEndDate);

		final double horizon = extractPaymentDates(floatSchedule)[floatSchedule.getNumberOfPeriods() - 1] + 0.25;
		final ModelSetup modelSetup = createModelSetup(horizon);

		final double parSwapRate = net.finmath.marketdata.products.Swap.getForwardSwapRate(
				fixSchedule,
				floatSchedule,
				modelSetup.forwardCurve,
				modelSetup.analyticModel
		);

		final double offMarketFixedRate = parSwapRate + 0.01;
		final double[] notionals = new double[] { 1.00, 0.95, 0.90, 0.85, 0.80 };

		final SwapLeg receiverFixedLeg = new SwapLeg(
				fixSchedule,
				null,
				notionals,
				createConstantArray(fixSchedule.getNumberOfPeriods(), offMarketFixedRate),
				false
		);

		final SwapLeg payerFloatLeg = new SwapLeg(
				floatSchedule,
				modelSetup.forwardCurve.getName(),
				notionals,
				createConstantArray(floatSchedule.getNumberOfPeriods(), 0.0),
				false
		);

		final Swap pdeSwap = new Swap(receiverFixedLeg, payerFloatLeg);

		final double[] pdeValues = pdeSwap.getValue(0.0, modelSetup.pdeModel);
		final double pdeValue = extractValueAtInitialState(pdeValues, modelSetup.pdeModel);

		final SimpleSwap monteCarloSwap = new SimpleSwap(
				extractFixingDates(floatSchedule),
				extractPaymentDates(floatSchedule),
				createConstantArray(floatSchedule.getNumberOfPeriods(), offMarketFixedRate),
				false,
				notionals
		);

		final double monteCarloValue = monteCarloSwap.getValue(0.0, modelSetup.monteCarloModel).getAverage();

		assertEquals(monteCarloValue, pdeValue, OFF_MARKET_SWAP_TOLERANCE);
	}

	private static ModelSetup createModelSetup(final double horizon) throws CalculationException {
		final DiscountCurve discountCurve = createFlatDiscountCurve();
		final ForwardCurve forwardCurve = new ForwardCurveFromDiscountCurve(
				discountCurve.getName(),
				REFERENCE_DATE,
				"1Y"
		);

		final AnalyticModel analyticModel = new AnalyticModelFromCurvesAndVols(
				new Curve[] { discountCurve, forwardCurve }
		);

		final FDMHullWhiteModel pdeModel = createPdeHullWhiteModel(
				horizon,
				analyticModel,
				discountCurve
		);

		final LIBORModelMonteCarloSimulationModel monteCarloModel = createMonteCarloHullWhiteModel(
				horizon,
				analyticModel,
				forwardCurve,
				discountCurve
		);

		return new ModelSetup(
				analyticModel,
				discountCurve,
				forwardCurve,
				pdeModel,
				monteCarloModel
		);
	}

	private static FDMHullWhiteModel createPdeHullWhiteModel(
			final double horizon,
			final AnalyticModel analyticModel,
			final DiscountCurve discountCurve) {

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				(int)Math.round(horizon / 0.01),
				0.01
		);

		final Grid stateGrid = new GridWithMandatoryPoint(
				NUMBER_OF_SPACE_STEPS,
				-0.25,
				0.25,
				INITIAL_STATE
		);

		final SpaceTimeDiscretization spaceTimeDiscretization = new SpaceTimeDiscretization(
				stateGrid,
				timeDiscretization,
				THETA,
				new double[] { INITIAL_STATE }
		);

		final ShortRateVolatilityModel volatilityModel = new ShortRateVolatilityModelAsGiven(
				new TimeDiscretizationFromArray(0.0),
				new double[] { VOLATILITY },
				new double[] { MEAN_REVERSION }
		);

		return new FDMHullWhiteModel(
				analyticModel,
				discountCurve,
				volatilityModel,
				spaceTimeDiscretization
		);
	}

	private static LIBORModelMonteCarloSimulationModel createMonteCarloHullWhiteModel(
			final double horizon,
			final AnalyticModel analyticModel,
			final ForwardCurve forwardCurve,
			final DiscountCurve discountCurve) throws CalculationException {

		final TimeDiscretization liborPeriodDiscretization = new TimeDiscretizationFromArray(
				0.0,
				10,
				1.0
		);

		final ShortRateVolatilityModel volatilityModel = new ShortRateVolatilityModelAsGiven(
				new TimeDiscretizationFromArray(0.0),
				new double[] { VOLATILITY },
				new double[] { MEAN_REVERSION }
		);

		final java.util.Map<String, Object> properties = new java.util.HashMap<>();
		properties.put("isInterpolateDiscountFactorsOnLiborPeriodDiscretization", Boolean.FALSE);

		final HullWhiteModel hullWhiteModel = new HullWhiteModel(
				liborPeriodDiscretization,
				analyticModel,
				forwardCurve,
				discountCurve,
				volatilityModel,
				properties
		);

		final TimeDiscretization simulationTimeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				(int)Math.round(horizon / 0.02),
				0.02
		);

		final BrownianMotion brownianMotion = new BrownianMotionLazyInit(
				simulationTimeDiscretization,
				2,
				NUMBER_OF_MONTE_CARLO_PATHS,
				31415
		);

		final EulerSchemeFromProcessModel process = new EulerSchemeFromProcessModel(
				hullWhiteModel,
				brownianMotion,
				EulerSchemeFromProcessModel.Scheme.EULER
		);

		return new LIBORMonteCarloSimulationFromLIBORModel(hullWhiteModel, process);
	}

	private static DiscountCurve createFlatDiscountCurve() {
		return DiscountCurveInterpolation.createDiscountCurveFromZeroRates(
				"discountCurve",
				REFERENCE_DATE,
				new double[] { 1.0, 10.0 },
				new double[] { FLAT_ZERO_RATE, FLAT_ZERO_RATE },
				new boolean[] { false, false },
				InterpolationMethod.LINEAR,
				ExtrapolationMethod.CONSTANT,
				InterpolationEntity.LOG_OF_VALUE_PER_TIME
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

	private static double[] extractFixingDates(final Schedule schedule) {
		final double[] fixingDates = new double[schedule.getNumberOfPeriods()];
		for(int i = 0; i < fixingDates.length; i++) {
			fixingDates[i] = schedule.getFixing(i);
		}
		return fixingDates;
	}

	private static double[] extractPaymentDates(final Schedule schedule) {
		final double[] paymentDates = new double[schedule.getNumberOfPeriods()];
		for(int i = 0; i < paymentDates.length; i++) {
			paymentDates[i] = schedule.getPayment(i);
		}
		return paymentDates;
	}

	private static double[] createConstantArray(final int length, final double value) {
		final double[] values = new double[length];
		java.util.Arrays.fill(values, value);
		return values;
	}

	private static Schedule createAnnualSchedule(
			final LocalDate referenceDate,
			final LocalDate startDate,
			final LocalDate endDate) {

		if(!startDate.isBefore(endDate)) {
			throw new IllegalArgumentException("Require startDate < endDate.");
		}

		final java.util.ArrayList<Period> periods = new java.util.ArrayList<>();

		LocalDate periodStart = startDate;
		while(periodStart.isBefore(endDate)) {
			final LocalDate periodEnd = periodStart.plusYears(1);
			if(periodEnd.isAfter(endDate)) {
				throw new IllegalArgumentException("End date must align with annual periods.");
			}
			periods.add(new Period(periodStart, periodEnd, periodStart, periodEnd));
			periodStart = periodEnd;
		}

		return new LocalDateSchedule(referenceDate, periods);
	}

	/**
	 * Small container bundling the PDE and Monte Carlo model setup.
	 *
	 * @author Alessandro Gnoatto
	 */
	private static final class ModelSetup {

		private final AnalyticModel analyticModel;
		private final DiscountCurve discountCurve;
		private final ForwardCurve forwardCurve;
		private final FDMHullWhiteModel pdeModel;
		private final LIBORModelMonteCarloSimulationModel monteCarloModel;

		private ModelSetup(
				final AnalyticModel analyticModel,
				final DiscountCurve discountCurve,
				final ForwardCurve forwardCurve,
				final FDMHullWhiteModel pdeModel,
				final LIBORModelMonteCarloSimulationModel monteCarloModel) {
			this.analyticModel = analyticModel;
			this.discountCurve = discountCurve;
			this.forwardCurve = forwardCurve;
			this.pdeModel = pdeModel;
			this.monteCarloModel = monteCarloModel;
		}
	}

	/**
	 * Minimal date-based schedule implementation for tests.
	 *
	 * @author Alessandro Gnoatto
	 */
	private static final class LocalDateSchedule implements Schedule {

		private final LocalDate referenceDate;
		private final List<Period> periods;

		private final double[] fixingTimes;
		private final double[] paymentTimes;
		private final double[] periodStartTimes;
		private final double[] periodEndTimes;
		private final double[] periodLengths;

		private LocalDateSchedule(
				final LocalDate referenceDate,
				final List<Period> periods) {

			if(referenceDate == null) {
				throw new IllegalArgumentException("referenceDate must not be null.");
			}
			if(periods == null || periods.isEmpty()) {
				throw new IllegalArgumentException("periods must not be null or empty.");
			}

			this.referenceDate = referenceDate;
			this.periods = java.util.Collections.unmodifiableList(new java.util.ArrayList<>(periods));

			fixingTimes = new double[periods.size()];
			paymentTimes = new double[periods.size()];
			periodStartTimes = new double[periods.size()];
			periodEndTimes = new double[periods.size()];
			periodLengths = new double[periods.size()];

			for(int i = 0; i < periods.size(); i++) {
				final Period period = periods.get(i);

				fixingTimes[i] = FloatingpointDate.getFloatingPointDateFromDate(referenceDate, period.getFixing());
				paymentTimes[i] = FloatingpointDate.getFloatingPointDateFromDate(referenceDate, period.getPayment());
				periodStartTimes[i] = FloatingpointDate.getFloatingPointDateFromDate(referenceDate, period.getPeriodStart());
				periodEndTimes[i] = FloatingpointDate.getFloatingPointDateFromDate(referenceDate, period.getPeriodEnd());
				periodLengths[i] = periodEndTimes[i] - periodStartTimes[i];
			}
		}

		@Override
		public LocalDate getReferenceDate() {
			return referenceDate;
		}

		@Override
		public List<Period> getPeriods() {
			return periods;
		}

		@Override
		public DayCountConvention getDaycountconvention() {
			return null;
		}

		@Override
		public int getNumberOfPeriods() {
			return periods.size();
		}

		@Override
		public Period getPeriod(final int periodIndex) {
			return periods.get(periodIndex);
		}

		@Override
		public double getFixing(final int periodIndex) {
			return fixingTimes[periodIndex];
		}

		@Override
		public double getPayment(final int periodIndex) {
			return paymentTimes[periodIndex];
		}

		@Override
		public double getPeriodStart(final int periodIndex) {
			return periodStartTimes[periodIndex];
		}

		@Override
		public double getPeriodEnd(final int periodIndex) {
			return periodEndTimes[periodIndex];
		}

		@Override
		public double getPeriodLength(final int periodIndex) {
			return periodLengths[periodIndex];
		}

		@Override
		public int getPeriodIndex(final double time) {
			for(int i = 0; i < getNumberOfPeriods(); i++) {
				if(time >= getPeriodStart(i) - 1E-12 && time <= getPeriodEnd(i) + 1E-12) {
					return i;
				}
			}
			return -1;
		}

		@Override
		public int getPeriodIndex(final LocalDate date) {
			for(int i = 0; i < getNumberOfPeriods(); i++) {
				final Period period = getPeriod(i);
				if((!date.isBefore(period.getPeriodStart())) && (!date.isAfter(period.getPeriodEnd()))) {
					return i;
				}
			}
			return -1;
		}

		@Override
		public Iterator<Period> iterator() {
			return periods.iterator();
		}
	}
}