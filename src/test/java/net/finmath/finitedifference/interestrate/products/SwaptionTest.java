package net.finmath.finitedifference.interestrate.products;

import static org.junit.Assert.assertEquals;

import java.time.LocalDate;
import java.util.Arrays;
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
import net.finmath.modelling.AbstractExercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.BrownianMotionLazyInit;
import net.finmath.montecarlo.interestrate.LIBORMonteCarloSimulationFromLIBORModel;
import net.finmath.montecarlo.interestrate.LIBORModelMonteCarloSimulationModel;
import net.finmath.montecarlo.interestrate.models.HullWhiteModel;
import net.finmath.montecarlo.interestrate.models.covariance.ShortRateVolatilityModel;
import net.finmath.montecarlo.interestrate.models.covariance.ShortRateVolatilityModelAsGiven;
import net.finmath.montecarlo.interestrate.products.BermudanSwaptionFromSwapSchedules;
import net.finmath.montecarlo.interestrate.products.BermudanSwaptionFromSwapSchedules.SwaptionType;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.time.FloatingpointDate;
import net.finmath.time.Period;
import net.finmath.time.Schedule;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;
import net.finmath.time.daycount.DayCountConvention;

/**
 * Tests for {@link Swaption} under {@link FDMHullWhiteModel}.
 *
 * <p>
 * The benchmark is the Monte Carlo Hull-White implementation from finmath,
 * using {@link BermudanSwaptionFromSwapSchedules}. A single exercise date is
 * used for the European cases, while multiple exercise dates are used for the
 * Bermudan case.
 * </p>
 *
 * <p>
 * American exercise is intentionally not benchmarked here, since there is no
 * like-for-like Monte Carlo American swaption product in the standard rates
 * branch. A natural later benchmark for the American PDE product is convergence
 * against finer Bermudan exercise meshes.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class SwaptionTest {

	private static final LocalDate REFERENCE_DATE = LocalDate.of(2021, 1, 1);

	private static final double THETA = 0.5;
	private static final double INITIAL_STATE = 0.0;

	private static final double FLAT_ZERO_RATE = 0.03;
	private static final double MEAN_REVERSION = 0.10;
	private static final double VOLATILITY = 0.01;

	private static final int NUMBER_OF_SPACE_STEPS = 241;
	private static final int NUMBER_OF_MONTE_CARLO_PATHS = 50000;

	private static final double EUROPEAN_TOLERANCE = 1.5E-2;
	private static final double BERMUDAN_TOLERANCE = 2.5E-2;

	@Test
	public void testEuropeanPayerSwaptionAgainstMonteCarloHullWhite() throws CalculationException {
		final LocalDate exerciseDate = REFERENCE_DATE.plusYears(1);
		final LocalDate swapEndDate = REFERENCE_DATE.plusYears(6);

		final Schedule fixSchedule = createAnnualSchedule(REFERENCE_DATE, exerciseDate, swapEndDate);
		final Schedule floatSchedule = createAnnualSchedule(REFERENCE_DATE, exerciseDate, swapEndDate);

		final ModelSetup modelSetup = createModelSetup(6.0);
		final double exerciseTime = toTime(exerciseDate);

		final double swapRate = net.finmath.marketdata.products.Swap.getForwardSwapRate(
				fixSchedule,
				floatSchedule,
				modelSetup.forwardCurve,
				modelSetup.analyticModel
		);

		final Swaption pdeSwaption = new Swaption(
				new EuropeanExercise(exerciseTime),
				new double[] { exerciseTime },
				SwaptionType.PAYER,
				null,
				new double[] { swapRate },
				new double[] { 1.0 },
				new Schedule[] { fixSchedule },
				new Schedule[] { floatSchedule }
		);

		final double[] pdeValues = pdeSwaption.getValue(0.0, modelSetup.pdeModel);
		final double pdeValue = extractValueAtInitialState(pdeValues, modelSetup.pdeModel);

		final BermudanSwaptionFromSwapSchedules monteCarloSwaption =
				new BermudanSwaptionFromSwapSchedules(
						REFERENCE_DATE.atStartOfDay(),
						SwaptionType.PAYER,
						new LocalDate[] { exerciseDate },
						swapEndDate,
						new double[] { swapRate },
						new double[] { 1.0 },
						new Schedule[] { fixSchedule },
						new Schedule[] { floatSchedule }
				);

		final double monteCarloValue = monteCarloSwaption.getValue(0.0, modelSetup.monteCarloModel).getAverage();

		assertEquals(monteCarloValue, pdeValue, EUROPEAN_TOLERANCE);
	}

	@Test
	public void testEuropeanReceiverSwaptionAgainstMonteCarloHullWhite() throws CalculationException {
		final LocalDate exerciseDate = REFERENCE_DATE.plusYears(1);
		final LocalDate swapEndDate = REFERENCE_DATE.plusYears(6);

		final Schedule fixSchedule = createAnnualSchedule(REFERENCE_DATE, exerciseDate, swapEndDate);
		final Schedule floatSchedule = createAnnualSchedule(REFERENCE_DATE, exerciseDate, swapEndDate);

		final ModelSetup modelSetup = createModelSetup(6.0);
		final double exerciseTime = toTime(exerciseDate);

		final double swapRate = net.finmath.marketdata.products.Swap.getForwardSwapRate(
				fixSchedule,
				floatSchedule,
				modelSetup.forwardCurve,
				modelSetup.analyticModel
		);

		final Swaption pdeSwaption = new Swaption(
				new EuropeanExercise(exerciseTime),
				new double[] { exerciseTime },
				SwaptionType.RECEIVER,
				null,
				new double[] { swapRate },
				new double[] { 1.0 },
				new Schedule[] { fixSchedule },
				new Schedule[] { floatSchedule }
		);

		final double[] pdeValues = pdeSwaption.getValue(0.0, modelSetup.pdeModel);
		final double pdeValue = extractValueAtInitialState(pdeValues, modelSetup.pdeModel);

		final BermudanSwaptionFromSwapSchedules monteCarloSwaption =
				new BermudanSwaptionFromSwapSchedules(
						REFERENCE_DATE.atStartOfDay(),
						SwaptionType.RECEIVER,
						new LocalDate[] { exerciseDate },
						swapEndDate,
						new double[] { swapRate },
						new double[] { 1.0 },
						new Schedule[] { fixSchedule },
						new Schedule[] { floatSchedule }
				);

		final double monteCarloValue = monteCarloSwaption.getValue(0.0, modelSetup.monteCarloModel).getAverage();

		assertEquals(monteCarloValue, pdeValue, EUROPEAN_TOLERANCE);
	}

	@Test
	public void testBermudanPayerSwaptionAgainstMonteCarloHullWhite() throws CalculationException {
		final LocalDate[] exerciseDates = new LocalDate[] {
				REFERENCE_DATE.plusYears(1),
				REFERENCE_DATE.plusYears(2),
				REFERENCE_DATE.plusYears(3)
		};
		final LocalDate swapEndDate = REFERENCE_DATE.plusYears(6);

		final Schedule[] fixSchedules = new Schedule[exerciseDates.length];
		final Schedule[] floatSchedules = new Schedule[exerciseDates.length];
		final double[] swapRates = new double[exerciseDates.length];
		final double[] notionals = new double[exerciseDates.length];
		Arrays.fill(notionals, 1.0);

		final ModelSetup modelSetup = createModelSetup(6.0);

		for(int i = 0; i < exerciseDates.length; i++) {
			fixSchedules[i] = createAnnualSchedule(REFERENCE_DATE, exerciseDates[i], swapEndDate);
			floatSchedules[i] = createAnnualSchedule(REFERENCE_DATE, exerciseDates[i], swapEndDate);

			swapRates[i] = net.finmath.marketdata.products.Swap.getForwardSwapRate(
					fixSchedules[i],
					floatSchedules[i],
					modelSetup.forwardCurve,
					modelSetup.analyticModel
			);
		}

		final double[] exerciseTimes = new double[exerciseDates.length];
		for(int i = 0; i < exerciseDates.length; i++) {
			exerciseTimes[i] = toTime(exerciseDates[i]);
		}

		final Exercise bermudanExercise = new BermudanExerciseSpecification(exerciseTimes);

		final Swaption pdeSwaption = new Swaption(
				bermudanExercise,
				exerciseTimes,
				SwaptionType.PAYER,
				null,
				swapRates,
				notionals,
				fixSchedules,
				floatSchedules
		);

		final double[] pdeValues = pdeSwaption.getValue(0.0, modelSetup.pdeModel);
		final double pdeValue = extractValueAtInitialState(pdeValues, modelSetup.pdeModel);

		final BermudanSwaptionFromSwapSchedules monteCarloSwaption =
				new BermudanSwaptionFromSwapSchedules(
						REFERENCE_DATE.atStartOfDay(),
						SwaptionType.PAYER,
						exerciseDates,
						swapEndDate,
						swapRates,
						notionals,
						fixSchedules,
						floatSchedules
				);

		final double monteCarloValue = monteCarloSwaption.getValue(0.0, modelSetup.monteCarloModel).getAverage();

		assertEquals(monteCarloValue, pdeValue, BERMUDAN_TOLERANCE);
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

		return new ModelSetup(analyticModel, discountCurve, forwardCurve, pdeModel, monteCarloModel);
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

	private static double toTime(final LocalDate date) {
		return FloatingpointDate.getFloatingPointDateFromDate(REFERENCE_DATE, date);
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
	 * Simple Bermudan exercise specification for tests.
	 *
	 * @author Alessandro Gnoatto
	 */
	private static final class BermudanExerciseSpecification extends AbstractExercise {

		private BermudanExerciseSpecification(final double[] exerciseTimes) {
			super(exerciseTimes[exerciseTimes.length - 1], exerciseTimes);
		}

		@Override
		public boolean isContinuousExercise() {
			return false;
		}

		@Override
		public boolean isExerciseAllowed(final double time) {
			return isScheduledExerciseTime(time);
		}

		@Override
		public boolean isEuropean() {
			return false;
		}

		@Override
		public boolean isAmerican() {
			return false;
		}

		@Override
		public boolean isBermudan() {
			return true;
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