package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.MonitoringType;
import net.finmath.modelling.products.TouchSettlementTiming;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for the discrete-monitoring branch of {@link TouchOption}.
 *
 * <p>
 * Current scope:
 * one-dimensional European cash touch / no-touch.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class TouchOptionDiscreteMonitoringTest {

	private static final double MATURITY = 1.0;
	private static final double PAYOFF = 10.0;

	private static final double S0 = 100.0;
	private static final double R = 0.05;
	private static final double Q = 0.00;
	private static final double SIGMA = 0.25;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 200;
	private static final int NUMBER_OF_SPACE_STEPS = 320;
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	private static final double[] COARSE_MONITORING_TIMES = new double[] { 0.50, 1.00 };
	private static final double[] FINE_MONITORING_TIMES = new double[] { 0.25, 0.50, 0.75, 1.00 };

	private static final double PRICE_TOLERANCE = 1E-8;
	private static final double STRUCTURAL_TOLERANCE = 1E-6;

	@Test
	public void testDiscreteNoTouchEventConditionKnocksOutOnlyBreachedNodes() {

		final FDMBlackScholesModel model = createBlackScholesModel(80.0, BarrierType.DOWN_OUT);

		final TouchOption option = new TouchOption(
				MATURITY,
				80.0,
				BarrierType.DOWN_OUT,
				PAYOFF,
				TouchSettlementTiming.AT_EXPIRY,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final double[] valuesAfterEvent = new double[] { 1.0, 2.0, 3.0, 4.0 };
		final double[] valuesBeforeEvent = option.applyEventCondition(
				0.50,
				valuesAfterEvent,
				createMinimalModelWithGrid(model, new double[] { 79.0, 80.0, 81.0, 82.0 })
		);

		assertArrayEquals(new double[] { 0.0, 0.0, 3.0, 4.0 }, valuesBeforeEvent, PRICE_TOLERANCE);
	}

	@Test
	public void testDiscreteOneTouchAtExpiryEventConditionActivatesDiscountedCash() {

		final double monitoringTime = 0.50;

		final FDMBlackScholesModel model = createBlackScholesModel(80.0, BarrierType.DOWN_IN);

		final TouchOption option = new TouchOption(
				MATURITY,
				80.0,
				BarrierType.DOWN_IN,
				PAYOFF,
				TouchSettlementTiming.AT_EXPIRY,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final double[] valuesAfterEvent = new double[] { 1.0, 2.0, 3.0, 4.0 };
		final double[] valuesBeforeEvent = option.applyEventCondition(
				monitoringTime,
				valuesAfterEvent,
				createMinimalModelWithGrid(model, new double[] { 79.0, 80.0, 81.0, 82.0 })
		);

		final double expectedActivatedValue =
				PAYOFF
				* model.getRiskFreeCurve().getDiscountFactor(MATURITY)
				/ model.getRiskFreeCurve().getDiscountFactor(monitoringTime);

		assertEquals(expectedActivatedValue, valuesBeforeEvent[0], PRICE_TOLERANCE);
		assertEquals(expectedActivatedValue, valuesBeforeEvent[1], PRICE_TOLERANCE);
		assertEquals(3.0, valuesBeforeEvent[2], PRICE_TOLERANCE);
		assertEquals(4.0, valuesBeforeEvent[3], PRICE_TOLERANCE);
	}

	@Test
	public void testDiscreteOneTouchAtHitEventConditionActivatesImmediateCash() {

		final FDMBlackScholesModel model = createBlackScholesModel(120.0, BarrierType.UP_IN);

		final TouchOption option = new TouchOption(
				MATURITY,
				120.0,
				BarrierType.UP_IN,
				PAYOFF,
				TouchSettlementTiming.AT_HIT,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final double[] valuesAfterEvent = new double[] { 1.0, 2.0, 3.0, 4.0 };
		final double[] valuesBeforeEvent = option.applyEventCondition(
				0.75,
				valuesAfterEvent,
				createMinimalModelWithGrid(model, new double[] { 118.0, 119.0, 120.0, 121.0 })
		);

		assertEquals(1.0, valuesBeforeEvent[0], PRICE_TOLERANCE);
		assertEquals(2.0, valuesBeforeEvent[1], PRICE_TOLERANCE);
		assertEquals(PAYOFF, valuesBeforeEvent[2], PRICE_TOLERANCE);
		assertEquals(PAYOFF, valuesBeforeEvent[3], PRICE_TOLERANCE);
	}

	@Test
	public void testDiscreteDownOutNoTouchMonotonicityAgainstContinuousMonitoring() {

		final double barrier = 80.0;
		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.DOWN_OUT);

		final TouchOption coarseDiscrete = new TouchOption(
				MATURITY,
				barrier,
				BarrierType.DOWN_OUT,
				PAYOFF,
				TouchSettlementTiming.AT_EXPIRY,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final TouchOption fineDiscrete = new TouchOption(
				MATURITY,
				barrier,
				BarrierType.DOWN_OUT,
				PAYOFF,
				TouchSettlementTiming.AT_EXPIRY,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		
		final TouchOption continuous = new TouchOption(
		        null,
		        MATURITY,
		        barrier,
		        BarrierType.DOWN_OUT,
		        PAYOFF,
		        TouchSettlementTiming.AT_EXPIRY,
		        new EuropeanExercise(MATURITY),
		        MonitoringType.CONTINUOUS,
		        null
		);

		final double coarseValue = extractValueAtSpot(coarseDiscrete.getValue(0.0, model), model, S0);
		final double fineValue = extractValueAtSpot(fineDiscrete.getValue(0.0, model), model, S0);
		final double continuousValue = extractValueAtSpot(continuous.getValue(0.0, model), model, S0);

		assertTrue(coarseValue + STRUCTURAL_TOLERANCE >= fineValue);
		assertTrue(fineValue + STRUCTURAL_TOLERANCE >= continuousValue);
	}

	@Test
	public void testDiscreteDownInOneTouchMonotonicityAgainstContinuousMonitoring() {

		final double barrier = 80.0;
		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.DOWN_IN);

		final TouchOption coarseDiscrete = new TouchOption(
				MATURITY,
				barrier,
				BarrierType.DOWN_IN,
				PAYOFF,
				TouchSettlementTiming.AT_EXPIRY,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final TouchOption fineDiscrete = new TouchOption(
				MATURITY,
				barrier,
				BarrierType.DOWN_IN,
				PAYOFF,
				TouchSettlementTiming.AT_EXPIRY,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final TouchOption continuous = new TouchOption(
		        null,
		        MATURITY,
		        barrier,
		        BarrierType.DOWN_IN,
		        PAYOFF,
		        TouchSettlementTiming.AT_EXPIRY,
		        new EuropeanExercise(MATURITY),
		        MonitoringType.CONTINUOUS,
		        null
		);

		final double coarseValue = extractValueAtSpot(coarseDiscrete.getValue(0.0, model), model, S0);
		final double fineValue = extractValueAtSpot(fineDiscrete.getValue(0.0, model), model, S0);
		final double continuousValue = extractValueAtSpot(continuous.getValue(0.0, model), model, S0);

		assertTrue(continuousValue + STRUCTURAL_TOLERANCE >= fineValue);
		assertTrue(fineValue + STRUCTURAL_TOLERANCE >= coarseValue);
	}

	@Test
	public void testDiscreteUpOutNoTouchMonotonicityAgainstContinuousMonitoring() {

		final double barrier = 120.0;
		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.UP_OUT);

		final TouchOption coarseDiscrete = new TouchOption(
				MATURITY,
				barrier,
				BarrierType.UP_OUT,
				PAYOFF,
				TouchSettlementTiming.AT_EXPIRY,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final TouchOption fineDiscrete = new TouchOption(
				MATURITY,
				barrier,
				BarrierType.UP_OUT,
				PAYOFF,
				TouchSettlementTiming.AT_EXPIRY,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final TouchOption continuous = new TouchOption(
		        null,
		        MATURITY,
		        barrier,
		        BarrierType.UP_OUT,
		        PAYOFF,
		        TouchSettlementTiming.AT_EXPIRY,
		        new EuropeanExercise(MATURITY),
		        MonitoringType.CONTINUOUS,
		        null
		);

		final double coarseValue = extractValueAtSpot(coarseDiscrete.getValue(0.0, model), model, S0);
		final double fineValue = extractValueAtSpot(fineDiscrete.getValue(0.0, model), model, S0);
		final double continuousValue = extractValueAtSpot(continuous.getValue(0.0, model), model, S0);

		assertTrue(coarseValue + STRUCTURAL_TOLERANCE >= fineValue);
		assertTrue(fineValue + STRUCTURAL_TOLERANCE >= continuousValue);
	}

	@Test
	public void testDiscreteUpInOneTouchMonotonicityAgainstContinuousMonitoring() {

		final double barrier = 120.0;
		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.UP_IN);

		final TouchOption coarseDiscrete = new TouchOption(
				MATURITY,
				barrier,
				BarrierType.UP_IN,
				PAYOFF,
				TouchSettlementTiming.AT_EXPIRY,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final TouchOption fineDiscrete = new TouchOption(
				MATURITY,
				barrier,
				BarrierType.UP_IN,
				PAYOFF,
				TouchSettlementTiming.AT_EXPIRY,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final TouchOption continuous = new TouchOption(
		        null,
		        MATURITY,
		        barrier,
		        BarrierType.UP_IN,
		        PAYOFF,
		        TouchSettlementTiming.AT_EXPIRY,
		        new EuropeanExercise(MATURITY),
		        MonitoringType.CONTINUOUS,
		        null
		);

		final double coarseValue = extractValueAtSpot(coarseDiscrete.getValue(0.0, model), model, S0);
		final double fineValue = extractValueAtSpot(fineDiscrete.getValue(0.0, model), model, S0);
		final double continuousValue = extractValueAtSpot(continuous.getValue(0.0, model), model, S0);

		assertTrue(continuousValue + STRUCTURAL_TOLERANCE >= fineValue);
		assertTrue(fineValue + STRUCTURAL_TOLERANCE >= coarseValue);
	}

	private FDMBlackScholesModel createBlackScholesModel(
			final double barrier,
			final BarrierType barrierType) {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final Grid sGrid = createTouchGrid(barrier, barrierType);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { S0 }
		);

		return new FDMBlackScholesModel(
				S0,
				riskFreeCurve,
				dividendCurve,
				SIGMA,
				spaceTime
		);
	}

	private FDMBlackScholesModel createMinimalModelWithGrid(
	        final FDMBlackScholesModel template,
	        final double[] gridNodes) {

	    final double deltaT = MATURITY / NUMBER_OF_TIME_STEPS;
	    final TimeDiscretization timeDiscretization =
	            new TimeDiscretizationFromArray(0.0, NUMBER_OF_TIME_STEPS, deltaT);

	    final Grid spaceGrid = new UniformGrid(
	            gridNodes.length - 1,
	            gridNodes[0],
	            gridNodes[gridNodes.length - 1]
	    );

	    final SpaceTimeDiscretization discretization = new SpaceTimeDiscretization(
	            spaceGrid,
	            timeDiscretization,
	            THETA,
	            new double[] { S0 }
	    );

	    return new FDMBlackScholesModel(
	            template.getInitialValue()[0],
	            template.getRiskFreeCurve(),
	            template.getDividendYieldCurve(),
	            SIGMA,
	            discretization
	    );
	}

	private Grid createTouchGrid(final double barrier, final BarrierType barrierType) {

		if(barrierType != BarrierType.DOWN_IN
				&& barrierType != BarrierType.DOWN_OUT
				&& barrierType != BarrierType.UP_IN
				&& barrierType != BarrierType.UP_OUT) {
			throw new IllegalArgumentException("Unsupported touch barrier type.");
		}

		final int effectiveStepsBetweenBarrierAndSpot =
				getEffectiveStepsBetweenBarrierAndSpot(barrier);

		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {
			final double deltaS = (S0 - barrier) / effectiveStepsBetweenBarrierAndSpot;
			final double sMin = barrier;
			final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else {
			final double deltaS = (barrier - S0) / effectiveStepsBetweenBarrierAndSpot;
			final double sMax = barrier;
			final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
	}

	private int getEffectiveStepsBetweenBarrierAndSpot(final double barrier) {
		final int naturalSteps = Math.max(1, (int) Math.round(Math.abs(S0 - barrier)));
		final int cappedByGrid = NUMBER_OF_SPACE_STEPS - 2;
		return Math.max(
				1,
				Math.min(
						Math.min(STEPS_BETWEEN_BARRIER_AND_SPOT, cappedByGrid),
						naturalSteps
				)
		);
	}

	private double extractValueAtSpot(
			final double[] valuesOnGrid,
			final FDMBlackScholesModel model,
			final double spot) {

		final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

		if(isGridNode(sNodes, spot)) {
			return valuesOnGrid[getGridIndex(sNodes, spot)];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, valuesOnGrid);
		return interpolation.value(spot);
	}

	private static boolean isGridNode(final double[] grid, final double value) {
		return getGridIndex(grid, value) >= 0;
	}

	private static int getGridIndex(final double[] grid, final double value) {
		final double tolerance = 1E-12;
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - value) < tolerance) {
				return i;
			}
		}
		return -1;
	}

	private static DiscountCurve createFlatDiscountCurve(final String name, final double rate) {
		final double[] times = new double[] { 0.0, 1.0 };
		final double[] zeroRates = new double[] { rate, rate };

		return DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
				name,
				LocalDate.of(2010, 8, 1),
				times,
				zeroRates,
				InterpolationMethod.LINEAR,
				ExtrapolationMethod.CONSTANT,
				InterpolationEntity.VALUE
		);
	}
}