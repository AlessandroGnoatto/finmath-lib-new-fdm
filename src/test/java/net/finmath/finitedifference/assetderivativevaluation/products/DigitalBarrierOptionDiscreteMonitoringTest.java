package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

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
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DigitalPayoffType;
import net.finmath.modelling.products.MonitoringType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for 1D European discretely monitored digital barrier options.
 *
 * <p>
 * Current discrete-monitoring milestone covered here:
 * </p>
 * <ul>
 *   <li>1D European cash-or-nothing and asset-or-nothing,</li>
 *   <li>knock-in event replacement by vanilla digital continuation,</li>
 *   <li>knock-out event zeroing on breached nodes,</li>
 *   <li>monitoring-frequency ordering versus continuous monitoring.</li>
 * </ul>
 */
public class DigitalBarrierOptionDiscreteMonitoringTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double SPOT = 100.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;
	private static final double VOLATILITY = 0.25;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 220;
	private static final int NUMBER_OF_SPACE_STEPS = 340;
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	private static final double CASH_PAYOFF = 10.0;

	private static final double[] COARSE_MONITORING_TIMES = new double[] { 0.50, 1.00 };
	private static final double[] FINE_MONITORING_TIMES = new double[] { 0.25, 0.50, 0.75, 1.00 };

	private static final double ORDERING_TOLERANCE = 1E-2;
	private static final double EVENT_TOLERANCE = 1E-10;
	private static final double BARRIER_TOLERANCE = 1E-12;

	@Test
	public void testDiscreteDownInCashPutOrderingAgainstVanilla() {

		final double barrier = 80.0;
		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.DOWN_IN);

		final DigitalBarrierOption coarseDiscrete = new DigitalBarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				CallOrPut.PUT,
				BarrierType.DOWN_IN,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final DigitalBarrierOption fineDiscrete = new DigitalBarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				CallOrPut.PUT,
				BarrierType.DOWN_IN,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final DigitalBarrierOption continuous = new DigitalBarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				CallOrPut.PUT,
				BarrierType.DOWN_IN,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new EuropeanExercise(MATURITY)
		);

		final DigitalOption vanilla = new DigitalOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.PUT,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new EuropeanExercise(MATURITY)
		);

		final double coarseValue = extractValueAtSpot(coarseDiscrete.getValue(0.0, model), model, SPOT);
		final double fineValue = extractValueAtSpot(fineDiscrete.getValue(0.0, model), model, SPOT);
		final double continuousValue = extractValueAtSpot(continuous.getValue(0.0, model), model, SPOT);
		final double vanillaValue = extractValueAtSpot(vanilla.getValue(0.0, model), model, SPOT);

		assertTrue(coarseValue >= -ORDERING_TOLERANCE);
		assertTrue(coarseValue <= fineValue + ORDERING_TOLERANCE);
		assertTrue(fineValue <= vanillaValue + ORDERING_TOLERANCE);
		assertTrue(continuousValue <= vanillaValue + ORDERING_TOLERANCE);
	}

	@Test
	public void testDiscreteUpOutCashCallOrderingAgainstVanilla() {

		final double barrier = 120.0;
		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.UP_OUT);

		final DigitalBarrierOption coarseDiscrete = new DigitalBarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				CallOrPut.CALL,
				BarrierType.UP_OUT,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final DigitalBarrierOption fineDiscrete = new DigitalBarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				CallOrPut.CALL,
				BarrierType.UP_OUT,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final DigitalBarrierOption continuous = new DigitalBarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				CallOrPut.CALL,
				BarrierType.UP_OUT,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new EuropeanExercise(MATURITY)
		);

		final DigitalOption vanilla = new DigitalOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.CALL,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new EuropeanExercise(MATURITY)
		);

		final double coarseValue = extractValueAtSpot(coarseDiscrete.getValue(0.0, model), model, SPOT);
		final double fineValue = extractValueAtSpot(fineDiscrete.getValue(0.0, model), model, SPOT);
		final double continuousValue = extractValueAtSpot(continuous.getValue(0.0, model), model, SPOT);
		final double vanillaValue = extractValueAtSpot(vanilla.getValue(0.0, model), model, SPOT);

		assertTrue(vanillaValue + ORDERING_TOLERANCE >= coarseValue);
		assertTrue(coarseValue + ORDERING_TOLERANCE >= fineValue);
		assertTrue(fineValue >= -ORDERING_TOLERANCE);
		assertTrue(vanillaValue + ORDERING_TOLERANCE >= continuousValue);
	}

	@Test
	public void testDiscreteDownInAssetCallOrderingAgainstVanilla() {

		final double barrier = 80.0;
		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.DOWN_IN);

		final DigitalBarrierOption coarseDiscrete = new DigitalBarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				CallOrPut.CALL,
				BarrierType.DOWN_IN,
				DigitalPayoffType.ASSET_OR_NOTHING,
				0.0,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final DigitalBarrierOption fineDiscrete = new DigitalBarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				CallOrPut.CALL,
				BarrierType.DOWN_IN,
				DigitalPayoffType.ASSET_OR_NOTHING,
				0.0,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final DigitalBarrierOption continuous = new DigitalBarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				CallOrPut.CALL,
				BarrierType.DOWN_IN,
				DigitalPayoffType.ASSET_OR_NOTHING,
				0.0,
				new EuropeanExercise(MATURITY)
		);

		final DigitalOption vanilla = new DigitalOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.CALL,
				DigitalPayoffType.ASSET_OR_NOTHING,
				0.0,
				new EuropeanExercise(MATURITY)
		);

		final double coarseValue = extractValueAtSpot(coarseDiscrete.getValue(0.0, model), model, SPOT);
		final double fineValue = extractValueAtSpot(fineDiscrete.getValue(0.0, model), model, SPOT);
		final double continuousValue = extractValueAtSpot(continuous.getValue(0.0, model), model, SPOT);
		final double vanillaValue = extractValueAtSpot(vanilla.getValue(0.0, model), model, SPOT);

		assertTrue(coarseValue >= -ORDERING_TOLERANCE);
		assertTrue(coarseValue <= fineValue + ORDERING_TOLERANCE);
		assertTrue(fineValue <= vanillaValue + ORDERING_TOLERANCE);
		assertTrue(continuousValue <= vanillaValue + ORDERING_TOLERANCE);
	}

	@Test
	public void testDiscreteDownInCashEventConditionReplacesByVanillaContinuationValue() {

		final double barrier = 80.0;
		final double monitoringTime = 0.50;
		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.DOWN_IN);

		final DigitalBarrierOption discreteKnockIn = new DigitalBarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				CallOrPut.PUT,
				BarrierType.DOWN_IN,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				new double[] { monitoringTime, MATURITY }
		);

		final DigitalOption vanilla = new DigitalOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.PUT,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new EuropeanExercise(MATURITY)
		);

		final double[] knockInValuesAtMonitoring = discreteKnockIn.getValue(monitoringTime, model);
		final double[] vanillaValuesAtMonitoring = vanilla.getValue(monitoringTime, model);
		final double[] xGrid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

		for(int i = 0; i < xGrid.length; i++) {
			if(xGrid[i] <= barrier + BARRIER_TOLERANCE) {
				assertEquals(vanillaValuesAtMonitoring[i], knockInValuesAtMonitoring[i], EVENT_TOLERANCE);
			}
		}
	}

	@Test
	public void testDiscreteUpOutCashEventConditionZeroesBreachedNodes() {

		final double barrier = 120.0;
		final double monitoringTime = 0.75;
		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.UP_OUT);

		final DigitalBarrierOption discreteKnockOut = new DigitalBarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				CallOrPut.CALL,
				BarrierType.UP_OUT,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				new double[] { monitoringTime, MATURITY }
		);

		final double[] valuesAtMonitoring = discreteKnockOut.getValue(monitoringTime, model);
		final double[] xGrid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

		for(int i = 0; i < xGrid.length; i++) {
			if(xGrid[i] >= barrier - BARRIER_TOLERANCE) {
				assertEquals(0.0, valuesAtMonitoring[i], EVENT_TOLERANCE);
			}
		}
	}

	private FDMBlackScholesModel createBlackScholesModel(
			final double barrier,
			final BarrierType barrierType) {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", RISK_FREE_RATE);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", DIVIDEND_YIELD);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final Grid sGrid = createBarrierAlignedGrid(barrier, barrierType);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { SPOT }
		);

		return new FDMBlackScholesModel(
				SPOT,
				riskFreeCurve,
				dividendCurve,
				VOLATILITY,
				spaceTime
		);
	}

	private Grid createBarrierAlignedGrid(final double barrier, final BarrierType barrierType) {

		final int effectiveStepsBetweenBarrierAndSpot =
				getEffectiveStepsBetweenBarrierAndSpot(barrier);

		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {
			final double deltaS = (SPOT - barrier) / effectiveStepsBetweenBarrierAndSpot;
			final double sMin = barrier;
			final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else if(barrierType == BarrierType.UP_IN || barrierType == BarrierType.UP_OUT) {
			final double deltaS = (barrier - SPOT) / effectiveStepsBetweenBarrierAndSpot;
			final double sMax = barrier;
			final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else {
			throw new IllegalArgumentException("Unsupported barrier type.");
		}
	}

	private int getEffectiveStepsBetweenBarrierAndSpot(final double barrier) {
		final int naturalSteps = Math.max(1, (int)Math.round(Math.abs(SPOT - barrier)));
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
		final int index = getGridIndex(sNodes, spot);
		if(index < 0) {
			throw new IllegalArgumentException("Spot is expected to be a grid node in this test.");
		}
		return valuesOnGrid[index];
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