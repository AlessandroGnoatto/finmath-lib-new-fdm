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
import net.finmath.modelling.products.MonitoringType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for 1D discretely monitored knock-in barrier options.
 *
 * <p>
 * Current scope:
 * European exercise only.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionDiscreteMonitoringKnockInTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

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

	private static final double PRICE_TOLERANCE = 1E-10;
	private static final double STRUCTURAL_TOLERANCE = 1E-6;

	@Test
	public void testDiscreteDownInPutMonotonicityAndVanillaUpperBound() {

		final double barrier = 80.0;
		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.DOWN_IN);

		final BarrierOption coarseDiscrete = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final BarrierOption fineDiscrete = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final EuropeanOption vanilla = new EuropeanOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.PUT
		);

		final double coarseValue = extractValueAtSpot(coarseDiscrete.getValue(0.0, model), model, S0);
		final double fineValue = extractValueAtSpot(fineDiscrete.getValue(0.0, model), model, S0);
		final double vanillaValue = extractValueAtSpot(vanilla.getValue(0.0, model), model, S0);

		assertTrue(coarseValue <= fineValue + STRUCTURAL_TOLERANCE);
		assertTrue(fineValue <= vanillaValue + STRUCTURAL_TOLERANCE);
		assertTrue(coarseValue >= -STRUCTURAL_TOLERANCE);
	}
	

	@Test
	public void testDiscreteUpInCallMonotonicityAgainstContinuousMonitoring() {

		final double barrier = 120.0;
		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.UP_IN);

		final BarrierOption coarseDiscrete = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final BarrierOption fineDiscrete = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final BarrierOption continuous = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.CONTINUOUS,
				null
		);

		final double coarseValue = extractValueAtSpot(coarseDiscrete.getValue(0.0, model), model, S0);
		final double fineValue = extractValueAtSpot(fineDiscrete.getValue(0.0, model), model, S0);
		final double continuousValue = extractValueAtSpot(continuous.getValue(0.0, model), model, S0);

		assertTrue(coarseValue <= fineValue + STRUCTURAL_TOLERANCE);
		assertTrue(fineValue <= continuousValue + STRUCTURAL_TOLERANCE);
	}

	@Test
	public void testDiscreteDownInEventConditionReplacesByVanillaContinuationValue() {

		final double barrier = 80.0;
		final double monitoringTime = 0.50;

		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.DOWN_IN);

		final BarrierOption discreteKnockIn = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				new double[] { monitoringTime, MATURITY }
		);

		final EuropeanOption vanilla = new EuropeanOption(null, MATURITY, STRIKE, CallOrPut.PUT);

		final double[][] knockInValues = discreteKnockIn.getValues(model);
		final double[][] vanillaValues = vanilla.getValues(model);

		final double[] xGrid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double tau = MATURITY - monitoringTime;
		final int timeIndex = model.getSpaceTimeDiscretization().getTimeDiscretization().getTimeIndex(tau);

		for(int i = 0; i < xGrid.length; i++) {
			if(xGrid[i] <= barrier + STRUCTURAL_TOLERANCE) {
				assertEquals(vanillaValues[i][timeIndex], knockInValues[i][timeIndex], PRICE_TOLERANCE);
			}
		}
	}

	@Test
	public void testDiscreteUpInEventConditionReplacesByVanillaContinuationValue() {

		final double barrier = 120.0;
		final double monitoringTime = 0.75;

		final FDMBlackScholesModel model = createBlackScholesModel(barrier, BarrierType.UP_IN);

		final BarrierOption discreteKnockIn = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				new double[] { monitoringTime, MATURITY }
		);

		final EuropeanOption vanilla = new EuropeanOption(null, MATURITY, STRIKE, CallOrPut.CALL);

		final double[][] knockInValues = discreteKnockIn.getValues(model);
		final double[][] vanillaValues = vanilla.getValues(model);

		final double[] xGrid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double tau = MATURITY - monitoringTime;
		final int timeIndex = model.getSpaceTimeDiscretization().getTimeDiscretization().getTimeIndex(tau);

		for(int i = 0; i < xGrid.length; i++) {
			if(xGrid[i] >= barrier - STRUCTURAL_TOLERANCE) {
				assertEquals(vanillaValues[i][timeIndex], knockInValues[i][timeIndex], PRICE_TOLERANCE);
			}
		}
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

		final Grid sGrid = createBarrierAlignedGrid(barrier, barrierType);

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

	private Grid createBarrierAlignedGrid(final double barrier, final BarrierType barrierType) {

		final int effectiveStepsBetweenBarrierAndSpot =
				getEffectiveStepsBetweenBarrierAndSpot(barrier);

		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {
			final double deltaS = (S0 - barrier) / effectiveStepsBetweenBarrierAndSpot;
			final double sMin = barrier;
			final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else if(barrierType == BarrierType.UP_IN || barrierType == BarrierType.UP_OUT) {
			final double deltaS = (barrier - S0) / effectiveStepsBetweenBarrierAndSpot;
			final double sMax = barrier;
			final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else {
			throw new IllegalArgumentException("Unsupported barrier type.");
		}
	}

	private int getEffectiveStepsBetweenBarrierAndSpot(final double barrier) {
		final int naturalSteps = Math.max(1, (int)Math.round(Math.abs(S0 - barrier)));
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