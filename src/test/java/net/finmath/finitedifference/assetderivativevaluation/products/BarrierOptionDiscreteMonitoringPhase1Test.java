package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertFalse;
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
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.MonitoringType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Phase-1 regression tests for discrete monitoring of 1D knock-out barrier options.
 *
 * <p>
 * These tests focus on the newly introduced product behavior:
 * </p>
 * <ul>
 *   <li>for {@link MonitoringType#DISCRETE}, the knock-out constraint is active
 *       only on monitoring dates,</li>
 *   <li>for {@link MonitoringType#CONTINUOUS}, the knock-out constraint remains
 *       active at all times,</li>
 *   <li>European, Bermudan, and American discretely monitored 1D knock-out
 *       options can be valued without producing invalid numbers.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionDiscreteMonitoringPhase1Test {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 2.0;

	private static final double S0 = 100.0;
	private static final double R = 0.05;
	private static final double Q = 0.00;
	private static final double SIGMA = 0.25;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 160;
	private static final int NUMBER_OF_SPACE_STEPS = 320;
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	private static final double[] COARSE_MONITORING_TIMES = new double[] { 0.50, 1.00 };
	private static final double[] FINE_MONITORING_TIMES = new double[] { 0.25, 0.50, 0.75, 1.00 };

	@Test
	public void testDownOutDiscreteConstraintIsActiveOnlyOnMonitoringDates() {
		final BarrierOption option = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				80.0,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_OUT,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		assertTrue(option.isConstraintActive(0.25, 79.0));
		assertTrue(option.isConstraintActive(0.50, 79.0));
		assertFalse(option.isConstraintActive(0.20, 79.0));
		assertFalse(option.isConstraintActive(0.60, 79.0));
		assertFalse(option.isConstraintActive(0.25, 81.0));
	}

	@Test
	public void testUpOutDiscreteConstraintIsActiveOnlyOnMonitoringDates() {
		final BarrierOption option = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				120.0,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_OUT,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		assertTrue(option.isConstraintActive(0.25, 121.0));
		assertTrue(option.isConstraintActive(0.50, 121.0));
		assertFalse(option.isConstraintActive(0.20, 121.0));
		assertFalse(option.isConstraintActive(0.60, 121.0));
		assertFalse(option.isConstraintActive(0.25, 119.0));
	}

	@Test
	public void testContinuousConstraintRemainsActiveAwayFromMonitoringDates() {
		final BarrierOption downOut = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				80.0,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_OUT,
				new EuropeanExercise(MATURITY)
		);

		final BarrierOption upOut = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				120.0,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_OUT,
				new EuropeanExercise(MATURITY)
		);

		assertTrue(downOut.isConstraintActive(0.20, 79.0));
		assertTrue(upOut.isConstraintActive(0.20, 121.0));
	}

	@Test
	public void testDiscreteDownOutPutValuationSmokeEuropean() {
		runDiscreteValuationSmokeTest(
				CallOrPut.PUT,
				BarrierType.DOWN_OUT,
				80.0,
				new EuropeanExercise(MATURITY),
				FINE_MONITORING_TIMES
		);
	}

	@Test
	public void testDiscreteDownOutPutValuationSmokeBermudan() {
		runDiscreteValuationSmokeTest(
				CallOrPut.PUT,
				BarrierType.DOWN_OUT,
				80.0,
				new BermudanExercise(FINE_MONITORING_TIMES),
				FINE_MONITORING_TIMES
		);
	}

	@Test
	public void testDiscreteDownOutPutValuationSmokeAmerican() {
		runDiscreteValuationSmokeTest(
				CallOrPut.PUT,
				BarrierType.DOWN_OUT,
				80.0,
				new AmericanExercise(MATURITY),
				FINE_MONITORING_TIMES
		);
	}

	@Test
	public void testDiscreteUpOutCallValuationSmokeEuropean() {
		runDiscreteValuationSmokeTest(
				CallOrPut.CALL,
				BarrierType.UP_OUT,
				120.0,
				new EuropeanExercise(MATURITY),
				FINE_MONITORING_TIMES
		);
	}

	@Test
	public void testDiscreteUpOutCallValuationSmokeBermudan() {
		runDiscreteValuationSmokeTest(
				CallOrPut.CALL,
				BarrierType.UP_OUT,
				120.0,
				new BermudanExercise(FINE_MONITORING_TIMES),
				FINE_MONITORING_TIMES
		);
	}

	@Test
	public void testDiscreteUpOutCallValuationSmokeAmerican() {
		runDiscreteValuationSmokeTest(
				CallOrPut.CALL,
				BarrierType.UP_OUT,
				120.0,
				new AmericanExercise(MATURITY),
				FINE_MONITORING_TIMES
		);
	}

	@Test
	public void testDiscreteMonitoringSpecificationAffectsOnlyMonitoringDatesNotExerciseType() {
		final BarrierOption european = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				80.0,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_OUT,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final BarrierOption bermudan = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				80.0,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_OUT,
				new BermudanExercise(FINE_MONITORING_TIMES),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final BarrierOption american = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				80.0,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_OUT,
				new AmericanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		assertTrue(european.isConstraintActive(0.50, 79.0));
		assertTrue(bermudan.isConstraintActive(0.50, 79.0));
		assertTrue(american.isConstraintActive(0.50, 79.0));

		assertFalse(european.isConstraintActive(0.25, 79.0));
		assertFalse(bermudan.isConstraintActive(0.25, 79.0));
		assertFalse(american.isConstraintActive(0.25, 79.0));
	}

	private void runDiscreteValuationSmokeTest(
			final CallOrPut callOrPut,
			final BarrierType barrierType,
			final double barrier,
			final Exercise exercise,
			final double[] monitoringTimes) {

		final FDMBlackScholesModel model = createBlackScholesModel(barrier, barrierType);

		final BarrierOption option = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				barrierType,
				exercise,
				MonitoringType.DISCRETE,
				monitoringTimes
		);

		final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double[] values = option.getValue(0.0, model);
		final double valueAtSpot = extractValueAtSpot(values, sNodes, S0);

		assertTrue(Double.isFinite(valueAtSpot));
		assertTrue(valueAtSpot >= -1E-10);
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

		final Grid sGrid = createKnockOutGrid(barrier, barrierType);

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

	private Grid createKnockOutGrid(final double barrier, final BarrierType barrierType) {

		if(barrierType != BarrierType.DOWN_OUT && barrierType != BarrierType.UP_OUT) {
			throw new IllegalArgumentException("This test supports knock-out barriers only.");
		}

		final int effectiveStepsBetweenBarrierAndSpot =
				getEffectiveStepsBetweenBarrierAndSpot(barrier);

		if(barrierType == BarrierType.DOWN_OUT) {
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

	private double extractValueAtSpot(final double[] valuesOnGrid, final double[] sNodes, final double spot) {
		assertTrue(
				"Spot must lie inside the grid domain.",
				spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
		);

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