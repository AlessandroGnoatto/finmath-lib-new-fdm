package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertTrue;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMCevModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

public class BarrierOptionCevAmericanKnockInTest {

	private static final double SPOT = 100.0;
	private static final double STRIKE = 100.0;
	private static final double MATURITY = 1.0;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.02;
	private static final double VOLATILITY = 0.20;
	private static final double EXPONENT = 0.8;
	private static final double REBATE = 0.0;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 200;
	private static final int NUMBER_OF_SPACE_STEPS = 400;
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	private static final double ORDERING_TOLERANCE = 1E-2;
	private static final double DOMINANCE_TOLERANCE = 1E-2;

	@Test
	public void testAmericanGreaterOrEqualBermudanGreaterOrEqualEuropeanDownAndInPut() {
		runOrderingTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testAmericanGreaterOrEqualBermudanGreaterOrEqualEuropeanUpAndInCall() {
		runOrderingTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
	}

	@Test
	public void testAmericanGreaterOrEqualBermudanGreaterOrEqualEuropeanDownAndInCall() {
		runOrderingTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testAmericanGreaterOrEqualBermudanGreaterOrEqualEuropeanUpAndInPut() {
		runOrderingTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
	}

	@Test
	public void testAmericanDownInPutLessOrEqualAmericanVanillaPut() {
		runAmericanKnockInLessOrEqualAmericanVanillaTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testAmericanUpInCallLessOrEqualAmericanVanillaCall() {
		runAmericanKnockInLessOrEqualAmericanVanillaTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
	}

	@Test
	public void testAmericanDownInCallLessOrEqualAmericanVanillaCall() {
		runAmericanKnockInLessOrEqualAmericanVanillaTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testAmericanUpInPutLessOrEqualAmericanVanillaPut() {
		runAmericanKnockInLessOrEqualAmericanVanillaTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
	}

	private void runOrderingTest(
			final CallOrPut callOrPut,
			final BarrierType barrierType,
			final double barrier) {

		final TestSetup setup = createSetup(barrier, barrierType);

		final BarrierOption european = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				barrierType
		);

		final BarrierOption bermudan = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				barrierType,
				new BermudanExercise(new double[] { 0.5, MATURITY })
		);

		final BarrierOption american = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				barrierType,
				new AmericanExercise(0.0, MATURITY)
		);

		final double europeanPrice = interpolateAtSpot(european.getValue(0.0, setup.model), setup.sNodes, SPOT);
		final double bermudanPrice = interpolateAtSpot(bermudan.getValue(0.0, setup.model), setup.sNodes, SPOT);
		final double americanPrice = interpolateAtSpot(american.getValue(0.0, setup.model), setup.sNodes, SPOT);

		assertTrue(bermudanPrice + ORDERING_TOLERANCE >= europeanPrice);
		assertTrue(americanPrice + ORDERING_TOLERANCE >= bermudanPrice);
	}

	private void runAmericanKnockInLessOrEqualAmericanVanillaTest(
			final CallOrPut callOrPut,
			final BarrierType knockInType,
			final double barrier) {

		final TestSetup knockInSetup = createSetup(barrier, knockInType);
		final TestSetup vanillaSetup = createVanillaSetup();

		final BarrierOption knockIn = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				knockInType,
				new AmericanExercise(0.0, MATURITY)
		);

		final AmericanOption vanilla = new AmericanOption(
				MATURITY,
				STRIKE,
				callOrPut
		);

		final double knockInPrice =
				interpolateAtSpot(knockIn.getValue(0.0, knockInSetup.model), knockInSetup.sNodes, SPOT);

		final double vanillaPrice =
				interpolateAtSpot(vanilla.getValue(0.0, vanillaSetup.model), vanillaSetup.sNodes, SPOT);

		assertTrue(knockInPrice <= vanillaPrice + DOMINANCE_TOLERANCE);
	}

	private TestSetup createSetup(final double barrier, final BarrierType barrierType) {

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final Grid sGrid = createGrid(barrier, barrierType);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { SPOT }
		);

		final FDMCevModel model = new FDMCevModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				EXPONENT,
				spaceTime
		);

		return new TestSetup(model, sGrid.getGrid());
	}

	private TestSetup createVanillaSetup() {

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final double forward =
				SPOT * Math.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * MATURITY);
		final double varianceProxy =
				SPOT * SPOT
				* Math.exp(2.0 * (RISK_FREE_RATE - DIVIDEND_YIELD) * MATURITY)
				* (Math.exp(VOLATILITY * VOLATILITY * MATURITY) - 1.0);

		final double sMin = Math.max(forward - 6.0 * Math.sqrt(varianceProxy), 0.0);
		final double sMax = forward + 6.0 * Math.sqrt(varianceProxy);

		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { SPOT }
		);

		final FDMCevModel model = new FDMCevModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				EXPONENT,
				spaceTime
		);

		return new TestSetup(model, sGrid.getGrid());
	}

	private Grid createGrid(final double barrier, final BarrierType barrierType) {

		if(barrierType == BarrierType.DOWN_OUT) {
			final double deltaS = (SPOT - barrier) / STEPS_BETWEEN_BARRIER_AND_SPOT;
			final double sMin = barrier;
			final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else if(barrierType == BarrierType.UP_OUT) {
			final double deltaS = (barrier - SPOT) / STEPS_BETWEEN_BARRIER_AND_SPOT;
			final double sMax = barrier;
			final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else if(barrierType == BarrierType.DOWN_IN) {
			final double deltaS = (SPOT - barrier) / STEPS_BETWEEN_BARRIER_AND_SPOT;
			final int extraStepsBelowBarrier = 160;
			final double sMin = barrier - extraStepsBelowBarrier * deltaS;
			final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else if(barrierType == BarrierType.UP_IN) {
			final double deltaS = (barrier - SPOT) / STEPS_BETWEEN_BARRIER_AND_SPOT;
			final int extraStepsAboveBarrier = 160;
			final double sMax = barrier + extraStepsAboveBarrier * deltaS;
			final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else {
			throw new IllegalArgumentException("Unsupported barrier type.");
		}
	}

	private double interpolateAtSpot(
			final double[] values,
			final double[] sNodes,
			final double spot) {

		assertTrue(
				"Spot must lie inside the grid domain.",
				spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
		);

		if(isGridNode(sNodes, spot)) {
			return values[getGridIndex(sNodes, spot)];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, values);
		return interpolation.value(spot);
	}

	private boolean isGridNode(final double[] grid, final double x) {
		for(final double node : grid) {
			if(Math.abs(node - x) < 1E-12) {
				return true;
			}
		}
		return false;
	}

	private int getGridIndex(final double[] grid, final double x) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - x) < 1E-12) {
				return i;
			}
		}
		throw new IllegalArgumentException("Point is not a grid node.");
	}

	private static class TestSetup {

		private final FDMCevModel model;
		private final double[] sNodes;

		private TestSetup(final FDMCevModel model, final double[] sNodes) {
			this.model = model;
			this.sNodes = sNodes;
		}
	}
}