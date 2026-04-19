package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.SwingQuantityMode;
import net.finmath.modelling.products.SwingStrikeFixingConvention;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural tests for the 1D FloatingStrikeSwingOption implementation.
 *
 * <p>
 * Focus:
 * </p>
 * <ul>
 *   <li>all-zero quantities give zero value,</li>
 *   <li>strikeScale = 0 reduces to the fixed-strike SwingOption,</li>
 *   <li>relaxing the global maximum cannot reduce value,</li>
 *   <li>a discretized quantity grid dominates bang-bang control,</li>
 *   <li>increasing the strike shift cannot increase a call value.</li>
 * </ul>
 *
 * <p>
 * These are structural tests, not tight regression tests.
 * </p>
 */
public class FloatingStrikeSwingOption1DStructuralTest {

	private static final double SPOT = 100.0;
	private static final double STRIKE = 100.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;
	private static final double VOLATILITY = 0.25;
	private static final double THETA = 0.5;

	private static final double SPACE_MIN = 20.0;
	private static final double SPACE_MAX = 200.0;

	private static final int NUMBER_OF_TIME_STEPS = 160;
	private static final int NUMBER_OF_SPACE_STEPS = 180;

	private static final double[] DECISION_TIMES = new double[] { 0.5, 1.0 };
	private static final double[] FIXING_TIMES = new double[] { 0.5, 1.0 };
	private static final double[] FIXING_WEIGHTS = new double[] { 1.0, 1.0 };

	private static final double ZERO_TOL = 1E-10;
	private static final double REDUCTION_TOL = 8E-2;
	private static final double ORDERING_TOL = 1E-8;

	@Test
	public void testZeroQuantitiesGiveZeroValue() {
		final BlackScholesSetup setup = createModel(SPOT, 1.0);

		final FloatingStrikeSwingOption swing = new FloatingStrikeSwingOption(
				DECISION_TIMES,
				FIXING_TIMES,
				FIXING_WEIGHTS,
				STRIKE,
				1.0,
				buildAccumulatorGrid(),
				0.0,
				0.0,
				0.0,
				0.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25,
				SwingStrikeFixingConvention.FIX_THEN_EXERCISE
		);

		final double value = valueAtSpot(
				swing.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		assertEquals(0.0, value, ZERO_TOL);
	}

	@Test
	public void testStrikeScaleZeroReducesToFixedStrikeSwing() {
		final BlackScholesSetup setup = createModel(SPOT, 1.0);

		final FloatingStrikeSwingOption floatingDegenerate = new FloatingStrikeSwingOption(
				DECISION_TIMES,
				FIXING_TIMES,
				FIXING_WEIGHTS,
				STRIKE,
				0.0,
				buildAccumulatorGrid(),
				0.0,
				1.0,
				0.0,
				2.0,
				CallOrPut.CALL,
				SwingQuantityMode.DISCRETE_QUANTITY_GRID,
				0.25,
				SwingStrikeFixingConvention.FIX_THEN_EXERCISE
		);

		final SwingOption fixedStrike = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				2.0,
				CallOrPut.CALL,
				SwingQuantityMode.DISCRETE_QUANTITY_GRID,
				0.25
		);

		final double floatingValue = valueAtSpot(
				floatingDegenerate.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		final double fixedValue = valueAtSpot(
				fixedStrike.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		assertEquals(fixedValue, floatingValue, REDUCTION_TOL);
	}

	@Test
	public void testRelaxingGlobalMaximumCannotReduceValue() {
		final BlackScholesSetup setup = createModel(SPOT, 1.0);

		final FloatingStrikeSwingOption tighterGlobalMax = new FloatingStrikeSwingOption(
				DECISION_TIMES,
				FIXING_TIMES,
				FIXING_WEIGHTS,
				0.0,
				1.0,
				buildAccumulatorGrid(),
				0.0,
				1.0,
				0.0,
				1.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25,
				SwingStrikeFixingConvention.FIX_THEN_EXERCISE
		);

		final FloatingStrikeSwingOption looserGlobalMax = new FloatingStrikeSwingOption(
				DECISION_TIMES,
				FIXING_TIMES,
				FIXING_WEIGHTS,
				0.0,
				1.0,
				buildAccumulatorGrid(),
				0.0,
				1.0,
				0.0,
				2.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25,
				SwingStrikeFixingConvention.FIX_THEN_EXERCISE
		);

		final double tighterValue = valueAtSpot(
				tighterGlobalMax.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		final double looserValue = valueAtSpot(
				looserGlobalMax.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		assertTrue(looserValue + ORDERING_TOL >= tighterValue);
	}

	@Test
	public void testDiscreteQuantityGridDominatesBangBang() {
		final BlackScholesSetup setup = createModel(SPOT, 1.0);

		final FloatingStrikeSwingOption bangBang = new FloatingStrikeSwingOption(
				DECISION_TIMES,
				FIXING_TIMES,
				FIXING_WEIGHTS,
				0.0,
				1.0,
				buildAccumulatorGrid(),
				0.0,
				1.0,
				0.0,
				1.5,
				CallOrPut.PUT,
				SwingQuantityMode.BANG_BANG,
				0.25,
				SwingStrikeFixingConvention.FIX_THEN_EXERCISE
		);

		final FloatingStrikeSwingOption discreteGrid = new FloatingStrikeSwingOption(
				DECISION_TIMES,
				FIXING_TIMES,
				FIXING_WEIGHTS,
				0.0,
				1.0,
				buildAccumulatorGrid(),
				0.0,
				1.0,
				0.0,
				1.5,
				CallOrPut.PUT,
				SwingQuantityMode.DISCRETE_QUANTITY_GRID,
				0.25,
				SwingStrikeFixingConvention.FIX_THEN_EXERCISE
		);

		final double bangBangValue = valueAtSpot(
				bangBang.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		final double discreteGridValue = valueAtSpot(
				discreteGrid.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		assertTrue(discreteGridValue + ORDERING_TOL >= bangBangValue);
	}

	@Test
	public void testIncreasingStrikeShiftCannotIncreaseCallValue() {
		final BlackScholesSetup setup = createModel(SPOT, 1.0);

		final FloatingStrikeSwingOption lowerShift = new FloatingStrikeSwingOption(
				DECISION_TIMES,
				FIXING_TIMES,
				FIXING_WEIGHTS,
				0.0,
				1.0,
				buildAccumulatorGrid(),
				0.0,
				1.0,
				0.0,
				2.0,
				CallOrPut.CALL,
				SwingQuantityMode.DISCRETE_QUANTITY_GRID,
				0.25,
				SwingStrikeFixingConvention.FIX_THEN_EXERCISE
		);

		final FloatingStrikeSwingOption higherShift = new FloatingStrikeSwingOption(
				DECISION_TIMES,
				FIXING_TIMES,
				FIXING_WEIGHTS,
				10.0,
				1.0,
				buildAccumulatorGrid(),
				0.0,
				1.0,
				0.0,
				2.0,
				CallOrPut.CALL,
				SwingQuantityMode.DISCRETE_QUANTITY_GRID,
				0.25,
				SwingStrikeFixingConvention.FIX_THEN_EXERCISE
		);

		final double lowerShiftValue = valueAtSpot(
				lowerShift.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		final double higherShiftValue = valueAtSpot(
				higherShift.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		assertTrue(lowerShiftValue + ORDERING_TOL >= higherShiftValue);
	}

	private BlackScholesSetup createModel(
			final double initialSpot,
			final double maturity) {

		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS, SPACE_MIN, SPACE_MAX);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						maturity / NUMBER_OF_TIME_STEPS
				);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { initialSpot }
		);

		final FDMBlackScholesModel model = new FDMBlackScholesModel(
				initialSpot,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				spaceTime
		);

		return new BlackScholesSetup(model, sGrid.getGrid());
	}

	private double[] buildAccumulatorGrid() {
		final int n = 81;
		final double[] grid = new double[n];
		final double aMin = 0.0;
		final double aMax = 400.0;
		final double da = (aMax - aMin) / (n - 1);

		for(int i = 0; i < n; i++) {
			grid[i] = aMin + i * da;
		}

		return grid;
	}

	private double valueAtSpot(
			final double[] values,
			final double[] sNodes,
			final double spot) {

		if(spot <= sNodes[0]) {
			return values[0];
		}
		if(spot >= sNodes[sNodes.length - 1]) {
			return values[sNodes.length - 1];
		}

		int upperIndex = 1;
		while(upperIndex < sNodes.length && sNodes[upperIndex] < spot) {
			upperIndex++;
		}
		final int lowerIndex = upperIndex - 1;

		final double xL = sNodes[lowerIndex];
		final double xU = sNodes[upperIndex];
		final double yL = values[lowerIndex];
		final double yU = values[upperIndex];

		if(Math.abs(xU - xL) < 1E-14) {
			return yL;
		}

		final double w = (spot - xL) / (xU - xL);
		return (1.0 - w) * yL + w * yU;
	}

	private static final class BlackScholesSetup {

		private final FDMBlackScholesModel model;
		private final double[] sNodes;

		private BlackScholesSetup(
				final FDMBlackScholesModel model,
				final double[] sNodes) {
			this.model = model;
			this.sNodes = sNodes;
		}
	}
}