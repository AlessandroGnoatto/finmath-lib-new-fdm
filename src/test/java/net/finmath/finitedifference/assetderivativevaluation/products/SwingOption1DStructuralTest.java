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
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural tests for the 1D fixed-strike SwingOption implementation.
 *
 * <p>
 * Focus:
 * </p>
 * <ul>
 *   <li>all-zero quantities give zero value,</li>
 *   <li>a fully deterministic quantity schedule reduces to a sum of European options,</li>
 *   <li>relaxing the global maximum cannot reduce value,</li>
 *   <li>relaxing the global minimum cannot reduce value,</li>
 *   <li>a discretized quantity grid must dominate bang-bang control.</li>
 * </ul>
 *
 * <p>
 * These are structural tests, not tight regression tests.
 * </p>
 */
public class SwingOption1DStructuralTest {

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

	private static final double ZERO_TOL = 1E-10;
	private static final double ORDERING_TOL = 1E-8;
	private static final double DETERMINISTIC_TOL = 5E-2;

	@Test
	public void testZeroQuantitiesGiveZeroValue() {
		final BlackScholesSetup setup = createModel(SPOT, DECISION_TIMES[DECISION_TIMES.length - 1]);

		final SwingOption swing = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				0.0,
				0.0,
				0.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final double value = valueAtSpot(
				swing.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		assertEquals(0.0, value, ZERO_TOL);
	}

	@Test
	public void testDeterministicScheduleMatchesSumOfEuropeans() {
		/*
		 * local min = local max = 1 at both decision dates,
		 * and global min = global max = 2.
		 *
		 * Hence exercise quantities are fully deterministic:
		 * 1 unit at t = 0.5 and 1 unit at t = 1.0.
		 *
		 * The swing value should therefore match the sum of two European option values
		 * with maturities 0.5 and 1.0 respectively.
		 */
		final BlackScholesSetup swingSetup = createModel(SPOT, DECISION_TIMES[DECISION_TIMES.length - 1]);

		final SwingOption swing = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				1.0,
				1.0,
				2.0,
				2.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final double swingValue = valueAtSpot(
				swing.getValue(0.0, swingSetup.model),
				swingSetup.sNodes,
				SPOT
		);

		double referenceValue = 0.0;
		for(final double decisionTime : DECISION_TIMES) {
			final BlackScholesSetup setupForEuropean = createModel(SPOT, decisionTime);

			final EuropeanOption european = new EuropeanOption(
					decisionTime,
					STRIKE,
					CallOrPut.CALL
			);

			referenceValue += valueAtSpot(
					european.getValue(0.0, setupForEuropean.model),
					setupForEuropean.sNodes,
					SPOT
			);
		}

		assertEquals(referenceValue, swingValue, DETERMINISTIC_TOL);
	}

	@Test
	public void testRelaxingGlobalMaximumCannotReduceValue() {
		final BlackScholesSetup setup = createModel(SPOT, DECISION_TIMES[DECISION_TIMES.length - 1]);

		final SwingOption tighterGlobalMax = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				1.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final SwingOption looserGlobalMax = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				2.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
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
	public void testRelaxingGlobalMinimumCannotReduceValue() {
		final BlackScholesSetup setup = createModel(SPOT, DECISION_TIMES[DECISION_TIMES.length - 1]);

		final SwingOption stricterGlobalMin = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				1.0,
				2.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final SwingOption looserGlobalMin = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				2.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final double stricterValue = valueAtSpot(
				stricterGlobalMin.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		final double looserValue = valueAtSpot(
				looserGlobalMin.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		assertTrue(looserValue + ORDERING_TOL >= stricterValue);
	}

	@Test
	public void testDiscreteQuantityGridDominatesBangBang() {
		final BlackScholesSetup setup = createModel(SPOT, DECISION_TIMES[DECISION_TIMES.length - 1]);

		final SwingOption bangBang = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				1.5,
				CallOrPut.PUT,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final SwingOption discreteGrid = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				1.5,
				CallOrPut.PUT,
				SwingQuantityMode.DISCRETE_QUANTITY_GRID,
				0.25
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

	private BlackScholesSetup createModel(
			final double initialSpot,
			final double maturity) {

		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS, SPACE_MIN, SPACE_MAX);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						getNumberOfTimeStepsForMaturity(maturity),
						maturity / getNumberOfTimeStepsForMaturity(maturity)
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

	private int getNumberOfTimeStepsForMaturity(final double maturity) {
		return Math.max(20,
				(int)Math.round(NUMBER_OF_TIME_STEPS * maturity / DECISION_TIMES[DECISION_TIMES.length - 1]));
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