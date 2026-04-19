package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Small structural regression tests for the 1D shout implementation.
 *
 * <p>
 * Focus:
 * </p>
 * <ul>
 *   <li>zero-shout reduces exactly to vanilla,</li>
 *   <li>one shout dominates vanilla,</li>
 *   <li>more shouts should not reduce value,</li>
 *   <li>continuous shout right dominates immediate reset into a better strike.</li>
 * </ul>
 */
public class ShoutOption1DStructuralTest {

	private static final double MATURITY = 1.0;
	private static final double INITIAL_STRIKE = 100.0;

	private static final double SPOT_ATM = 100.0;
	private static final double SPOT_HIGH = 120.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;
	private static final double VOLATILITY = 0.25;
	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 160;
	private static final int NUMBER_OF_SPACE_STEPS = 180;

	private static final double SPACE_MIN = 20.0;
	private static final double SPACE_MAX = 200.0;

	private static final double TOL = 1E-8;

	@Test
	public void testZeroShoutEqualsVanillaPut() {
		final BlackScholesSetup setup = createModel();

		final ShoutOption zeroShout = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildStrikeGrid(),
				0,
				CallOrPut.PUT
		);

		final EuropeanOption vanilla = new EuropeanOption(
				MATURITY,
				INITIAL_STRIKE,
				CallOrPut.PUT
		);

		final double shoutValue = valueAtSpot(zeroShout.getValue(0.0, setup.model), setup.sNodes, SPOT_ATM);
		final double vanillaValue = valueAtSpot(vanilla.getValue(0.0, setup.model), setup.sNodes, SPOT_ATM);

		assertEquals(vanillaValue, shoutValue, TOL);
	}

	@Test
	public void testOneShoutDominatesVanillaPut() {
		final BlackScholesSetup setup = createModel();

		final ShoutOption oneShout = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildStrikeGrid(),
				1,
				CallOrPut.PUT
		);

		final EuropeanOption vanilla = new EuropeanOption(
				MATURITY,
				INITIAL_STRIKE,
				CallOrPut.PUT
		);

		final double shoutValue = valueAtSpot(oneShout.getValue(0.0, setup.model), setup.sNodes, SPOT_ATM);
		final double vanillaValue = valueAtSpot(vanilla.getValue(0.0, setup.model), setup.sNodes, SPOT_ATM);

		assertTrue(shoutValue + TOL >= vanillaValue);
	}

	@Test
	public void testTwoShoutsDominateOneShout() {
		final BlackScholesSetup setup = createModel();

		final ShoutOption oneShout = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildStrikeGrid(),
				1,
				CallOrPut.PUT
		);

		final ShoutOption twoShouts = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildStrikeGrid(),
				2,
				CallOrPut.PUT
		);

		final double oneShoutValue = valueAtSpot(oneShout.getValue(0.0, setup.model), setup.sNodes, SPOT_ATM);
		final double twoShoutValue = valueAtSpot(twoShouts.getValue(0.0, setup.model), setup.sNodes, SPOT_ATM);

		assertTrue(twoShoutValue + TOL >= oneShoutValue);
	}

	@Test
	public void testOneShoutDominatesImmediateResetVanillaPut() {
		final BlackScholesSetup setup = createModel();

		final ShoutOption oneShout = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildStrikeGrid(),
				1,
				CallOrPut.PUT
		);

		/*
		 * With continuous shout right and reset rule K* = S,
		 * at S = 120 the holder can shout immediately into a vanilla put with strike 120.
		 */
		final EuropeanOption immediateResetVanilla = new EuropeanOption(
				MATURITY,
				SPOT_HIGH,
				CallOrPut.PUT
		);

		final double shoutValue = valueAtSpot(oneShout.getValue(0.0, setup.model), setup.sNodes, SPOT_HIGH);
		final double resetVanillaValue = valueAtSpot(
				immediateResetVanilla.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT_HIGH
		);

		assertTrue(shoutValue + TOL >= resetVanillaValue);
	}

	private BlackScholesSetup createModel() {
		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS, SPACE_MIN, SPACE_MAX);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { SPOT_ATM }
		);

		final FDMBlackScholesModel model = new FDMBlackScholesModel(
				SPOT_ATM,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				spaceTime
		);

		return new BlackScholesSetup(model, sGrid.getGrid());
	}

	private double[] buildStrikeGrid() {
		final int n = 181;
		final double[] grid = new double[n];
		final double dx = (SPACE_MAX - SPACE_MIN) / (n - 1);

		for(int i = 0; i < n; i++) {
			grid[i] = SPACE_MIN + i * dx;
		}
		return grid;
	}

	private double valueAtSpot(
			final double[] values,
			final double[] sNodes,
			final double spot) {

		for(int i = 0; i < sNodes.length; i++) {
			if(Math.abs(sNodes[i] - spot) < 1E-12) {
				return values[i];
			}
		}
		throw new IllegalArgumentException("Spot is not a grid node in this test setup.");
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