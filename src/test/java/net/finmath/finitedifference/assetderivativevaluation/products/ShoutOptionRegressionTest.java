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
 * Regression tests for ShoutOption under the 1D Black-Scholes model.
 *
 * <p>
 * Focus:
 * </p>
 * <ul>
 *   <li>0 shouts = vanilla exactly,</li>
 *   <li>1 shout >= vanilla,</li>
 *   <li>2 shouts >= 1 shout,</li>
 *   <li>1 shout dominates immediate-reset vanilla intuition,</li>
 *   <li>stabilization under time/space refinement,</li>
 *   <li>stabilization under strike-grid refinement.</li>
 * </ul>
 *
 * <p>
 * These are regression-style tests, not closed-form shout formula tests.
 * The fine configurations serve as internal numerical references.
 * </p>
 */
public class ShoutOptionRegressionTest {

	private static final double MATURITY = 1.0;
	private static final double INITIAL_STRIKE = 100.0;

	private static final double SPOT_ATM = 100.0;
	private static final double SPOT_HIGH = 120.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;
	private static final double VOLATILITY = 0.25;
	private static final double THETA = 0.5;

	private static final double SPACE_MIN = 20.0;
	private static final double SPACE_MAX = 200.0;

	private static final int TIME_STEPS_COARSE = 50;
	private static final int SPACE_STEPS_COARSE = 80;

	private static final int TIME_STEPS_MEDIUM = 100;
	private static final int SPACE_STEPS_MEDIUM = 120;

	private static final int TIME_STEPS_FINE = 160;
	private static final int SPACE_STEPS_FINE = 180;

	private static final int TIME_STEPS_REFERENCE = 240;
	private static final int SPACE_STEPS_REFERENCE = 260;

	private static final double ZERO_SHOUT_TOL = 1E-12;
	private static final double ORDERING_TOL = 1E-10;
	private static final double RESET_DOMINANCE_TOL = 1E-8;

	private static final double COARSE_REF_TOL = 3.5E-1;
	private static final double MEDIUM_REF_TOL = 2.0E-1;
	private static final double FINE_REF_TOL = 1.0E-1;

	private static final double COARSE_STRIKE_REF_TOL = 1.2;
	private static final double MEDIUM_STRIKE_REF_TOL = 4.0E-1;
	private static final double FINE_STRIKE_REF_TOL = 2.0E-1;

	@Test
	public void testBlackScholesZeroShoutEqualsVanillaPut() {
		final BlackScholesSetup setup = createModel(SPOT_ATM, TIME_STEPS_FINE, SPACE_STEPS_FINE);

		final ShoutOption zeroShout = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildStrikeGrid(41),
				0,
				CallOrPut.PUT
		);

		final EuropeanOption vanilla = new EuropeanOption(
				MATURITY,
				INITIAL_STRIKE,
				CallOrPut.PUT
		);

		final double shoutValue = interpolate1DAtSpot(
				zeroShout.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT_ATM
		);

		final double vanillaValue = interpolate1DAtSpot(
				vanilla.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT_ATM
		);

		assertEquals(vanillaValue, shoutValue, ZERO_SHOUT_TOL);
	}

	@Test
	public void testBlackScholesOneShoutDominatesVanillaPut() {
		final BlackScholesSetup setup = createModel(SPOT_ATM, TIME_STEPS_FINE, SPACE_STEPS_FINE);

		final ShoutOption oneShout = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildStrikeGrid(41),
				1,
				CallOrPut.PUT
		);

		final EuropeanOption vanilla = new EuropeanOption(
				MATURITY,
				INITIAL_STRIKE,
				CallOrPut.PUT
		);

		final double shoutValue = interpolate1DAtSpot(
				oneShout.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT_ATM
		);

		final double vanillaValue = interpolate1DAtSpot(
				vanilla.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT_ATM
		);

		assertTrue(shoutValue + ORDERING_TOL >= vanillaValue);
	}

	@Test
	public void testBlackScholesTwoShoutsDominateOneShoutPut() {
		final BlackScholesSetup setup = createModel(SPOT_ATM, TIME_STEPS_FINE, SPACE_STEPS_FINE);

		final ShoutOption oneShout = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildStrikeGrid(41),
				1,
				CallOrPut.PUT
		);

		final ShoutOption twoShouts = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildStrikeGrid(41),
				2,
				CallOrPut.PUT
		);

		final double oneShoutValue = interpolate1DAtSpot(
				oneShout.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT_ATM
		);

		final double twoShoutValue = interpolate1DAtSpot(
				twoShouts.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT_ATM
		);

		assertTrue(twoShoutValue + ORDERING_TOL >= oneShoutValue);
	}

	@Test
	public void testBlackScholesOneShoutDominatesImmediateResetVanillaPut() {
		final BlackScholesSetup setup = createModel(SPOT_HIGH, TIME_STEPS_FINE, SPACE_STEPS_FINE);

		final ShoutOption oneShout = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildStrikeGrid(41),
				1,
				CallOrPut.PUT
		);

		/*
		 * With reset rule K* = S and S = 120, the holder may shout immediately
		 * into a vanilla put with strike 120. The continuous shout right should dominate that.
		 */
		final EuropeanOption immediateResetVanilla = new EuropeanOption(
				MATURITY,
				SPOT_HIGH,
				CallOrPut.PUT
		);

		final double shoutValue = interpolate1DAtSpot(
				oneShout.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT_HIGH
		);

		final double resetVanillaValue = interpolate1DAtSpot(
				immediateResetVanilla.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT_HIGH
		);

		assertTrue(shoutValue + RESET_DOMINANCE_TOL >= resetVanillaValue);
	}

	@Test
	public void testBlackScholesOneShoutStabilizesWithTimeAndSpaceRefinement() {
		final double referenceValue = priceOneShoutPut(
				SPOT_ATM,
				TIME_STEPS_REFERENCE,
				SPACE_STEPS_REFERENCE,
				41
		);

		final double coarseValue = priceOneShoutPut(
				SPOT_ATM,
				TIME_STEPS_COARSE,
				SPACE_STEPS_COARSE,
				41
		);

		final double mediumValue = priceOneShoutPut(
				SPOT_ATM,
				TIME_STEPS_MEDIUM,
				SPACE_STEPS_MEDIUM,
				41
		);

		final double fineValue = priceOneShoutPut(
				SPOT_ATM,
				TIME_STEPS_FINE,
				SPACE_STEPS_FINE,
				41
		);

		final double coarseError = Math.abs(coarseValue - referenceValue);
		final double mediumError = Math.abs(mediumValue - referenceValue);
		final double fineError = Math.abs(fineValue - referenceValue);

		assertTrue(
				"Coarse one-shout error too large versus reference: " + coarseError,
				coarseError < COARSE_REF_TOL
		);

		assertTrue(
				"Medium one-shout error too large versus reference: " + mediumError,
				mediumError < MEDIUM_REF_TOL
		);

		assertTrue(
				"Fine one-shout error too large versus reference: " + fineError,
				fineError < FINE_REF_TOL
		);

		assertTrue(
				"Medium refinement should not be materially worse than coarse refinement.",
				mediumError <= coarseError + 5.0E-2
		);

		assertTrue(
				"Fine refinement should not be materially worse than medium refinement.",
				fineError <= mediumError + 5.0E-2
		);
	}

	@Test
	public void testBlackScholesOneShoutStabilizesWithStrikeGridRefinement() {
		final double referenceValue = priceOneShoutPut(
				SPOT_ATM,
				TIME_STEPS_FINE,
				SPACE_STEPS_FINE,
				81
		);

		final double coarseStrikeValue = priceOneShoutPut(
				SPOT_ATM,
				TIME_STEPS_FINE,
				SPACE_STEPS_FINE,
				21
		);

		final double mediumStrikeValue = priceOneShoutPut(
				SPOT_ATM,
				TIME_STEPS_FINE,
				SPACE_STEPS_FINE,
				41
		);

		final double fineStrikeValue = priceOneShoutPut(
				SPOT_ATM,
				TIME_STEPS_FINE,
				SPACE_STEPS_FINE,
				61
		);

		final double coarseError = Math.abs(coarseStrikeValue - referenceValue);
		final double mediumError = Math.abs(mediumStrikeValue - referenceValue);
		final double fineError = Math.abs(fineStrikeValue - referenceValue);

		assertTrue(
				"Coarse strike-grid one-shout error too large versus reference: " + coarseError,
				coarseError < COARSE_STRIKE_REF_TOL
		);

		assertTrue(
				"Medium strike-grid one-shout error too large versus reference: " + mediumError,
				mediumError < MEDIUM_STRIKE_REF_TOL
		);

		assertTrue(
				"Fine strike-grid one-shout error too large versus reference: " + fineError,
				fineError < FINE_STRIKE_REF_TOL
		);

		assertTrue(
				"Medium strike-grid refinement should not be materially worse than coarse.",
				mediumError <= coarseError + 5.0E-2
		);

		assertTrue(
				"Fine strike-grid refinement should not be materially worse than medium.",
				fineError <= mediumError + 5.0E-2
		);
	}

	private double priceOneShoutPut(
			final double spot,
			final int timeSteps,
			final int spaceSteps,
			final int strikeGridSize) {

		final BlackScholesSetup setup = createModel(spot, timeSteps, spaceSteps);

		final ShoutOption oneShout = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildStrikeGrid(strikeGridSize),
				1,
				CallOrPut.PUT
		);

		return interpolate1DAtSpot(
				oneShout.getValue(0.0, setup.model),
				setup.sNodes,
				spot
		);
	}

	private BlackScholesSetup createModel(
			final double initialSpot,
			final int numberOfTimeSteps,
			final int numberOfSpaceSteps) {

		final Grid sGrid = new UniformGrid(numberOfSpaceSteps, SPACE_MIN, SPACE_MAX);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						numberOfTimeSteps,
						MATURITY / numberOfTimeSteps
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

	private double[] buildStrikeGrid(final int n) {
		final double[] grid = new double[n];
		final double dx = (SPACE_MAX - SPACE_MIN) / (n - 1);

		for(int i = 0; i < n; i++) {
			grid[i] = SPACE_MIN + i * dx;
		}

		return grid;
	}

	private double interpolate1DAtSpot(
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