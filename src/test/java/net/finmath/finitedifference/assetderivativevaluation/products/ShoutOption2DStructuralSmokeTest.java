package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Fast structural smoke tests for the 2D shout recursion.
 *
 * Focus:
 * - 0 shouts = vanilla
 * - 1 shout >= vanilla
 *
 * These tests intentionally use small grids and a small strike grid.
 */
public class ShoutOption2DStructuralSmokeTest {

	private enum ModelType {
		HESTON,
		SABR
	}

	private static final double MATURITY = 1.0;
	private static final double INITIAL_STRIKE = 100.0;
	private static final double SPOT_ATM = 100.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;

	private static final double HESTON_VOLATILITY = 0.25;
	private static final double HESTON_INITIAL_VARIANCE = HESTON_VOLATILITY * HESTON_VOLATILITY;
	private static final double HESTON_KAPPA = 1.5;
	private static final double HESTON_THETA_V = HESTON_INITIAL_VARIANCE;
	private static final double HESTON_XI = 0.30;
	private static final double HESTON_RHO = -0.70;

	private static final double SABR_INITIAL_ALPHA = 0.20;
	private static final double SABR_BETA = 1.0;
	private static final double SABR_NU = 0.30;
	private static final double SABR_RHO = -0.50;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 35;
	private static final int NUMBER_OF_SPACE_STEPS_S = 70;
	private static final int NUMBER_OF_SPACE_STEPS_SECOND = 25;

	private static final double SPACE_MIN = 20.0;
	private static final double SPACE_MAX = 200.0;

	private static final double ZERO_SHOUT_TOL = 1E-12;
	private static final double ORDERING_TOL_2D = 2E-1;

	@Test
	public void testHestonZeroShoutEqualsVanillaPut() {
		runZeroShoutEqualsVanillaTest(ModelType.HESTON);
	}

	@Test
	public void testSabrZeroShoutEqualsVanillaPut() {
		runZeroShoutEqualsVanillaTest(ModelType.SABR);
	}

	@Test
	public void testHestonOneShoutDominatesVanillaPut() {
		runOneShoutDominatesVanillaTest(ModelType.HESTON);
	}

	@Test
	public void testSabrOneShoutDominatesVanillaPut() {
		runOneShoutDominatesVanillaTest(ModelType.SABR);
	}

	private void runZeroShoutEqualsVanillaTest(final ModelType modelType) {
		final TwoDimensionalSetup setup = createTwoDimensionalSetup(modelType, SPOT_ATM);

		final ShoutOption zeroShout = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildSmokeStrikeGrid(),
				0,
				CallOrPut.PUT
		);

		final EuropeanOption vanilla = new EuropeanOption(
				MATURITY,
				INITIAL_STRIKE,
				CallOrPut.PUT
		);

		final double shoutValue = interpolate2DAtInitialState(
				zeroShout.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT_ATM,
				setup.initialSecondState
		);

		final double vanillaValue = interpolate2DAtInitialState(
				vanilla.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT_ATM,
				setup.initialSecondState
		);

		assertEquals(vanillaValue, shoutValue, ZERO_SHOUT_TOL);
	}

	private void runOneShoutDominatesVanillaTest(final ModelType modelType) {
		final TwoDimensionalSetup setup = createTwoDimensionalSetup(modelType, SPOT_ATM);

		final ShoutOption oneShout = new ShoutOption(
				MATURITY,
				INITIAL_STRIKE,
				buildSmokeStrikeGrid(),
				1,
				CallOrPut.PUT
		);

		final EuropeanOption vanilla = new EuropeanOption(
				MATURITY,
				INITIAL_STRIKE,
				CallOrPut.PUT
		);

		final double shoutValue = interpolate2DAtInitialState(
				oneShout.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT_ATM,
				setup.initialSecondState
		);

		final double vanillaValue = interpolate2DAtInitialState(
				vanilla.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT_ATM,
				setup.initialSecondState
		);

		assertTrue(shoutValue + ORDERING_TOL_2D >= vanillaValue);
	}

	private double[] buildSmokeStrikeGrid() {
		final int n = 21;
		final double[] grid = new double[n];
		final double dx = (SPACE_MAX - SPACE_MIN) / (n - 1);

		for(int i = 0; i < n; i++) {
			grid[i] = SPACE_MIN + i * dx;
		}

		return grid;
	}

	private TwoDimensionalSetup createTwoDimensionalSetup(
			final ModelType modelType,
			final double initialSpot) {

		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_S, SPACE_MIN, SPACE_MAX);
		final Grid secondGrid = createSecondGrid(modelType);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, secondGrid },
				timeDiscretization,
				THETA,
				new double[] { initialSpot, getInitialSecondState(modelType) }
		);

		return new TwoDimensionalSetup(
				createTwoDimensionalModel(modelType, initialSpot, spaceTime),
				sGrid.getGrid(),
				secondGrid.getGrid(),
				getInitialSecondState(modelType)
		);
	}

	private Grid createSecondGrid(final ModelType modelType) {
		if(modelType == ModelType.HESTON) {
			final double vMax = Math.max(
					4.0 * HESTON_THETA_V,
					HESTON_INITIAL_VARIANCE + 4.0 * HESTON_XI * Math.sqrt(MATURITY)
			);
			return new UniformGrid(NUMBER_OF_SPACE_STEPS_SECOND, 0.0, vMax);
		}
		else if(modelType == ModelType.SABR) {
			final double alphaMax = Math.max(4.0 * SABR_INITIAL_ALPHA, 1.0);
			return new UniformGrid(NUMBER_OF_SPACE_STEPS_SECOND, 0.0, alphaMax);
		}
		else {
			throw new IllegalArgumentException("Unsupported model type: " + modelType);
		}
	}

	private FiniteDifferenceEquityModel createTwoDimensionalModel(
			final ModelType modelType,
			final double initialSpot,
			final SpaceTimeDiscretization spaceTime) {

		if(modelType == ModelType.HESTON) {
			return new FDMHestonModel(
					initialSpot,
					HESTON_INITIAL_VARIANCE,
					RISK_FREE_RATE,
					DIVIDEND_YIELD,
					HESTON_KAPPA,
					HESTON_THETA_V,
					HESTON_XI,
					HESTON_RHO,
					spaceTime
			);
		}
		else if(modelType == ModelType.SABR) {
			return new FDMSabrModel(
					initialSpot,
					SABR_INITIAL_ALPHA,
					RISK_FREE_RATE,
					DIVIDEND_YIELD,
					SABR_BETA,
					SABR_NU,
					SABR_RHO,
					spaceTime
			);
		}
		else {
			throw new IllegalArgumentException("Unsupported model type: " + modelType);
		}
	}

	private double getInitialSecondState(final ModelType modelType) {
		if(modelType == ModelType.HESTON) {
			return HESTON_INITIAL_VARIANCE;
		}
		if(modelType == ModelType.SABR) {
			return SABR_INITIAL_ALPHA;
		}
		throw new IllegalArgumentException("Unsupported model type: " + modelType);
	}

	private double interpolate2DAtInitialState(
			final double[] flattenedValues,
			final double[] sNodes,
			final double[] secondNodes,
			final double spot,
			final double secondState) {

		final int nS = sNodes.length;

		final int i0 = getLowerBracketIndex(sNodes, spot);
		final int i1 = Math.min(i0 + 1, sNodes.length - 1);

		final int j0 = getLowerBracketIndex(secondNodes, secondState);
		final int j1 = Math.min(j0 + 1, secondNodes.length - 1);

		final double x0 = sNodes[i0];
		final double x1 = sNodes[i1];
		final double y0 = secondNodes[j0];
		final double y1 = secondNodes[j1];

		final double f00 = flattenedValues[flatten(i0, j0, nS)];
		final double f10 = flattenedValues[flatten(i1, j0, nS)];
		final double f01 = flattenedValues[flatten(i0, j1, nS)];
		final double f11 = flattenedValues[flatten(i1, j1, nS)];

		final double wx = Math.abs(x1 - x0) < 1E-14 ? 0.0 : (spot - x0) / (x1 - x0);
		final double wy = Math.abs(y1 - y0) < 1E-14 ? 0.0 : (secondState - y0) / (y1 - y0);

		return (1.0 - wx) * (1.0 - wy) * f00
				+ wx * (1.0 - wy) * f10
				+ (1.0 - wx) * wy * f01
				+ wx * wy * f11;
	}

	private int getLowerBracketIndex(final double[] grid, final double x) {
		if(x <= grid[0]) {
			return 0;
		}
		if(x >= grid[grid.length - 1]) {
			return grid.length - 2;
		}

		int upperIndex = 1;
		while(upperIndex < grid.length && grid[upperIndex] < x) {
			upperIndex++;
		}
		return upperIndex - 1;
	}

	private int flatten(final int iS, final int i2, final int numberOfSNodes) {
		return iS + i2 * numberOfSNodes;
	}

	private static final class TwoDimensionalSetup {

		private final FiniteDifferenceEquityModel model;
		private final double[] sNodes;
		private final double[] secondNodes;
		private final double initialSecondState;

		private TwoDimensionalSetup(
				final FiniteDifferenceEquityModel model,
				final double[] sNodes,
				final double[] secondNodes,
				final double initialSecondState) {
			this.model = model;
			this.sNodes = sNodes;
			this.secondNodes = secondNodes;
			this.initialSecondState = initialSecondState;
		}
	}
}