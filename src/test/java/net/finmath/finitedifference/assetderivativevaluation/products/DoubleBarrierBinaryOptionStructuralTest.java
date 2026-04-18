package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.DoubleBarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural tests for DoubleBarrierBinaryOption.
 *
 * <p>
 * The focus here is the exercise semantics, especially for KNOCK_OUT:
 * </p>
 * <ul>
 *   <li>European <= Bermudan <= American,</li>
 *   <li>for KNOCK_OUT, American value inside the alive band should equal the cash payoff
 *       since immediate exercise is optimal under the implemented product semantics,</li>
 *   <li>outside-band endpoint semantics for KNOCK_OUT / KNOCK_IN / KIKO / KOKI.</li>
 * </ul>
 */
public class DoubleBarrierBinaryOptionStructuralTest {

	private enum ModelType {
		BLACK_SCHOLES,
		HESTON,
		SABR
	}

	private static final double MATURITY = 1.0;
	private static final double CASH_PAYOFF = 10.0;

	private static final double LOWER_BARRIER = 80.0;
	private static final double UPPER_BARRIER = 120.0;

	private static final double SPOT = 100.0;
	private static final double LOWER_OUTSIDE_SPOT = 75.0;
	private static final double UPPER_OUTSIDE_SPOT = 125.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;

	private static final double BS_VOLATILITY = 0.25;

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

	private static final int NUMBER_OF_TIME_STEPS_1D = 160;
	private static final int NUMBER_OF_SPACE_STEPS_1D = 200;

	private static final int NUMBER_OF_TIME_STEPS_2D = 120;
	private static final int NUMBER_OF_SPACE_STEPS_S_2D = 300;
	private static final int NUMBER_OF_SPACE_STEPS_SECOND_2D = 100;

	private static final double SPACE_MIN = 0.0;
	private static final double SPACE_MAX = 200.0;

	private static final double ORDERING_TOL_1D = 1E-8;
	private static final double ORDERING_TOL_2D = 5E-2;

	private static final double VALUE_TOL_1D = 1E-5;
	private static final double VALUE_TOL_2D = 1E-1;

	/*
	 * ============================================================
	 * KNOCK-OUT EXERCISE SEMANTICS
	 * ============================================================
	 */

	@Test
	public void testBlackScholesKnockOutExerciseSemantics() {
		runKnockOutExerciseSemanticsTest(ModelType.BLACK_SCHOLES);
	}

	@Test
	public void testHestonKnockOutExerciseSemantics() {
		runKnockOutExerciseSemanticsTest(ModelType.HESTON);
	}

	@Test
	public void testSabrKnockOutExerciseSemantics() {
		runKnockOutExerciseSemanticsTest(ModelType.SABR);
	}

	/*
	 * ============================================================
	 * ALREADY KNOCKED OUT
	 * ============================================================
	 */

	@Test
	public void testBlackScholesAlreadyKnockedOutIsZero() {
		runAlreadyKnockedOutIsZeroTest(ModelType.BLACK_SCHOLES, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedOutIsZeroTest(ModelType.BLACK_SCHOLES, UPPER_OUTSIDE_SPOT);
	}

	@Test
	public void testHestonAlreadyKnockedOutIsZero() {
		runAlreadyKnockedOutIsZeroTest(ModelType.HESTON, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedOutIsZeroTest(ModelType.HESTON, UPPER_OUTSIDE_SPOT);
	}

	@Test
	public void testSabrAlreadyKnockedOutIsZero() {
		runAlreadyKnockedOutIsZeroTest(ModelType.SABR, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedOutIsZeroTest(ModelType.SABR, UPPER_OUTSIDE_SPOT);
	}

	/*
	 * ============================================================
	 * ALREADY KNOCKED IN
	 * ============================================================
	 */

	@Test
	public void testBlackScholesAlreadyKnockedInIsCash() {
		runAlreadyKnockedInIsCashTest(ModelType.BLACK_SCHOLES, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedInIsCashTest(ModelType.BLACK_SCHOLES, UPPER_OUTSIDE_SPOT);
	}

	@Test
	public void testHestonAlreadyKnockedInIsCash() {
		runAlreadyKnockedInIsCashTest(ModelType.HESTON, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedInIsCashTest(ModelType.HESTON, UPPER_OUTSIDE_SPOT);
	}

	@Test
	public void testSabrAlreadyKnockedInIsCash() {
		runAlreadyKnockedInIsCashTest(ModelType.SABR, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedInIsCashTest(ModelType.SABR, UPPER_OUTSIDE_SPOT);
	}

	/*
	 * ============================================================
	 * KIKO / KOKI ENDPOINT SEMANTICS
	 * ============================================================
	 */

	@Test
	public void testBlackScholesKIKOEndpointSemantics() {
		final BlackScholesSetup lowerSetup = createBlackScholesSetup(LOWER_OUTSIDE_SPOT);
		final BlackScholesSetup upperSetup = createBlackScholesSetup(UPPER_OUTSIDE_SPOT);

		final DoubleBarrierBinaryOption kiko = new DoubleBarrierBinaryOption(
				MATURITY,
				CASH_PAYOFF,
				LOWER_BARRIER,
				UPPER_BARRIER,
				DoubleBarrierType.KIKO,
				new AmericanExercise(0.0, MATURITY)
		);

		final double lowerValue = interpolate1DAtSpot(
				kiko.getValue(0.0, lowerSetup.model),
				lowerSetup.sNodes,
				LOWER_OUTSIDE_SPOT
		);

		final double upperValue = interpolate1DAtSpot(
				kiko.getValue(0.0, upperSetup.model),
				upperSetup.sNodes,
				UPPER_OUTSIDE_SPOT
		);

		assertEquals(CASH_PAYOFF, lowerValue, VALUE_TOL_1D);
		assertEquals(0.0, upperValue, VALUE_TOL_1D);
	}

	@Test
	public void testBlackScholesKOKIEndpointSemantics() {
		final BlackScholesSetup lowerSetup = createBlackScholesSetup(LOWER_OUTSIDE_SPOT);
		final BlackScholesSetup upperSetup = createBlackScholesSetup(UPPER_OUTSIDE_SPOT);

		final DoubleBarrierBinaryOption koki = new DoubleBarrierBinaryOption(
				MATURITY,
				CASH_PAYOFF,
				LOWER_BARRIER,
				UPPER_BARRIER,
				DoubleBarrierType.KOKI,
				new AmericanExercise(0.0, MATURITY)
		);

		final double lowerValue = interpolate1DAtSpot(
				koki.getValue(0.0, lowerSetup.model),
				lowerSetup.sNodes,
				LOWER_OUTSIDE_SPOT
		);

		final double upperValue = interpolate1DAtSpot(
				koki.getValue(0.0, upperSetup.model),
				upperSetup.sNodes,
				UPPER_OUTSIDE_SPOT
		);

		assertEquals(0.0, lowerValue, VALUE_TOL_1D);
		assertEquals(CASH_PAYOFF, upperValue, VALUE_TOL_1D);
	}

	/*
	 * ============================================================
	 * CORE STRUCTURAL CHECKS
	 * ============================================================
	 */

	private void runKnockOutExerciseSemanticsTest(final ModelType modelType) {
		final DoubleBarrierBinaryOption european = new DoubleBarrierBinaryOption(
				MATURITY,
				CASH_PAYOFF,
				LOWER_BARRIER,
				UPPER_BARRIER,
				DoubleBarrierType.KNOCK_OUT,
				new EuropeanExercise(MATURITY)
		);

		final DoubleBarrierBinaryOption bermudan = new DoubleBarrierBinaryOption(
				MATURITY,
				CASH_PAYOFF,
				LOWER_BARRIER,
				UPPER_BARRIER,
				DoubleBarrierType.KNOCK_OUT,
				new BermudanExercise(new double[] { 0.5, MATURITY })
		);

		final DoubleBarrierBinaryOption american = new DoubleBarrierBinaryOption(
				MATURITY,
				CASH_PAYOFF,
				LOWER_BARRIER,
				UPPER_BARRIER,
				DoubleBarrierType.KNOCK_OUT,
				new AmericanExercise(0.0, MATURITY)
		);

		if(modelType == ModelType.BLACK_SCHOLES) {
			final BlackScholesSetup setup = createBlackScholesSetup(SPOT);

			final double europeanValue = interpolate1DAtSpot(
					european.getValue(0.0, setup.model),
					setup.sNodes,
					SPOT
			);

			final double bermudanValue = interpolate1DAtSpot(
					bermudan.getValue(0.0, setup.model),
					setup.sNodes,
					SPOT
			);

			final double americanValue = interpolate1DAtSpot(
					american.getValue(0.0, setup.model),
					setup.sNodes,
					SPOT
			);

			assertTrue(bermudanValue + ORDERING_TOL_1D >= europeanValue);
			assertTrue(americanValue + ORDERING_TOL_1D >= bermudanValue);

			assertTrue(europeanValue <= CASH_PAYOFF + VALUE_TOL_1D);
			assertTrue(bermudanValue <= CASH_PAYOFF + VALUE_TOL_1D);
			assertTrue(americanValue <= CASH_PAYOFF + VALUE_TOL_1D);

			assertEquals(CASH_PAYOFF, americanValue, VALUE_TOL_1D);
		}
		else {
			final TwoDimensionalSetup setup = createTwoDimensionalSetup(modelType, SPOT);

			final double europeanValue = interpolate2DAtInitialState(
					european.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					SPOT,
					setup.initialSecondState
			);

			final double bermudanValue = interpolate2DAtInitialState(
					bermudan.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					SPOT,
					setup.initialSecondState
			);

			final double americanValue = interpolate2DAtInitialState(
					american.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					SPOT,
					setup.initialSecondState
			);

			assertTrue(bermudanValue + ORDERING_TOL_2D >= europeanValue);
			assertTrue(americanValue + ORDERING_TOL_2D >= bermudanValue);

			assertTrue(europeanValue <= CASH_PAYOFF + VALUE_TOL_2D);
			assertTrue(bermudanValue <= CASH_PAYOFF + VALUE_TOL_2D);
			assertTrue(americanValue <= CASH_PAYOFF + VALUE_TOL_2D);

			assertEquals(CASH_PAYOFF, americanValue, VALUE_TOL_2D);
		}
	}

	private void runAlreadyKnockedOutIsZeroTest(
			final ModelType modelType,
			final double outsideSpot) {

		final DoubleBarrierBinaryOption knockOut = new DoubleBarrierBinaryOption(
				MATURITY,
				CASH_PAYOFF,
				LOWER_BARRIER,
				UPPER_BARRIER,
				DoubleBarrierType.KNOCK_OUT,
				new AmericanExercise(0.0, MATURITY)
		);

		if(modelType == ModelType.BLACK_SCHOLES) {
			final BlackScholesSetup setup = createBlackScholesSetup(outsideSpot);

			final double value = interpolate1DAtSpot(
					knockOut.getValue(0.0, setup.model),
					setup.sNodes,
					outsideSpot
			);

			assertEquals(0.0, value, VALUE_TOL_1D);
		}
		else {
			final TwoDimensionalSetup setup = createTwoDimensionalSetup(modelType, outsideSpot);

			final double value = interpolate2DAtInitialState(
					knockOut.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					outsideSpot,
					setup.initialSecondState
			);

			assertEquals(0.0, value, VALUE_TOL_2D);
		}
	}

	private void runAlreadyKnockedInIsCashTest(
			final ModelType modelType,
			final double outsideSpot) {

		final DoubleBarrierBinaryOption knockIn = new DoubleBarrierBinaryOption(
				MATURITY,
				CASH_PAYOFF,
				LOWER_BARRIER,
				UPPER_BARRIER,
				DoubleBarrierType.KNOCK_IN,
				new AmericanExercise(0.0, MATURITY)
		);

		if(modelType == ModelType.BLACK_SCHOLES) {
			final BlackScholesSetup setup = createBlackScholesSetup(outsideSpot);

			final double value = interpolate1DAtSpot(
					knockIn.getValue(0.0, setup.model),
					setup.sNodes,
					outsideSpot
			);

			assertEquals(CASH_PAYOFF, value, VALUE_TOL_1D);
		}
		else {
			final TwoDimensionalSetup setup = createTwoDimensionalSetup(modelType, outsideSpot);

			final double value = interpolate2DAtInitialState(
					knockIn.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					outsideSpot,
					setup.initialSecondState
			);

			assertEquals(CASH_PAYOFF, value, VALUE_TOL_2D);
		}
	}

	/*
	 * ============================================================
	 * MODEL SETUPS
	 * ============================================================
	 */

	private BlackScholesSetup createBlackScholesSetup(final double initialSpot) {
		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_1D, SPACE_MIN, SPACE_MAX);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS_1D,
						MATURITY / NUMBER_OF_TIME_STEPS_1D
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
				BS_VOLATILITY,
				spaceTime
		);

		return new BlackScholesSetup(model, sGrid.getGrid());
	}

	private TwoDimensionalSetup createTwoDimensionalSetup(
			final ModelType modelType,
			final double initialSpot) {

		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_S_2D, SPACE_MIN, SPACE_MAX);
		final Grid secondGrid = createSecondGrid(modelType);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS_2D,
						MATURITY / NUMBER_OF_TIME_STEPS_2D
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
			return new UniformGrid(NUMBER_OF_SPACE_STEPS_SECOND_2D, 0.0, vMax);
		}
		else if(modelType == ModelType.SABR) {
			final double alphaMax = Math.max(4.0 * SABR_INITIAL_ALPHA, 1.0);
			return new UniformGrid(NUMBER_OF_SPACE_STEPS_SECOND_2D, 0.0, alphaMax);
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

	/*
	 * ============================================================
	 * INTERPOLATION HELPERS
	 * ============================================================
	 */

	private double interpolate1DAtSpot(
			final double[] values,
			final double[] sNodes,
			final double spot) {

		final int exactIndex = findExactIndex(sNodes, spot);
		if(exactIndex >= 0) {
			return values[exactIndex];
		}

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

		final double w = (spot - xL) / (xU - xL);
		return (1.0 - w) * yL + w * yU;
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

	private int findExactIndex(final double[] grid, final double x) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - x) < 1E-12) {
				return i;
			}
		}
		return -1;
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

	/*
	 * ============================================================
	 * SMALL SETUP HOLDERS
	 * ============================================================
	 */

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