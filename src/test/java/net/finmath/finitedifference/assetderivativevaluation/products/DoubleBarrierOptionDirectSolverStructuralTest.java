package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DoubleBarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural regression tests for the direct DoubleBarrierOption implementation.
 *
 * <p>
 * Covered checks:
 * </p>
 * <ul>
 *   <li>European / Bermudan / American ordering,</li>
 *   <li>double-barrier option value <= corresponding vanilla value,</li>
 *   <li>already-knocked-in endpoint behavior: knock-in equals vanilla,</li>
 *   <li>already-knocked-out endpoint behavior: knock-out is zero,</li>
 *   <li>coverage across Black-Scholes, Heston, and SABR.</li>
 * </ul>
 *
 * <p>
 * This class is intended as a structural regression test for the new direct
 * double-barrier solvers, not as an analytic accuracy test.
 * </p>
 */
public class DoubleBarrierOptionDirectSolverStructuralTest {

	private enum ModelType {
		BLACK_SCHOLES,
		HESTON,
		SABR
	}

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;

	private static final double LOWER_BARRIER = 80.0;
	private static final double UPPER_BARRIER = 120.0;

	private static final double SPOT = 100.0;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;

	/*
	 * Black-Scholes
	 */
	private static final double BS_VOLATILITY = 0.25;

	/*
	 * Heston
	 */
	private static final double HESTON_VOLATILITY = 0.25;
	private static final double HESTON_INITIAL_VARIANCE = HESTON_VOLATILITY * HESTON_VOLATILITY;
	private static final double HESTON_KAPPA = 1.5;
	private static final double HESTON_THETA_V = HESTON_INITIAL_VARIANCE;
	private static final double HESTON_XI = 0.30;
	private static final double HESTON_RHO = -0.70;

	/*
	 * SABR
	 */
	private static final double SABR_INITIAL_ALPHA = 0.20;
	private static final double SABR_BETA = 1.0;
	private static final double SABR_NU = 0.30;
	private static final double SABR_RHO = -0.50;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS_1D = 160;
	private static final int NUMBER_OF_SPACE_STEPS_1D = 240;

	private static final int NUMBER_OF_TIME_STEPS_2D = 120;
	private static final int NUMBER_OF_SPACE_STEPS_S_2D = 220;
	private static final int NUMBER_OF_SPACE_STEPS_SECOND_2D = 100;

	/*
	 * Wide endpoint grids.
	 * Step size = 1.0 for the 1D setup, so 75, 80, 100, 120, 125 are all nodes.
	 */
	private static final int WIDE_NUMBER_OF_SPACE_STEPS_1D = 200;
	private static final double WIDE_GRID_MIN_1D = 0.0;
	private static final double WIDE_GRID_MAX_1D = 200.0;

	/*
	 * For 2D we keep the same wide spot band, with many spot nodes.
	 */
	private static final int WIDE_NUMBER_OF_SPACE_STEPS_2D = 300;
	private static final double WIDE_GRID_MIN_2D = 0.0;
	private static final double WIDE_GRID_MAX_2D = 200.0;

	private static final double LOWER_OUTSIDE_SPOT = 75.0;
	private static final double UPPER_OUTSIDE_SPOT = 125.0;

	private static final double ORDERING_TOL_1D = 1E-8;
	private static final double DOMINANCE_TOL_1D = 1E-6;

	private static final double ORDERING_TOL_2D = 5E-2;
	private static final double DOMINANCE_TOL_2D = 5E-2;

	private static final double ENDPOINT_TOL_1D = 1E-5;
	private static final double ENDPOINT_TOL_2D = 1E-1;

	/*
	 * ============================================================
	 * BLACK-SCHOLES 1D: ORDERING + VANILLA DOMINANCE
	 * ============================================================
	 */

	@Test
	public void testBlackScholesKnockOutCallOrderingAndDominance() {
		runOrderingAndVanillaDominanceTest(ModelType.BLACK_SCHOLES, DoubleBarrierType.KNOCK_OUT, CallOrPut.CALL);
	}

	@Test
	public void testBlackScholesKnockOutPutOrderingAndDominance() {
		runOrderingAndVanillaDominanceTest(ModelType.BLACK_SCHOLES, DoubleBarrierType.KNOCK_OUT, CallOrPut.PUT);
	}

	@Test
	public void testBlackScholesKnockInCallOrderingAndDominance() {
		runOrderingAndVanillaDominanceTest(ModelType.BLACK_SCHOLES, DoubleBarrierType.KNOCK_IN, CallOrPut.CALL);
	}

	@Test
	public void testBlackScholesKnockInPutOrderingAndDominance() {
		runOrderingAndVanillaDominanceTest(ModelType.BLACK_SCHOLES, DoubleBarrierType.KNOCK_IN, CallOrPut.PUT);
	}

	/*
	 * ============================================================
	 * HESTON / SABR 2D: ORDERING + VANILLA DOMINANCE
	 * ============================================================
	 */

	@Test
	public void testHestonKnockOutCallOrderingAndDominance() {
		runOrderingAndVanillaDominanceTest(ModelType.HESTON, DoubleBarrierType.KNOCK_OUT, CallOrPut.CALL);
	}

	@Test
	public void testHestonKnockInPutOrderingAndDominance() {
		runOrderingAndVanillaDominanceTest(ModelType.HESTON, DoubleBarrierType.KNOCK_IN, CallOrPut.PUT);
	}

	@Test
	public void testSabrKnockOutCallOrderingAndDominance() {
		runOrderingAndVanillaDominanceTest(ModelType.SABR, DoubleBarrierType.KNOCK_OUT, CallOrPut.CALL);
	}

	@Test
	public void testSabrKnockInPutOrderingAndDominance() {
		runOrderingAndVanillaDominanceTest(ModelType.SABR, DoubleBarrierType.KNOCK_IN, CallOrPut.PUT);
	}

	/*
	 * ============================================================
	 * ENDPOINT TESTS: ALREADY KNOCKED IN / KNOCKED OUT
	 * ============================================================
	 */

	@Test
	public void testBlackScholesAlreadyKnockedInMatchesVanilla() {
		runAlreadyKnockedInMatchesVanillaTest(ModelType.BLACK_SCHOLES, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedInMatchesVanillaTest(ModelType.BLACK_SCHOLES, UPPER_OUTSIDE_SPOT);
	}

	@Test
	public void testBlackScholesAlreadyKnockedOutIsZero() {
		runAlreadyKnockedOutIsZeroTest(ModelType.BLACK_SCHOLES, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedOutIsZeroTest(ModelType.BLACK_SCHOLES, UPPER_OUTSIDE_SPOT);
	}

	@Test
	public void testHestonAlreadyKnockedInMatchesVanilla() {
		runAlreadyKnockedInMatchesVanillaTest(ModelType.HESTON, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedInMatchesVanillaTest(ModelType.HESTON, UPPER_OUTSIDE_SPOT);
	}

	@Test
	public void testHestonAlreadyKnockedOutIsZero() {
		runAlreadyKnockedOutIsZeroTest(ModelType.HESTON, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedOutIsZeroTest(ModelType.HESTON, UPPER_OUTSIDE_SPOT);
	}

	@Test
	public void testSabrAlreadyKnockedInMatchesVanilla() {
		runAlreadyKnockedInMatchesVanillaTest(ModelType.SABR, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedInMatchesVanillaTest(ModelType.SABR, UPPER_OUTSIDE_SPOT);
	}

	@Test
	public void testSabrAlreadyKnockedOutIsZero() {
		runAlreadyKnockedOutIsZeroTest(ModelType.SABR, LOWER_OUTSIDE_SPOT);
		runAlreadyKnockedOutIsZeroTest(ModelType.SABR, UPPER_OUTSIDE_SPOT);
	}

	/*
	 * ============================================================
	 * CORE REGRESSION LOGIC
	 * ============================================================
	 */

	private void runOrderingAndVanillaDominanceTest(
			final ModelType modelType,
			final DoubleBarrierType doubleBarrierType,
			final CallOrPut callOrPut) {

		if(modelType == ModelType.BLACK_SCHOLES) {
			final BlackScholesSetup setup = createBlackScholesInsideBandSetup();

			final DoubleBarrierOption europeanBarrier = new DoubleBarrierOption(
					MATURITY,
					STRIKE,
					LOWER_BARRIER,
					UPPER_BARRIER,
					callOrPut,
					doubleBarrierType,
					new EuropeanExercise(MATURITY)
			);

			final DoubleBarrierOption bermudanBarrier = new DoubleBarrierOption(
					MATURITY,
					STRIKE,
					LOWER_BARRIER,
					UPPER_BARRIER,
					callOrPut,
					doubleBarrierType,
					new BermudanExercise(new double[] { 0.5, MATURITY })
			);

			final DoubleBarrierOption americanBarrier = new DoubleBarrierOption(
					MATURITY,
					STRIKE,
					LOWER_BARRIER,
					UPPER_BARRIER,
					callOrPut,
					doubleBarrierType,
					new AmericanExercise(0.0, MATURITY)
			);

			final FiniteDifferenceEquityProduct europeanVanilla = new EuropeanOption(
					MATURITY,
					STRIKE,
					callOrPut
			);

			final FiniteDifferenceEquityProduct bermudanVanilla = new BermudanOption(
					null,
					new double[] { 0.5, MATURITY },
					STRIKE,
					callOrPut
			);

			final FiniteDifferenceEquityProduct americanVanilla = new AmericanOption(
					MATURITY,
					STRIKE,
					callOrPut
			);

			final double europeanBarrierPrice =
					interpolate1DAtSpot(europeanBarrier.getValue(0.0, setup.model), setup.sNodes, SPOT);
			final double bermudanBarrierPrice =
					interpolate1DAtSpot(bermudanBarrier.getValue(0.0, setup.model), setup.sNodes, SPOT);
			final double americanBarrierPrice =
					interpolate1DAtSpot(americanBarrier.getValue(0.0, setup.model), setup.sNodes, SPOT);

			final double europeanVanillaPrice =
					interpolate1DAtSpot(europeanVanilla.getValue(0.0, setup.model), setup.sNodes, SPOT);
			final double bermudanVanillaPrice =
					interpolate1DAtSpot(bermudanVanilla.getValue(0.0, setup.model), setup.sNodes, SPOT);
			final double americanVanillaPrice =
					interpolate1DAtSpot(americanVanilla.getValue(0.0, setup.model), setup.sNodes, SPOT);

			assertTrue(
					"Bermudan barrier must be >= European barrier for " + doubleBarrierType + " " + callOrPut,
					bermudanBarrierPrice + ORDERING_TOL_1D >= europeanBarrierPrice
			);

			assertTrue(
					"American barrier must be >= Bermudan barrier for " + doubleBarrierType + " " + callOrPut,
					americanBarrierPrice + ORDERING_TOL_1D >= bermudanBarrierPrice
			);

			assertTrue(
					"European double barrier must be <= European vanilla for " + doubleBarrierType + " " + callOrPut,
					europeanBarrierPrice <= europeanVanillaPrice + DOMINANCE_TOL_1D
			);

			assertTrue(
					"Bermudan double barrier must be <= Bermudan vanilla for " + doubleBarrierType + " " + callOrPut,
					bermudanBarrierPrice <= bermudanVanillaPrice + DOMINANCE_TOL_1D
			);

			assertTrue(
					"American double barrier must be <= American vanilla for " + doubleBarrierType + " " + callOrPut,
					americanBarrierPrice <= americanVanillaPrice + DOMINANCE_TOL_1D
			);

			assertTrue(europeanBarrierPrice >= -1E-10);
			assertTrue(bermudanBarrierPrice >= -1E-10);
			assertTrue(americanBarrierPrice >= -1E-10);
		}
		else {
			final TwoDimensionalSetup setup = createTwoDimensionalInsideBandSetup(modelType);

			final DoubleBarrierOption europeanBarrier = new DoubleBarrierOption(
					MATURITY,
					STRIKE,
					LOWER_BARRIER,
					UPPER_BARRIER,
					callOrPut,
					doubleBarrierType,
					new EuropeanExercise(MATURITY)
			);

			final DoubleBarrierOption bermudanBarrier = new DoubleBarrierOption(
					MATURITY,
					STRIKE,
					LOWER_BARRIER,
					UPPER_BARRIER,
					callOrPut,
					doubleBarrierType,
					new BermudanExercise(new double[] { 0.5, MATURITY })
			);

			final DoubleBarrierOption americanBarrier = new DoubleBarrierOption(
					MATURITY,
					STRIKE,
					LOWER_BARRIER,
					UPPER_BARRIER,
					callOrPut,
					doubleBarrierType,
					new AmericanExercise(0.0, MATURITY)
			);

			final FiniteDifferenceEquityProduct europeanVanilla = new EuropeanOption(
					MATURITY,
					STRIKE,
					callOrPut
			);

			final FiniteDifferenceEquityProduct bermudanVanilla = new BermudanOption(
					null,
					new double[] { 0.5, MATURITY },
					STRIKE,
					callOrPut
			);

			final FiniteDifferenceEquityProduct americanVanilla = new AmericanOption(
					MATURITY,
					STRIKE,
					callOrPut
			);

			final double europeanBarrierPrice = interpolate2DAtInitialState(
					europeanBarrier.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					SPOT,
					setup.initialSecondState
			);

			final double bermudanBarrierPrice = interpolate2DAtInitialState(
					bermudanBarrier.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					SPOT,
					setup.initialSecondState
			);

			final double americanBarrierPrice = interpolate2DAtInitialState(
					americanBarrier.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					SPOT,
					setup.initialSecondState
			);

			final double europeanVanillaPrice = interpolate2DAtInitialState(
					europeanVanilla.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					SPOT,
					setup.initialSecondState
			);

			final double bermudanVanillaPrice = interpolate2DAtInitialState(
					bermudanVanilla.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					SPOT,
					setup.initialSecondState
			);

			final double americanVanillaPrice = interpolate2DAtInitialState(
					americanVanilla.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					SPOT,
					setup.initialSecondState
			);

			assertTrue(
					"Bermudan barrier must be >= European barrier for " + modelType + " "
							+ doubleBarrierType + " " + callOrPut,
					bermudanBarrierPrice + ORDERING_TOL_2D >= europeanBarrierPrice
			);

			assertTrue(
					"American barrier must be >= Bermudan barrier for " + modelType + " "
							+ doubleBarrierType + " " + callOrPut,
					americanBarrierPrice + ORDERING_TOL_2D >= bermudanBarrierPrice
			);

			assertTrue(
					"European double barrier must be <= European vanilla for " + modelType + " "
							+ doubleBarrierType + " " + callOrPut,
					europeanBarrierPrice <= europeanVanillaPrice + DOMINANCE_TOL_2D
			);

			assertTrue(
					"Bermudan double barrier must be <= Bermudan vanilla for " + modelType + " "
							+ doubleBarrierType + " " + callOrPut,
					bermudanBarrierPrice <= bermudanVanillaPrice + DOMINANCE_TOL_2D
			);

			assertTrue(
					"American double barrier must be <= American vanilla for " + modelType + " "
							+ doubleBarrierType + " " + callOrPut,
					americanBarrierPrice <= americanVanillaPrice + DOMINANCE_TOL_2D
			);

			assertTrue(europeanBarrierPrice >= -1E-10);
			assertTrue(bermudanBarrierPrice >= -1E-10);
			assertTrue(americanBarrierPrice >= -1E-10);
		}
	}

	private void runAlreadyKnockedInMatchesVanillaTest(
			final ModelType modelType,
			final double outsideSpot) {

		final Exercise americanExercise = new AmericanExercise(0.0, MATURITY);

		if(modelType == ModelType.BLACK_SCHOLES) {
			final BlackScholesSetup setup = createBlackScholesWideSetup(outsideSpot);

			final DoubleBarrierOption knockIn = new DoubleBarrierOption(
					MATURITY,
					STRIKE,
					LOWER_BARRIER,
					UPPER_BARRIER,
					CallOrPut.PUT,
					DoubleBarrierType.KNOCK_IN,
					americanExercise
			);

			final AmericanOption vanilla = new AmericanOption(
					MATURITY,
					STRIKE,
					CallOrPut.PUT
			);

			final double knockInValue = interpolate1DAtSpot(
					knockIn.getValue(0.0, setup.model),
					setup.sNodes,
					outsideSpot
			);

			final double vanillaValue = interpolate1DAtSpot(
					vanilla.getValue(0.0, setup.model),
					setup.sNodes,
					outsideSpot
			);

			assertEquals(vanillaValue, knockInValue, ENDPOINT_TOL_1D);
		}
		else {
			final TwoDimensionalSetup setup = createTwoDimensionalWideSetup(modelType, outsideSpot);

			final DoubleBarrierOption knockIn = new DoubleBarrierOption(
					MATURITY,
					STRIKE,
					LOWER_BARRIER,
					UPPER_BARRIER,
					CallOrPut.PUT,
					DoubleBarrierType.KNOCK_IN,
					americanExercise
			);

			final AmericanOption vanilla = new AmericanOption(
					MATURITY,
					STRIKE,
					CallOrPut.PUT
			);

			final double knockInValue = interpolate2DAtInitialState(
					knockIn.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					outsideSpot,
					setup.initialSecondState
			);

			final double vanillaValue = interpolate2DAtInitialState(
					vanilla.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					outsideSpot,
					setup.initialSecondState
			);

			assertEquals(vanillaValue, knockInValue, ENDPOINT_TOL_2D);
		}
	}

	private void runAlreadyKnockedOutIsZeroTest(
			final ModelType modelType,
			final double outsideSpot) {

		final Exercise americanExercise = new AmericanExercise(0.0, MATURITY);

		if(modelType == ModelType.BLACK_SCHOLES) {
			final BlackScholesSetup setup = createBlackScholesWideSetup(outsideSpot);

			final DoubleBarrierOption knockOut = new DoubleBarrierOption(
					MATURITY,
					STRIKE,
					LOWER_BARRIER,
					UPPER_BARRIER,
					CallOrPut.CALL,
					DoubleBarrierType.KNOCK_OUT,
					americanExercise
			);

			final double knockOutValue = interpolate1DAtSpot(
					knockOut.getValue(0.0, setup.model),
					setup.sNodes,
					outsideSpot
			);

			assertEquals(0.0, knockOutValue, ENDPOINT_TOL_1D);
		}
		else {
			final TwoDimensionalSetup setup = createTwoDimensionalWideSetup(modelType, outsideSpot);

			final DoubleBarrierOption knockOut = new DoubleBarrierOption(
					MATURITY,
					STRIKE,
					LOWER_BARRIER,
					UPPER_BARRIER,
					CallOrPut.CALL,
					DoubleBarrierType.KNOCK_OUT,
					americanExercise
			);

			final double knockOutValue = interpolate2DAtInitialState(
					knockOut.getValue(0.0, setup.model),
					setup.sNodes,
					setup.secondNodes,
					outsideSpot,
					setup.initialSecondState
			);

			assertEquals(0.0, knockOutValue, ENDPOINT_TOL_2D);
		}
	}

	/*
	 * ============================================================
	 * 1D BLACK-SCHOLES SETUPS
	 * ============================================================
	 */

	private BlackScholesSetup createBlackScholesInsideBandSetup() {
		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", RISK_FREE_RATE);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", DIVIDEND_YIELD);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS_1D,
						MATURITY / NUMBER_OF_TIME_STEPS_1D
				);

		/*
		 * Same style as the existing European analytic double-barrier test:
		 * lower barrier, spot, and upper barrier are all nodes.
		 */
		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_1D, 40.0, 160.0);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { SPOT }
		);

		final FDMBlackScholesModel model = new FDMBlackScholesModel(
				SPOT,
				riskFreeCurve,
				dividendCurve,
				BS_VOLATILITY,
				spaceTime
		);

		return new BlackScholesSetup(model, sGrid.getGrid());
	}

	private BlackScholesSetup createBlackScholesWideSetup(final double initialSpot) {
		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", RISK_FREE_RATE);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", DIVIDEND_YIELD);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS_1D,
						MATURITY / NUMBER_OF_TIME_STEPS_1D
				);

		final Grid sGrid = new UniformGrid(
				WIDE_NUMBER_OF_SPACE_STEPS_1D,
				WIDE_GRID_MIN_1D,
				WIDE_GRID_MAX_1D
		);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { initialSpot }
		);

		final FDMBlackScholesModel model = new FDMBlackScholesModel(
				initialSpot,
				riskFreeCurve,
				dividendCurve,
				BS_VOLATILITY,
				spaceTime
		);

		return new BlackScholesSetup(model, sGrid.getGrid());
	}

	/*
	 * ============================================================
	 * 2D HESTON / SABR SETUPS
	 * ============================================================
	 */

	private TwoDimensionalSetup createTwoDimensionalInsideBandSetup(final ModelType modelType) {
		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS_2D,
						MATURITY / NUMBER_OF_TIME_STEPS_2D
				);

		/*
		 * Use a uniform spot grid on [0, 200] so 80, 100, 120 are all nodes.
		 */
		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_S_2D, 0.0, 200.0);
		final Grid secondGrid = createSecondGrid(modelType);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, secondGrid },
				timeDiscretization,
				THETA,
				new double[] { SPOT, getInitialSecondState(modelType) }
		);

		return new TwoDimensionalSetup(
				createTwoDimensionalModel(modelType, SPOT, spaceTime),
				sGrid.getGrid(),
				secondGrid.getGrid(),
				getInitialSecondState(modelType)
		);
	}

	private TwoDimensionalSetup createTwoDimensionalWideSetup(
			final ModelType modelType,
			final double initialSpot) {

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS_2D,
						MATURITY / NUMBER_OF_TIME_STEPS_2D
				);

		final Grid sGrid = new UniformGrid(
				WIDE_NUMBER_OF_SPACE_STEPS_2D,
				WIDE_GRID_MIN_2D,
				WIDE_GRID_MAX_2D
		);

		final Grid secondGrid = createSecondGrid(modelType);

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
			throw new IllegalArgumentException("Second grid requested for unsupported model type: " + modelType);
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

	private double interpolate2DAtInitialState(
			final double[] flattenedValues,
			final double[] sNodes,
			final double[] secondNodes,
			final double spot,
			final double secondState) {

		assertTrue(
				"Spot must lie inside the grid domain.",
				spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12
		);

		assertTrue(
				"Second state must lie inside the grid domain.",
				secondState >= secondNodes[0] - 1E-12
						&& secondState <= secondNodes[secondNodes.length - 1] + 1E-12
		);

		final int nS = sNodes.length;
		final int n2 = secondNodes.length;

		final double[][] surface = new double[nS][n2];
		for(int j = 0; j < n2; j++) {
			for(int i = 0; i < nS; i++) {
				surface[i][j] = flattenedValues[flatten(i, j, nS)];
			}
		}

		final BiLinearInterpolation interpolation =
				new BiLinearInterpolation(sNodes, secondNodes, surface);

		return interpolation.apply(spot, secondState);
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

	private int flatten(final int iS, final int i2, final int numberOfSNodes) {
		return iS + i2 * numberOfSNodes;
	}

	/*
	 * ============================================================
	 * CURVE HELPER
	 * ============================================================
	 */

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