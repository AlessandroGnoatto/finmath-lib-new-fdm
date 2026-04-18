package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import it.univr.fima.correction.BarrierOptions;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DoubleBarrierType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for DoubleBarrierOption.
 *
 * <p>
 * Focus:
 * </p>
 * <ul>
 *   <li>European Black-Scholes analytic comparison,</li>
 *   <li>European KI + KO ~= vanilla,</li>
 *   <li>monotonicity with respect to widening the alive band,</li>
 *   <li>grid convergence against the Black-Scholes analytic benchmark.</li>
 * </ul>
 *
 * <p>
 * Compared with the structural test class, this one is more numerical and
 * should be treated as a regression suite rather than a smoke test.
 * </p>
 */
public class DoubleBarrierOptionRegressionTest {

	private enum ModelType {
		BLACK_SCHOLES,
		HESTON,
		SABR
	}

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double SPOT = 100.0;

	private static final double LOWER_BARRIER = 80.0;
	private static final double UPPER_BARRIER = 120.0;

	private static final double LOWER_BARRIER_NARROW = 85.0;
	private static final double UPPER_BARRIER_NARROW = 115.0;

	private static final double LOWER_BARRIER_WIDE = 75.0;
	private static final double UPPER_BARRIER_WIDE = 125.0;

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

	/*
	 * 1D regression grids.
	 * We keep [0, 200] so 75, 80, 85, 100, 115, 120, 125 are all nodes
	 * whenever the number of steps is a multiple of 40.
	 */
	private static final double SPACE_MIN_1D = 0.0;
	private static final double SPACE_MAX_1D = 200.0;

	private static final int ANALYTIC_TIME_STEPS = 240;
	private static final int ANALYTIC_SPACE_STEPS = 240;

	private static final int COARSE_TIME_STEPS = 60;
	private static final int COARSE_SPACE_STEPS = 80;

	private static final int MEDIUM_TIME_STEPS = 120;
	private static final int MEDIUM_SPACE_STEPS = 160;

	private static final int FINE_TIME_STEPS = 240;
	private static final int FINE_SPACE_STEPS = 320;

	/*
	 * 2D diagnostic grids.
	 */
	private static final int TIME_STEPS_2D = 120;
	private static final int SPACE_STEPS_S_2D = 220;
	private static final int SPACE_STEPS_SECOND_2D = 100;

	private static final double SPACE_MIN_2D = 0.0;
	private static final double SPACE_MAX_2D = 200.0;

	/*
	 * Tolerances
	 */
	private static final double PARITY_TOL_1D = 1.0E-2;
	private static final double PARITY_TOL_2D = 2.5E-1;

	private static final double MONOTONICITY_TOL = 1.0E-8;

	private static final double ANALYTIC_TOL_1D_KNOCK_OUT = 2.0E-2;
	private static final double ANALYTIC_TOL_1D_KNOCK_IN  = 1.2E-1;

	private static final double COARSE_ERROR_TOL_KNOCK_OUT = 1.2E-1;
	private static final double MEDIUM_ERROR_TOL_KNOCK_OUT = 5.0E-2;
	private static final double FINE_ERROR_TOL_KNOCK_OUT   = 2.0E-2;

	private static final double COARSE_ERROR_TOL_KNOCK_IN = 1.5E-1;
	private static final double MEDIUM_ERROR_TOL_KNOCK_IN = 7.0E-2;
	private static final double FINE_ERROR_TOL_KNOCK_IN   = 8.0E-2;


	/*
	 * ============================================================
	 * EUROPEAN BLACK-SCHOLES VS ANALYTIC
	 * ============================================================
	 */

	@Test
	public void testBlackScholesEuropeanAnalyticKnockOutCall() {
		runBlackScholesAnalyticComparison(CallOrPut.CALL, DoubleBarrierType.KNOCK_OUT);
	}

	@Test
	public void testBlackScholesEuropeanAnalyticKnockOutPut() {
		runBlackScholesAnalyticComparison(CallOrPut.PUT, DoubleBarrierType.KNOCK_OUT);
	}

	@Test
	public void testBlackScholesEuropeanAnalyticKnockInCall() {
		runBlackScholesAnalyticComparison(CallOrPut.CALL, DoubleBarrierType.KNOCK_IN);
	}

	@Test
	public void testBlackScholesEuropeanAnalyticKnockInPut() {
		runBlackScholesAnalyticComparison(CallOrPut.PUT, DoubleBarrierType.KNOCK_IN);
	}

	/*
	 * ============================================================
	 * KI + KO ~= VANILLA
	 * ============================================================
	 */

	@Test
	public void testBlackScholesEuropeanKnockInPlusKnockOutMatchesVanillaCall() {
		runEuropeanParityCheck1D(CallOrPut.CALL);
	}

	@Test
	public void testBlackScholesEuropeanKnockInPlusKnockOutMatchesVanillaPut() {
		runEuropeanParityCheck1D(CallOrPut.PUT);
	}

	@Test
	public void testHestonEuropeanKnockInPlusKnockOutMatchesVanillaCall() {
		runEuropeanParityCheck2D(ModelType.HESTON, CallOrPut.CALL);
	}

	@Test
	public void testSabrEuropeanKnockInPlusKnockOutMatchesVanillaPut() {
		runEuropeanParityCheck2D(ModelType.SABR, CallOrPut.PUT);
	}

	/*
	 * ============================================================
	 * MONOTONICITY
	 * ============================================================
	 */

	@Test
	public void testBlackScholesEuropeanBandMonotonicity() {
		runBandMonotonicityTest(new EuropeanExercise(MATURITY));
	}

	@Test
	public void testBlackScholesAmericanBandMonotonicity() {
		runBandMonotonicityTest(new AmericanExercise(0.0, MATURITY));
	}

	/*
	 * ============================================================
	 * CONVERGENCE
	 * ============================================================
	 */

	@Test
	public void testBlackScholesConvergenceKnockOutCall() {
		runBlackScholesConvergenceTest(CallOrPut.CALL, DoubleBarrierType.KNOCK_OUT);
	}

	@Test
	public void testBlackScholesConvergenceKnockInPut() {
		runBlackScholesConvergenceTest(CallOrPut.PUT, DoubleBarrierType.KNOCK_IN);
	}

	/*
	 * ============================================================
	 * ANALYTIC COMPARISON
	 * ============================================================
	 */

	private void runBlackScholesAnalyticComparison(
			final CallOrPut callOrPut,
			final DoubleBarrierType barrierType) {

		final FDMBlackScholesModel model = createBlackScholesModel(
				SPOT,
				ANALYTIC_TIME_STEPS,
				ANALYTIC_SPACE_STEPS
		);

		final DoubleBarrierOption option = new DoubleBarrierOption(
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				callOrPut,
				barrierType,
				new EuropeanExercise(MATURITY)
		);

		final double numericalValue = interpolate1DAtSpot(
				option.getValue(0.0, model),
				model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
				SPOT
		);

		final double analyticValue = analyticDoubleBarrierValue(
				callOrPut,
				barrierType,
				LOWER_BARRIER,
				UPPER_BARRIER
		);

		final double tolerance =
				barrierType == DoubleBarrierType.KNOCK_IN
				? ANALYTIC_TOL_1D_KNOCK_IN
				: ANALYTIC_TOL_1D_KNOCK_OUT;

		assertEquals(
				"Black-Scholes European analytic comparison failed for " + barrierType + " " + callOrPut,
				analyticValue,
				numericalValue,
				tolerance
		);
	}

	/*
	 * ============================================================
	 * EUROPEAN PARITY CHECKS
	 * ============================================================
	 */

	private void runEuropeanParityCheck1D(final CallOrPut callOrPut) {
		final FDMBlackScholesModel model = createBlackScholesModel(
				SPOT,
				ANALYTIC_TIME_STEPS,
				ANALYTIC_SPACE_STEPS
		);

		final DoubleBarrierOption knockIn = new DoubleBarrierOption(
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				callOrPut,
				DoubleBarrierType.KNOCK_IN,
				new EuropeanExercise(MATURITY)
		);

		final DoubleBarrierOption knockOut = new DoubleBarrierOption(
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				callOrPut,
				DoubleBarrierType.KNOCK_OUT,
				new EuropeanExercise(MATURITY)
		);

		final EuropeanOption vanilla = new EuropeanOption(
				MATURITY,
				STRIKE,
				callOrPut
		);

		final double knockInValue = interpolate1DAtSpot(
				knockIn.getValue(0.0, model),
				model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
				SPOT
		);

		final double knockOutValue = interpolate1DAtSpot(
				knockOut.getValue(0.0, model),
				model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
				SPOT
		);

		final double vanillaValue = interpolate1DAtSpot(
				vanilla.getValue(0.0, model),
				model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
				SPOT
		);

		assertEquals(
				"1D European KI + KO must match vanilla for " + callOrPut,
				vanillaValue,
				knockInValue + knockOutValue,
				PARITY_TOL_1D
		);
	}

	private void runEuropeanParityCheck2D(
			final ModelType modelType,
			final CallOrPut callOrPut) {

		final TwoDimensionalSetup setup = createTwoDimensionalSetup(modelType);

		final DoubleBarrierOption knockIn = new DoubleBarrierOption(
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				callOrPut,
				DoubleBarrierType.KNOCK_IN,
				new EuropeanExercise(MATURITY)
		);

		final DoubleBarrierOption knockOut = new DoubleBarrierOption(
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				callOrPut,
				DoubleBarrierType.KNOCK_OUT,
				new EuropeanExercise(MATURITY)
		);

		final EuropeanOption vanilla = new EuropeanOption(
				MATURITY,
				STRIKE,
				callOrPut
		);

		final double knockInValue = interpolate2DAtInitialState(
				knockIn.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT,
				setup.initialSecondState
		);

		final double knockOutValue = interpolate2DAtInitialState(
				knockOut.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT,
				setup.initialSecondState
		);

		final double vanillaValue = interpolate2DAtInitialState(
				vanilla.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT,
				setup.initialSecondState
		);

		assertEquals(
				modelType + " European KI + KO should approximately match vanilla for " + callOrPut,
				vanillaValue,
				knockInValue + knockOutValue,
				PARITY_TOL_2D
		);
	}

	/*
	 * ============================================================
	 * MONOTONICITY
	 * ============================================================
	 */

	private void runBandMonotonicityTest(final Exercise exercise) {
		final FDMBlackScholesModel model = createBlackScholesModel(
				SPOT,
				ANALYTIC_TIME_STEPS,
				ANALYTIC_SPACE_STEPS
		);

		final DoubleBarrierOption knockOutNarrow = new DoubleBarrierOption(
				MATURITY,
				STRIKE,
				LOWER_BARRIER_NARROW,
				UPPER_BARRIER_NARROW,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_OUT,
				exercise
		);

		final DoubleBarrierOption knockOutWide = new DoubleBarrierOption(
				MATURITY,
				STRIKE,
				LOWER_BARRIER_WIDE,
				UPPER_BARRIER_WIDE,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_OUT,
				exercise
		);

		final DoubleBarrierOption knockInNarrow = new DoubleBarrierOption(
				MATURITY,
				STRIKE,
				LOWER_BARRIER_NARROW,
				UPPER_BARRIER_NARROW,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_IN,
				exercise
		);

		final DoubleBarrierOption knockInWide = new DoubleBarrierOption(
				MATURITY,
				STRIKE,
				LOWER_BARRIER_WIDE,
				UPPER_BARRIER_WIDE,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_IN,
				exercise
		);

		final double knockOutNarrowValue = interpolate1DAtSpot(
				knockOutNarrow.getValue(0.0, model),
				model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
				SPOT
		);

		final double knockOutWideValue = interpolate1DAtSpot(
				knockOutWide.getValue(0.0, model),
				model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
				SPOT
		);

		final double knockInNarrowValue = interpolate1DAtSpot(
				knockInNarrow.getValue(0.0, model),
				model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
				SPOT
		);

		final double knockInWideValue = interpolate1DAtSpot(
				knockInWide.getValue(0.0, model),
				model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
				SPOT
		);

		assertTrue(
				"Wider band must increase knock-out value.",
				knockOutWideValue + MONOTONICITY_TOL >= knockOutNarrowValue
		);

		assertTrue(
				"Wider band must decrease knock-in value.",
				knockInNarrowValue + MONOTONICITY_TOL >= knockInWideValue
		);
	}

	/*
	 * ============================================================
	 * CONVERGENCE
	 * ============================================================
	 */

	private void runBlackScholesConvergenceTest(
			final CallOrPut callOrPut,
			final DoubleBarrierType barrierType) {

		final double analyticValue = analyticDoubleBarrierValue(
				callOrPut,
				barrierType,
				LOWER_BARRIER,
				UPPER_BARRIER
		);

		final double coarseValue = priceBlackScholesDoubleBarrier(
				callOrPut,
				barrierType,
				COARSE_TIME_STEPS,
				COARSE_SPACE_STEPS
		);

		final double mediumValue = priceBlackScholesDoubleBarrier(
				callOrPut,
				barrierType,
				MEDIUM_TIME_STEPS,
				MEDIUM_SPACE_STEPS
		);

		final double fineValue = priceBlackScholesDoubleBarrier(
				callOrPut,
				barrierType,
				FINE_TIME_STEPS,
				FINE_SPACE_STEPS
		);

		final double coarseError = Math.abs(coarseValue - analyticValue);
		final double mediumError = Math.abs(mediumValue - analyticValue);
		final double fineError = Math.abs(fineValue - analyticValue);

		final double coarseTol =
				barrierType == DoubleBarrierType.KNOCK_IN
				? COARSE_ERROR_TOL_KNOCK_IN
				: COARSE_ERROR_TOL_KNOCK_OUT;

		final double mediumTol =
				barrierType == DoubleBarrierType.KNOCK_IN
				? MEDIUM_ERROR_TOL_KNOCK_IN
				: MEDIUM_ERROR_TOL_KNOCK_OUT;

		final double fineTol =
				barrierType == DoubleBarrierType.KNOCK_IN
				? FINE_ERROR_TOL_KNOCK_IN
				: FINE_ERROR_TOL_KNOCK_OUT;

		assertTrue(
				"Coarse grid error too large for " + barrierType + " " + callOrPut + ": " + coarseError,
				coarseError < coarseTol
		);

		assertTrue(
				"Medium grid error too large for " + barrierType + " " + callOrPut + ": " + mediumError,
				mediumError < mediumTol
		);

		assertTrue(
				"Fine grid error too large for " + barrierType + " " + callOrPut + ": " + fineError,
				fineError < fineTol
		);
		if(barrierType == DoubleBarrierType.KNOCK_OUT) {
			assertTrue(
					"Medium grid should not be worse than coarse grid by more than numerical noise.",
					mediumError <= coarseError + 1.0E-4
			);

			assertTrue(
					"Fine grid should not be worse than medium grid by more than numerical noise.",
					fineError <= mediumError + 1.0E-4
			);
		}
		else {
			assertTrue(
					"Fine KI grid should remain in the same error band as medium KI grid.",
					fineError <= mediumError + 5.0E-2
			);
		}
	}

	/*
	 * ============================================================
	 * PRICING HELPERS
	 * ============================================================
	 */

	private double priceBlackScholesDoubleBarrier(
			final CallOrPut callOrPut,
			final DoubleBarrierType barrierType,
			final int numberOfTimeSteps,
			final int numberOfSpaceSteps) {

		final FDMBlackScholesModel model = createBlackScholesModel(
				SPOT,
				numberOfTimeSteps,
				numberOfSpaceSteps
		);

		final DoubleBarrierOption option = new DoubleBarrierOption(
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				callOrPut,
				barrierType,
				new EuropeanExercise(MATURITY)
		);

		return interpolate1DAtSpot(
				option.getValue(0.0, model),
				model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
				SPOT
		);
	}

	private double analyticDoubleBarrierValue(
			final CallOrPut callOrPut,
			final DoubleBarrierType barrierType,
			final double lowerBarrier,
			final double upperBarrier) {

		final boolean isCall = callOrPut == CallOrPut.CALL;

		final BarrierOptions.DoubleBarrierType analyticBarrierType;
		if(barrierType == DoubleBarrierType.KNOCK_IN) {
			analyticBarrierType = BarrierOptions.DoubleBarrierType.KNOCK_IN;
		}
		else if(barrierType == DoubleBarrierType.KNOCK_OUT) {
			analyticBarrierType = BarrierOptions.DoubleBarrierType.KNOCK_OUT;
		}
		else {
			throw new IllegalArgumentException("Unsupported barrier type for analytic regression: " + barrierType);
		}

		return BarrierOptions.blackScholesDoubleBarrierOptionValue(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				BS_VOLATILITY,
				MATURITY,
				STRIKE,
				isCall,
				lowerBarrier,
				upperBarrier,
				analyticBarrierType,
				40
		);
	}

	/*
	 * ============================================================
	 * MODEL SETUPS
	 * ============================================================
	 */

	private FDMBlackScholesModel createBlackScholesModel(
			final double initialSpot,
			final int numberOfTimeSteps,
			final int numberOfSpaceSteps) {

		final Grid sGrid = new UniformGrid(numberOfSpaceSteps, SPACE_MIN_1D, SPACE_MAX_1D);

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

		return new FDMBlackScholesModel(
				initialSpot,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				BS_VOLATILITY,
				spaceTime
		);
	}

	private TwoDimensionalSetup createTwoDimensionalSetup(final ModelType modelType) {
		final Grid sGrid = new UniformGrid(SPACE_STEPS_S_2D, SPACE_MIN_2D, SPACE_MAX_2D);
		final Grid secondGrid = createSecondGrid(modelType);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						TIME_STEPS_2D,
						MATURITY / TIME_STEPS_2D
				);

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

	private Grid createSecondGrid(final ModelType modelType) {
		if(modelType == ModelType.HESTON) {
			final double vMax = Math.max(
					4.0 * HESTON_THETA_V,
					HESTON_INITIAL_VARIANCE + 4.0 * HESTON_XI * Math.sqrt(MATURITY)
			);
			return new UniformGrid(SPACE_STEPS_SECOND_2D, 0.0, vMax);
		}
		else if(modelType == ModelType.SABR) {
			final double alphaMax = Math.max(4.0 * SABR_INITIAL_ALPHA, 1.0);
			return new UniformGrid(SPACE_STEPS_SECOND_2D, 0.0, alphaMax);
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

		final int index = findGridIndex(sNodes, spot);
		if(index >= 0) {
			return values[index];
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

	private int findGridIndex(final double[] grid, final double x) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - x) < 1E-12) {
				return i;
			}
		}
		return -1;
	}

	private int flatten(final int iS, final int i2, final int numberOfSNodes) {
		return iS + i2 * numberOfSNodes;
	}

	/*
	 * ============================================================
	 * SMALL SETUP HOLDER
	 * ============================================================
	 */

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