package net.finmath.finitedifference;

import org.junit.Assert;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.products.AsianOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.AsianStrike;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloAssetModel;
import net.finmath.montecarlo.assetderivativevaluation.models.BlackScholesModel;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.montecarlo.process.MonteCarloProcessFromProcessModel;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Compares finite-difference and Monte Carlo prices for arithmetic Asian options
 * under Black-Scholes, split by contract family:
 *
 * <ul>
 *   <li>fixed-strike call</li>
 *   <li>fixed-strike put</li>
 *   <li>floating-strike call</li>
 *   <li>floating-strike put</li>
 * </ul>
 *
 * <p>
 * The finite-difference result is interpolated bilinearly on the lifted (S,I)-grid.
 * For fixed strike the interpolation point is (S0, I0 = 0).
 * For floating strike the interpolation point uses the first positive I node, consistent
 * with the current product / solver behavior.
 * </p>
 */
public class AsianOptionBlackScholesFdmVsMonteCarloTest {

	private static final double SPOT = 100.0;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD_RATE = 0.0;
	private static final double VOLATILITY = 0.30;
	private static final double MATURITY = 2.0;

	private static final double[] STRIKES = new double[] { 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0 };

	private static final int NUMBER_OF_TIME_STEPS = 200;
	private static final int NUMBER_OF_PATHS = 20000;
	private static final int SEED = 31415;

	/*
	 * Current baseline PDE setup.
	 * Adjust these independently later if you want different grids per contract family.
	 */
	private static final int NS = 150;
	private static final double S_MIN = 0.0;
	private static final double S_MAX = 2.0 * SPOT;
	private static final int I_GRID_MULTIPLIER = 4;

	/*
	 * Tolerances.
	 * These can later be specialized by contract family if needed.
	 */
	private static final double MAX_RELATIVE_ERROR_TOLERANCE_FIXED_CALL = 0.20;
	private static final double AVERAGE_RELATIVE_ERROR_TOLERANCE_FIXED_CALL = 0.10;

	private static final double MAX_RELATIVE_ERROR_TOLERANCE_FIXED_PUT = 0.20;
	private static final double AVERAGE_RELATIVE_ERROR_TOLERANCE_FIXED_PUT = 0.10;

	private static final double MAX_RELATIVE_ERROR_TOLERANCE_FLOATING_CALL = 0.20;
	private static final double AVERAGE_RELATIVE_ERROR_TOLERANCE_FLOATING_CALL = 0.15;

	private static final double MAX_RELATIVE_ERROR_TOLERANCE_FLOATING_PUT = 0.20;
	private static final double AVERAGE_RELATIVE_ERROR_TOLERANCE_FLOATING_PUT = 0.15;

	@Test
	public void testAsianOptionFdmVsMonteCarloFixedStrikeCall() throws Exception {
		runFixedStrikeTest(
				CallOrPut.CALL,
				MAX_RELATIVE_ERROR_TOLERANCE_FIXED_CALL,
				AVERAGE_RELATIVE_ERROR_TOLERANCE_FIXED_CALL
		);
	}

	@Test
	public void testAsianOptionFdmVsMonteCarloFixedStrikePut() throws Exception {
		runFixedStrikeTest(
				CallOrPut.PUT,
				MAX_RELATIVE_ERROR_TOLERANCE_FIXED_PUT,
				AVERAGE_RELATIVE_ERROR_TOLERANCE_FIXED_PUT
		);
	}

	@Test
	public void testAsianOptionFdmVsMonteCarloFloatingStrikeCall() throws Exception {
		runFloatingStrikeTest(
				CallOrPut.CALL,
				MAX_RELATIVE_ERROR_TOLERANCE_FLOATING_CALL,
				AVERAGE_RELATIVE_ERROR_TOLERANCE_FLOATING_CALL
		);
	}

	@Test
	public void testAsianOptionFdmVsMonteCarloFloatingStrikePut() throws Exception {
		runFloatingStrikeTest(
				CallOrPut.PUT,
				MAX_RELATIVE_ERROR_TOLERANCE_FLOATING_PUT,
				AVERAGE_RELATIVE_ERROR_TOLERANCE_FLOATING_PUT
		);
	}

	private void runFixedStrikeTest(
			final CallOrPut callOrPut,
			final double maxRelativeErrorTolerance,
			final double averageRelativeErrorTolerance) throws Exception {

		final TestSetup setup = createSetup();

		double maxRelativeError = 0.0;
		double averageRelativeError = 0.0;
		int numberOfCases = 0;

		System.out.println("FIXED STRIKE " + callOrPut);
		System.out.println("Strike\tFDM\tMC\tAbsDiff\tRelDiff");

		for(final double strike : STRIKES) {

			final AsianOption fdmAsian = new AsianOption(
					null,
					MATURITY,
					strike,
					callOrPut,
					AsianStrike.FIXED_STRIKE,
					new EuropeanExercise(MATURITY)
			);

			final long tStart = System.currentTimeMillis();
			final double[] fdmValueVector = fdmAsian.getValue(0.0, setup.fdmModel);
			final long tEnd = System.currentTimeMillis();

			if(fdmValueVector.length != setup.nS * setup.nI) {
				throw new IllegalStateException(
						"Unexpected FD value vector length. Got " + fdmValueVector.length
						+ ", expected " + (setup.nS * setup.nI) + "."
				);
			}

			final double valueFdm = interpolateValue(
					fdmValueVector,
					setup.sNodes,
					setup.iNodes,
					setup.nS,
					setup.nI,
					SPOT,
					0.0
			);

			final net.finmath.montecarlo.assetderivativevaluation.myproducts.AsianOption mcAsian =
					new net.finmath.montecarlo.assetderivativevaluation.myproducts.AsianOption(
							MATURITY,
							strike,
							setup.averagingTimes,
							0,
							callOrPut,
							AsianStrike.FIXED_STRIKE
					);

			final double valueMc = mcAsian.getValue(0.0, setup.mcSimulation).getAverage();

			final double absDiff = Math.abs(valueFdm - valueMc);
			final double relativeError = absDiff / Math.max(Math.abs(valueMc), 1.0);

			maxRelativeError = Math.max(maxRelativeError, relativeError);
			averageRelativeError += relativeError;
			numberOfCases++;

			System.out.println(
					String.format(
							"%.2f\t%.8f\t%.8f\t%.8f\t%.4f%%   (FD %.3fs)",
							strike,
							valueFdm,
							valueMc,
							absDiff,
							100.0 * relativeError,
							(tEnd - tStart) / 1000.0
					)
			);
		}

		averageRelativeError /= numberOfCases;

		System.out.println(String.format("%s fixed max relative error: %.4f%%", callOrPut, 100.0 * maxRelativeError));
		System.out.println(String.format("%s fixed avg relative error: %.4f%%", callOrPut, 100.0 * averageRelativeError));

		Assert.assertTrue(
				callOrPut + " fixed maximum relative error too large: " + (100.0 * maxRelativeError) + "%",
				maxRelativeError < maxRelativeErrorTolerance
		);

		Assert.assertTrue(
				callOrPut + " fixed average relative error too large: " + (100.0 * averageRelativeError) + "%",
				averageRelativeError < averageRelativeErrorTolerance
		);
	}

	private void runFloatingStrikeTest(
			final CallOrPut callOrPut,
			final double maxRelativeErrorTolerance,
			final double averageRelativeErrorTolerance) throws Exception {

		final TestSetup setup = createSetup();

		System.out.println("FLOATING STRIKE " + callOrPut);
		System.out.println("FDM\tMC\tAbsDiff\tRelDiff");

		final AsianOption fdmAsian = new AsianOption(
				null,
				MATURITY,
				callOrPut,
				AsianStrike.FLOATING_STRIKE
		);

		final long tStart = System.currentTimeMillis();
		final double[] fdmValueVector = fdmAsian.getValue(0.0, setup.fdmModel);
		final long tEnd = System.currentTimeMillis();

		if(fdmValueVector.length != setup.nS * setup.nI) {
			throw new IllegalStateException(
					"Unexpected FD value vector length. Got " + fdmValueVector.length
					+ ", expected " + (setup.nS * setup.nI) + "."
			);
		}

		final double iForInterpolation = setup.iNodes[1];
		final double valueFdm = interpolateValue(
				fdmValueVector,
				setup.sNodes,
				setup.iNodes,
				setup.nS,
				setup.nI,
				SPOT,
				iForInterpolation
		);

		final net.finmath.montecarlo.assetderivativevaluation.myproducts.AsianOption mcAsian =
				new net.finmath.montecarlo.assetderivativevaluation.myproducts.AsianOption(
						MATURITY,
						Double.NaN,
						setup.averagingTimes,
						0,
						callOrPut,
						AsianStrike.FLOATING_STRIKE
				);

		final double valueMc = mcAsian.getValue(0.0, setup.mcSimulation).getAverage();

		final double absDiff = Math.abs(valueFdm - valueMc);
		final double relativeError = absDiff / Math.max(Math.abs(valueMc), 1.0);

		System.out.println(
				String.format(
						"%.8f\t%.8f\t%.8f\t%.4f%%   (FD %.3fs)",
						valueFdm,
						valueMc,
						absDiff,
						100.0 * relativeError,
						(tEnd - tStart) / 1000.0
				)
		);

		System.out.println(String.format("%s floating max relative error: %.4f%%", callOrPut, 100.0 * relativeError));
		System.out.println(String.format("%s floating avg relative error: %.4f%%", callOrPut, 100.0 * relativeError));

		Assert.assertTrue(
				callOrPut + " floating maximum relative error too large: " + (100.0 * relativeError) + "%",
				relativeError < maxRelativeErrorTolerance
		);

		Assert.assertTrue(
				callOrPut + " floating average relative error too large: " + (100.0 * relativeError) + "%",
				relativeError < averageRelativeErrorTolerance
		);
	}

	private TestSetup createSetup() {

		final double dt = MATURITY / NUMBER_OF_TIME_STEPS;

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, NUMBER_OF_TIME_STEPS, dt);

		final Grid sGrid = new UniformGrid(NS - 1, S_MIN, S_MAX);

		final SpaceTimeDiscretization discretization = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				0.5,
				new double[] { SPOT }
		);

		final FDMBlackScholesModel fdmModel = new FDMBlackScholesModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_RATE,
				VOLATILITY,
				discretization
		);

		final BlackScholesModel mcModel = new BlackScholesModel(SPOT, RISK_FREE_RATE, VOLATILITY);

		final BrownianMotionFromMersenneRandomNumbers brownianMotion =
				new BrownianMotionFromMersenneRandomNumbers(timeDiscretization, 1, NUMBER_OF_PATHS, SEED);

		final MonteCarloProcessFromProcessModel process =
				new EulerSchemeFromProcessModel(mcModel, brownianMotion);

		final MonteCarloAssetModel mcSimulation =
				new MonteCarloAssetModel(mcModel, process);

		final double[] averagingTimesArray = new double[NUMBER_OF_TIME_STEPS];
		for(int i = 0; i < NUMBER_OF_TIME_STEPS; i++) {
			averagingTimesArray[i] = timeDiscretization.getTime(i + 1);
		}
		final TimeDiscretization averagingTimes =
				new TimeDiscretizationFromArray(averagingTimesArray);

		final double[] sNodes = sGrid.getGrid();
		final double iMax = MATURITY * sNodes[sNodes.length - 1];
		final int nI = I_GRID_MULTIPLIER * sNodes.length;
		final Grid iGrid = new UniformGrid(nI - 1, 0.0, iMax);
		final double[] iNodes = iGrid.getGrid();

		return new TestSetup(
				timeDiscretization,
				averagingTimes,
				fdmModel,
				mcSimulation,
				sNodes,
				iNodes,
				NS,
				nI
		);
	}

	private double interpolateValue(
			final double[] fdmValueVector,
			final double[] sNodes,
			final double[] iNodes,
			final int nS,
			final int nI,
			final double s,
			final double i) {

		final double[][] valuesSI = new double[nS][nI];
		for(int j = 0; j < nI; j++) {
			for(int k = 0; k < nS; k++) {
				valuesSI[k][j] = fdmValueVector[k + j * nS];
			}
		}

		final BiLinearInterpolation interpolation = new BiLinearInterpolation(sNodes, iNodes, valuesSI);
		return interpolation.apply(s, i);
	}

	private static class TestSetup {
		private final TimeDiscretization timeDiscretization;
		private final TimeDiscretization averagingTimes;
		private final FDMBlackScholesModel fdmModel;
		private final MonteCarloAssetModel mcSimulation;
		private final double[] sNodes;
		private final double[] iNodes;
		private final int nS;
		private final int nI;

		private TestSetup(
				final TimeDiscretization timeDiscretization,
				final TimeDiscretization averagingTimes,
				final FDMBlackScholesModel fdmModel,
				final MonteCarloAssetModel mcSimulation,
				final double[] sNodes,
				final double[] iNodes,
				final int nS,
				final int nI) {
			this.timeDiscretization = timeDiscretization;
			this.averagingTimes = averagingTimes;
			this.fdmModel = fdmModel;
			this.mcSimulation = mcSimulation;
			this.sNodes = sNodes;
			this.iNodes = iNodes;
			this.nS = nS;
			this.nI = nI;
		}
	}
}