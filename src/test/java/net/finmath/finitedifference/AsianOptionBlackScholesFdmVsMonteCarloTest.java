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
 * under Black-Scholes across multiple strikes.
 *
 * <p>
 * The finite-difference result is interpolated bilinearly on the lifted (S,I)-grid
 * at (S0, I0=0).
 * </p>
 */
public class AsianOptionBlackScholesFdmVsMonteCarloTest {

	@Test
	public void testAsianOptionFdmVsMonteCarloAcrossStrikes() throws Exception {

		/*
		 * Model parameters
		 */
		final double spot = 100.0;
		final double riskFreeRate = 0.05;
		final double dividendYieldRate = 0.0;
		final double volatility = 0.30;
		final double maturity = 2.0;

		/*
		 * Strike sweep for fixed-strike options
		 */
		final double[] strikes = new double[] { 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0 };

		/*
		 * Shared time discretization
		 */
		final int numberOfTimeSteps = 100;
		final double dt = maturity / numberOfTimeSteps;

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, numberOfTimeSteps, dt);

		/*
		 * PDE setup
		 */
		final int nS = 250;
		final double sMin = 0.0;
		final double sMax = 4.0 * spot;

		final Grid sGrid = new UniformGrid(nS - 1, sMin, sMax);

		final SpaceTimeDiscretization discretization = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				0.5,
				new double[] { spot }
		);

		final FDMBlackScholesModel fdmModel = new FDMBlackScholesModel(
				spot,
				riskFreeRate,
				dividendYieldRate,
				volatility,
				discretization
		);

		/*
		 * MC setup
		 */
		final int numberOfPaths = 20000;
		final int seed = 31415;

		final BlackScholesModel mcModel = new BlackScholesModel(spot, riskFreeRate, volatility);

		final BrownianMotionFromMersenneRandomNumbers brownianMotion =
				new BrownianMotionFromMersenneRandomNumbers(timeDiscretization, 1, numberOfPaths, seed);

		final MonteCarloProcessFromProcessModel process =
				new EulerSchemeFromProcessModel(mcModel, brownianMotion);

		final MonteCarloAssetModel mcSimulation =
				new MonteCarloAssetModel(mcModel, process);

		/*
		 * Averaging dates for the MC benchmark:
		 * exclude t=0.0 and reuse exact times from the simulation grid.
		 */
		final double[] averagingTimesArray = new double[numberOfTimeSteps];
		for(int i = 0; i < numberOfTimeSteps; i++) {
			averagingTimesArray[i] = timeDiscretization.getTime(i+1);
		}
		final TimeDiscretization averagingTimes =
				new TimeDiscretizationFromArray(averagingTimesArray);

		/*
		 * Lifted FD grid reconstruction.
		 * Must match the product logic.
		 */
		final double[] sNodes = sGrid.getGrid();
		final double iMax = maturity * sNodes[sNodes.length - 1];
		final int nI = 2 * sNodes.length;
		final Grid iGrid = new UniformGrid(nI - 1, 0.0, iMax);
		final double[] iNodes = iGrid.getGrid();

		/*
		 * Fixed-strike tests
		 */
		double maxRelativeErrorFixed = 0.0;
		double averageRelativeErrorFixed = 0.0;
		int numberOfFixedCases = 0;

		System.out.println("FIXED STRIKE");
		System.out.println("Type\tStrike\tFDM\tMC\tAbsDiff\tRelDiff");

		for(final double strike : strikes) {
			for(final CallOrPut callOrPut : new CallOrPut[] { CallOrPut.CALL, CallOrPut.PUT }) {

				final AsianOption fdmAsian = new AsianOption(
						null,
						maturity,
						strike,
						callOrPut,
						AsianStrike.FIXED_STRIKE,
						new EuropeanExercise(maturity)
				);

				final long tStart = System.currentTimeMillis();
				final double[] fdmValueVector = fdmAsian.getValue(0.0, fdmModel);
				final long tEnd = System.currentTimeMillis();

				if(fdmValueVector.length != nS * nI) {
					throw new IllegalStateException(
							"Unexpected FD value vector length. Got " + fdmValueVector.length
							+ ", expected " + (nS * nI) + "."
					);
				}

				final double[][] valuesSI = new double[nS][nI];
				for(int j = 0; j < nI; j++) {
					for(int i = 0; i < nS; i++) {
						valuesSI[i][j] = fdmValueVector[i + j * nS];
					}
				}

				final BiLinearInterpolation interp = new BiLinearInterpolation(sNodes, iNodes, valuesSI);
				final double valueFdm = interp.apply(spot, 0.0);

				final net.finmath.montecarlo.assetderivativevaluation.myproducts.AsianOption mcAsian =
						new net.finmath.montecarlo.assetderivativevaluation.myproducts.AsianOption(
								maturity,
								strike,
								averagingTimes,
								0,
								callOrPut,
								AsianStrike.FIXED_STRIKE
						);

				final double valueMc = mcAsian.getValue(0.0, mcSimulation).getAverage();

				final double absDiff = Math.abs(valueFdm - valueMc);
				final double relativeError = absDiff / Math.max(Math.abs(valueMc), 1.0);

				maxRelativeErrorFixed = Math.max(maxRelativeErrorFixed, relativeError);
				averageRelativeErrorFixed += relativeError;
				numberOfFixedCases++;

				System.out.println(
						String.format(
								"%s\t%.2f\t%.8f\t%.8f\t%.8f\t%.4f%%   (FD %.3fs)",
								callOrPut,
								strike,
								valueFdm,
								valueMc,
								absDiff,
								100.0 * relativeError,
								(tEnd - tStart) / 1000.0
						)
				);
			}
		}

		averageRelativeErrorFixed /= numberOfFixedCases;

		System.out.println(String.format("Fixed max relative error: %.4f%%", 100.0 * maxRelativeErrorFixed));
		System.out.println(String.format("Fixed avg relative error: %.4f%%", 100.0 * averageRelativeErrorFixed));

		/*
		 * Floating-strike tests
		 */
		double maxRelativeErrorFloating = 0.0;
		double averageRelativeErrorFloating = 0.0;
		int numberOfFloatingCases = 0;

		System.out.println("FLOATING STRIKE");
		System.out.println("Type\tFDM\tMC\tAbsDiff\tRelDiff");

		for(final CallOrPut callOrPut : new CallOrPut[] { CallOrPut.CALL, CallOrPut.PUT }) {

			final AsianOption fdmAsian = new AsianOption(
					null,
					maturity,
					callOrPut,
					AsianStrike.FLOATING_STRIKE
			);

			final long tStart = System.currentTimeMillis();
			final double[] fdmValueVector = fdmAsian.getValue(0.0, fdmModel);
			final long tEnd = System.currentTimeMillis();

			if(fdmValueVector.length != nS * nI) {
				throw new IllegalStateException(
						"Unexpected FD value vector length. Got " + fdmValueVector.length
						+ ", expected " + (nS * nI) + "."
				);
			}

			final double[][] valuesSI = new double[nS][nI];
			for(int j = 0; j < nI; j++) {
				for(int i = 0; i < nS; i++) {
					valuesSI[i][j] = fdmValueVector[i + j * nS];
				}
			}

			final BiLinearInterpolation interp = new BiLinearInterpolation(sNodes, iNodes, valuesSI);
			
			final double iForInterpolation = iNodes[1];
			final double valueFdm = interp.apply(spot, iForInterpolation);
			//final double valueFdm = interp.apply(spot, 0.0);

			final net.finmath.montecarlo.assetderivativevaluation.myproducts.AsianOption mcAsian =
					new net.finmath.montecarlo.assetderivativevaluation.myproducts.AsianOption(
							maturity,
							Double.NaN,
							averagingTimes,
							0,
							callOrPut,
							AsianStrike.FLOATING_STRIKE
					);

			final double valueMc = mcAsian.getValue(0.0, mcSimulation).getAverage();

			final double absDiff = Math.abs(valueFdm - valueMc);
			final double relativeError = absDiff / Math.max(Math.abs(valueMc), 1.0);

			maxRelativeErrorFloating = Math.max(maxRelativeErrorFloating, relativeError);
			averageRelativeErrorFloating += relativeError;
			numberOfFloatingCases++;

			System.out.println(
					String.format(
							"%s\t%.8f\t%.8f\t%.8f\t%.4f%%   (FD %.3fs)",
							callOrPut,
							valueFdm,
							valueMc,
							absDiff,
							100.0 * relativeError,
							(tEnd - tStart) / 1000.0
					)
			);
		}

		averageRelativeErrorFloating /= numberOfFloatingCases;

		System.out.println(String.format("Floating max relative error: %.4f%%", 100.0 * maxRelativeErrorFloating));
		System.out.println(String.format("Floating avg relative error: %.4f%%", 100.0 * averageRelativeErrorFloating));

		/*
		 * Suggested thresholds for the current development stage.
		 * Tighten these once the grid / boundaries are finalized.
		 */
		final double maxRelativeErrorToleranceFixed = 0.22;      // 22%
		final double averageRelativeErrorToleranceFixed = 0.10;  // 10%

		final double maxRelativeErrorToleranceFloating = 0.20;      // 20%
		final double averageRelativeErrorToleranceFloating = 0.12;  // 12%

		Assert.assertTrue(
				"Maximum fixed-strike relative error too large: " + (100.0 * maxRelativeErrorFixed) + "%",
				maxRelativeErrorFixed < maxRelativeErrorToleranceFixed
		);

		Assert.assertTrue(
				"Average fixed-strike relative error too large: " + (100.0 * averageRelativeErrorFixed) + "%",
				averageRelativeErrorFixed < averageRelativeErrorToleranceFixed
		);

		Assert.assertTrue(
				"Maximum floating-strike relative error too large: " + (100.0 * maxRelativeErrorFloating) + "%",
				maxRelativeErrorFloating < maxRelativeErrorToleranceFloating
		);

		Assert.assertTrue(
				"Average floating-strike relative error too large: " + (100.0 * averageRelativeErrorFloating) + "%",
				averageRelativeErrorFloating < averageRelativeErrorToleranceFloating
		);
	}
}