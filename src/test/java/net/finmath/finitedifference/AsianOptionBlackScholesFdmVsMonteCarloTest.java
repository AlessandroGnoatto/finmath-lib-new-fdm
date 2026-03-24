package net.finmath.finitedifference;

import org.junit.Assert;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.products.AsianOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloAssetModel;
import net.finmath.montecarlo.assetderivativevaluation.models.BlackScholesModel;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.montecarlo.process.MonteCarloProcessFromProcessModel;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Compares finite-difference and Monte Carlo prices for fixed-strike arithmetic Asian calls
 * under Black-Scholes across multiple strikes.
 *
 * <p>The finite-difference result is interpolated bilinearly on the lifted (S,I)-grid
 * at (S0, I0=0).</p>
 *
 * <p>The test checks relative differences across a strike sweep, which is more robust
 * than a single-strike equality test while the PDE implementation is still being tuned.</p>
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
		 * Strike sweep
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
		final int nS = 200;
		final double sMin = 0.0;
		final double sMax = 3.0 * spot;

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

		final TimeDiscretization averagingTimes = timeDiscretization;

		/*
		 * Lifted FD grid reconstruction.
		 * Must match the product logic.
		 */
		final double[] sNodes = sGrid.getGrid();
		final double iMax = maturity * sNodes[sNodes.length - 1];
		final int nI = 2 * sNodes.length;
		final Grid iGrid = new UniformGrid(nI - 1, 0.0, iMax);
		final double[] iNodes = iGrid.getGrid();

		double maxRelativeError = 0.0;
		double averageRelativeError = 0.0;

		System.out.println("Strike\tFDM\tMC\tAbsDiff\tRelDiff");

		for(final double strike : strikes) {

			final AsianOption fdmAsian = new AsianOption(maturity, strike);

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

			final net.finmath.montecarlo.assetderivativevaluation.products.AsianOption mcAsian =
					new net.finmath.montecarlo.assetderivativevaluation.products.AsianOption(
							maturity, strike, averagingTimes
					);

			final double valueMc = mcAsian.getValue(0.0, mcSimulation).getAverage();

			final double absDiff = Math.abs(valueFdm - valueMc);

			/*
			 * Use a floor to avoid meaningless huge percentages for very small MC values.
			 */
			final double relativeError = absDiff / Math.max(Math.abs(valueMc), 1.0);

			maxRelativeError = Math.max(maxRelativeError, relativeError);
			averageRelativeError += relativeError;

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

		averageRelativeError /= strikes.length;

		System.out.println(String.format("Max relative error: %.4f%%", 100.0 * maxRelativeError));
		System.out.println(String.format("Avg relative error: %.4f%%", 100.0 * averageRelativeError));

		/*
		 * Suggested thresholds for the current development stage.
		 * Tighten these once the grid / boundaries are finalized.
		 */
		final double maxRelativeErrorTolerance = 0.10;   // 10%
		final double averageRelativeErrorTolerance = 0.06; // 6%

		Assert.assertTrue(
				"Maximum relative error too large: " + (100.0 * maxRelativeError) + "%",
				maxRelativeError < maxRelativeErrorTolerance
		);

		Assert.assertTrue(
				"Average relative error too large: " + (100.0 * averageRelativeError) + "%",
				averageRelativeError < averageRelativeErrorTolerance
		);
	}
}