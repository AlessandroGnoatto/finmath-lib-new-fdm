package net.finmath.finitedifference;


import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Assert;

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
 * Compares the finite difference price of a fixed-strike arithmetic Asian option under Black-Scholes
 * with the finmath Monte Carlo implementation.
 *
 * <p>Assumption: the averaging times in the Monte Carlo product coincide with the PDE time discretization
 * (up to the usual convention that MC averaging times typically exclude 0.0).</p>
 *
 * <p>The finite difference result is interpolated bilinearly on the lifted (S,I)-grid at (S0, I0=0).</p>
 */
public class AsianOptionBlackScholesFdmVsMonteCarloTest {

	@Test
	public void testAsianOptionFdmVsMonteCarlo() throws Exception {

		/*
		 * Model parameters
		 */
		final double spot = 100.0;
		final double riskFreeRate = 0.05;
		final double dividendYieldRate = 0.0;
		final double volatility = 0.30;

		/*
		 * Product parameters
		 */
		final double maturity = 2.0;
		final double strike = 95.0;

		final int numberOfTimeSteps = 100;
		final double dt = maturity / numberOfTimeSteps;

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, numberOfTimeSteps, dt);
		
		final int nS = 150; // grid points
		final double sMin = 0.0 * spot;
		final double sMax = 3 * spot;

		final Grid sGrid = new UniformGrid(nS - 1, sMin, sMax);

		final SpaceTimeDiscretization discretization = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				0.5,               // Crank-Nicolson
				new double[] { spot }
		);

		final FDMBlackScholesModel fdmModel = new FDMBlackScholesModel(
				spot,
				riskFreeRate,
				dividendYieldRate,
				volatility,
				discretization
		);

		final AsianOption fdmAsian = new AsianOption(maturity, strike);

		/*
		 * Get FD grid values at evaluation time (tau = T - t; evaluationTime = 0 => tau = T).
		 * The solver returns a vector over the lifted (S,I) grid, flattened with index k = i + j*nS.
		 */
		double tStart = System.currentTimeMillis();
		final double[] fdmValueVector = fdmAsian.getValue(0.0, fdmModel);
		double tEnd = System.currentTimeMillis();
		System.out.println("Computation required " + (tEnd - tStart) / 1000.0);
		
		/*
		 * Reconstruct the lifted (S,I) grids used by AsianOption (must match AsianOption's logic):
		 * - S grid is reused
		 * - I grid is UniformGrid with nI = nS points on [0, T * Smax]
		 */
		final double[] sNodes = sGrid.getGrid();
		final double iMax = maturity * sNodes[sNodes.length - 1];
		final int nI = sNodes.length * 2;
		final Grid iGrid = new UniformGrid(nI - 1, 0.0, iMax);
		final double[] iNodes = iGrid.getGrid();

		if (fdmValueVector.length != nS * nI) {
			throw new IllegalStateException(
					"Unexpected FD value vector length. Got " + fdmValueVector.length + ", expected " + (nS * nI) + "."
			);
		}

		/*
		 * Reshape into values[iS][iI] for bilinear interpolation.
		 */
		final double[][] valuesSI = new double[nS][nI];
		for (int j = 0; j < nI; j++) {
			for (int i = 0; i < nS; i++) {
				final int k = i + j * nS;
				valuesSI[i][j] = fdmValueVector[k];
			}
		}

		/*
		 * Interpolate at (S0, I0=0).
		 */
		final BiLinearInterpolation interp =
				new BiLinearInterpolation(sNodes, iNodes, valuesSI);
		final double valueFdm = interp.apply(spot, 0.0);
		System.out.println("Value FDM: " + valueFdm);
		/*
		 * Monte Carlo benchmark (finmath-lib)
		 */
		final int numberOfPaths = 10000;
		final int seed = 31415;

		final BlackScholesModel mcModel = new BlackScholesModel(spot, riskFreeRate, volatility);

		final BrownianMotionFromMersenneRandomNumbers brownianMotion =
				new BrownianMotionFromMersenneRandomNumbers(timeDiscretization, 1, numberOfPaths, seed);

		final MonteCarloProcessFromProcessModel process =
				new EulerSchemeFromProcessModel(mcModel, brownianMotion);

		final MonteCarloAssetModel mcSimulation =
				new MonteCarloAssetModel(mcModel, process);

		final TimeDiscretization averagingTimes = timeDiscretization;

		final net.finmath.montecarlo.assetderivativevaluation.products.AsianOption mcAsian =
				new net.finmath.montecarlo.assetderivativevaluation.products.AsianOption(maturity, strike, averagingTimes);

		final double valueMc = mcAsian.getValue(0.0, mcSimulation).getAverage();
		System.out.println("Value Monte Carlo: " + valueMc);
		/*
		 * Assertion: PDE (with a moderate grid) vs MC (statistical error).
		 * Tune tolerance depending on your chosen grid resolution and MC paths.
		 */
		final double tolerance = 2.5e-2;
		Assert.assertEquals(valueMc, valueFdm, tolerance);
	}
}