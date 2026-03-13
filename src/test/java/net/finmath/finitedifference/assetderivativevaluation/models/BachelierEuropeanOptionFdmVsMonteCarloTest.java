package net.finmath.finitedifference.assetderivativevaluation.models;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.RationalFunctionInterpolation;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloAssetModel;
import net.finmath.montecarlo.assetderivativevaluation.products.AbstractAssetMonteCarloProduct;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Compares a European call option price under a Bachelier (normal) model across:
 * <ul>
 *   <li>finite difference (this project),</li>
 *   <li>finmath Monte Carlo.</li>
 * </ul>
 *
 * <p>
 * The finite difference solver returns values on the full S-grid; the test extracts the price at {@code S0}
 * via finmath's {@link LinearInterpolation}.
 * </p>
 */
public class BachelierEuropeanOptionFdmVsMonteCarloTest {

	@Test
	public void testEuropeanCallBachelierFiniteDifferenceVsMonteCarlo() throws Exception {

		// --- Contract ---
		final double maturity = 1.0;
		final double strike = 100.0;

		// --- Model parameters ---
		final double s0 = 100.0;

		/*
		 * Use r=q=0 to avoid any ambiguity across model conventions
		 * (drift specification differs across some Bachelier implementations).
		 */
		final double r = 0.0;
		final double q = 0.0;

		// Normal volatility (units of underlying per sqrt(year))
		final double sigmaN = 10.0;

		// --- Curves for the FD model ---
		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", r);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", q);

		// --- FD grid (1D: S) + time discretization in time-to-maturity ---
		final int nTimeSteps = 50;
		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, nTimeSteps, maturity / nTimeSteps);

		// Range: +/- 6 stdev around S0
		final double stdev = sigmaN * Math.sqrt(maturity);
		final double sMin = s0 - 6.0 * stdev;
		final double sMax = s0 + 6.0 * stdev;

		/*
		 * Use an even number of steps so the midpoint is exactly S0
		 * for a symmetric uniform grid: nodes = steps+1, midpoint index = steps/2.
		 */
		final int nStepsS = 200;
		final Grid sGrid = new UniformGrid(nStepsS, sMin, sMax);

		final double theta = 0.5; // Crank-Nicolson
		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				theta,
				new double[] { s0 }
		);

		// --- FD model + product ---
		final FDMBachelierModel fdmModel = new FDMBachelierModel(
				s0,
				riskFreeCurve,
				dividendCurve,
				sigmaN,
				spaceTime
		);

		final EuropeanOption fdmProduct = new EuropeanOption(maturity, strike);

		// FD returns values on the full grid (including boundaries)
		final double[] fdValuesOnGrid = fdmProduct.getValue(0.0, fdmModel);
		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();

		final RationalFunctionInterpolation interp = new RationalFunctionInterpolation(sNodes, fdValuesOnGrid);
		final double fdPrice = interp.getValue(s0);

		// --- finmath Monte Carlo price ---
		final int numberOfPaths = 50_000;
		final int seed = 31415;

		final TimeDiscretization mcTimes = new TimeDiscretizationFromArray(0.0, 100, maturity / 100.0);

		final BrownianMotionFromMersenneRandomNumbers brownianMotion =
				new BrownianMotionFromMersenneRandomNumbers(mcTimes, 1, numberOfPaths, seed);

		final net.finmath.montecarlo.assetderivativevaluation.models.BachelierModel mcModel =
				new net.finmath.montecarlo.assetderivativevaluation.models.BachelierModel(s0, r, sigmaN);

		final EulerSchemeFromProcessModel process = new EulerSchemeFromProcessModel(mcModel, brownianMotion);
		final MonteCarloAssetModel mcSimulation = new MonteCarloAssetModel(process);

		final AbstractAssetMonteCarloProduct mcProduct =
				new net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption(maturity, strike);

		final double mcPrice = mcProduct.getValue(mcSimulation);

		// --- Checks / reporting ---
		System.out.println("FD price = " + fdPrice);
		System.out.println("MC price = " + mcPrice);

		assertTrue(fdPrice >= 0.0);
		assertTrue(mcPrice >= 0.0);

		// MC statistical error + FD grid error (tune if you change grid / paths)
		final double tolerance = 0.35;
		assertEquals("FD vs MC", mcPrice, fdPrice, tolerance);
	}

	private static DiscountCurve createFlatDiscountCurve(final String name, final double rate) {
		final double[] times = new double[] { 0.0, 1.0 };
		final double[] zeroRates = new double[] { rate, rate };
		final InterpolationMethod interpolationMethod = InterpolationMethod.LINEAR;
		final InterpolationEntity interpolationEntity = InterpolationEntity.VALUE;
		final ExtrapolationMethod extrapolationMethod = ExtrapolationMethod.CONSTANT;

		return DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
				name,
				LocalDate.of(2010, 8, 1),
				times,
				zeroRates,
				interpolationMethod,
				extrapolationMethod,
				interpolationEntity
		);
	}
}