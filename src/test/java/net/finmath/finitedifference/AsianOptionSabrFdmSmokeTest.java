package net.finmath.finitedifference;

import org.junit.Assert;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.products.AsianOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.AsianStrike;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Smoke / sanity test for arithmetic Asian options under SABR.
 *
 * <p>
 * This test checks that the 3D lifted PDE solver:
 * </p>
 * <ul>
 *   <li>runs without exception,</li>
 *   <li>returns finite values,</li>
 *   <li>returns a finite and non-negative interpolated price at the initial state,</li>
 *   <li>preserves basic strike monotonicity for calls.</li>
 * </ul>
 *
 * <p>
 * We do not require pointwise non-negativity on the whole 3D grid, since the
 * current ADI discretization may exhibit tiny local undershoots.
 * </p>
 *
 * <p>
 * Interpolation is trilinear on the lifted (S, alpha, I)-grid.
 * The pricing point is (S0, alpha0, I0 = 0).
 * </p>
 */
public class AsianOptionSabrFdmSmokeTest {

	private static final double SPOT = 100.0;
	private static final double INITIAL_ALPHA = 0.20;

	private static final double RISK_FREE_RATE = 0.02;
	private static final double DIVIDEND_YIELD_RATE = 0.01;

	private static final double BETA = 0.50;
	private static final double NU = 0.40;
	private static final double RHO = -0.30;

	private static final double MATURITY = 1.0;

	private static final double[] STRIKES = new double[] { 80.0, 100.0, 120.0 };

	private static final int NUMBER_OF_TIME_STEPS = 40;

	private static final int NS = 40;
	private static final int NALPHA = 20;
	private static final int I_GRID_MULTIPLIER = 4;

	private static final double S_MIN = 0.0;
	private static final double S_MAX = 2.0 * SPOT;

	private static final double ALPHA_MIN = 0.0;
	private static final double ALPHA_MAX = 1.0;

	@Test
	public void testAsianOptionSabrFixedStrikeCallSmoke() throws Exception {

		final TestSetup setup = createSetup();

		double previousValue = Double.POSITIVE_INFINITY;

		System.out.println("SABR ASIAN FIXED STRIKE CALL");
		System.out.println("Strike\tFDM\tGridMin\tGridMax");

		for(final double strike : STRIKES) {

			final AsianOption option = new AsianOption(
					null,
					MATURITY,
					strike,
					CallOrPut.CALL,
					AsianStrike.FIXED_STRIKE,
					new EuropeanExercise(MATURITY)
			);

			final double[] values = option.getValue(0.0, setup.fdmModel);

			Assert.assertEquals(
					"Unexpected FD vector length.",
					setup.nS * setup.nAlpha * setup.nI,
					values.length
			);

			double minValue = Double.POSITIVE_INFINITY;
			double maxValue = Double.NEGATIVE_INFINITY;

			for(final double value : values) {
				Assert.assertTrue("Non-finite PDE value detected.", Double.isFinite(value));
				minValue = Math.min(minValue, value);
				maxValue = Math.max(maxValue, value);
			}

			final double price = interpolateValue(
					values,
					setup.sNodes,
					setup.alphaNodes,
					setup.iNodes,
					setup.nS,
					setup.nAlpha,
					setup.nI,
					SPOT,
					INITIAL_ALPHA,
					0.0
			);

			System.out.println(String.format("%.2f\t%.8f\t%.8e\t%.8e", strike, price, minValue, maxValue));

			Assert.assertTrue("Interpolated price must be finite.", Double.isFinite(price));
			Assert.assertTrue("Interpolated price must be non-negative.", price >= -1E-10);

			Assert.assertTrue(
					"Call price should be non-increasing in strike.",
					price <= previousValue + 1E-8
			);

			previousValue = price;
		}
	}

	@Test
	public void testAsianOptionSabrFixedStrikePutSmoke() throws Exception {

		final TestSetup setup = createSetup();

		System.out.println("SABR ASIAN FIXED STRIKE PUT");
		System.out.println("Strike\tFDM\tGridMin\tGridMax");

		for(final double strike : STRIKES) {

			final AsianOption option = new AsianOption(
					null,
					MATURITY,
					strike,
					CallOrPut.PUT,
					AsianStrike.FIXED_STRIKE,
					new EuropeanExercise(MATURITY)
			);

			final double[] values = option.getValue(0.0, setup.fdmModel);

			Assert.assertEquals(
					"Unexpected FD vector length.",
					setup.nS * setup.nAlpha * setup.nI,
					values.length
			);

			double minValue = Double.POSITIVE_INFINITY;
			double maxValue = Double.NEGATIVE_INFINITY;

			for(final double value : values) {
				Assert.assertTrue("Non-finite PDE value detected.", Double.isFinite(value));
				minValue = Math.min(minValue, value);
				maxValue = Math.max(maxValue, value);
			}

			final double price = interpolateValue(
					values,
					setup.sNodes,
					setup.alphaNodes,
					setup.iNodes,
					setup.nS,
					setup.nAlpha,
					setup.nI,
					SPOT,
					INITIAL_ALPHA,
					0.0
			);

			System.out.println(String.format("%.2f\t%.8f\t%.8e\t%.8e", strike, price, minValue, maxValue));

			Assert.assertTrue("Interpolated price must be finite.", Double.isFinite(price));
			Assert.assertTrue("Interpolated price should be non-negative.", price >= -1E-10);
		}
	}

	private TestSetup createSetup() {

		final double dt = MATURITY / NUMBER_OF_TIME_STEPS;

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, NUMBER_OF_TIME_STEPS, dt);

		final Grid sGrid = new UniformGrid(NS - 1, S_MIN, S_MAX);
		final Grid alphaGrid = new UniformGrid(NALPHA - 1, ALPHA_MIN, ALPHA_MAX);

		final SpaceTimeDiscretization discretization = new SpaceTimeDiscretization(
				new Grid[] { sGrid, alphaGrid },
				timeDiscretization,
				0.5,
				new double[] { SPOT, INITIAL_ALPHA }
		);

		final FDMSabrModel fdmModel = new FDMSabrModel(
				SPOT,
				INITIAL_ALPHA,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_RATE,
				BETA,
				NU,
				RHO,
				discretization
		);

		final double[] sNodes = sGrid.getGrid();
		final double[] alphaNodes = alphaGrid.getGrid();

		final double iMax = MATURITY * sNodes[sNodes.length - 1];
		final int nI = I_GRID_MULTIPLIER * sNodes.length;
		final Grid iGrid = new UniformGrid(nI - 1, 0.0, iMax);
		final double[] iNodes = iGrid.getGrid();

		return new TestSetup(
				fdmModel,
				sNodes,
				alphaNodes,
				iNodes,
				NS,
				NALPHA,
				nI
		);
	}

	private double interpolateValue(
			final double[] valueVector,
			final double[] sNodes,
			final double[] alphaNodes,
			final double[] iNodes,
			final int nS,
			final int nAlpha,
			final int nI,
			final double s,
			final double alpha,
			final double i) {

		final int iS = findLeftIndex(sNodes, s);
		final int iA = findLeftIndex(alphaNodes, alpha);
		final int iI = findLeftIndex(iNodes, i);

		final double s0 = sNodes[iS];
		final double s1 = sNodes[iS + 1];
		final double a0 = alphaNodes[iA];
		final double a1 = alphaNodes[iA + 1];
		final double i0 = iNodes[iI];
		final double i1 = iNodes[iI + 1];

		final double ws = (s - s0) / (s1 - s0);
		final double wa = (alpha - a0) / (a1 - a0);
		final double wi = (i - i0) / (i1 - i0);

		final double c000 = valueVector[flatten(iS,     iA,     iI,     nS, nAlpha)];
		final double c100 = valueVector[flatten(iS + 1, iA,     iI,     nS, nAlpha)];
		final double c010 = valueVector[flatten(iS,     iA + 1, iI,     nS, nAlpha)];
		final double c110 = valueVector[flatten(iS + 1, iA + 1, iI,     nS, nAlpha)];
		final double c001 = valueVector[flatten(iS,     iA,     iI + 1, nS, nAlpha)];
		final double c101 = valueVector[flatten(iS + 1, iA,     iI + 1, nS, nAlpha)];
		final double c011 = valueVector[flatten(iS,     iA + 1, iI + 1, nS, nAlpha)];
		final double c111 = valueVector[flatten(iS + 1, iA + 1, iI + 1, nS, nAlpha)];

		return
				(1.0 - ws) * (1.0 - wa) * (1.0 - wi) * c000
				+ ws         * (1.0 - wa) * (1.0 - wi) * c100
				+ (1.0 - ws) * wa         * (1.0 - wi) * c010
				+ ws         * wa         * (1.0 - wi) * c110
				+ (1.0 - ws) * (1.0 - wa) * wi         * c001
				+ ws         * (1.0 - wa) * wi         * c101
				+ (1.0 - ws) * wa         * wi         * c011
				+ ws         * wa         * wi         * c111;
	}

	private int findLeftIndex(final double[] grid, final double x) {
		if(x <= grid[0]) {
			return 0;
		}
		if(x >= grid[grid.length - 1]) {
			return grid.length - 2;
		}

		for(int i = 0; i < grid.length - 1; i++) {
			if(x >= grid[i] && x <= grid[i + 1]) {
				return i;
			}
		}

		throw new IllegalArgumentException("Point outside grid.");
	}

	private int flatten(
			final int iS,
			final int iA,
			final int iI,
			final int nS,
			final int nAlpha) {
		return iS + nS * (iA + nAlpha * iI);
	}

	private static class TestSetup {
		private final FDMSabrModel fdmModel;
		private final double[] sNodes;
		private final double[] alphaNodes;
		private final double[] iNodes;
		private final int nS;
		private final int nAlpha;
		private final int nI;

		private TestSetup(
				final FDMSabrModel fdmModel,
				final double[] sNodes,
				final double[] alphaNodes,
				final double[] iNodes,
				final int nS,
				final int nAlpha,
				final int nI) {
			this.fdmModel = fdmModel;
			this.sNodes = sNodes;
			this.alphaNodes = alphaNodes;
			this.iNodes = iNodes;
			this.nS = nS;
			this.nAlpha = nAlpha;
			this.nI = nI;
		}
	}
}