package net.finmath.finitedifference;

import org.junit.Assert;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
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
 * Debug test for the Heston fixed-strike arithmetic Asian call.
 *
 * <p>
 * This test is intentionally small and diagnostic-oriented. It prints:
 * </p>
 * <ul>
 *   <li>interpolated price at the initial state,</li>
 *   <li>global minimum and maximum over the full 3D grid,</li>
 *   <li>the grid locations where min and max are attained.</li>
 * </ul>
 *
 * <p>
 * The purpose is to localize instability regions in the lifted (S, v, I) grid.
 * </p>
 */
public class AsianOptionHestonFixedStrikeCallDebugTest {

	private static final double SPOT = 100.0;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD_RATE = 0.0;

	private static final double VOLATILITY = 0.30;
	private static final double VOLATILITY_SQUARED = VOLATILITY * VOLATILITY;

	private static final double VOLATILITY_OF_VOLATILITY = 0.25;
	private static final double CORRELATION = -0.5;
	private static final double MEAN_REVERSION = 1.0;
	private static final double LONG_RUN_VARIANCE = VOLATILITY_SQUARED;

	private static final double MATURITY = 2.0;
	private static final double STRIKE = 100.0;

	/*
	 * Smaller setup for fast debugging.
	 */
	private static final int NUMBER_OF_TIME_STEPS = 40;

	private static final int NS = 40;
	private static final int NV = 20;
	private static final int I_GRID_MULTIPLIER = 4;

	private static final double S_MIN = 0.0;
	private static final double S_MAX = 2.0 * SPOT;

	private static final double V_MIN = 0.0;
	private static final double V_MAX = 4.0 * VOLATILITY_SQUARED;

	@Test
	public void testAsianOptionHestonFixedStrikeCallDebug() throws Exception {

		final TestSetup setup = createSetup();

		final AsianOption option = new AsianOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.CALL,
				AsianStrike.FIXED_STRIKE,
				new EuropeanExercise(MATURITY)
		);

		final long tStart = System.currentTimeMillis();
		final double[] values = option.getValue(0.0, setup.fdmModel);
		final long tEnd = System.currentTimeMillis();

		Assert.assertEquals(
				"Unexpected FD vector length.",
				setup.nS * setup.nV * setup.nI,
				values.length
		);

		double minValue = Double.POSITIVE_INFINITY;
		double maxValue = Double.NEGATIVE_INFINITY;

		int minIS = -1;
		int minIV = -1;
		int minII = -1;

		int maxIS = -1;
		int maxIV = -1;
		int maxII = -1;

		for(int iI = 0; iI < setup.nI; iI++) {
			for(int iV = 0; iV < setup.nV; iV++) {
				for(int iS = 0; iS < setup.nS; iS++) {

					final int index = flatten(iS, iV, iI, setup.nS, setup.nV);
					final double value = values[index];

					Assert.assertTrue("Non-finite PDE value detected.", Double.isFinite(value));

					if(value < minValue) {
						minValue = value;
						minIS = iS;
						minIV = iV;
						minII = iI;
					}

					if(value > maxValue) {
						maxValue = value;
						maxIS = iS;
						maxIV = iV;
						maxII = iI;
					}
				}
			}
		}

		final double price = interpolateValue(
				values,
				setup.sNodes,
				setup.vNodes,
				setup.iNodes,
				setup.nS,
				setup.nV,
				setup.nI,
				SPOT,
				VOLATILITY_SQUARED,
				0.0
		);

		System.out.println("HESTON ASIAN FIXED STRIKE CALL DEBUG");
		System.out.println(String.format("Runtime: %.3f s", (tEnd - tStart) / 1000.0));
		System.out.println(String.format("Interpolated price at (S0,v0,I0) = %.10f", price));
		System.out.println(String.format("Grid min = %.10f", minValue));
		System.out.println(String.format("Grid max = %.10f", maxValue));

		System.out.println(String.format(
				"Min located at indices (iS,iV,iI) = (%d,%d,%d), state = (%.10f, %.10f, %.10f)",
				minIS,
				minIV,
				minII,
				setup.sNodes[minIS],
				setup.vNodes[minIV],
				setup.iNodes[minII]
		));

		System.out.println(String.format(
				"Max located at indices (iS,iV,iI) = (%d,%d,%d), state = (%.10f, %.10f, %.10f)",
				maxIS,
				maxIV,
				maxII,
				setup.sNodes[maxIS],
				setup.vNodes[maxIV],
				setup.iNodes[maxII]
		));

		/*
		 * Also inspect a few slices around the minimum location.
		 */
		printLocalNeighborhood(values, setup, minIS, minIV, minII);

		Assert.assertTrue("Interpolated price must be finite.", Double.isFinite(price));
		Assert.assertTrue("Interpolated price should be non-negative.", price >= -1E-10);
	}

	private TestSetup createSetup() {

		final double dt = MATURITY / NUMBER_OF_TIME_STEPS;

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, NUMBER_OF_TIME_STEPS, dt);

		final Grid sGrid = new UniformGrid(NS - 1, S_MIN, S_MAX);
		final Grid vGrid = new UniformGrid(NV - 1, V_MIN, V_MAX);

		final SpaceTimeDiscretization discretization = new SpaceTimeDiscretization(
				new Grid[] { sGrid, vGrid },
				timeDiscretization,
				0.5,
				new double[] { SPOT, VOLATILITY_SQUARED }
		);

		final FDMHestonModel fdmModel = new FDMHestonModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_RATE,
				VOLATILITY_OF_VOLATILITY,
				CORRELATION,
				MEAN_REVERSION,
				LONG_RUN_VARIANCE,
				VOLATILITY_SQUARED,
				discretization
		);

		final double[] sNodes = sGrid.getGrid();
		final double[] vNodes = vGrid.getGrid();

		final double iMax = MATURITY * sNodes[sNodes.length - 1];
		final int nI = I_GRID_MULTIPLIER * sNodes.length;
		final Grid iGrid = new UniformGrid(nI - 1, 0.0, iMax);
		final double[] iNodes = iGrid.getGrid();

		return new TestSetup(
				fdmModel,
				sNodes,
				vNodes,
				iNodes,
				NS,
				NV,
				nI
		);
	}

	private double interpolateValue(
			final double[] valueVector,
			final double[] sNodes,
			final double[] vNodes,
			final double[] iNodes,
			final int nS,
			final int nV,
			final int nI,
			final double s,
			final double v,
			final double i) {

		final int iS = findLeftIndex(sNodes, s);
		final int iV = findLeftIndex(vNodes, v);
		final int iI = findLeftIndex(iNodes, i);

		final double s0 = sNodes[iS];
		final double s1 = sNodes[iS + 1];
		final double v0 = vNodes[iV];
		final double v1 = vNodes[iV + 1];
		final double i0 = iNodes[iI];
		final double i1 = iNodes[iI + 1];

		final double ws = (s - s0) / (s1 - s0);
		final double wv = (v - v0) / (v1 - v0);
		final double wi = (i - i0) / (i1 - i0);

		final double c000 = valueVector[flatten(iS,     iV,     iI,     nS, nV)];
		final double c100 = valueVector[flatten(iS + 1, iV,     iI,     nS, nV)];
		final double c010 = valueVector[flatten(iS,     iV + 1, iI,     nS, nV)];
		final double c110 = valueVector[flatten(iS + 1, iV + 1, iI,     nS, nV)];
		final double c001 = valueVector[flatten(iS,     iV,     iI + 1, nS, nV)];
		final double c101 = valueVector[flatten(iS + 1, iV,     iI + 1, nS, nV)];
		final double c011 = valueVector[flatten(iS,     iV + 1, iI + 1, nS, nV)];
		final double c111 = valueVector[flatten(iS + 1, iV + 1, iI + 1, nS, nV)];

		return
				(1.0 - ws) * (1.0 - wv) * (1.0 - wi) * c000
				+ ws         * (1.0 - wv) * (1.0 - wi) * c100
				+ (1.0 - ws) * wv         * (1.0 - wi) * c010
				+ ws         * wv         * (1.0 - wi) * c110
				+ (1.0 - ws) * (1.0 - wv) * wi         * c001
				+ ws         * (1.0 - wv) * wi         * c101
				+ (1.0 - ws) * wv         * wi         * c011
				+ ws         * wv         * wi         * c111;
	}

	private void printLocalNeighborhood(
			final double[] valueVector,
			final TestSetup setup,
			final int centerIS,
			final int centerIV,
			final int centerII) {

		System.out.println("Local neighborhood around minimum:");

		for(int iI = Math.max(0, centerII - 1); iI <= Math.min(setup.nI - 1, centerII + 1); iI++) {
			System.out.println(String.format("Slice iI=%d, I=%.10f", iI, setup.iNodes[iI]));

			for(int iV = Math.max(0, centerIV - 1); iV <= Math.min(setup.nV - 1, centerIV + 1); iV++) {
				final StringBuilder row = new StringBuilder();
				row.append(String.format("  iV=%d, v=%.10f : ", iV, setup.vNodes[iV]));

				for(int iS = Math.max(0, centerIS - 2); iS <= Math.min(setup.nS - 1, centerIS + 2); iS++) {
					final double value = valueVector[flatten(iS, iV, iI, setup.nS, setup.nV)];
					row.append(String.format("[%d: %.6f] ", iS, value));
				}

				System.out.println(row.toString());
			}
		}
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
			final int iV,
			final int iI,
			final int nS,
			final int nV) {
		return iS + nS * (iV + nV * iI);
	}

	private static class TestSetup {
		private final FDMHestonModel fdmModel;
		private final double[] sNodes;
		private final double[] vNodes;
		private final double[] iNodes;
		private final int nS;
		private final int nV;
		private final int nI;

		private TestSetup(
				final FDMHestonModel fdmModel,
				final double[] sNodes,
				final double[] vNodes,
				final double[] iNodes,
				final int nS,
				final int nV,
				final int nI) {
			this.fdmModel = fdmModel;
			this.sNodes = sNodes;
			this.vNodes = vNodes;
			this.iNodes = iNodes;
			this.nS = nS;
			this.nV = nV;
			this.nI = nI;
		}
	}
}