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
 * Debug / smoke test for fixed-strike arithmetic Asian calls under SABR.
 *
 * The lifted PDE state is (S, alpha, I).
 * We print:
 * - interpolated price at the initial state,
 * - global min/max over the full 3D grid,
 * - the grid location of min/max,
 * - a small local neighborhood around the minimum.
 */
public class AsianOptionSabrFixedStrikeCallDebugTest {

	private static final double SPOT = 100.0;
	private static final double INITIAL_ALPHA = 0.20;

	private static final double RISK_FREE_RATE = 0.02;
	private static final double DIVIDEND_YIELD_RATE = 0.01;

	private static final double BETA = 0.50;
	private static final double NU = 0.40;
	private static final double RHO = -0.30;

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;

	/*
	 * Small setup for quick debugging.
	 */
	private static final int NUMBER_OF_TIME_STEPS = 40;

	private static final int NS = 40;
	private static final int NALPHA = 20;
	private static final int I_GRID_MULTIPLIER = 4;

	private static final double S_MIN = 0.0;
	private static final double S_MAX = 2.0 * SPOT;

	private static final double ALPHA_MIN = 0.0;
	private static final double ALPHA_MAX = 1.0;

	@Test
	public void testAsianOptionSabrFixedStrikeCallDebug() throws Exception {

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
				setup.nS * setup.nAlpha * setup.nI,
				values.length
		);

		double minValue = Double.POSITIVE_INFINITY;
		double maxValue = Double.NEGATIVE_INFINITY;

		int minIS = -1;
		int minIA = -1;
		int minII = -1;

		int maxIS = -1;
		int maxIA = -1;
		int maxII = -1;

		for(int iI = 0; iI < setup.nI; iI++) {
			for(int iA = 0; iA < setup.nAlpha; iA++) {
				for(int iS = 0; iS < setup.nS; iS++) {

					final int index = flatten(iS, iA, iI, setup.nS, setup.nAlpha);
					final double value = values[index];

					Assert.assertTrue("Non-finite PDE value detected.", Double.isFinite(value));

					if(value < minValue) {
						minValue = value;
						minIS = iS;
						minIA = iA;
						minII = iI;
					}

					if(value > maxValue) {
						maxValue = value;
						maxIS = iS;
						maxIA = iA;
						maxII = iI;
					}
				}
			}
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

		System.out.println("SABR ASIAN FIXED STRIKE CALL DEBUG");
		System.out.println(String.format("Runtime: %.3f s", (tEnd - tStart) / 1000.0));
		System.out.println(String.format("Interpolated price at (S0,alpha0,I0) = %.10f", price));
		System.out.println(String.format("Grid min = %.10f", minValue));
		System.out.println(String.format("Grid max = %.10f", maxValue));

		System.out.println(String.format(
				"Min located at indices (iS,iA,iI) = (%d,%d,%d), state = (%.10f, %.10f, %.10f)",
				minIS,
				minIA,
				minII,
				setup.sNodes[minIS],
				setup.alphaNodes[minIA],
				setup.iNodes[minII]
		));

		System.out.println(String.format(
				"Max located at indices (iS,iA,iI) = (%d,%d,%d), state = (%.10f, %.10f, %.10f)",
				maxIS,
				maxIA,
				maxII,
				setup.sNodes[maxIS],
				setup.alphaNodes[maxIA],
				setup.iNodes[maxII]
		));

		printLocalNeighborhood(values, setup, minIS, minIA, minII);

		Assert.assertTrue("Interpolated price must be finite.", Double.isFinite(price));
		Assert.assertTrue("Interpolated price should be non-negative.", price >= -1E-10);
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

	private void printLocalNeighborhood(
			final double[] valueVector,
			final TestSetup setup,
			final int centerIS,
			final int centerIA,
			final int centerII) {

		System.out.println("Local neighborhood around minimum:");

		for(int iI = Math.max(0, centerII - 1); iI <= Math.min(setup.nI - 1, centerII + 1); iI++) {
			System.out.println(String.format("Slice iI=%d, I=%.10f", iI, setup.iNodes[iI]));

			for(int iA = Math.max(0, centerIA - 1); iA <= Math.min(setup.nAlpha - 1, centerIA + 1); iA++) {
				final StringBuilder row = new StringBuilder();
				row.append(String.format("  iA=%d, alpha=%.10f : ", iA, setup.alphaNodes[iA]));

				for(int iS = Math.max(0, centerIS - 2); iS <= Math.min(setup.nS - 1, centerIS + 2); iS++) {
					final double value = valueVector[flatten(iS, iA, iI, setup.nS, setup.nAlpha)];
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