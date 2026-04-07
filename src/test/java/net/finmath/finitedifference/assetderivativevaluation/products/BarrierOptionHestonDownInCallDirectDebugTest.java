package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.grids.BarrierAlignedSpotGridFactory;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Focused debug test for the direct 2D Heston down-in call case.
 */
public class BarrierOptionHestonDownInCallDirectDebugTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double BARRIER = 80.0;
	private static final double REBATE = 0.0;

	private static final double S0 = 100.0;
	private static final double R = 0.05;
	private static final double Q = 0.00;

	private static final double VOLATILITY = 0.25;
	private static final double VOLATILITY_SQUARED = VOLATILITY * VOLATILITY;
	private static final double KAPPA = 1.5;
	private static final double THETA_H = VOLATILITY_SQUARED;
	private static final double XI = 0.30;
	private static final double RHO = -0.70;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS_S = 160;
	private static final int NUMBER_OF_SPACE_STEPS_V = 100;

	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
	private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

	@Test
	public void testDownInEuropeanCallDirectHestonDebug() throws Exception {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS);

		final Grid sGrid = createSpotGrid();
		final double vMin = 0.0;
		final double vMax = Math.max(4.0 * THETA_H, VOLATILITY_SQUARED + 4.0 * XI * Math.sqrt(MATURITY));
		final Grid vGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_V, vMin, vMax);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, vGrid },
				timeDiscretization,
				THETA,
				new double[] { S0, VOLATILITY_SQUARED }
		);

		final FDMHestonModel fdmModel = new FDMHestonModel(
				S0,
				VOLATILITY_SQUARED,
				riskFreeCurve,
				dividendCurve,
				KAPPA,
				THETA_H,
				XI,
				RHO,
				spaceTime
		);

		final EuropeanOption vanilla = new EuropeanOption(null, MATURITY, STRIKE, CallOrPut.CALL);
		final BarrierOption knockIn = new BarrierOption(
				MATURITY,
				STRIKE,
				BARRIER,
				REBATE,
				CallOrPut.CALL,
				BarrierType.DOWN_IN
		);

		final long start = System.currentTimeMillis();

		final double[] vanillaValuesOnGrid = vanilla.getValue(0.0, fdmModel);
		final double[] knockInValuesOnGrid = knockIn.getValue(0.0, fdmModel);

		final long end = System.currentTimeMillis();

		final double[] sNodes = sGrid.getGrid();
		final double[] vNodes = vGrid.getGrid();
		final int nS = sNodes.length;
		final int nV = vNodes.length;

		final int v0Index = getNearestGridIndex(vNodes, VOLATILITY_SQUARED);

		final double[] vanillaSliceAtInitialVariance = extractSliceAtSecondStateIndex(vanillaValuesOnGrid, nS, nV, v0Index);
		final double[] knockInSliceAtInitialVariance = extractSliceAtSecondStateIndex(knockInValuesOnGrid, nS, nV, v0Index);

		final double vanillaPrice = interpolateAlongSpot(sNodes, vanillaSliceAtInitialVariance, S0);
		final double knockInPrice = interpolateAlongSpot(sNodes, knockInSliceAtInitialVariance, S0);

		final double gridMin = getMin(knockInValuesOnGrid);
		final double gridMax = getMax(knockInValuesOnGrid);
		final int minIndex = getMinIndex(knockInValuesOnGrid);

		final int minIS = minIndex % nS;
		final int minIV = minIndex / nS;

		final int barrierIndex = getGridIndex(sNodes, BARRIER);

		System.out.println("HESTON DIRECT DOWN-IN CALL DEBUG");
		System.out.println("Runtime        = " + (end - start) / 1000.0 + " s");
		System.out.println("Vanilla price  = " + vanillaPrice);
		System.out.println("Knock-in price = " + knockInPrice);
		System.out.println("Grid min       = " + gridMin);
		System.out.println("Grid max       = " + gridMax);
		System.out.println("S0 on grid     = " + (getGridIndex(sNodes, S0) >= 0));
		System.out.println("Barrier on grid= " + (barrierIndex >= 0));
		System.out.println(
				"Min located at indices (iS,iV) = (" + minIS + "," + minIV + "), state = ("
						+ sNodes[minIS] + ", " + vNodes[minIV] + ")"
		);

		printNeighborhood(knockInValuesOnGrid, sNodes, vNodes, nS, nV, minIS, minIV, 4, 2);

		printSpotSlice(knockInValuesOnGrid, sNodes, vNodes, nS, 0, "Knock-in slice at v=0");
		if(nV > 1) {
			printSpotSlice(knockInValuesOnGrid, sNodes, vNodes, nS, 1, "Knock-in slice at first positive v");
		}
		printSpotSlice(knockInValuesOnGrid, sNodes, vNodes, nS, v0Index, "Knock-in slice at v closest to v0");

		printSpotSlice(vanillaValuesOnGrid, sNodes, vNodes, nS, 0, "Vanilla slice at v=0");
		if(nV > 1) {
			printSpotSlice(vanillaValuesOnGrid, sNodes, vNodes, nS, 1, "Vanilla slice at first positive v");
		}
		printSpotSlice(vanillaValuesOnGrid, sNodes, vNodes, nS, v0Index, "Vanilla slice at v closest to v0");

		if(barrierIndex >= 0) {
			printVarianceColumn(knockInValuesOnGrid, sNodes, vNodes, nS, nV, barrierIndex, "Knock-in variance column at barrier");
			if(barrierIndex + 1 < nS) {
				printVarianceColumn(knockInValuesOnGrid, sNodes, vNodes, nS, nV, barrierIndex + 1, "Knock-in variance column at first interior spot node");
			}
			if(barrierIndex + 2 < nS) {
				printVarianceColumn(knockInValuesOnGrid, sNodes, vNodes, nS, nV, barrierIndex + 2, "Knock-in variance column at second interior spot node");
			}

			printVarianceColumn(vanillaValuesOnGrid, sNodes, vNodes, nS, nV, barrierIndex, "Vanilla variance column at barrier");
			if(barrierIndex + 1 < nS) {
				printVarianceColumn(vanillaValuesOnGrid, sNodes, vNodes, nS, nV, barrierIndex + 1, "Vanilla variance column at first interior spot node");
			}
			if(barrierIndex + 2 < nS) {
				printVarianceColumn(vanillaValuesOnGrid, sNodes, vNodes, nS, nV, barrierIndex + 2, "Vanilla variance column at second interior spot node");
			}
		}

		assertTrue("Vanilla price must be finite.", Double.isFinite(vanillaPrice));
		assertTrue("Knock-in price must be finite.", Double.isFinite(knockInPrice));
		assertTrue("Vanilla price must be non-negative.", vanillaPrice >= -1E-10);
		assertTrue("Knock-in price must be finite.", knockInPrice > -1.0);
		assertTrue("Knock-in price should not exceed vanilla price.", knockInPrice <= vanillaPrice + 1E-8);
	}

	private Grid createSpotGrid() {
		final double deltaS = Math.abs(BARRIER - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;

		final double sMin = Math.max(1E-8, BARRIER - 8.0 * deltaS);
		final double sMax = Math.max(3.0 * S0, S0 + 12.0 * deltaS);

		final int numberOfSteps = Math.max(
				NUMBER_OF_SPACE_STEPS_S,
				(int)Math.round((sMax - sMin) / deltaS)
		);

		return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
				numberOfSteps,
				sMin,
				sMax,
				BARRIER,
				BARRIER_CLUSTERING_EXPONENT
		);
	}

	private static double[] extractSliceAtSecondStateIndex(
			final double[] valuesOnGrid,
			final int nS,
			final int nSecondState,
			final int secondStateIndex) {

		if(secondStateIndex < 0 || secondStateIndex >= nSecondState) {
			throw new IllegalArgumentException("secondStateIndex out of range.");
		}

		final double[] slice = new double[nS];
		for(int i = 0; i < nS; i++) {
			slice[i] = valuesOnGrid[i + secondStateIndex * nS];
		}
		return slice;
	}

	private static double interpolateAlongSpot(
			final double[] sNodes,
			final double[] values,
			final double s) {

		final int index = getGridIndex(sNodes, s);
		if(index >= 0) {
			return values[index];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, values);
		return interpolation.value(s);
	}

	private static void printNeighborhood(
			final double[] values,
			final double[] sNodes,
			final double[] vNodes,
			final int nS,
			final int nV,
			final int centerIS,
			final int centerIV,
			final int radiusS,
			final int radiusV) {

		System.out.println("Local neighborhood around minimum:");

		final int vStart = Math.max(0, centerIV - radiusV);
		final int vEnd = Math.min(nV - 1, centerIV + radiusV);
		final int sStart = Math.max(0, centerIS - radiusS);
		final int sEnd = Math.min(nS - 1, centerIS + radiusS);

		for(int j = vStart; j <= vEnd; j++) {
			System.out.println("  iV=" + j + ", v=" + vNodes[j]);
			final StringBuilder row = new StringBuilder("    ");
			for(int i = sStart; i <= sEnd; i++) {
				row.append("[")
					.append(i)
					.append(": S=").append(sNodes[i])
					.append(", V=").append(values[i + j * nS])
					.append("] ");
			}
			System.out.println(row.toString());
		}
	}

	private static void printSpotSlice(
			final double[] values,
			final double[] sNodes,
			final double[] vNodes,
			final int nS,
			final int vIndex,
			final String title) {

		System.out.println(title + " (iV=" + vIndex + ", v=" + vNodes[vIndex] + ")");
		final int stride = Math.max(1, nS / 12);

		final StringBuilder row = new StringBuilder("  ");
		for(int i = 0; i < nS; i += stride) {
			row.append("[")
				.append(i)
				.append(": S=").append(sNodes[i])
				.append(", V=").append(values[i + vIndex * nS])
				.append("] ");
		}

		if((nS - 1) % stride != 0) {
			final int i = nS - 1;
			row.append("[")
				.append(i)
				.append(": S=").append(sNodes[i])
				.append(", V=").append(values[i + vIndex * nS])
				.append("] ");
		}

		System.out.println(row.toString());
	}

	private static void printVarianceColumn(
			final double[] values,
			final double[] sNodes,
			final double[] vNodes,
			final int nS,
			final int nV,
			final int sIndex,
			final String title) {

		System.out.println(title + " (iS=" + sIndex + ", S=" + sNodes[sIndex] + ")");
		final int stride = Math.max(1, nV / 12);

		final StringBuilder row = new StringBuilder("  ");
		for(int j = 0; j < nV; j += stride) {
			row.append("[")
				.append(j)
				.append(": v=").append(vNodes[j])
				.append(", V=").append(values[sIndex + j * nS])
				.append("] ");
		}

		if((nV - 1) % stride != 0) {
			final int j = nV - 1;
			row.append("[")
				.append(j)
				.append(": v=").append(vNodes[j])
				.append(", V=").append(values[sIndex + j * nS])
				.append("] ");
		}

		System.out.println(row.toString());
	}

	private static double getMin(final double[] values) {
		double min = values[0];
		for(int i = 1; i < values.length; i++) {
			min = Math.min(min, values[i]);
		}
		return min;
	}

	private static double getMax(final double[] values) {
		double max = values[0];
		for(int i = 1; i < values.length; i++) {
			max = Math.max(max, values[i]);
		}
		return max;
	}

	private static int getMinIndex(final double[] values) {
		int minIndex = 0;
		double min = values[0];
		for(int i = 1; i < values.length; i++) {
			if(values[i] < min) {
				min = values[i];
				minIndex = i;
			}
		}
		return minIndex;
	}

	private static int getGridIndex(final double[] grid, final double value) {
		final double tolerance = 1E-12;
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - value) < tolerance) {
				return i;
			}
		}
		return -1;
	}

	private static int getNearestGridIndex(final double[] grid, final double value) {
		int bestIndex = 0;
		double bestDistance = Math.abs(grid[0] - value);

		for(int i = 1; i < grid.length; i++) {
			final double distance = Math.abs(grid[i] - value);
			if(distance < bestDistance) {
				bestDistance = distance;
				bestIndex = i;
			}
		}

		return bestIndex;
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