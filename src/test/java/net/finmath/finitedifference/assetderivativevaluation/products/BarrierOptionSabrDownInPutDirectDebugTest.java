package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
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
 * Focused debug test for the direct 2D SABR down-in put case.
 */
public class BarrierOptionSabrDownInPutDirectDebugTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double BARRIER = 80.0;
	private static final double REBATE = 0.0;

	private static final double FORWARD0 = 100.0;
	private static final double R = 0.00;
	private static final double Q = 0.00;

	private static final double ALPHA0 = 0.25;
	private static final double BETA = 0.50;
	private static final double NU = 0.40;
	private static final double RHO = -0.30;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS_F = 180;
	private static final int NUMBER_OF_SPACE_STEPS_ALPHA = 100;

	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
	private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

	@Test
	public void testDownInEuropeanPutDirectSabrDebug() throws Exception {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS);

		final Grid forwardGrid = createForwardGrid();
		final double alphaMin = 0.0;
		final double alphaMax = Math.max(4.0 * ALPHA0, ALPHA0 + 4.0 * NU * Math.sqrt(MATURITY));
		final Grid alphaGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_ALPHA, alphaMin, alphaMax);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { forwardGrid, alphaGrid },
				timeDiscretization,
				THETA,
				new double[] { FORWARD0, ALPHA0 }
		);

		final FDMSabrModel fdmModel = new FDMSabrModel(
				FORWARD0,
				ALPHA0,
				riskFreeCurve,
				dividendCurve,
				BETA,
				NU,
				RHO,
				spaceTime
		);

		final EuropeanOption vanilla = new EuropeanOption(null, MATURITY, STRIKE, CallOrPut.PUT);
		final BarrierOption knockIn = new BarrierOption(
				MATURITY,
				STRIKE,
				BARRIER,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_IN
		);

		final long start = System.currentTimeMillis();

		final double[] vanillaValuesOnGrid = vanilla.getValue(0.0, fdmModel);
		final double[] knockInValuesOnGrid = knockIn.getValue(0.0, fdmModel);

		final long end = System.currentTimeMillis();

		final double[] fNodes = forwardGrid.getGrid();
		final double[] aNodes = alphaGrid.getGrid();

		final int nF = fNodes.length;
		final int nA = aNodes.length;

		final int alpha0Index = getNearestGridIndex(aNodes, ALPHA0);

		final double[] vanillaSliceAtInitialAlpha =
				extractSliceAtSecondStateIndex(vanillaValuesOnGrid, nF, nA, alpha0Index);
		final double[] knockInSliceAtInitialAlpha =
				extractSliceAtSecondStateIndex(knockInValuesOnGrid, nF, nA, alpha0Index);

		final double vanillaPrice = interpolateAlongFirstState(fNodes, vanillaSliceAtInitialAlpha, FORWARD0);
		final double knockInPrice = interpolateAlongFirstState(fNodes, knockInSliceAtInitialAlpha, FORWARD0);

		final double gridMin = getMin(knockInValuesOnGrid);
		final double gridMax = getMax(knockInValuesOnGrid);
		final int minIndex = getMinIndex(knockInValuesOnGrid);

		final int minIF = minIndex % nF;
		final int minIA = minIndex / nF;

		final int barrierIndex = getGridIndex(fNodes, BARRIER);

		System.out.println("SABR DIRECT DOWN-IN PUT DEBUG");
		System.out.println("Runtime        = " + (end - start) / 1000.0 + " s");
		System.out.println("Vanilla price  = " + vanillaPrice);
		System.out.println("Knock-in price = " + knockInPrice);
		System.out.println("Grid min       = " + gridMin);
		System.out.println("Grid max       = " + gridMax);
		System.out.println("F0 on grid     = " + (getGridIndex(fNodes, FORWARD0) >= 0));
		System.out.println("Barrier on grid= " + (barrierIndex >= 0));
		System.out.println(
				"Min located at indices (iF,iA) = (" + minIF + "," + minIA + "), state = ("
						+ fNodes[minIF] + ", " + aNodes[minIA] + ")"
		);

		printNeighborhood(knockInValuesOnGrid, fNodes, aNodes, nF, nA, minIF, minIA, 4, 3);

		printFirstStateSlice(knockInValuesOnGrid, fNodes, aNodes, nF, 0, "Knock-in slice at alpha=0");
		if(nA > 1) {
			printFirstStateSlice(knockInValuesOnGrid, fNodes, aNodes, nF, 1, "Knock-in slice at first positive alpha");
		}
		printFirstStateSlice(knockInValuesOnGrid, fNodes, aNodes, nF, alpha0Index, "Knock-in slice at alpha closest to alpha0");

		printFirstStateSlice(vanillaValuesOnGrid, fNodes, aNodes, nF, 0, "Vanilla slice at alpha=0");
		if(nA > 1) {
			printFirstStateSlice(vanillaValuesOnGrid, fNodes, aNodes, nF, 1, "Vanilla slice at first positive alpha");
		}
		printFirstStateSlice(vanillaValuesOnGrid, fNodes, aNodes, nF, alpha0Index, "Vanilla slice at alpha closest to alpha0");

		if(barrierIndex >= 0) {
			printSecondStateColumn(knockInValuesOnGrid, fNodes, aNodes, nF, nA, barrierIndex, "Knock-in alpha column at barrier");
			if(barrierIndex + 1 < nF) {
				printSecondStateColumn(knockInValuesOnGrid, fNodes, aNodes, nF, nA, barrierIndex + 1, "Knock-in alpha column at first interior node");
			}
			if(barrierIndex + 2 < nF) {
				printSecondStateColumn(knockInValuesOnGrid, fNodes, aNodes, nF, nA, barrierIndex + 2, "Knock-in alpha column at second interior node");
			}

			printSecondStateColumn(vanillaValuesOnGrid, fNodes, aNodes, nF, nA, barrierIndex, "Vanilla alpha column at barrier");
			if(barrierIndex + 1 < nF) {
				printSecondStateColumn(vanillaValuesOnGrid, fNodes, aNodes, nF, nA, barrierIndex + 1, "Vanilla alpha column at first interior node");
			}
			if(barrierIndex + 2 < nF) {
				printSecondStateColumn(vanillaValuesOnGrid, fNodes, aNodes, nF, nA, barrierIndex + 2, "Vanilla alpha column at second interior node");
			}
		}

		assertTrue(Double.isFinite(vanillaPrice));
		assertTrue(Double.isFinite(knockInPrice));
		assertTrue(vanillaPrice >= -1E-10);
		assertTrue(knockInPrice >= -1E-4);
		assertTrue(knockInPrice <= vanillaPrice + 1E-8);
	}

	private Grid createForwardGrid() {

		final double deltaF = Math.abs(BARRIER - FORWARD0) / STEPS_BETWEEN_BARRIER_AND_SPOT;

		final double fMin = Math.max(0.0, BARRIER - deltaF);
		final double desiredFMax = Math.max(3.0 * FORWARD0, FORWARD0 + 12.0 * deltaF);

		final int numberOfSteps = Math.max(
				NUMBER_OF_SPACE_STEPS_F,
				(int)Math.round((desiredFMax - fMin) / deltaF)
		);

		final double fMax = fMin + numberOfSteps * deltaF;

		return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
				numberOfSteps,
				fMin,
				fMax,
				BARRIER,
				BARRIER_CLUSTERING_EXPONENT
		);
	}

	private static double[] extractSliceAtSecondStateIndex(
			final double[] valuesOnGrid,
			final int nFirstState,
			final int nSecondState,
			final int secondStateIndex) {

		if(secondStateIndex < 0 || secondStateIndex >= nSecondState) {
			throw new IllegalArgumentException("secondStateIndex out of range.");
		}

		final double[] slice = new double[nFirstState];
		for(int i = 0; i < nFirstState; i++) {
			slice[i] = valuesOnGrid[i + secondStateIndex * nFirstState];
		}
		return slice;
	}

	private static double interpolateAlongFirstState(
			final double[] nodes,
			final double[] values,
			final double x) {

		final int index = getGridIndex(nodes, x);
		if(index >= 0) {
			return values[index];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(nodes, values);
		return interpolation.value(x);
	}

	private static void printNeighborhood(
			final double[] values,
			final double[] firstStateNodes,
			final double[] secondStateNodes,
			final int nFirstState,
			final int nSecondState,
			final int centerIF,
			final int centerIS,
			final int radiusF,
			final int radiusS) {

		System.out.println("Local neighborhood around minimum:");

		final int sStart = Math.max(0, centerIS - radiusS);
		final int sEnd = Math.min(nSecondState - 1, centerIS + radiusS);
		final int fStart = Math.max(0, centerIF - radiusF);
		final int fEnd = Math.min(nFirstState - 1, centerIF + radiusF);

		for(int j = sStart; j <= sEnd; j++) {
			System.out.println("  iA=" + j + ", alpha=" + secondStateNodes[j]);
			final StringBuilder row = new StringBuilder("    ");
			for(int i = fStart; i <= fEnd; i++) {
				row.append("[")
					.append(i)
					.append(": F=").append(firstStateNodes[i])
					.append(", V=").append(values[i + j * nFirstState])
					.append("] ");
			}
			System.out.println(row.toString());
		}
	}

	private static void printFirstStateSlice(
			final double[] values,
			final double[] firstStateNodes,
			final double[] secondStateNodes,
			final int nFirstState,
			final int secondStateIndex,
			final String title) {

		System.out.println(title + " (iA=" + secondStateIndex + ", alpha=" + secondStateNodes[secondStateIndex] + ")");
		final int stride = Math.max(1, nFirstState / 12);

		final StringBuilder row = new StringBuilder("  ");
		for(int i = 0; i < nFirstState; i += stride) {
			row.append("[")
				.append(i)
				.append(": F=").append(firstStateNodes[i])
				.append(", V=").append(values[i + secondStateIndex * nFirstState])
				.append("] ");
		}

		if((nFirstState - 1) % stride != 0) {
			final int i = nFirstState - 1;
			row.append("[")
				.append(i)
				.append(": F=").append(firstStateNodes[i])
				.append(", V=").append(values[i + secondStateIndex * nFirstState])
				.append("] ");
		}

		System.out.println(row.toString());
	}

	private static void printSecondStateColumn(
			final double[] values,
			final double[] firstStateNodes,
			final double[] secondStateNodes,
			final int nFirstState,
			final int nSecondState,
			final int firstStateIndex,
			final String title) {

		System.out.println(title + " (iF=" + firstStateIndex + ", F=" + firstStateNodes[firstStateIndex] + ")");
		final int stride = Math.max(1, nSecondState / 12);

		final StringBuilder row = new StringBuilder("  ");
		for(int j = 0; j < nSecondState; j += stride) {
			row.append("[")
				.append(j)
				.append(": alpha=").append(secondStateNodes[j])
				.append(", V=").append(values[firstStateIndex + j * nFirstState])
				.append("] ");
		}

		if((nSecondState - 1) % stride != 0) {
			final int j = nSecondState - 1;
			row.append("[")
				.append(j)
				.append(": alpha=").append(secondStateNodes[j])
				.append(", V=").append(values[firstStateIndex + j * nFirstState])
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