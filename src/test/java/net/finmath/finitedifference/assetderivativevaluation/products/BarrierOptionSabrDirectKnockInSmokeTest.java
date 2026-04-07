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
 * Smoke test for direct 2D SABR knock-in pricing.
 *
 * <p>
 * This test does not compare to Monte Carlo. It checks that the direct pre-hit
 * SABR implementation:
 * </p>
 * <ul>
 *   <li>returns finite prices,</li>
 *   <li>respects {@code 0 <= knockIn <= vanilla},</li>
 *   <li>runs for all four European knock-in types.</li>
 * </ul>
 */
public class BarrierOptionSabrDirectKnockInSmokeTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
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
	public void testDownInEuropeanPutDirectSabrSmoke() throws Exception {
		runSmokeTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testDownInEuropeanCallDirectSabrSmoke() throws Exception {
		runSmokeTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testUpInEuropeanPutDirectSabrSmoke() throws Exception {
		runSmokeTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
	}

	@Test
	public void testUpInEuropeanCallDirectSabrSmoke() throws Exception {
		runSmokeTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
	}

	private void runSmokeTest(
			final CallOrPut callOrPut,
			final BarrierType barrierType,
			final double barrier) throws Exception {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS);

		final Grid forwardGrid = createForwardGrid(barrier, barrierType);

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

		final EuropeanOption vanilla = new EuropeanOption(null, MATURITY, STRIKE, callOrPut);
		final BarrierOption knockIn = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				barrierType
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

		System.out.println("SABR DIRECT KNOCK-IN SMOKE");
		System.out.println("Type           = " + barrierType + " " + callOrPut);
		System.out.println("Runtime        = " + (end - start) / 1000.0 + " s");
		System.out.println("Grid min       = " + gridMin);
		System.out.println("Grid max       = " + gridMax);
		System.out.println("Vanilla price  = " + vanillaPrice);
		System.out.println("Knock-in price = " + knockInPrice);
		System.out.println("F0 on grid     = " + (getGridIndex(fNodes, FORWARD0) >= 0));
		System.out.println("Barrier on grid= " + (getGridIndex(fNodes, barrier) >= 0));
		System.out.println(
				"Min located at indices (iF,iA) = (" + minIF + "," + minIA + "), state = ("
						+ fNodes[minIF] + ", " + aNodes[minIA] + ")"
		);

		assertTrue(Double.isFinite(vanillaPrice));
		assertTrue(Double.isFinite(knockInPrice));
		assertTrue(Double.isFinite(gridMin));
		assertTrue(Double.isFinite(gridMax));

		assertTrue("Vanilla price must be non-negative.", vanillaPrice >= -1E-10);
		assertTrue("Knock-in price must be non-negative up to a small tolerance.", knockInPrice >= -1E-6);
		assertTrue("Knock-in price must not exceed vanilla price.", knockInPrice <= vanillaPrice + 1E-8);

		assertTrue("Barrier must be exactly on the forward grid.", getGridIndex(fNodes, barrier) >= 0);
		assertTrue("Initial alpha must lie inside the alpha grid domain.",
				ALPHA0 >= aNodes[0] - 1E-12 && ALPHA0 <= aNodes[aNodes.length - 1] + 1E-12);
	}

	private Grid createForwardGrid(final double barrier, final BarrierType barrierType) {

		final double deltaF = Math.abs(barrier - FORWARD0) / STEPS_BETWEEN_BARRIER_AND_SPOT;

		if(barrierType == BarrierType.DOWN_IN) {

			final double rawFMin = Math.max(0.0, barrier - deltaF);
			final double desiredFMax = Math.max(3.0 * FORWARD0, FORWARD0 + 12.0 * deltaF);

			final int numberOfSteps = Math.max(
					NUMBER_OF_SPACE_STEPS_F,
					(int)Math.round((desiredFMax - rawFMin) / deltaF)
			);

			final double fMax = rawFMin + numberOfSteps * deltaF;

			return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
					numberOfSteps,
					rawFMin,
					fMax,
					barrier,
					BARRIER_CLUSTERING_EXPONENT
			);
		}
		else if(barrierType == BarrierType.UP_IN) {

			final double rawFMax = barrier + deltaF;
			final double desiredFMin = 0.0;

			final int numberOfSteps = Math.max(
					NUMBER_OF_SPACE_STEPS_F,
					(int)Math.round((rawFMax - desiredFMin) / deltaF)
			);

			final double fMin = Math.max(0.0, rawFMax - numberOfSteps * deltaF);

			return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
					numberOfSteps,
					fMin,
					rawFMax,
					barrier,
					BARRIER_CLUSTERING_EXPONENT
			);
		}
		else {
			throw new IllegalArgumentException("Smoke test grid requested for non knock-in barrier type.");
		}
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