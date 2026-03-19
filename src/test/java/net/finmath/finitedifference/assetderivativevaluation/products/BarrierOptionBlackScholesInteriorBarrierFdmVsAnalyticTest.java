package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import it.univr.fima.correction.BarrierOptions;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
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
 * Compares finite-difference prices of European knock-out barrier options under
 * Black-Scholes with analytic prices, using a grid where the barrier lies strictly
 * inside the spatial domain and is no longer an outer boundary.
 *
 * <p>
 * This test validates the internal-state-constraint treatment introduced for
 * knock-out barrier options.
 * </p>
 *
 * <p>
 * The suite covers:
 * </p>
 * <ul>
 *   <li>down-and-out call,</li>
 *   <li>down-and-out put,</li>
 *   <li>up-and-out call,</li>
 *   <li>up-and-out put.</li>
 * </ul>
 *
 * <p>
 * In addition to comparing FD prices against the analytic benchmark, the test also
 * checks that the nodes in the knocked-out region are equal to the rebate.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionBlackScholesInteriorBarrierFdmVsAnalyticTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

	private static final double S0 = 100.0;
	private static final double R = 0.05;
	private static final double Q = 0.00;
	private static final double SIGMA = 0.25;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS = 220;

	/**
	 * Number of space intervals between the barrier and S0.
	 * This ensures that both S0 and the barrier are grid nodes.
	 */
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	/**
	 * Number of extra steps beyond the barrier toward the outer boundary.
	 * This ensures the barrier is strictly interior to the grid.
	 */
	private static final int EXTRA_STEPS_BEYOND_BARRIER = 20;

	@Test
	public void testDownAndOutEuropeanCallBlackScholesInteriorBarrierFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.CALL, BarrierType.DOWN_OUT, 80.0, 0.35);
	}

	@Test
	public void testDownAndOutEuropeanPutBlackScholesInteriorBarrierFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.PUT, BarrierType.DOWN_OUT, 80.0, 0.35);
	}

	@Test
	public void testUpAndOutEuropeanCallBlackScholesInteriorBarrierFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.CALL, BarrierType.UP_OUT, 120.0, 0.35);
	}

	@Test
	public void testUpAndOutEuropeanPutBlackScholesInteriorBarrierFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.PUT, BarrierType.UP_OUT, 120.0, 0.35);
	}

	private void runBarrierTest(
			final CallOrPut callOrPut,
			final BarrierType barrierType,
			final double barrier,
			final double tolerance) {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS);

		final Grid sGrid = createInteriorBarrierGrid(barrier, barrierType);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { S0 }
		);

		final FDMBlackScholesModel fdmModel = new FDMBlackScholesModel(
				S0,
				riskFreeCurve,
				dividendCurve,
				SIGMA,
				spaceTime
		);

		final double callOrPutSign = callOrPut == CallOrPut.CALL ? 1.0 : -1.0;

		final BarrierOption fdmProduct = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPutSign,
				barrierType
		);

		final double[] fdValuesOnGrid = fdmProduct.getValue(0.0, fdmModel);
		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();

		final double fdPrice;
		if(isGridNode(sNodes, S0)) {
			fdPrice = fdValuesOnGrid[getGridIndex(sNodes, S0)];
		}
		else {
			final PolynomialSplineFunction interpolation =
					new LinearInterpolator().interpolate(sNodes, fdValuesOnGrid);
			fdPrice = interpolation.value(S0);
		}

		final double analyticPrice = BarrierOptions.blackScholesBarrierOptionValue(
				S0,
				R,
				Q,
				SIGMA,
				MATURITY,
				STRIKE,
				callOrPut == CallOrPut.CALL,
				REBATE,
				barrier,
				mapBarrierType(barrierType)
		);

		System.out.println("Type           = " + barrierType + " " + callOrPut);
		System.out.println("Barrier         = " + barrier);
		System.out.println("FD price        = " + fdPrice);
		System.out.println("Analytic price  = " + analyticPrice);

		assertTrue(fdPrice >= -1E-10);
		assertTrue(analyticPrice >= -1E-10);

		assertKnockedOutRegionPinnedToRebate(sNodes, fdValuesOnGrid, barrierType, barrier, REBATE);

		assertEquals(
				"FD vs analytic barrier price for " + barrierType + " " + callOrPut,
				analyticPrice,
				fdPrice,
				tolerance
		);
	}

	private Grid createInteriorBarrierGrid(final double barrier, final BarrierType barrierType) {

		/*
		 * We construct a uniform grid where:
		 * - the barrier is a grid node,
		 * - S0 is a grid node,
		 * - the barrier is strictly interior to the spatial domain.
		 */
		if(barrierType == BarrierType.DOWN_OUT) {
			final double deltaS = (S0 - barrier) / STEPS_BETWEEN_BARRIER_AND_SPOT;

			final double sMin = barrier - EXTRA_STEPS_BEYOND_BARRIER * deltaS;
			final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;

			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else if(barrierType == BarrierType.UP_OUT) {
			final double deltaS = (barrier - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;

			final double sMax = barrier + EXTRA_STEPS_BEYOND_BARRIER * deltaS;
			final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;

			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else {
			throw new IllegalArgumentException("This interior-barrier test supports only knock-out options.");
		}
	}

	private static void assertKnockedOutRegionPinnedToRebate(
			final double[] sNodes,
			final double[] fdValuesOnGrid,
			final BarrierType barrierType,
			final double barrier,
			final double rebate) {

		final double tolerance = 1E-10;

		for(int i = 0; i < sNodes.length; i++) {
			if(barrierType == BarrierType.DOWN_OUT && sNodes[i] <= barrier) {
				assertEquals(
						"Knocked-out down-region node should equal rebate at S=" + sNodes[i],
						rebate,
						fdValuesOnGrid[i],
						tolerance
				);
			}
			else if(barrierType == BarrierType.UP_OUT && sNodes[i] >= barrier) {
				assertEquals(
						"Knocked-out up-region node should equal rebate at S=" + sNodes[i],
						rebate,
						fdValuesOnGrid[i],
						tolerance
				);
			}
		}
	}

	private static boolean isGridNode(final double[] grid, final double value) {
		return getGridIndex(grid, value) >= 0;
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

	private static BarrierOptions.BarrierType mapBarrierType(final BarrierType barrierType) {
		switch(barrierType) {
		case DOWN_OUT:
			return BarrierOptions.BarrierType.DOWN_OUT;
		case UP_OUT:
			return BarrierOptions.BarrierType.UP_OUT;
		default:
			throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
		}
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