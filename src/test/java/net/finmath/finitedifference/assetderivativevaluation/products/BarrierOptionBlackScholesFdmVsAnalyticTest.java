package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import it.univr.fima.correction.BarrierOptions;
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
 * Compares finite-difference prices of European barrier options under Black-Scholes
 * with the closed-form prices provided by {@link BarrierOptions}.
 *
 * <p>
 * The test suite covers all 8 standard barrier combinations:
 * </p>
 * <ul>
 *   <li>call / put,</li>
 *   <li>down / up,</li>
 *   <li>in / out.</li>
 * </ul>
 *
 * <p>
 * Implementation notes:
 * <ul>
 *   <li>Uses Apache Commons Math linear interpolation when S0 is not exactly a grid node.</li>
 *   <li>Ensures the barrier coincides with the relevant grid boundary by construction.</li>
 *   <li>Prints FD and analytic prices for easy diagnostics.</li>
 * </ul>
 * </p>
 */
public class BarrierOptionBlackScholesFdmVsAnalyticTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

	private static final double S0 = 100.0;
	private static final double R = 0.05;
	private static final double Q = 0.00;
	private static final double SIGMA = 0.25;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS = 200;

	/**
	 * Number of space intervals between the barrier and S0.
	 * This ensures that S0 is aligned to a grid node when possible.
	 */
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	@Test
	public void testDownAndOutEuropeanCallBlackScholesFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.CALL, BarrierType.DOWN_OUT, 80.0, 0.35);
	}

	@Test
	public void testDownAndInEuropeanCallBlackScholesFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0, 0.35);
	}

	@Test
	public void testUpAndOutEuropeanCallBlackScholesFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.CALL, BarrierType.UP_OUT, 120.0, 0.35);
	}

	@Test
	public void testUpAndInEuropeanCallBlackScholesFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0, 0.35);
	}

	@Test
	public void testDownAndOutEuropeanPutBlackScholesFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.PUT, BarrierType.DOWN_OUT, 80.0, 0.35);
	}

	@Test
	public void testDownAndInEuropeanPutBlackScholesFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0, 0.35);
	}

	@Test
	public void testUpAndOutEuropeanPutBlackScholesFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.PUT, BarrierType.UP_OUT, 120.0, 0.35);
	}

	@Test
	public void testUpAndInEuropeanPutBlackScholesFiniteDifferenceVsAnalytic() {
		runBarrierTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0, 0.35);
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

		final Grid sGrid = createGrid(barrier, barrierType);

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
		
		double callOrPutSign = 1.0;
		if(callOrPut == CallOrPut.CALL) {
			callOrPutSign = 1.0;
		}
		else if(callOrPut == CallOrPut.PUT) {
			callOrPutSign = -1.0;
		}
		
		final BarrierOption fdmProduct = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPutSign,
				barrierType
		);

		// Solve via FD
		final double[] fdValuesOnGrid = fdmProduct.getValue(0.0, fdmModel);
		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();

		final double fdPrice;
		if(isGridNode(sNodes, S0)) {
			fdPrice = fdValuesOnGrid[getGridIndex(sNodes, S0)];
		}
		else {
			// fallback linear interpolation using Apache Commons Math
			final PolynomialSplineFunction interpolation =
					new LinearInterpolator().interpolate(sNodes, fdValuesOnGrid);
			fdPrice = interpolation.value(S0);
		}

		// Analytic closed-form from finmath.functions (your corrected version)
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
		System.out.println("FD price       = " + fdPrice);
		System.out.println("Analytic price = " + analyticPrice);

		// allow tiny negative numerical values
		assertTrue(fdPrice >= -1E-10);
		assertTrue(analyticPrice >= -1E-10);

		assertEquals(
				"FD vs analytic barrier price for " + barrierType + " " + callOrPut,
				analyticPrice,
				fdPrice,
				tolerance
		);
	}

	private Grid createGrid(final double barrier, final BarrierType barrierType) {

		/*
		 * We construct a uniform grid that places the barrier exactly on the chosen
		 * boundary (lower for DOWN_*, upper for UP_*). The spacing is chosen so that
		 * there are STEPS_BETWEEN_BARRIER_AND_SPOT intervals between barrier and S0
		 * (so S0 ideally falls on a node).
		 */
		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {
			final double deltaS = (S0 - barrier) / STEPS_BETWEEN_BARRIER_AND_SPOT;
			final double sMin = barrier;
			final double sMax = sMin + NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else {
			final double deltaS = (barrier - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;
			final double sMax = barrier;
			final double sMin = sMax - NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS, sMin, sMax);
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
		case DOWN_IN:
			return BarrierOptions.BarrierType.DOWN_IN;
		case DOWN_OUT:
			return BarrierOptions.BarrierType.DOWN_OUT;
		case UP_IN:
			return BarrierOptions.BarrierType.UP_IN;
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