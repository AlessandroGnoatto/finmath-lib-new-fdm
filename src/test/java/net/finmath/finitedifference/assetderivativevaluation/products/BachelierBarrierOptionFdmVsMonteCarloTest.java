package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBachelierModel;
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
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloAssetModel;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Unified Bachelier finite-difference vs Monte Carlo regression test for all 8
 * standard single-barrier European options.
 *
 * <p>
 * Grid policy:
 * </p>
 * <ul>
 *   <li>knock-out options use the traditional barrier-on-boundary grid,</li>
 *   <li>knock-in options use an interior-barrier grid with asymmetric tail
 *       extension where needed.</li>
 * </ul>
 *
 * <p>
 * This mirrors the current production implementation:
 * </p>
 * <ul>
 *   <li>knock-outs are solved directly on the original grid,</li>
 *   <li>1D knock-ins are solved directly through a coupled two-state PDE using
 *       an auxiliary interior-barrier grid.</li>
 * </ul>
 *
 * <p>
 * Barrier monitoring in Monte Carlo is discrete on the simulation time grid.
 * Hence agreement improves when the Monte Carlo time grid is refined.
 * </p>
 */
public class BachelierBarrierOptionFdmVsMonteCarloTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

	private static final double S0 = 100.0;
	private static final double R = 0.0;
	private static final double Q = 0.0;
	private static final double SIGMA_N = 10.0;

	private static final int FD_NUMBER_OF_TIME_STEPS = 100;
	private static final int FD_NUMBER_OF_SPACE_STEPS = 300;
	private static final double THETA = 0.5;

	private static final int MC_NUMBER_OF_TIME_STEPS = 1000;
	private static final int MC_NUMBER_OF_PATHS = 10_000;
	private static final int SEED = 31415;

	private static final double RELATIVE_TOLERANCE = 0.15;
	private static final double ABSOLUTE_TOLERANCE = 1.0E-3;
	/**
	 * Number of space intervals between the barrier and S0.
	 * This keeps the local spacing near the barrier under control and often aligns S0 with a grid node.
	 */
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	/**
	 * Default interior extension for knock-in grids.
	 */
	private static final int DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS = 40;

	/**
	 * Extra lower-tail depth for DOWN_IN PUT.
	 */
	private static final int DOWN_IN_PUT_EXTRA_STEPS = 160;

	/**
	 * Extra upper-tail depth for UP_IN CALL.
	 */
	private static final int UP_IN_CALL_EXTRA_STEPS = 160;

	@Test
	public void testDownAndOutEuropeanCallBachelierFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.CALL, BarrierType.DOWN_OUT, 80.0);
	}

	@Test
	public void testDownAndInEuropeanCallBachelierFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testUpAndOutEuropeanCallBachelierFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.CALL, BarrierType.UP_OUT, 120.0);
	}

	@Test
	public void testUpAndInEuropeanCallBachelierFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
	}

	@Test
	public void testDownAndOutEuropeanPutBachelierFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.PUT, BarrierType.DOWN_OUT, 80.0);
	}

	@Test
	public void testDownAndInEuropeanPutBachelierFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testUpAndOutEuropeanPutBachelierFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.PUT, BarrierType.UP_OUT, 120.0);
	}

	@Test
	public void testUpAndInEuropeanPutBachelierFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
	}

	private void runBarrierTest(
			final CallOrPut callOrPut,
			final BarrierType barrierType,
			final double barrier) throws Exception {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization fdTimeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						FD_NUMBER_OF_TIME_STEPS,
						MATURITY / FD_NUMBER_OF_TIME_STEPS
				);

		final Grid sGrid = createGrid(barrier, barrierType, callOrPut);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				fdTimeDiscretization,
				THETA,
				new double[] { S0 }
		);

		final FDMBachelierModel fdmModel = new FDMBachelierModel(
				S0,
				riskFreeCurve,
				dividendCurve,
				SIGMA_N,
				spaceTime
		);

		final BarrierOption fdmProduct = new BarrierOption(
				null,
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				barrierType
		);

		final double[] fdValuesOnGrid = fdmProduct.getValue(0.0, fdmModel);
		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();

		final double fdPrice;
		final int s0Index = getGridIndex(sNodes, S0);
		if(s0Index >= 0) {
			fdPrice = fdValuesOnGrid[s0Index];
		}
		else {
			final PolynomialSplineFunction interpolation =
					new LinearInterpolator().interpolate(sNodes, fdValuesOnGrid);
			fdPrice = interpolation.value(S0);
		}

		final TimeDiscretization mcTimes =
				new TimeDiscretizationFromArray(
						0.0,
						MC_NUMBER_OF_TIME_STEPS,
						MATURITY / MC_NUMBER_OF_TIME_STEPS
				);

		final BrownianMotionFromMersenneRandomNumbers brownianMotion =
				new BrownianMotionFromMersenneRandomNumbers(mcTimes, 1, MC_NUMBER_OF_PATHS, SEED);

		final net.finmath.montecarlo.assetderivativevaluation.models.BachelierModel mcModel =
				new net.finmath.montecarlo.assetderivativevaluation.models.BachelierModel(S0, R, SIGMA_N);

		final EulerSchemeFromProcessModel process =
				new EulerSchemeFromProcessModel(mcModel, brownianMotion);

		final MonteCarloAssetModel mcSimulation = new MonteCarloAssetModel(process);

		final net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption mcProduct =
				new net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption(
						MATURITY,
						STRIKE,
						barrier,
						REBATE,
						callOrPut,
						barrierType
				);

		final double mcPrice = mcProduct.getValue(mcSimulation);

		System.out.println("Type           = " + barrierType + " " + callOrPut);
		System.out.println("Grid min       = " + sNodes[0]);
		System.out.println("Grid max       = " + sNodes[sNodes.length - 1]);
		System.out.println("FD price       = " + fdPrice);
		System.out.println("MC price       = " + mcPrice);

		assertTrue(fdPrice >= -1E-10);
		assertTrue(mcPrice >= -1E-10);

		final double relativeError;
		if(Math.abs(mcPrice) > 1E-12) {
			relativeError = Math.abs(fdPrice - mcPrice) / Math.abs(mcPrice);
		}
		else {
			/*
			 * If the Monte Carlo benchmark is numerically zero, fall back to an absolute check.
			 */
			relativeError = Math.abs(fdPrice - mcPrice);
		}

		final double absoluteError = Math.abs(fdPrice - mcPrice);
		final double scale = Math.max(Math.abs(fdPrice), Math.abs(mcPrice));
		final double admissibleError = Math.max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * scale);

		System.out.println("Absolute error = " + absoluteError);
		System.out.println("Admissible     = " + admissibleError);

		assertTrue(
				"Error exceeds tolerance for " + barrierType + " " + callOrPut
				+ ": absoluteError = " + absoluteError
				+ ", admissibleError = " + admissibleError,
				absoluteError <= admissibleError
		);
	}

	private Grid createGrid(
			final double barrier,
			final BarrierType barrierType,
			final CallOrPut callOrPut) {

		if(barrierType == BarrierType.DOWN_OUT || barrierType == BarrierType.UP_OUT) {
			return createKnockOutGrid(barrier, barrierType);
		}
		else if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.UP_IN) {
			return createKnockInInteriorGrid(barrier, barrierType, callOrPut);
		}
		else {
			throw new IllegalArgumentException("Unsupported barrier type: " + barrierType);
		}
	}

	private Grid createKnockOutGrid(final double barrier, final BarrierType barrierType) {

		/*
		 * Traditional knock-out grid:
		 * place the barrier exactly on the relevant outer boundary.
		 */
		if(barrierType == BarrierType.DOWN_OUT) {
			final double deltaS = (S0 - barrier) / STEPS_BETWEEN_BARRIER_AND_SPOT;
			final double sMin = barrier;
			final double sMax = sMin + FD_NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(FD_NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else if(barrierType == BarrierType.UP_OUT) {
			final double deltaS = (barrier - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;
			final double sMax = barrier;
			final double sMin = sMax - FD_NUMBER_OF_SPACE_STEPS * deltaS;
			return new UniformGrid(FD_NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else {
			throw new IllegalArgumentException("Knock-out grid requested for non knock-out type: " + barrierType);
		}
	}

	private Grid createKnockInInteriorGrid(
			final double barrier,
			final BarrierType barrierType,
			final CallOrPut callOrPut) {

		/*
		 * Interior-barrier grid for direct knock-in pricing:
		 * - keep the same local spacing near the barrier as the Black-Scholes unified test,
		 * - place the barrier on an interior node,
		 * - extend asymmetrically for tail-sensitive cases.
		 */
		if(barrierType == BarrierType.DOWN_IN) {
			final double deltaS = (S0 - barrier) / STEPS_BETWEEN_BARRIER_AND_SPOT;

			final int extraStepsBelowBarrier =
					callOrPut == CallOrPut.PUT
					? DOWN_IN_PUT_EXTRA_STEPS
					: DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS;

			final double sMin = barrier - extraStepsBelowBarrier * deltaS;
			final double sMax = sMin + FD_NUMBER_OF_SPACE_STEPS * deltaS;

			return new UniformGrid(FD_NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else if(barrierType == BarrierType.UP_IN) {
			final double deltaS = (barrier - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;

			final int extraStepsAboveBarrier =
					callOrPut == CallOrPut.CALL
					? UP_IN_CALL_EXTRA_STEPS
					: DEFAULT_INTERIOR_BARRIER_EXTRA_STEPS;

			final double sMax = barrier + extraStepsAboveBarrier * deltaS;
			final double sMin = sMax - FD_NUMBER_OF_SPACE_STEPS * deltaS;

			return new UniformGrid(FD_NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else {
			throw new IllegalArgumentException("Knock-in interior grid requested for non knock-in type: " + barrierType);
		}
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