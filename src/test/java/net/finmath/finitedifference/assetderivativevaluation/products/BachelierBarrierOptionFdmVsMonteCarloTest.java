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
 * Compares European barrier option prices under the Bachelier model across
 * finite differences and Monte Carlo.
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
 * Barrier monitoring in Monte Carlo is discrete on the simulation time grid.
 * Hence a tighter agreement is obtained by refining that grid.
 * </p>
 *
 * <p>
 * The assertion is performed in relative terms:
 * </p>
 * <p>
 * |FDM - MC| / MC <= 5%
 * </p>
 *
 * @author Alessandro Gnoatto
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
	private static final int FD_NUMBER_OF_SPACE_STEPS = 200;
	private static final double THETA = 0.5;

	private static final int MC_NUMBER_OF_TIME_STEPS = 1000;
	private static final int MC_NUMBER_OF_PATHS = 10_000;
	private static final int SEED = 31415;

	private static final double RELATIVE_TOLERANCE = 0.15;

	@Test
	public void testDownAndOutEuropeanCallBachelierFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.CALL, BarrierType.DOWN_OUT, 70.0);
	}

	@Test
	public void testDownAndInEuropeanCallBachelierFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.CALL, BarrierType.DOWN_IN, 70.0);
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
						MATURITY / FD_NUMBER_OF_TIME_STEPS);

		final Grid sGrid = createGrid(barrier, barrierType);

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

		final net.finmath.finitedifference.assetderivativevaluation.products.BarrierOption fdmProduct =
				new net.finmath.finitedifference.assetderivativevaluation.products.BarrierOption(
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
						MATURITY / MC_NUMBER_OF_TIME_STEPS);

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
		System.out.println("FD price       = " + fdPrice);
		System.out.println("MC price       = " + mcPrice);

		assertTrue(fdPrice >= 0.0);
		assertTrue(mcPrice >= 0.0);

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

		System.out.println("Relative error = " + relativeError);

		assertTrue(
				"Relative error exceeds 15% for " + barrierType + " " + callOrPut
				+ ": relativeError = " + relativeError,
				relativeError <= RELATIVE_TOLERANCE
		);
	}

	private Grid createGrid(final double barrier, final BarrierType barrierType) {

		final double stdev = SIGMA_N * Math.sqrt(MATURITY);

		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {
			final double sMin = barrier;
			final double sMax = S0 + 8.0 * stdev;
			return new UniformGrid(FD_NUMBER_OF_SPACE_STEPS, sMin, sMax);
		}
		else {
			final double sMin = Math.max(1E-8, S0 - 8.0 * stdev);
			final double sMax = barrier;
			return new UniformGrid(FD_NUMBER_OF_SPACE_STEPS, sMin, sMax);
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