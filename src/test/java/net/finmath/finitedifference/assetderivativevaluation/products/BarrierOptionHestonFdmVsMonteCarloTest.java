package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
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
import net.finmath.montecarlo.assetderivativevaluation.models.HestonModel.Scheme;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Compares finite-difference prices of European barrier options under a Heston model
 * with Monte Carlo benchmark prices.
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
 * </p>
 * <ul>
 *   <li>Uses Apache Commons Math linear interpolation in the asset direction.</li>
 *   <li>Constructs the spot grid so that the barrier coincides with the relevant boundary.</li>
 *   <li>Uses a uniform variance grid for the second PDE dimension.</li>
 *   <li>Monte Carlo barrier monitoring is discrete on the simulation time grid.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionHestonFdmVsMonteCarloTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

	private static final double S0 = 100.0;
	private static final double R = 0.05;
	private static final double Q = 0.00;

	// Heston parameters
	private static final double VOLATILITY = 0.25;
	private static final double VOLATILITY_SQUARED = VOLATILITY * VOLATILITY;
	private static final double KAPPA = 1.5;
	private static final double THETA_H = VOLATILITY_SQUARED;
	private static final double XI = 0.30;
	private static final double RHO = -0.70;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 80;
	private static final int NUMBER_OF_SPACE_STEPS_S = 160;
	private static final int NUMBER_OF_SPACE_STEPS_V = 60;

	/**
	 * Number of space intervals between the barrier and S0.
	 * This ensures that S0 is aligned to a grid node in the S direction.
	 */
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	@Test
	public void testDownAndOutEuropeanCallHestonFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.CALL, BarrierType.DOWN_OUT, 80.0, 0.10);
	}

	@Test
	public void testDownAndInEuropeanCallHestonFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0, 0.10);
	}

	@Test
	public void testUpAndOutEuropeanCallHestonFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.CALL, BarrierType.UP_OUT, 120.0, 0.10);
	}

	@Test
	public void testUpAndInEuropeanCallHestonFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0, 0.10);
	}

	@Test
	public void testDownAndOutEuropeanPutHestonFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.PUT, BarrierType.DOWN_OUT, 80.0, 0.10);
	}

	@Test
	public void testDownAndInEuropeanPutHestonFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0, 0.10);
	}

	@Test
	public void testUpAndOutEuropeanPutHestonFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.PUT, BarrierType.UP_OUT, 120.0, 0.10);
	}

	@Test
	public void testUpAndInEuropeanPutHestonFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0, 0.10);
	}

	private void runBarrierTest(
			final CallOrPut callOrPut,
			final BarrierType barrierType,
			final double barrier,
			final double relativeTolerance) throws Exception {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS);

		final Grid sGrid = createSpotGrid(barrier, barrierType);

		/*
		 * Variance grid.
		 * Choose a sufficiently wide interval around the long-run mean theta.
		 */
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
				VOLATILITY,
				riskFreeCurve,
				dividendCurve,
				KAPPA,
				THETA_H,
				XI,
				RHO,
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

		/*
		 * FD solver returns the full flattened grid.
		 * Extract the time-0 slice and then interpolate in the spot dimension
		 * at the initial variance level.
		 */
		final double[] fdValuesOnGrid = fdmProduct.getValue(0.0, fdmModel);
		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();
		final double[] vNodes = spaceTime.getSpaceGrid(1).getGrid();

		final int v0Index = getNearestGridIndex(vNodes, VOLATILITY_SQUARED);

		final double[] fdSliceAtInitialVariance = new double[sNodes.length];
		for(int i = 0; i < sNodes.length; i++) {
			final int flatIndex = i + v0Index * sNodes.length;
			fdSliceAtInitialVariance[i] = fdValuesOnGrid[flatIndex];
		}

		final double fdPrice;
		final int s0Index = getGridIndex(sNodes, S0);
		if(s0Index >= 0) {
			fdPrice = fdSliceAtInitialVariance[s0Index];
		}
		else {
			final PolynomialSplineFunction interpolation =
					new LinearInterpolator().interpolate(sNodes, fdSliceAtInitialVariance);
			fdPrice = interpolation.value(S0);
		}

		// --- Monte Carlo benchmark ---
		final int numberOfPaths = 50_000;
		final int seed = 31415;
		final int mcNumberOfTimeSteps = 500;

		final TimeDiscretization mcTimes =
				new TimeDiscretizationFromArray(
						0.0,
						mcNumberOfTimeSteps,
						MATURITY / mcNumberOfTimeSteps);

		final BrownianMotionFromMersenneRandomNumbers brownianMotion =
				new BrownianMotionFromMersenneRandomNumbers(mcTimes, 2, numberOfPaths, seed);
		
		final net.finmath.montecarlo.assetderivativevaluation.models.HestonModel mcModel =
				new net.finmath.montecarlo.assetderivativevaluation.models.HestonModel(
						S0,
						R,
						VOLATILITY_SQUARED,
						THETA_H,
						KAPPA,
						XI,
						RHO,
						Scheme.FULL_TRUNCATION
				);

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

		assertTrue(fdPrice >= -1E-10);
		assertTrue(mcPrice >= -1E-10);

		final double denominator = Math.max(Math.abs(mcPrice), 1E-8);
		final double relativeError = Math.abs(fdPrice - mcPrice) / denominator;

		System.out.println("Relative error = " + relativeError);

		assertEquals(
				"Relative FD vs MC barrier price error for " + barrierType + " " + callOrPut,
				0.0,
				relativeError,
				relativeTolerance
		);
	}

	private Grid createSpotGrid(final double barrier, final BarrierType barrierType) {
		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {
			final double deltaS = (S0 - barrier) / STEPS_BETWEEN_BARRIER_AND_SPOT;
			final double sMin = barrier;
			final double sMax = sMin + NUMBER_OF_SPACE_STEPS_S * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS_S, sMin, sMax);
		}
		else {
			final double deltaS = (barrier - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;
			final double sMax = barrier;
			final double sMin = sMax - NUMBER_OF_SPACE_STEPS_S * deltaS;
			return new UniformGrid(NUMBER_OF_SPACE_STEPS_S, sMin, sMax);
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