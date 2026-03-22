
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
 * Focused regression test for the two Heston barrier cases that are still problematic:
 * <ul>
 *   <li>UP_OUT CALL</li>
 *   <li>DOWN_OUT PUT</li>
 * </ul>
 *
 * <p>
 * This is intended to isolate boundary-condition / continuation-region issues
 * in the direct Heston knock-out PDE solver.
 * </p>
 */
public class BarrierOptionHestonOutliersFdmVsMonteCarloTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

	private static final double S0 = 100.0;
	private static final double R = 0.05;
	private static final double Q = 0.00;

	private static final double VOLATILITY = 0.25;
	private static final double V0 = VOLATILITY * VOLATILITY;
	private static final double KAPPA = 1.5;
	private static final double THETA_H = V0;
	private static final double XI = 0.30;
	private static final double RHO = -0.70;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 80;
	private static final int NUMBER_OF_SPACE_STEPS_S = 160;
	private static final int NUMBER_OF_SPACE_STEPS_V = 40;

	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	@Test
	public void testUpAndOutEuropeanCallHestonFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.CALL, BarrierType.UP_OUT, 120.0, 0.20);
	}

	@Test
	public void testDownAndOutEuropeanPutHestonFiniteDifferenceVsMonteCarlo() throws Exception {
		runBarrierTest(CallOrPut.PUT, BarrierType.DOWN_OUT, 80.0, 0.20);
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

		final double vMin = 0.0;
		final double vMax = Math.max(4.0 * THETA_H, V0 + 4.0 * XI * Math.sqrt(MATURITY));
		final Grid vGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_V, vMin, vMax);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, vGrid },
				timeDiscretization,
				THETA,
				new double[] { S0, V0 }
		);

		final FDMHestonModel fdmModel = new FDMHestonModel(
				S0,
				V0,
				riskFreeCurve,
				dividendCurve,
				KAPPA,
				THETA_H,
				XI,
				RHO,
				spaceTime
		);

		final BarrierOption fdmProduct = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				barrierType
		);

		final double[] fdValuesOnGrid = fdmProduct.getValue(0.0, fdmModel);
		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();
		final double[] vNodes = spaceTime.getSpaceGrid(1).getGrid();

		assertTrue("S0 must lie inside the spot grid domain.", S0 >= sNodes[0] - 1E-12 && S0 <= sNodes[sNodes.length - 1] + 1E-12);

		final int v0Index = getNearestGridIndex(vNodes, V0);

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
						R - Q,
						VOLATILITY,
						R,
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
		System.out.println("Barrier        = " + barrier);
		System.out.println("Grid min       = " + sNodes[0]);
		System.out.println("Grid max       = " + sNodes[sNodes.length - 1]);
		System.out.println("Grid size S    = " + sNodes.length);
		System.out.println("Grid size V    = " + vNodes.length);
		System.out.println("S0 on grid     = " + (s0Index >= 0));
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

		final double deltaS = Math.abs(barrier - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;

		if(barrierType == BarrierType.DOWN_OUT) {
			final double sMin = barrier;
			final double desiredSMax = Math.max(3.0 * S0, S0 + 12.0 * deltaS);
			final int numberOfSteps = Math.max(
					NUMBER_OF_SPACE_STEPS_S,
					(int)Math.round((desiredSMax - sMin) / deltaS)
			);
			final double sMax = sMin + numberOfSteps * deltaS;
			return new UniformGrid(numberOfSteps, sMin, sMax);
		}
		else if(barrierType == BarrierType.UP_OUT) {
			final double sMax = barrier;
			final double desiredSMin = 0.0;
			final int numberOfSteps = Math.max(
					NUMBER_OF_SPACE_STEPS_S,
					(int)Math.round((sMax - desiredSMin) / deltaS)
			);
			final double sMin = Math.max(0.0, sMax - numberOfSteps * deltaS);
			return new UniformGrid(numberOfSteps, sMin, sMax);
		}
		else {
			throw new IllegalArgumentException("This focused regression test supports only knock-out options.");
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