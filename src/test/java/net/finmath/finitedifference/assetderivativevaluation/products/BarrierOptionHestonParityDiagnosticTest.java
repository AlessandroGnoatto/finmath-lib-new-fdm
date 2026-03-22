package net.finmath.finitedifference.assetderivativevaluation.products;

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

public class BarrierOptionHestonParityDiagnosticTest {

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
	private static final int NUMBER_OF_SPACE_STEPS_V = 40;
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	@Test
	public void diagnoseUpOutCall() throws Exception {
		runParityDiagnostic(CallOrPut.CALL, BarrierType.UP_OUT, BarrierType.UP_IN, 120.0);
	}

	@Test
	public void diagnoseDownOutPut() throws Exception {
		runParityDiagnostic(CallOrPut.PUT, BarrierType.DOWN_OUT, BarrierType.DOWN_IN, 80.0);
	}

	private void runParityDiagnostic(
			final CallOrPut callOrPut,
			final BarrierType outType,
			final BarrierType inType,
			final double barrier) throws Exception {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, NUMBER_OF_TIME_STEPS, MATURITY / NUMBER_OF_TIME_STEPS);

		final Grid sGrid = createSpotGrid(barrier, outType);
		final double vMin = 0.0;
		final double vMax = Math.max(4.0 * THETA_H, V0 + 4.0 * XI * Math.sqrt(MATURITY));
		final Grid vGrid = new UniformGrid(40, vMin, vMax);

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

		final EuropeanOption vanilla = new EuropeanOption(MATURITY, STRIKE, callOrPut);
		final BarrierOption outOption = new BarrierOption(MATURITY, STRIKE, barrier, REBATE, callOrPut, outType);
		final BarrierOption inOption = new BarrierOption(MATURITY, STRIKE, barrier, REBATE, callOrPut, inType);

		final double vanillaFd = extractAtS0V0(vanilla.getValue(0.0, fdmModel), spaceTime, S0, V0);
		final double outFd = extractAtS0V0(outOption.getValue(0.0, fdmModel), spaceTime, S0, V0);
		final double inFdDirect = extractAtS0V0(inOption.getValue(0.0, fdmModel), spaceTime, S0, V0);
		final double inFdParity = vanillaFd - outFd;

		final int numberOfPaths = 50_000;
		final int seed = 31415;
		final int mcNumberOfTimeSteps = 500;

		final TimeDiscretization mcTimes =
				new TimeDiscretizationFromArray(0.0, mcNumberOfTimeSteps, MATURITY / mcNumberOfTimeSteps);

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

		final EulerSchemeFromProcessModel process = new EulerSchemeFromProcessModel(mcModel, brownianMotion);
		final MonteCarloAssetModel mcSimulation = new MonteCarloAssetModel(process);

		final net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption mcOut =
				new net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption(
						MATURITY, STRIKE, barrier, REBATE, callOrPut, outType);

		final net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption mcIn =
				new net.finmath.montecarlo.assetderivativevaluation.myproducts.BarrierOption(
						MATURITY, STRIKE, barrier, REBATE, callOrPut, inType);

		final double outMc = mcOut.getValue(mcSimulation);
		final double inMc = mcIn.getValue(mcSimulation);

		System.out.println("====================================");
		System.out.println("Case            = " + outType + " " + callOrPut);
		System.out.println("Vanilla FD       = " + vanillaFd);
		System.out.println("Out FD           = " + outFd);
		System.out.println("In FD direct     = " + inFdDirect);
		System.out.println("In FD parity     = " + inFdParity);
		System.out.println("Out MC           = " + outMc);
		System.out.println("In MC            = " + inMc);
		System.out.println("Parity residual  = " + Math.abs(vanillaFd - (outFd + inFdDirect)));

		assertTrue(vanillaFd >= -1E-10);
		assertTrue(outFd >= -1E-10);
		assertTrue(inFdDirect >= -1E-10);
		assertTrue(outMc >= -1E-10);
		assertTrue(inMc >= -1E-10);
	}

	private double extractAtS0V0(
			final double[] valuesOnGrid,
			final SpaceTimeDiscretization spaceTime,
			final double s0,
			final double v0) {

		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();
		final double[] vNodes = spaceTime.getSpaceGrid(1).getGrid();

		final int v0Index = getNearestGridIndex(vNodes, v0);

		final double[] fdSliceAtInitialVariance = new double[sNodes.length];
		for(int i = 0; i < sNodes.length; i++) {
			final int flatIndex = i + v0Index * sNodes.length;
			fdSliceAtInitialVariance[i] = valuesOnGrid[flatIndex];
		}

		final int s0Index = getGridIndex(sNodes, s0);
		if(s0Index >= 0) {
			return fdSliceAtInitialVariance[s0Index];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, fdSliceAtInitialVariance);
		return interpolation.value(s0);
	}

	private Grid createSpotGrid(final double barrier, final BarrierType barrierType) {
		final double deltaS = Math.abs(barrier - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;

		if(barrierType == BarrierType.DOWN_OUT || barrierType == BarrierType.DOWN_IN) {
			final double sMin = barrier;
			final double desiredSMax = Math.max(3.0 * S0, S0 + 12.0 * deltaS);
			final int numberOfSteps = Math.max(160, (int)Math.round((desiredSMax - sMin) / deltaS));
			final double sMax = sMin + numberOfSteps * deltaS;
			return new UniformGrid(numberOfSteps, sMin, sMax);
		}
		else {
			final double sMax = barrier;
			final double desiredSMin = 0.0;
			final int numberOfSteps = Math.max(160, (int)Math.round((sMax - desiredSMin) / deltaS));
			return new UniformGrid(numberOfSteps, desiredSMin, sMax);
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
		return DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
				name,
				LocalDate.of(2010, 8, 1),
				times,
				zeroRates,
				InterpolationMethod.LINEAR,
				ExtrapolationMethod.CONSTANT,
				InterpolationEntity.VALUE
		);
	}
}