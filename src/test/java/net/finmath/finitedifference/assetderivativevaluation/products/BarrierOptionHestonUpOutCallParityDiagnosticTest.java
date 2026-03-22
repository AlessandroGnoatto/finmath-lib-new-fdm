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

/**
 * Tiny diagnostic for Heston UP_OUT CALL:
 * compares direct out pricing with parity-out pricing on the same FD setup.
 *
 * <p>
 * Computes:
 * </p>
 * <ul>
 *   <li>vanilla call FD,</li>
 *   <li>direct up-and-out call FD,</li>
 *   <li>direct up-and-in call FD,</li>
 *   <li>parity out FD = vanilla FD - up-and-in FD,</li>
 *   <li>MC up-and-out call benchmark.</li>
 * </ul>
 */
public class BarrierOptionHestonUpOutCallParityDiagnosticTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double BARRIER = 140.0;
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

	private static final int NUMBER_OF_TIME_STEPS = 320;
	private static final int NUMBER_OF_SPACE_STEPS_S = 320;
	private static final int NUMBER_OF_SPACE_STEPS_V = 40;
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	private static final int MC_NUMBER_OF_TIME_STEPS = 1000;
	private static final int MC_NUMBER_OF_PATHS = 50_000;
	private static final int SEED = 31415;

	@Test
	public void diagnoseUpOutCallParityOut() throws Exception {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final Grid sGrid = createUpBarrierSpotGrid();
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

		final EuropeanOption vanillaCall = new EuropeanOption(
				MATURITY,
				STRIKE,
				CallOrPut.CALL
		);

		final BarrierOption upOutCall = new BarrierOption(
				MATURITY,
				STRIKE,
				BARRIER,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_OUT
		);

		final BarrierOption upInCall = new BarrierOption(
				MATURITY,
				STRIKE,
				BARRIER,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_IN
		);

		final double vanillaFd = extractAtS0V0(vanillaCall.getValue(0.0, fdmModel), spaceTime, S0, V0);
		final double directOutFd = extractAtS0V0(upOutCall.getValue(0.0, fdmModel), spaceTime, S0, V0);
		final double directInFd = extractAtS0V0(upInCall.getValue(0.0, fdmModel), spaceTime, S0, V0);
		final double parityOutFd = vanillaFd - directInFd;

		final double mcOut = getMonteCarloUpOutCallPrice();

		System.out.println("====================================");
		System.out.println("UP_OUT CALL parity diagnostic");
		System.out.println("Vanilla FD        = " + vanillaFd);
		System.out.println("Direct OUT FD     = " + directOutFd);
		System.out.println("Direct IN FD      = " + directInFd);
		System.out.println("Parity OUT FD     = " + parityOutFd);
		System.out.println("MC OUT            = " + mcOut);
		System.out.println("------------------------------------");
		System.out.println("Direct OUT abs err = " + Math.abs(directOutFd - mcOut));
		System.out.println("Direct OUT rel err = " + Math.abs(directOutFd - mcOut) / Math.max(Math.abs(mcOut), 1E-8));
		System.out.println("Parity OUT abs err = " + Math.abs(parityOutFd - mcOut));
		System.out.println("Parity OUT rel err = " + Math.abs(parityOutFd - mcOut) / Math.max(Math.abs(mcOut), 1E-8));
		System.out.println("------------------------------------");
		System.out.println("Vanilla - (IN + OUT direct) residual = " + Math.abs(vanillaFd - (directInFd + directOutFd)));

		assertTrue(vanillaFd >= -1E-10);
		assertTrue(directOutFd >= -1E-10);
		assertTrue(directInFd >= -1E-10);
		assertTrue(parityOutFd >= -1E-10);
		assertTrue(mcOut >= -1E-10);
	}

	private double getMonteCarloUpOutCallPrice() throws Exception {

		final TimeDiscretization mcTimes =
				new TimeDiscretizationFromArray(
						0.0,
						MC_NUMBER_OF_TIME_STEPS,
						MATURITY / MC_NUMBER_OF_TIME_STEPS
				);

		final BrownianMotionFromMersenneRandomNumbers brownianMotion =
				new BrownianMotionFromMersenneRandomNumbers(mcTimes, 2, MC_NUMBER_OF_PATHS, SEED);

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
						BARRIER,
						REBATE,
						CallOrPut.CALL,
						BarrierType.UP_OUT
				);

		return mcProduct.getValue(mcSimulation);
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

		return interpolateOnSlice(sNodes, fdSliceAtInitialVariance, s0);
	}

	private Grid createUpBarrierSpotGrid() {
		final double deltaS = Math.abs(BARRIER - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;
		final double sMax = BARRIER;
		final double desiredSMin = 0.0;
		final int numberOfSteps = Math.max(
				NUMBER_OF_SPACE_STEPS_S,
				(int)Math.round((sMax - desiredSMin) / deltaS)
		);
		final double sMin = Math.max(0.0, sMax - numberOfSteps * deltaS);
		return new UniformGrid(numberOfSteps, sMin, sMax);
	}

	private static double interpolateOnSlice(
			final double[] sNodes,
			final double[] values,
			final double stock) {

		final int gridIndex = getGridIndex(sNodes, stock);
		if(gridIndex >= 0) {
			return values[gridIndex];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, values);
		return interpolation.value(stock);
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