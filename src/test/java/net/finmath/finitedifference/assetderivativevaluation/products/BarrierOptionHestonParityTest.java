package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
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
 * Tests European in-out parity for barrier options under Heston,
 * using finmath's Fourier pricer as the vanilla anchor.
 *
 * <p>
 * For European barriers:
 * </p>
 *
 * <pre>
 * Vanilla = Knock-In + Knock-Out
 * </pre>
 *
 * <p>
 * Here "Vanilla" is taken from the finmath Fourier Heston pricer, not from
 * FD on the barrier-aligned grid. This avoids contamination from a vanilla
 * PDE solve on a barrier-specific spatial grid.
 * </p>
 */
public class BarrierOptionHestonParityTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

	private static final double S0 = 100.0;
	private static final double R = 0.05;
	private static final double Q = 0.00;

	private static final double VOLATILITY = 0.25;
	private static final double VOLATILITY_SQUARED = VOLATILITY * VOLATILITY;
	private static final double KAPPA = 1.5;
	private static final double THETA_H = VOLATILITY_SQUARED;
	private static final double XI = 0.30;
	private static final double RHO = -0.70;

	private static final double THETA = 1.0/3.0;

	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS_S = 160;
	private static final int NUMBER_OF_SPACE_STEPS_V = 100;

	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 60;
	private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

	/*
	 * Since the vanilla anchor now comes from Fourier, this can be made tighter
	 * than when vanilla was computed on the same barrier-shaped FD grid.
	 */
	private static final double PARITY_RELATIVE_TOLERANCE = 0.05;

	@Test
	public void testDownBarrierCallParity() throws Exception {
		runParityTest(CallOrPut.CALL, 80.0, BarrierType.DOWN_IN, BarrierType.DOWN_OUT);
	}

	@Test
	public void testDownBarrierPutParity() throws Exception {
		runParityTest(CallOrPut.PUT, 80.0, BarrierType.DOWN_IN, BarrierType.DOWN_OUT);
	}

	@Test
	public void testUpBarrierCallParity() throws Exception {
		runParityTest(CallOrPut.CALL, 120.0, BarrierType.UP_IN, BarrierType.UP_OUT);
	}

	@Test
	public void testUpBarrierPutParity() throws Exception {
		runParityTest(CallOrPut.PUT, 120.0, BarrierType.UP_IN, BarrierType.UP_OUT);
	}

	private void runParityTest(
			final CallOrPut callOrPut,
			final double barrier,
			final BarrierType knockInType,
			final BarrierType knockOutType) throws Exception {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS);

		final Grid sGrid = createSpotGrid(barrier, knockOutType);

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
				VOLATILITY_SQUARED,
				riskFreeCurve,
				dividendCurve,
				KAPPA,
				THETA_H,
				XI,
				RHO,
				spaceTime
		);

		final BarrierOption knockIn = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				knockInType
		);

		final BarrierOption knockOut = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				knockOutType
		);

		final double[] knockInValuesOnGrid = knockIn.getValue(0.0, fdmModel);
		final double[] knockOutValuesOnGrid = knockOut.getValue(0.0, fdmModel);

		final double knockInPrice = interpolatePriceAtInitialState(knockInValuesOnGrid, fdmModel);
		final double knockOutPrice = interpolatePriceAtInitialState(knockOutValuesOnGrid, fdmModel);

		final double vanillaFourierPrice = getFourierVanillaPrice(callOrPut);

		final double parityResidual = vanillaFourierPrice - (knockInPrice + knockOutPrice);
		final double relativeParityResidual =
				Math.abs(parityResidual) / Math.max(Math.abs(vanillaFourierPrice), 1E-8);

		final double knockInParityPrice = vanillaFourierPrice - knockOutPrice;
		final double directVsParityDifference = knockInPrice - knockInParityPrice;
		final double relativeDirectVsParityDifference =
				Math.abs(directVsParityDifference) / Math.max(Math.abs(vanillaFourierPrice), 1E-8);

		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();

		System.out.println("HESTON PARITY TEST (FOURIER VANILLA)");
		System.out.println("Call/Put                 = " + callOrPut);
		System.out.println("Barrier                  = " + barrier);
		System.out.println("Knock-in type            = " + knockInType);
		System.out.println("Knock-out type           = " + knockOutType);
		System.out.println("Grid min                 = " + sNodes[0]);
		System.out.println("Grid max                 = " + sNodes[sNodes.length - 1]);
		System.out.println("Barrier on grid          = " + (getGridIndex(sNodes, barrier) >= 0));
		System.out.println("S0 on grid               = " + (getGridIndex(sNodes, S0) >= 0));
		System.out.println("Vanilla (Fourier)        = " + vanillaFourierPrice);
		System.out.println("Knock-In direct          = " + knockInPrice);
		System.out.println("Knock-Out direct         = " + knockOutPrice);
		System.out.println("Knock-In via parity      = " + knockInParityPrice);
		System.out.println("Residual                 = " + parityResidual);
		System.out.println("RelResidual              = " + relativeParityResidual);
		System.out.println("Direct minus parity KI   = " + directVsParityDifference);
		System.out.println("RelDirect minus parity KI= " + relativeDirectVsParityDifference);

		assertTrue(Double.isFinite(vanillaFourierPrice));
		assertTrue(Double.isFinite(knockInPrice));
		assertTrue(Double.isFinite(knockOutPrice));

		assertTrue(vanillaFourierPrice >= -1E-10);
		assertTrue(knockInPrice >= -1E-10);
		assertTrue(knockOutPrice >= -1E-10);

		assertTrue(
				"Knock-in price should not exceed vanilla price.",
				knockInPrice <= vanillaFourierPrice + 1E-8
		);

		assertTrue(
				"Knock-out price should not exceed vanilla price.",
				knockOutPrice <= vanillaFourierPrice + 1E-8
		);

		assertEquals(
				"European in-out parity residual too large for "
						+ knockInType + " / " + knockOutType + " " + callOrPut,
				0.0,
				relativeParityResidual,
				PARITY_RELATIVE_TOLERANCE
		);
	}

	/**
	 * Uses finmath's Fourier Heston pricer for the vanilla anchor.
	 *
	 * <p>
	 * The existing Heston Fourier tests in the repo price the call directly with:
	 * </p>
	 *
	 * <pre>
	 * new net.finmath.fouriermethod.models.HestonModel(
	 *     s0, r-q, sqrt(v0), r, thetaV, kappa, xi, rho)
	 * </pre>
	 *
	 * <p>
	 * and
	 * </p>
	 *
	 * <pre>
	 * new net.finmath.fouriermethod.products.EuropeanOption(maturity, strike)
	 * </pre>
	 *
	 * <p>
	 * For puts, we recover the price from call-put parity.
	 * </p>
	 */
	private double getFourierVanillaPrice(final CallOrPut callOrPut) throws Exception {

		final net.finmath.fouriermethod.models.HestonModel fourierModel =
				new net.finmath.fouriermethod.models.HestonModel(
						S0,
						R - Q,
						Math.sqrt(VOLATILITY_SQUARED),
						R,
						THETA_H,
						KAPPA,
						XI,
						RHO
				);

		final net.finmath.fouriermethod.products.EuropeanOption fourierCall =
				new net.finmath.fouriermethod.products.EuropeanOption(MATURITY, STRIKE);

		final double callPrice = fourierCall.getValue(fourierModel);

		if(callOrPut == CallOrPut.CALL) {
			return callPrice;
		}

		final double discountFactor = Math.exp(-R * MATURITY);
		final double dividendDiscountFactor = Math.exp(-Q * MATURITY);

		return callPrice - S0 * dividendDiscountFactor + STRIKE * discountFactor;
	}

	private double interpolatePriceAtInitialState(
			final double[] valuesOnGrid,
			final FDMHestonModel model) {

		final SpaceTimeDiscretization spaceTime = model.getSpaceTimeDiscretization();
		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();
		final double[] vNodes = spaceTime.getSpaceGrid(1).getGrid();

		final int nS = sNodes.length;
		final int v0Index = getNearestGridIndex(vNodes, VOLATILITY_SQUARED);

		final double[] sliceAtInitialVariance = new double[nS];
		for(int i = 0; i < nS; i++) {
			final int flatIndex = i + v0Index * nS;
			sliceAtInitialVariance[i] = valuesOnGrid[flatIndex];
		}

		final int s0Index = getGridIndex(sNodes, S0);
		if(s0Index >= 0) {
			return sliceAtInitialVariance[s0Index];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, sliceAtInitialVariance);
		return interpolation.value(S0);
	}

	private Grid createSpotGrid(final double barrier, final BarrierType barrierType) {

		final double deltaS = Math.abs(barrier - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;
		final boolean isKnockIn =
				barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.UP_IN;

		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {

			final double sMin = Math.max(1E-8, barrier - 8.0 * deltaS);
			final double sMax = Math.max(3.0 * S0, S0 + 12.0 * deltaS);

			final int numberOfSteps = Math.max(
					NUMBER_OF_SPACE_STEPS_S,
					(int)Math.round((sMax - sMin) / deltaS)
			);

			if(isKnockIn) {
				return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
						numberOfSteps,
						sMin,
						sMax,
						barrier,
						BARRIER_CLUSTERING_EXPONENT
				);
			}
			else {
				return BarrierAlignedSpotGridFactory.createBarrierAlignedUniformGrid(
						numberOfSteps,
						sMin,
						sMax,
						barrier
				);
			}
		}
		else {
			final double sMin = 0.0;
			final double sMax = Math.max(2.0 * S0, barrier + 40.0 * deltaS);

			final int numberOfSteps = Math.max(
					NUMBER_OF_SPACE_STEPS_S,
					(int)Math.round((sMax - sMin) / deltaS)
			);

			if(isKnockIn) {
				return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
						numberOfSteps,
						sMin,
						sMax,
						barrier,
						BARRIER_CLUSTERING_EXPONENT
				);
			}
			else {
				return BarrierAlignedSpotGridFactory.createBarrierAlignedUniformGrid(
						numberOfSteps,
						sMin,
						sMax,
						barrier
				);
			}
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