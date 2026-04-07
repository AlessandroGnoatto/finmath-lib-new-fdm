package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
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
 * Tests European in-out parity for barrier options under SABR.
 *
 * <p>
 * For European barriers, the identity
 * </p>
 *
 * <pre>
 * Vanilla = Knock-In + Knock-Out
 * </pre>
 *
 * <p>
 * should hold numerically up to PDE discretization error.
 * </p>
 */
public class BarrierOptionSabrParityTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

	private static final double S0 = 100.0;
	private static final double R = 0.05;
	private static final double Q = 0.00;

	private static final double INITIAL_ALPHA = 0.25;
	private static final double BETA = 0.50;
	private static final double NU = 0.40;
	private static final double RHO = -0.30;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS_S = 160;
	private static final int NUMBER_OF_SPACE_STEPS_ALPHA = 100;

	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

	/*
	 * Relative tolerance for parity:
	 * | vanilla - (in + out) | / max(|vanilla|, 1e-8)
	 */
	private static final double PARITY_RELATIVE_TOLERANCE = 0.03;

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

		final double alphaMin = 0.0;
		final double alphaMax = Math.max(1.0, 4.0 * INITIAL_ALPHA);
		final Grid alphaGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_ALPHA, alphaMin, alphaMax);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, alphaGrid },
				timeDiscretization,
				THETA,
				new double[] { S0, INITIAL_ALPHA }
		);

		final FDMSabrModel fdmModel = new FDMSabrModel(
				S0,
				INITIAL_ALPHA,
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

		final double vanillaPrice = interpolatePriceAtInitialState(vanilla.getValue(0.0, fdmModel), fdmModel);
		final double knockInPrice = interpolatePriceAtInitialState(knockIn.getValue(0.0, fdmModel), fdmModel);
		final double knockOutPrice = interpolatePriceAtInitialState(knockOut.getValue(0.0, fdmModel), fdmModel);

		final double parityResidual = vanillaPrice - (knockInPrice + knockOutPrice);
		final double relativeParityResidual = Math.abs(parityResidual) / Math.max(Math.abs(vanillaPrice), 1E-8);

		System.out.println("SABR PARITY TEST");
		System.out.println("Call/Put       = " + callOrPut);
		System.out.println("Barrier        = " + barrier);
		System.out.println("Knock-in type  = " + knockInType);
		System.out.println("Knock-out type = " + knockOutType);
		System.out.println("Vanilla        = " + vanillaPrice);
		System.out.println("Knock-In       = " + knockInPrice);
		System.out.println("Knock-Out      = " + knockOutPrice);
		System.out.println("Residual       = " + parityResidual);
		System.out.println("RelResidual    = " + relativeParityResidual);

		assertTrue(Double.isFinite(vanillaPrice));
		assertTrue(Double.isFinite(knockInPrice));
		assertTrue(Double.isFinite(knockOutPrice));

		assertTrue(vanillaPrice >= -1E-10);
		assertTrue(knockInPrice >= -1E-10);
		assertTrue(knockOutPrice >= -1E-10);

		assertTrue(
				"Knock-in price should not exceed vanilla price.",
				knockInPrice <= vanillaPrice + 1E-8
		);

		assertTrue(
				"Knock-out price should not exceed vanilla price.",
				knockOutPrice <= vanillaPrice + 1E-8
		);

		assertEquals(
				"European in-out parity residual too large for " + knockInType + " / " + knockOutType + " " + callOrPut,
				0.0,
				relativeParityResidual,
				PARITY_RELATIVE_TOLERANCE
		);
	}

	private double interpolatePriceAtInitialState(
			final double[] valuesOnGrid,
			final FDMSabrModel model) {

		final SpaceTimeDiscretization spaceTime = model.getSpaceTimeDiscretization();
		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();
		final double[] alphaNodes = spaceTime.getSpaceGrid(1).getGrid();

		final int nS = sNodes.length;
		final int nAlpha = alphaNodes.length;

		final int alphaIndex = getNearestGridIndex(alphaNodes, INITIAL_ALPHA);

		final double[] sliceAtInitialAlpha = new double[nS];
		for(int i = 0; i < nS; i++) {
			final int flatIndex = i + alphaIndex * nS;
			sliceAtInitialAlpha[i] = valuesOnGrid[flatIndex];
		}

		final int s0Index = getGridIndex(sNodes, S0);
		if(s0Index >= 0) {
			return sliceAtInitialAlpha[s0Index];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, sliceAtInitialAlpha);
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
			final double sMax = barrier + 8.0 * deltaS;

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