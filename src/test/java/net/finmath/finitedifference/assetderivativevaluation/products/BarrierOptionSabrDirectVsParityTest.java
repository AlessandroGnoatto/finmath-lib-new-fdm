package net.finmath.finitedifference.assetderivativevaluation.products;

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
 * Compares direct 2D SABR knock-in pricing against the parity value
 * {@code vanilla - knockOut} on the same model setup.
 *
 * <p>
 * This test is intended as a diagnostic. It does not assume the direct solver
 * is already accurate enough to pass a tight tolerance. Instead it prints the
 * discrepancy for all four European knock-in types and checks only basic
 * finiteness / no-explosion conditions.
 * </p>
 */
public class BarrierOptionSabrDirectVsParityTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

	private static final double FORWARD0 = 100.0;
	private static final double R = 0.00;
	private static final double Q = 0.00;

	private static final double ALPHA0 = 0.25;
	private static final double BETA = 0.50;
	private static final double NU = 0.40;
	private static final double RHO = -0.30;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS_F = 180;
	private static final int NUMBER_OF_SPACE_STEPS_ALPHA = 100;

	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
	private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

	@Test
	public void testDownInEuropeanPutDirectVsParitySabr() throws Exception {
		runComparison(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testDownInEuropeanCallDirectVsParitySabr() throws Exception {
		runComparison(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testUpInEuropeanPutDirectVsParitySabr() throws Exception {
		runComparison(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
	}

	@Test
	public void testUpInEuropeanCallDirectVsParitySabr() throws Exception {
		runComparison(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
	}

	private void runComparison(
			final CallOrPut callOrPut,
			final BarrierType knockInType,
			final double barrier) throws Exception {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS);

		final Grid forwardGrid = createForwardGrid(barrier, knockInType);

		final double alphaMin = 0.0;
		final double alphaMax = Math.max(4.0 * ALPHA0, ALPHA0 + 4.0 * NU * Math.sqrt(MATURITY));
		final Grid alphaGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_ALPHA, alphaMin, alphaMax);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { forwardGrid, alphaGrid },
				timeDiscretization,
				THETA,
				new double[] { FORWARD0, ALPHA0 }
		);

		final FDMSabrModel fdmModel = new FDMSabrModel(
				FORWARD0,
				ALPHA0,
				riskFreeCurve,
				dividendCurve,
				BETA,
				NU,
				RHO,
				spaceTime
		);

		final EuropeanOption vanilla = new EuropeanOption(null, MATURITY, STRIKE, callOrPut);

		final BarrierOption directKnockIn = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				knockInType
		);

		final BarrierType knockOutType =
				knockInType == BarrierType.DOWN_IN ? BarrierType.DOWN_OUT : BarrierType.UP_OUT;

		final BarrierOption knockOut = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				knockOutType
		);

		final long start = System.currentTimeMillis();

		final double[] vanillaValuesOnGrid = vanilla.getValue(0.0, fdmModel);
		final double[] directKnockInValuesOnGrid = directKnockIn.getValue(0.0, fdmModel);
		final double[] knockOutValuesOnGrid = knockOut.getValue(0.0, fdmModel);

		final long end = System.currentTimeMillis();

		final double[] fNodes = forwardGrid.getGrid();
		final double[] aNodes = alphaGrid.getGrid();

		final int nF = fNodes.length;
		final int nA = aNodes.length;
		final int alpha0Index = getNearestGridIndex(aNodes, ALPHA0);

		final double vanillaPrice = interpolateAlongFirstState(
				fNodes,
				extractSliceAtSecondStateIndex(vanillaValuesOnGrid, nF, nA, alpha0Index),
				FORWARD0
		);

		final double directKnockInPrice = interpolateAlongFirstState(
				fNodes,
				extractSliceAtSecondStateIndex(directKnockInValuesOnGrid, nF, nA, alpha0Index),
				FORWARD0
		);

		final double knockOutPrice = interpolateAlongFirstState(
				fNodes,
				extractSliceAtSecondStateIndex(knockOutValuesOnGrid, nF, nA, alpha0Index),
				FORWARD0
		);

		final double parityKnockInPrice = vanillaPrice - knockOutPrice;
		final double residual = directKnockInPrice - parityKnockInPrice;
		final double relResidual = Math.abs(residual) / Math.max(Math.abs(parityKnockInPrice), 1E-10);

		final double gridMin = getMin(directKnockInValuesOnGrid);
		final int minIndex = getMinIndex(directKnockInValuesOnGrid);
		final int minIF = minIndex % nF;
		final int minIA = minIndex / nF;

		System.out.println("SABR DIRECT VS PARITY");
		System.out.println("Call/Put       = " + callOrPut);
		System.out.println("Barrier        = " + barrier);
		System.out.println("Knock-in type  = " + knockInType);
		System.out.println("Knock-out type = " + knockOutType);
		System.out.println("Runtime        = " + (end - start) / 1000.0 + " s");
		System.out.println("Vanilla        = " + vanillaPrice);
		System.out.println("Direct KI      = " + directKnockInPrice);
		System.out.println("Parity KI      = " + parityKnockInPrice);
		System.out.println("Knock-Out      = " + knockOutPrice);
		System.out.println("Residual       = " + residual);
		System.out.println("RelResidual    = " + relResidual);
		System.out.println("Grid min       = " + gridMin);
		System.out.println("F0 on grid     = " + (getGridIndex(fNodes, FORWARD0) >= 0));
		System.out.println("Barrier on grid= " + (getGridIndex(fNodes, barrier) >= 0));
		System.out.println(
				"Min located at indices (iF,iA) = (" + minIF + "," + minIA + "), state = ("
						+ fNodes[minIF] + ", " + aNodes[minIA] + ")"
		);

		assertTrue(Double.isFinite(vanillaPrice));
		assertTrue(Double.isFinite(directKnockInPrice));
		assertTrue(Double.isFinite(parityKnockInPrice));
		assertTrue(Double.isFinite(knockOutPrice));
		assertTrue(Double.isFinite(gridMin));

		assertTrue("Vanilla price must be non-negative.", vanillaPrice >= -1E-10);
		assertTrue("Knock-out price must be non-negative.", knockOutPrice >= -1E-10);
		assertTrue("Direct knock-in price must be non-negative up to a small tolerance.", directKnockInPrice >= -5E-2);
		assertTrue("Direct knock-in should not explode above vanilla.", directKnockInPrice <= vanillaPrice + 1.0);
		assertTrue("Barrier must be exactly on the forward grid.", getGridIndex(fNodes, barrier) >= 0);
	}

	private Grid createForwardGrid(final double barrier, final BarrierType barrierType) {

		final double deltaF = Math.abs(barrier - FORWARD0) / STEPS_BETWEEN_BARRIER_AND_SPOT;

		if(barrierType == BarrierType.DOWN_IN) {
			final double fMin = Math.max(0.0, barrier - deltaF);
			final double desiredFMax = Math.max(3.0 * FORWARD0, FORWARD0 + 12.0 * deltaF);

			final int numberOfSteps = Math.max(
					NUMBER_OF_SPACE_STEPS_F,
					(int)Math.round((desiredFMax - fMin) / deltaF)
			);

			final double fMax = fMin + numberOfSteps * deltaF;

			return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
					numberOfSteps,
					fMin,
					fMax,
					barrier,
					BARRIER_CLUSTERING_EXPONENT
			);
		}
		else if(barrierType == BarrierType.UP_IN) {
			final double fMax = barrier + deltaF;
			final double desiredFMin = 0.0;

			final int numberOfSteps = Math.max(
					NUMBER_OF_SPACE_STEPS_F,
					(int)Math.round((fMax - desiredFMin) / deltaF)
			);

			final double fMin = Math.max(0.0, fMax - numberOfSteps * deltaF);

			return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
					numberOfSteps,
					fMin,
					fMax,
					barrier,
					BARRIER_CLUSTERING_EXPONENT
			);
		}
		else {
			throw new IllegalArgumentException("Comparison grid requested for non knock-in barrier type.");
		}
	}

	private static double[] extractSliceAtSecondStateIndex(
			final double[] valuesOnGrid,
			final int nFirstState,
			final int nSecondState,
			final int secondStateIndex) {

		if(secondStateIndex < 0 || secondStateIndex >= nSecondState) {
			throw new IllegalArgumentException("secondStateIndex out of range.");
		}

		final double[] slice = new double[nFirstState];
		for(int i = 0; i < nFirstState; i++) {
			slice[i] = valuesOnGrid[i + secondStateIndex * nFirstState];
		}
		return slice;
	}

	private static double interpolateAlongFirstState(
			final double[] nodes,
			final double[] values,
			final double x) {

		final int index = getGridIndex(nodes, x);
		if(index >= 0) {
			return values[index];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(nodes, values);
		return interpolation.value(x);
	}

	private static double getMin(final double[] values) {
		double min = values[0];
		for(int i = 1; i < values.length; i++) {
			min = Math.min(min, values[i]);
		}
		return min;
	}

	private static int getMinIndex(final double[] values) {
		int minIndex = 0;
		double min = values[0];
		for(int i = 1; i < values.length; i++) {
			if(values[i] < min) {
				min = values[i];
				minIndex = i;
			}
		}
		return minIndex;
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