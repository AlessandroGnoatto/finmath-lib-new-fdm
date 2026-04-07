package net.finmath.finitedifference.assetderivativevaluation.products;

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
 * Smoke test for direct 2D knock-in barrier pricing under Heston.
 *
 * <p>
 * This test does not compare against Monte Carlo. Instead it checks that the newly
 * introduced direct pre-hit knock-in PDE path produces numerically sane values:
 * </p>
 * <ul>
 *   <li>finite prices,</li>
 *   <li>non-negative prices,</li>
 *   <li>knock-in prices not exceeding vanilla prices,</li>
 *   <li>no large negative pockets on the grid.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionHestonDirectKnockInSmokeTest {

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

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS_S = 160;
	private static final int NUMBER_OF_SPACE_STEPS_V = 100;

	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;
	private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

	private static final double MAX_ALLOWED_NEGATIVE_GRID_VALUE = 5E-2;

	@Test
	public void testDownInEuropeanCallDirectHestonSmoke() throws Exception {
		runSmokeTest(CallOrPut.CALL, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testDownInEuropeanPutDirectHestonSmoke() throws Exception {
		runSmokeTest(CallOrPut.PUT, BarrierType.DOWN_IN, 80.0);
	}

	@Test
	public void testUpInEuropeanCallDirectHestonSmoke() throws Exception {
		runSmokeTest(CallOrPut.CALL, BarrierType.UP_IN, 120.0);
	}

	@Test
	public void testUpInEuropeanPutDirectHestonSmoke() throws Exception {
		runSmokeTest(CallOrPut.PUT, BarrierType.UP_IN, 120.0);
	}

	private void runSmokeTest(
			final CallOrPut callOrPut,
			final BarrierType barrierType,
			final double barrier) throws Exception {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS);

		final Grid sGrid = createSpotGrid(barrier, barrierType);

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

		final EuropeanOption vanilla = new EuropeanOption(null, MATURITY, STRIKE, callOrPut);

		final BarrierOption knockIn = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut,
				barrierType
		);

		final long start = System.currentTimeMillis();

		final double[] vanillaValuesOnGrid = vanilla.getValue(0.0, fdmModel);
		final double[] knockInValuesOnGrid = knockIn.getValue(0.0, fdmModel);

		final long end = System.currentTimeMillis();

		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();
		final double[] vNodes = spaceTime.getSpaceGrid(1).getGrid();
		final int nS = sNodes.length;
		final int nV = vNodes.length;

		final int v0Index = getNearestGridIndex(vNodes, VOLATILITY_SQUARED);

		final double[] vanillaSliceAtInitialVariance = extractSliceAtSecondStateIndex(vanillaValuesOnGrid, nS, nV, v0Index);
		final double[] knockInSliceAtInitialVariance = extractSliceAtSecondStateIndex(knockInValuesOnGrid, nS, nV, v0Index);

		final double vanillaPrice = interpolateAlongSpot(sNodes, vanillaSliceAtInitialVariance, S0);
		final double knockInPrice = interpolateAlongSpot(sNodes, knockInSliceAtInitialVariance, S0);

		final double gridMin = getMin(knockInValuesOnGrid);
		final double gridMax = getMax(knockInValuesOnGrid);
		final int minIndex = getMinIndex(knockInValuesOnGrid);

		final int minIS = minIndex % nS;
		final int minIV = minIndex / nS;

		System.out.println("HESTON DIRECT KNOCK-IN SMOKE");
		System.out.println("Type           = " + barrierType + " " + callOrPut);
		System.out.println("Runtime        = " + (end - start) / 1000.0 + " s");
		System.out.println("Grid min       = " + gridMin);
		System.out.println("Grid max       = " + gridMax);
		System.out.println("Vanilla price  = " + vanillaPrice);
		System.out.println("Knock-in price = " + knockInPrice);
		System.out.println("S0 on grid     = " + (getGridIndex(sNodes, S0) >= 0));
		System.out.println("Barrier on grid= " + (getGridIndex(sNodes, barrier) >= 0));
		System.out.println(
				"Min located at indices (iS,iV) = (" + minIS + "," + minIV + "), state = ("
						+ sNodes[minIS] + ", " + vNodes[minIV] + ")"
		);

		assertTrue("Vanilla price must be finite.", Double.isFinite(vanillaPrice));
		assertTrue("Knock-in price must be finite.", Double.isFinite(knockInPrice));

		assertTrue("Vanilla price must be non-negative.", vanillaPrice >= -1E-10);
		assertTrue("Knock-in price must be non-negative.", knockInPrice >= -1E-10);

		assertTrue(
				"Knock-in price should not exceed vanilla price.",
				knockInPrice <= vanillaPrice + 1E-8
		);

		assertTrue(
				"Grid minimum is too negative for a smoke test.",
				gridMin > -MAX_ALLOWED_NEGATIVE_GRID_VALUE
		);
	}

	private Grid createSpotGrid(final double barrier, final BarrierType barrierType) {

		final double deltaS = Math.abs(barrier - S0) / STEPS_BETWEEN_BARRIER_AND_SPOT;

		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {

			final double sMin = Math.max(1E-8, barrier - 8.0 * deltaS);
			final double sMax = Math.max(3.0 * S0, S0 + 12.0 * deltaS);

			final int numberOfSteps = Math.max(
					NUMBER_OF_SPACE_STEPS_S,
					(int)Math.round((sMax - sMin) / deltaS)
			);

			return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
					numberOfSteps,
					sMin,
					sMax,
					barrier,
					BARRIER_CLUSTERING_EXPONENT
			);
		}
		else {
			final double sMin = 0.0;
			final double sMax = barrier + 8.0 * deltaS;

			final int numberOfSteps = Math.max(
					NUMBER_OF_SPACE_STEPS_S,
					(int)Math.round((sMax - sMin) / deltaS)
			);

			return BarrierAlignedSpotGridFactory.createBarrierAlignedClusteredGrid(
					numberOfSteps,
					sMin,
					sMax,
					barrier,
					BARRIER_CLUSTERING_EXPONENT
			);
		}
	}

	private static double[] extractSliceAtSecondStateIndex(
			final double[] valuesOnGrid,
			final int nS,
			final int nSecondState,
			final int secondStateIndex) {

		if(secondStateIndex < 0 || secondStateIndex >= nSecondState) {
			throw new IllegalArgumentException("secondStateIndex out of range.");
		}

		final double[] slice = new double[nS];
		for(int i = 0; i < nS; i++) {
			slice[i] = valuesOnGrid[i + secondStateIndex * nS];
		}
		return slice;
	}

	private static double interpolateAlongSpot(
			final double[] sNodes,
			final double[] values,
			final double s) {

		final int index = getGridIndex(sNodes, s);
		if(index >= 0) {
			return values[index];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, values);
		return interpolation.value(s);
	}

	private static double getMin(final double[] values) {
		double min = values[0];
		for(int i = 1; i < values.length; i++) {
			min = Math.min(min, values[i]);
		}
		return min;
	}

	private static double getMax(final double[] values) {
		double max = values[0];
		for(int i = 1; i < values.length; i++) {
			max = Math.max(max, values[i]);
		}
		return max;
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