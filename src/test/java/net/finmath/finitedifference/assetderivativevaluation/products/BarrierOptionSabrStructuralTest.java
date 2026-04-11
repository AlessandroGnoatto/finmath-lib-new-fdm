package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.grids.BarrierAlignedSpotGridFactory;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Structural tests for {@link BarrierOption} under {@link FDMSabrModel}.
 *
 * <p>
 * These tests deliberately avoid external benchmark values and check only:
 * </p>
 * <ul>
 *   <li>in-out parity,</li>
 *   <li>far-barrier convergence to the vanilla price.</li>
 * </ul>
 *
 * <p>
 * Benchmark comparisons against Monte Carlo should live in a separate test class.
 * </p>
 */
public class BarrierOptionSabrStructuralTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

	private static final double SPOT = 100.0;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.02;

	private static final double INITIAL_VOLATILITY = 0.20;
	private static final double BETA = 1.0;
	private static final double NU = 0.30;
	private static final double RHO = -0.50;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS_S = 160;
	private static final int NUMBER_OF_SPACE_STEPS_VOL = 80;
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	private static final double BARRIER_CLUSTERING_EXPONENT = 2.0;

	private static final double PARITY_TOLERANCE = 2.5E-1;
	private static final double FAR_BARRIER_TOLERANCE = 2.5E-1;

	@Test
	public void testDownBarrierCallSatisfiesInOutParity() {
		assertInOutParity(CallOrPut.CALL, 80.0, BarrierType.DOWN_OUT, BarrierType.DOWN_IN);
	}

	@Test
	public void testUpBarrierPutSatisfiesInOutParity() {
		assertInOutParity(CallOrPut.PUT, 120.0, BarrierType.UP_OUT, BarrierType.UP_IN);
	}

	@Test
	public void testFarDownOutCallApproachesVanilla() {
		assertFarBarrierApproachesVanilla(CallOrPut.CALL, 20.0, BarrierType.DOWN_OUT);
	}

	@Test
	public void testFarUpOutPutApproachesVanilla() {
		assertFarBarrierApproachesVanilla(CallOrPut.PUT, 200.0, BarrierType.UP_OUT);
	}

	private void assertInOutParity(
			final CallOrPut callOrPut,
			final double barrier,
			final BarrierType outType,
			final BarrierType inType) {

		final TestSetup setup = createSetup(barrier, outType);

		final BarrierOption outOption = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut == CallOrPut.CALL ? 1.0 : -1.0,
				outType
		);

		final BarrierOption inOption = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut == CallOrPut.CALL ? 1.0 : -1.0,
				inType
		);

		final EuropeanOption vanillaOption = new EuropeanOption(
				MATURITY,
				STRIKE,
				callOrPut == CallOrPut.CALL ? 1.0 : -1.0
		);

		final double outPrice = interpolateAtSpotAndInitialVolatility(
				outOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY
		);

		final double inPrice = interpolateAtSpotAndInitialVolatility(
				inOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY
		);

		final double vanillaPrice = interpolateAtSpotAndInitialVolatility(
				vanillaOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY
		);

		assertTrue(outPrice >= -1E-10);
		assertTrue(inPrice >= -1E-10);
		assertTrue(vanillaPrice >= -1E-10);

		assertEquals(
				"Barrier in-out parity failed for " + outType + " / " + inType + " " + callOrPut,
				vanillaPrice,
				inPrice + outPrice,
				PARITY_TOLERANCE
		);
	}

	private void assertFarBarrierApproachesVanilla(
			final CallOrPut callOrPut,
			final double barrier,
			final BarrierType barrierType) {

		final TestSetup setup = createSetup(barrier, barrierType);

		final BarrierOption barrierOption = new BarrierOption(
				MATURITY,
				STRIKE,
				barrier,
				REBATE,
				callOrPut == CallOrPut.CALL ? 1.0 : -1.0,
				barrierType
		);

		final EuropeanOption vanillaOption = new EuropeanOption(
				MATURITY,
				STRIKE,
				callOrPut == CallOrPut.CALL ? 1.0 : -1.0
		);

		final double barrierPrice = interpolateAtSpotAndInitialVolatility(
				barrierOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY
		);

		final double vanillaPrice = interpolateAtSpotAndInitialVolatility(
				vanillaOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY
		);

		assertTrue(barrierPrice >= -1E-10);
		assertTrue(vanillaPrice >= -1E-10);

		assertEquals(
				"Far-barrier limit failed for " + barrierType + " " + callOrPut,
				vanillaPrice,
				barrierPrice,
				FAR_BARRIER_TOLERANCE
		);
	}

	private TestSetup createSetup(final double barrier, final BarrierType barrierType) {

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				NUMBER_OF_TIME_STEPS,
				MATURITY / NUMBER_OF_TIME_STEPS
		);

		final Grid sGrid = createSpotGrid(barrier, barrierType);

		final double volMax = Math.max(4.0 * INITIAL_VOLATILITY, 1.0);
		final Grid volGrid = new UniformGrid(
				NUMBER_OF_SPACE_STEPS_VOL,
				0.0,
				volMax
		);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, volGrid },
				timeDiscretization,
				THETA,
				new double[] { SPOT, INITIAL_VOLATILITY }
		);

		final FDMSabrModel model = new FDMSabrModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				INITIAL_VOLATILITY,
				BETA,
				NU,
				RHO,
				spaceTime
		);

		return new TestSetup(model, sGrid.getGrid(), volGrid.getGrid());
	}

	private Grid createSpotGrid(final double barrier, final BarrierType barrierType) {

		final double deltaS = Math.abs(barrier - SPOT) / STEPS_BETWEEN_BARRIER_AND_SPOT;

		final boolean isKnockIn =
				barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.UP_IN;

		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {

			final double sMin = Math.max(1E-8, barrier - 8.0 * deltaS);
			final double sMax = Math.max(3.0 * SPOT, SPOT + 12.0 * deltaS);

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

	private double interpolateAtSpotAndInitialVolatility(
			final double[] values,
			final double[] sNodes,
			final double[] volNodes,
			final double spot,
			final double volatility) {

		assertTrue("Spot must lie inside the grid domain.",
				spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12);
		assertTrue("Volatility must lie inside the grid domain.",
				volatility >= volNodes[0] - 1E-12 && volatility <= volNodes[volNodes.length - 1] + 1E-12);

		final int iS = getLeftIndex(sNodes, spot);
		final int iVol = getLeftIndex(volNodes, volatility);

		if(iS == sNodes.length - 1 && iVol == volNodes.length - 1) {
			return values[flatten(iS, iVol, sNodes.length)];
		}
		if(iS == sNodes.length - 1) {
			return linearInterpolate(
					volNodes[iVol],
					volNodes[iVol + 1],
					values[flatten(iS, iVol, sNodes.length)],
					values[flatten(iS, iVol + 1, sNodes.length)],
					volatility
			);
		}
		if(iVol == volNodes.length - 1) {
			return linearInterpolate(
					sNodes[iS],
					sNodes[iS + 1],
					values[flatten(iS, iVol, sNodes.length)],
					values[flatten(iS + 1, iVol, sNodes.length)],
					spot
			);
		}

		final double s0 = sNodes[iS];
		final double s1 = sNodes[iS + 1];
		final double a0 = volNodes[iVol];
		final double a1 = volNodes[iVol + 1];

		final double f00 = values[flatten(iS, iVol, sNodes.length)];
		final double f10 = values[flatten(iS + 1, iVol, sNodes.length)];
		final double f01 = values[flatten(iS, iVol + 1, sNodes.length)];
		final double f11 = values[flatten(iS + 1, iVol + 1, sNodes.length)];

		final double wS = (spot - s0) / (s1 - s0);
		final double wA = (volatility - a0) / (a1 - a0);

		return (1.0 - wS) * (1.0 - wA) * f00
				+ wS * (1.0 - wA) * f10
				+ (1.0 - wS) * wA * f01
				+ wS * wA * f11;
	}

	private int getLeftIndex(final double[] grid, final double x) {
		if(x <= grid[0]) {
			return 0;
		}
		if(x >= grid[grid.length - 1]) {
			return grid.length - 1;
		}

		for(int i = 0; i < grid.length - 1; i++) {
			if(x >= grid[i] - 1E-12 && x <= grid[i + 1] + 1E-12) {
				return i;
			}
		}

		throw new IllegalArgumentException("Point is outside the grid.");
	}

	private double linearInterpolate(
			final double x0,
			final double x1,
			final double y0,
			final double y1,
			final double x) {

		if(Math.abs(x1 - x0) < 1E-14) {
			return y0;
		}

		return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
	}

	private int flatten(final int iS, final int iVol, final int numberOfSNodes) {
		return iS + iVol * numberOfSNodes;
	}

	private static class TestSetup {

		private final FDMSabrModel model;
		private final double[] sNodes;
		private final double[] volNodes;

		private TestSetup(
				final FDMSabrModel model,
				final double[] sNodes,
				final double[] volNodes) {
			this.model = model;
			this.sNodes = sNodes;
			this.volNodes = volNodes;
		}
	}
}