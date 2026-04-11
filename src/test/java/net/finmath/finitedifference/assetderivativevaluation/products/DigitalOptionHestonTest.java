package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.GridWithMandatoryPoint;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Minimal smoke tests for {@link DigitalOption} under the {@link FDMHestonModel}.
 *
 * @author Alessandro Gnoatto
 */
public class DigitalOptionHestonTest {

	private static final double SPOT = 100.0;
	private static final double STRIKE = 100.0;
	private static final double MATURITY = 1.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.02;

	private static final double VOLATILITY = 0.20;
	private static final double INITIAL_VARIANCE = VOLATILITY * VOLATILITY;
	private static final double LONG_RUN_VARIANCE = VOLATILITY * VOLATILITY;
	private static final double KAPPA = 1.0;
	private static final double XI = 0.30;
	private static final double RHO = -0.5;

	private static final double CASH_PAYOFF = 10.0;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS_S = 120;
	private static final int NUMBER_OF_SPACE_STEPS_V = 60;
	private static final int NUMBER_OF_STANDARD_DEVIATIONS = 6;

	private static final double TOLERANCE = 1E-1;

	@Test
	public void testEuropeanCashOrNothingCallIsFiniteAndNonNegative() {

		final TestSetup setup = createSetup();

		final DigitalOption option = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.CALL,
				DigitalOption.DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF);

		final double[] values = option.getValue(0.0, setup.model);
		final double price = interpolateAtSpotAndInitialVariance(values, setup.sNodes, setup.vNodes, SPOT, INITIAL_VARIANCE);

		assertTrue(Double.isFinite(price));
		assertTrue(price >= -1E-10);
		assertTrue(price <= CASH_PAYOFF + 1E-10);
	}

	@Test
	public void testOneDateBermudanEqualsEuropeanCashOrNothingCall() {

		final TestSetup setup = createSetup();

		final DigitalOption europeanOption = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.CALL,
				DigitalOption.DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF);

		final DigitalOption bermudanOption = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.CALL,
				DigitalOption.DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new BermudanExercise(new double[] { MATURITY }));

		final double europeanPrice = interpolateAtSpotAndInitialVariance(
				europeanOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.vNodes,
				SPOT,
				INITIAL_VARIANCE);

		final double bermudanPrice = interpolateAtSpotAndInitialVariance(
				bermudanOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.vNodes,
				SPOT,
				INITIAL_VARIANCE);

		assertEquals(europeanPrice, bermudanPrice, TOLERANCE);
	}

	@Test
	public void testAmericanGreaterOrEqualBermudanGreaterOrEqualEuropeanCashOrNothingPut() {

		final TestSetup setup = createSetup();

		final DigitalOption europeanOption = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.PUT,
				DigitalOption.DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF);

		final DigitalOption bermudanOption = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.PUT,
				DigitalOption.DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new BermudanExercise(new double[] { 0.5, MATURITY }));

		final DigitalOption americanOption = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.PUT,
				DigitalOption.DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new AmericanExercise(0.0, MATURITY));

		final double europeanPrice = interpolateAtSpotAndInitialVariance(
				europeanOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.vNodes,
				SPOT,
				INITIAL_VARIANCE);

		final double bermudanPrice = interpolateAtSpotAndInitialVariance(
				bermudanOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.vNodes,
				SPOT,
				INITIAL_VARIANCE);

		final double americanPrice = interpolateAtSpotAndInitialVariance(
				americanOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.vNodes,
				SPOT,
				INITIAL_VARIANCE);

		assertTrue(bermudanPrice + 1E-12 >= europeanPrice);
		assertTrue(americanPrice + 1E-12 >= bermudanPrice);
	}

	private TestSetup createSetup() {

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				NUMBER_OF_TIME_STEPS,
				MATURITY / NUMBER_OF_TIME_STEPS
		);

		final double forward = SPOT * Math.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * MATURITY);
		final double varianceProxy =
				SPOT * SPOT
				* Math.exp(2.0 * (RISK_FREE_RATE - DIVIDEND_YIELD) * MATURITY)
				* (Math.exp(VOLATILITY * VOLATILITY * MATURITY) - 1.0);

		final double sMin = Math.max(forward - NUMBER_OF_STANDARD_DEVIATIONS * Math.sqrt(varianceProxy), 0.0);
		final double sMax = forward + NUMBER_OF_STANDARD_DEVIATIONS * Math.sqrt(varianceProxy);

		final Grid sGrid = new GridWithMandatoryPoint(
				NUMBER_OF_SPACE_STEPS_S,
				sMin,
				sMax,
				STRIKE
		);

		final double vMax = Math.max(4.0 * LONG_RUN_VARIANCE, 4.0 * INITIAL_VARIANCE);
		final Grid vGrid = new UniformGrid(
				NUMBER_OF_SPACE_STEPS_V,
				0.0,
				vMax		
		);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, vGrid },
				timeDiscretization,
				THETA,
				new double[] { SPOT, INITIAL_VARIANCE }
		);

		final FDMHestonModel model = new FDMHestonModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				INITIAL_VARIANCE,
				KAPPA,
				LONG_RUN_VARIANCE,
				XI,
				RHO,
				spaceTime
		);

		return new TestSetup(model, sGrid.getGrid(), vGrid.getGrid());
	}

	private double interpolateAtSpotAndInitialVariance(
			final double[] values,
			final double[] sNodes,
			final double[] vNodes,
			final double spot,
			final double variance) {

		assertTrue("Spot must lie inside the grid domain.",
				spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12);
		assertTrue("Variance must lie inside the grid domain.",
				variance >= vNodes[0] - 1E-12 && variance <= vNodes[vNodes.length - 1] + 1E-12);

		final int iS = getLeftIndex(sNodes, spot);
		final int iV = getLeftIndex(vNodes, variance);

		if(iS == sNodes.length - 1 && iV == vNodes.length - 1) {
			return values[flatten(iS, iV, sNodes.length)];
		}
		if(iS == sNodes.length - 1) {
			return linearInterpolate(
					vNodes[iV],
					vNodes[iV + 1],
					values[flatten(iS, iV, sNodes.length)],
					values[flatten(iS, iV + 1, sNodes.length)],
					variance);
		}
		if(iV == vNodes.length - 1) {
			return linearInterpolate(
					sNodes[iS],
					sNodes[iS + 1],
					values[flatten(iS, iV, sNodes.length)],
					values[flatten(iS + 1, iV, sNodes.length)],
					spot);
		}

		final double s0 = sNodes[iS];
		final double s1 = sNodes[iS + 1];
		final double v0 = vNodes[iV];
		final double v1 = vNodes[iV + 1];

		final double f00 = values[flatten(iS, iV, sNodes.length)];
		final double f10 = values[flatten(iS + 1, iV, sNodes.length)];
		final double f01 = values[flatten(iS, iV + 1, sNodes.length)];
		final double f11 = values[flatten(iS + 1, iV + 1, sNodes.length)];

		final double wS = (spot - s0) / (s1 - s0);
		final double wV = (variance - v0) / (v1 - v0);

		return (1.0 - wS) * (1.0 - wV) * f00
				+ wS * (1.0 - wV) * f10
				+ (1.0 - wS) * wV * f01
				+ wS * wV * f11;
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

	private int flatten(final int iS, final int iV, final int numberOfSNodes) {
		return iS + iV * numberOfSNodes;
	}
	private static class TestSetup {

		private final FDMHestonModel model;
		private final double[] sNodes;
		private final double[] vNodes;

		private TestSetup(
				final FDMHestonModel model,
				final double[] sNodes,
				final double[] vNodes) {
			this.model = model;
			this.sNodes = sNodes;
			this.vNodes = vNodes;
		}
	}
}