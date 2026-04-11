package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
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
 * Minimal smoke tests for {@link DigitalOption} under the {@link FDMSabrModel}.
 *
 * @author Alessandro Gnoatto
 */
public class DigitalOptionSabrTest {

	private static final double SPOT = 100.0;
	private static final double STRIKE = 100.0;
	private static final double MATURITY = 1.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.02;

	private static final double INITIAL_VOLATILITY = 0.20;
	private static final double BETA = 1.0;
	private static final double NU = 0.30;
	private static final double RHO = -0.5;

	private static final double CASH_PAYOFF = 10.0;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS_S = 120;
	private static final int NUMBER_OF_SPACE_STEPS_VOL = 60;
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
		final double price = interpolateAtSpotAndInitialVolatility(
				values,
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY);

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

		final double europeanPrice = interpolateAtSpotAndInitialVolatility(
				europeanOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY);

		final double bermudanPrice = interpolateAtSpotAndInitialVolatility(
				bermudanOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY);

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

		final double europeanPrice = interpolateAtSpotAndInitialVolatility(
				europeanOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY);

		final double bermudanPrice = interpolateAtSpotAndInitialVolatility(
				bermudanOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY);

		final double americanPrice = interpolateAtSpotAndInitialVolatility(
				americanOption.getValue(0.0, setup.model),
				setup.sNodes,
				setup.volNodes,
				SPOT,
				INITIAL_VOLATILITY);

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
				* (Math.exp(INITIAL_VOLATILITY * INITIAL_VOLATILITY * MATURITY) - 1.0);

		final double sMin = Math.max(forward - NUMBER_OF_STANDARD_DEVIATIONS * Math.sqrt(varianceProxy), 0.0);
		final double sMax = forward + NUMBER_OF_STANDARD_DEVIATIONS * Math.sqrt(varianceProxy);

		final Grid sGrid = new GridWithMandatoryPoint(
				NUMBER_OF_SPACE_STEPS_S,
				sMin,
				sMax,
				STRIKE
		);

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
					volatility);
		}
		if(iVol == volNodes.length - 1) {
			return linearInterpolate(
					sNodes[iS],
					sNodes[iS + 1],
					values[flatten(iS, iVol, sNodes.length)],
					values[flatten(iS + 1, iVol, sNodes.length)],
					spot);
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

	private int flatten(final int iS, final int iA, final int numberOfSNodes) {
		return iS + iA * numberOfSNodes;
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