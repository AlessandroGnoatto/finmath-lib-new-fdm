package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMCevModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.GridWithMandatoryPoint;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DigitalPayoffType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Minimal smoke tests for {@link DigitalOption} under the {@link FDMCevModel}.
 *
 * @author Alessandro Gnoatto
 */
public class DigitalOptionCevTest {

	private static final double SPOT = 100.0;
	private static final double STRIKE = 100.0;
	private static final double MATURITY = 1.0;
	private static final double VOLATILITY = 0.20;
	private static final double EXPONENT = 0.8;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.02;
	private static final double CASH_PAYOFF = 10.0;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 200;
	private static final int NUMBER_OF_SPACE_STEPS = 400;
	private static final int NUMBER_OF_STANDARD_DEVIATIONS = 6;

	private static final double TOLERANCE = 5E-2;

	@Test
	public void testEuropeanCashOrNothingCallIsFiniteAndNonNegative() {

		final TestSetup setup = createSetup();

		final DigitalOption option = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.CALL,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF);

		final double[] valuesOnGrid = option.getValue(0.0, setup.model);
		final double price = interpolateAtSpot(valuesOnGrid, setup.sNodes, SPOT);

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
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF);

		final DigitalOption bermudanOption = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.CALL,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new BermudanExercise(new double[] { MATURITY }));

		final double europeanPrice =
				interpolateAtSpot(europeanOption.getValue(0.0, setup.model), setup.sNodes, SPOT);
		final double bermudanPrice =
				interpolateAtSpot(bermudanOption.getValue(0.0, setup.model), setup.sNodes, SPOT);

		assertEquals(europeanPrice, bermudanPrice, TOLERANCE);
	}

	@Test
	public void testAmericanGreaterOrEqualBermudanGreaterOrEqualEuropeanCashOrNothingPut() {

		final TestSetup setup = createSetup();

		final DigitalOption europeanOption = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.PUT,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF);

		final DigitalOption bermudanOption = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.PUT,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new BermudanExercise(new double[] { 0.5, MATURITY }));

		final DigitalOption americanOption = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.PUT,
				DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF,
				new AmericanExercise(0.0, MATURITY));

		final double europeanPrice =
				interpolateAtSpot(europeanOption.getValue(0.0, setup.model), setup.sNodes, SPOT);
		final double bermudanPrice =
				interpolateAtSpot(bermudanOption.getValue(0.0, setup.model), setup.sNodes, SPOT);
		final double americanPrice =
				interpolateAtSpot(americanOption.getValue(0.0, setup.model), setup.sNodes, SPOT);

		assertTrue(bermudanPrice + 1E-12 >= europeanPrice);
		assertTrue(americanPrice + 1E-12 >= bermudanPrice);
	}

	private TestSetup createSetup() {

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final double forward =
				SPOT * Math.exp((RISK_FREE_RATE - DIVIDEND_YIELD) * MATURITY);
		final double varianceProxy =
				SPOT * SPOT
				* Math.exp(2.0 * (RISK_FREE_RATE - DIVIDEND_YIELD) * MATURITY)
				* (Math.exp(VOLATILITY * VOLATILITY * MATURITY) - 1.0);

		final double sMin =
				Math.max(forward - NUMBER_OF_STANDARD_DEVIATIONS * Math.sqrt(varianceProxy), 0.0);
		final double sMax =
				forward + NUMBER_OF_STANDARD_DEVIATIONS * Math.sqrt(varianceProxy);

		final Grid sGrid = new GridWithMandatoryPoint(
				NUMBER_OF_SPACE_STEPS,
				sMin,
				sMax,
				STRIKE
		);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { SPOT }
		);

		final FDMCevModel model = new FDMCevModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				EXPONENT,
				spaceTime
		);

		return new TestSetup(model, sGrid.getGrid());
	}

	private double interpolateAtSpot(
			final double[] values,
			final double[] sNodes,
			final double spot) {

		assertTrue("Spot must lie inside the grid domain.",
				spot >= sNodes[0] - 1E-12 && spot <= sNodes[sNodes.length - 1] + 1E-12);

		if(isGridNode(sNodes, spot)) {
			return values[getGridIndex(sNodes, spot)];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, values);
		return interpolation.value(spot);
	}

	private boolean isGridNode(final double[] grid, final double x) {
		for(final double node : grid) {
			if(Math.abs(node - x) < 1E-12) {
				return true;
			}
		}
		return false;
	}

	private int getGridIndex(final double[] grid, final double x) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - x) < 1E-12) {
				return i;
			}
		}
		throw new IllegalArgumentException("Point is not a grid node.");
	}

	private static class TestSetup {

		private final FDMCevModel model;
		private final double[] sNodes;

		private TestSetup(final FDMCevModel model, final double[] sNodes) {
			this.model = model;
			this.sNodes = sNodes;
		}
	}
}