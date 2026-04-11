package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.GridWithMandatoryPoint;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Unit tests for {@link DigitalOption} under the {@link FDMBlackScholesModel}.
 *
 * <p>
 * The finite-difference prices are compared against analytic Black-Scholes values for:
 * </p>
 * <ul>
 *   <li>cash-or-nothing call,</li>
 *   <li>cash-or-nothing put,</li>
 *   <li>asset-or-nothing call,</li>
 *   <li>asset-or-nothing put.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class DigitalOptionBlackScholesTest {

	private static final double SPOT = 100.0;
	private static final double STRIKE = 100.0;
	private static final double MATURITY = 1.0;
	private static final double VOLATILITY = 0.20;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.02;
	private static final double CASH_PAYOFF = 10.0;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 200;
	private static final int NUMBER_OF_SPACE_STEPS = 400;
	private static final int NUMBER_OF_STANDARD_DEVIATIONS = 6;

	private static final double CASH_DIGITAL_TOLERANCE = 2.5E-1;
	private static final double ASSET_DIGITAL_TOLERANCE = 3.5E-1;
	@Test
	public void testCashOrNothingCall() {

		final TestSetup setup = createSetup();

		final DigitalOption option = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.CALL,
				DigitalOption.DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF);

		final double[] fdValuesOnGrid = option.getValue(0.0, setup.model);
		final double fdPrice = interpolateAtSpot(fdValuesOnGrid, setup.sNodes, SPOT);

		final double analyticPrice = cashOrNothingCallValue(
				SPOT,
				STRIKE,
				MATURITY,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				CASH_PAYOFF);

		assertTrue(fdPrice >= -1E-10);
		assertEquals(analyticPrice, fdPrice, CASH_DIGITAL_TOLERANCE);
	}

	@Test
	public void testCashOrNothingPut() {

		final TestSetup setup = createSetup();

		final DigitalOption option = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.PUT,
				DigitalOption.DigitalPayoffType.CASH_OR_NOTHING,
				CASH_PAYOFF);

		final double[] fdValuesOnGrid = option.getValue(0.0, setup.model);
		final double fdPrice = interpolateAtSpot(fdValuesOnGrid, setup.sNodes, SPOT);

		final double analyticPrice = cashOrNothingPutValue(
				SPOT,
				STRIKE,
				MATURITY,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				CASH_PAYOFF);

		assertTrue(fdPrice >= -1E-10);
		assertEquals(analyticPrice, fdPrice, CASH_DIGITAL_TOLERANCE);
	}

	@Test
	public void testAssetOrNothingCall() {

		final TestSetup setup = createSetup();

		final DigitalOption option = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.CALL,
				DigitalOption.DigitalPayoffType.ASSET_OR_NOTHING,
				0.0);

		final double[] fdValuesOnGrid = option.getValue(0.0, setup.model);
		final double fdPrice = interpolateAtSpot(fdValuesOnGrid, setup.sNodes, SPOT);

		final double analyticPrice = assetOrNothingCallValue(
				SPOT,
				STRIKE,
				MATURITY,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY);

		assertTrue(fdPrice >= -1E-10);
		assertEquals(analyticPrice, fdPrice, ASSET_DIGITAL_TOLERANCE);
	}

	@Test
	public void testAssetOrNothingPut() {

		final TestSetup setup = createSetup();

		final DigitalOption option = new DigitalOption(
				MATURITY,
				STRIKE,
				CallOrPut.PUT,
				DigitalOption.DigitalPayoffType.ASSET_OR_NOTHING,
				0.0);

		final double[] fdValuesOnGrid = option.getValue(0.0, setup.model);
		final double fdPrice = interpolateAtSpot(fdValuesOnGrid, setup.sNodes, SPOT);

		final double analyticPrice = assetOrNothingPutValue(
				SPOT,
				STRIKE,
				MATURITY,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY);

		assertTrue(fdPrice >= -1E-10);
		assertEquals(analyticPrice, fdPrice, ASSET_DIGITAL_TOLERANCE);
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
		final double variance =
				SPOT * SPOT
				* Math.exp(2.0 * (RISK_FREE_RATE - DIVIDEND_YIELD) * MATURITY)
				* (Math.exp(VOLATILITY * VOLATILITY * MATURITY) - 1.0);

		final double sMin =
				Math.max(forward - NUMBER_OF_STANDARD_DEVIATIONS * Math.sqrt(variance), 0.0);
		final double sMax =
				forward + NUMBER_OF_STANDARD_DEVIATIONS * Math.sqrt(variance);

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

		final FDMBlackScholesModel model = new FDMBlackScholesModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
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

	private static double cashOrNothingCallValue(
			final double spot,
			final double strike,
			final double maturity,
			final double riskFreeRate,
			final double dividendYield,
			final double volatility,
			final double cashPayoff) {

		final double forward = spot * Math.exp((riskFreeRate - dividendYield) * maturity);
		final double payoffUnit = Math.exp(-riskFreeRate * maturity);

		return cashPayoff
				* payoffUnit
				* AnalyticFormulas.blackScholesDigitalOptionValue(
						forward,
						0.0,
						volatility,
						maturity,
						strike);
	}

	private static double cashOrNothingPutValue(
			final double spot,
			final double strike,
			final double maturity,
			final double riskFreeRate,
			final double dividendYield,
			final double volatility,
			final double cashPayoff) {

		final double payoffUnit = Math.exp(-riskFreeRate * maturity);

		return cashPayoff * payoffUnit
				- cashOrNothingCallValue(
						spot,
						strike,
						maturity,
						riskFreeRate,
						dividendYield,
						volatility,
						cashPayoff);
	}

	private static double assetOrNothingCallValue(
			final double spot,
			final double strike,
			final double maturity,
			final double riskFreeRate,
			final double dividendYield,
			final double volatility) {

		final double forward = spot * Math.exp((riskFreeRate - dividendYield) * maturity);
		final double payoffUnit = Math.exp(-riskFreeRate * maturity);

		final double vanillaCall = AnalyticFormulas.blackScholesGeneralizedOptionValue(
				forward,
				volatility,
				maturity,
				strike,
				payoffUnit);

		final double unitCashDigitalCall = cashOrNothingCallValue(
				spot,
				strike,
				maturity,
				riskFreeRate,
				dividendYield,
				volatility,
				1.0);

		return vanillaCall + strike * unitCashDigitalCall;
	}

	private static double assetOrNothingPutValue(
			final double spot,
			final double strike,
			final double maturity,
			final double riskFreeRate,
			final double dividendYield,
			final double volatility) {

		return spot * Math.exp(-dividendYield * maturity)
				- assetOrNothingCallValue(
						spot,
						strike,
						maturity,
						riskFreeRate,
						dividendYield,
						volatility);
	}

	private static class TestSetup {

		private final FDMBlackScholesModel model;
		private final double[] sNodes;

		private TestSetup(final FDMBlackScholesModel model, final double[] sNodes) {
			this.model = model;
			this.sNodes = sNodes;
		}
	}
}