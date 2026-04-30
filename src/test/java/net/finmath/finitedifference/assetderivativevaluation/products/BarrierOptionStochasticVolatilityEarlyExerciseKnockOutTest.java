package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.BarrierAlignedSpotGridFactory;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.Exercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for 2D stochastic-volatility knock-out barrier options with
 * Bermudan and American exercise.
 *
 * <p>
 * These tests are intended to protect the newly opened support for:
 * </p>
 * <ul>
 *   <li>2D Heston knock-outs with Bermudan and American exercise,</li>
 *   <li>2D SABR knock-outs with Bermudan and American exercise.</li>
 * </ul>
 *
 * <p>
 * The structural properties checked are:
 * </p>
 * <ul>
 *   <li>valuation no longer throws for Bermudan / American knock-outs,</li>
 *   <li>European <= Bermudan <= American at the reference state,</li>
 *   <li>the value in the already-knocked-out region remains equal to the rebate
 *       under all exercise styles.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionStochasticVolatilityEarlyExerciseKnockOutTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 2.5;

	private static final double SPOT = 100.0;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.02;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 160;
	private static final int NUMBER_OF_SPACE_STEPS_S = 220;
	private static final int NUMBER_OF_SPACE_STEPS_Y = 120;
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	private static final double ORDERING_TOLERANCE = 5.0E-3;
	private static final double REBATE_TOLERANCE = 1.0E-8;

	/* Heston parameters */
	private static final double HESTON_INITIAL_VOLATILITY = 0.25;
	private static final double HESTON_INITIAL_VARIANCE =
			HESTON_INITIAL_VOLATILITY * HESTON_INITIAL_VOLATILITY;
	private static final double HESTON_KAPPA = 1.5;
	private static final double HESTON_LONG_RUN_VARIANCE = HESTON_INITIAL_VARIANCE;
	private static final double HESTON_XI = 0.30;
	private static final double HESTON_RHO = -0.70;

	/* SABR parameters */
	private static final double SABR_INITIAL_ALPHA = 0.25;
	private static final double SABR_BETA = 0.7;
	private static final double SABR_NU = 0.40;
	private static final double SABR_RHO = -0.35;

	private static final double HESTON_DOWN_OUT_BARRIER = 80.0;
	private static final double HESTON_ALREADY_KNOCKED_OUT_SPOT = 78.0;

	private static final double SABR_UP_OUT_BARRIER = 120.0;
	private static final double SABR_ALREADY_KNOCKED_OUT_SPOT = 122.0;

	@Test
	public void testHestonDownOutPutBermudanAndAmericanOrdering() {
		final HestonSetup setup = createHestonSetup(HESTON_DOWN_OUT_BARRIER, BarrierType.DOWN_OUT);

		final BarrierOption european = new BarrierOption(
				MATURITY,
				STRIKE,
				HESTON_DOWN_OUT_BARRIER,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_OUT
		);

		final BarrierOption bermudan = new BarrierOption(
				MATURITY,
				STRIKE,
				HESTON_DOWN_OUT_BARRIER,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_OUT,
				new BermudanExercise(new double[] { 0.25, 0.50, 0.75, 1.00 })
		);

		final BarrierOption american = new BarrierOption(
				MATURITY,
				STRIKE,
				HESTON_DOWN_OUT_BARRIER,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_OUT,
				new AmericanExercise(MATURITY)
		);

		final double europeanPrice = interpolateAt(
				european.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				HESTON_INITIAL_VARIANCE
		);

		final double bermudanPrice = interpolateAt(
				bermudan.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				HESTON_INITIAL_VARIANCE
		);

		final double americanPrice = interpolateAt(
				american.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				HESTON_INITIAL_VARIANCE
		);

		assertTrue(Double.isFinite(europeanPrice));
		assertTrue(Double.isFinite(bermudanPrice));
		assertTrue(Double.isFinite(americanPrice));

		assertTrue(bermudanPrice + ORDERING_TOLERANCE >= europeanPrice);
		assertTrue(americanPrice + ORDERING_TOLERANCE >= bermudanPrice);
	}

	@Test
	public void testHestonDownOutPutKnockedOutRegionStaysAtRebateUnderEarlyExercise() {
		final HestonSetup setup = createHestonSetup(HESTON_DOWN_OUT_BARRIER, BarrierType.DOWN_OUT);

		assertAlreadyKnockedOutValueEqualsRebate(
				new BarrierOption(
						MATURITY,
						STRIKE,
						HESTON_DOWN_OUT_BARRIER,
						REBATE,
						CallOrPut.PUT,
						BarrierType.DOWN_OUT
				),
				setup.model,
				setup.sNodes,
				setup.yNodes,
				HESTON_ALREADY_KNOCKED_OUT_SPOT,
				HESTON_INITIAL_VARIANCE
		);

		assertAlreadyKnockedOutValueEqualsRebate(
				new BarrierOption(
						MATURITY,
						STRIKE,
						HESTON_DOWN_OUT_BARRIER,
						REBATE,
						CallOrPut.PUT,
						BarrierType.DOWN_OUT,
						new BermudanExercise(new double[] { 0.25, 0.50, 0.75, 1.00 })
				),
				setup.model,
				setup.sNodes,
				setup.yNodes,
				HESTON_ALREADY_KNOCKED_OUT_SPOT,
				HESTON_INITIAL_VARIANCE
		);

		assertAlreadyKnockedOutValueEqualsRebate(
				new BarrierOption(
						MATURITY,
						STRIKE,
						HESTON_DOWN_OUT_BARRIER,
						REBATE,
						CallOrPut.PUT,
						BarrierType.DOWN_OUT,
						new AmericanExercise(MATURITY)
				),
				setup.model,
				setup.sNodes,
				setup.yNodes,
				HESTON_ALREADY_KNOCKED_OUT_SPOT,
				HESTON_INITIAL_VARIANCE
		);
	}

	@Test
	public void testSabrUpOutCallBermudanAndAmericanOrdering() {
		final SabrSetup setup = createSabrSetup(SABR_UP_OUT_BARRIER, BarrierType.UP_OUT);

		final BarrierOption european = new BarrierOption(
				MATURITY,
				STRIKE,
				SABR_UP_OUT_BARRIER,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_OUT
		);

		final BarrierOption bermudan = new BarrierOption(
				MATURITY,
				STRIKE,
				SABR_UP_OUT_BARRIER,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_OUT,
				new BermudanExercise(new double[] { 0.25, 0.50, 0.75, 1.00 })
		);

		final BarrierOption american = new BarrierOption(
				MATURITY,
				STRIKE,
				SABR_UP_OUT_BARRIER,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_OUT,
				new AmericanExercise(MATURITY)
		);

		final double europeanPrice = interpolateAt(
				european.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				SABR_INITIAL_ALPHA
		);

		final double bermudanPrice = interpolateAt(
				bermudan.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				SABR_INITIAL_ALPHA
		);

		final double americanPrice = interpolateAt(
				american.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				SABR_INITIAL_ALPHA
		);

		assertTrue(Double.isFinite(europeanPrice));
		assertTrue(Double.isFinite(bermudanPrice));
		assertTrue(Double.isFinite(americanPrice));

		assertTrue(bermudanPrice + ORDERING_TOLERANCE >= europeanPrice);
		assertTrue(americanPrice + ORDERING_TOLERANCE >= bermudanPrice);
	}

	@Test
	public void testSabrUpOutCallKnockedOutRegionStaysAtRebateUnderEarlyExercise() {
		final SabrSetup setup = createSabrSetup(SABR_UP_OUT_BARRIER, BarrierType.UP_OUT);

		assertAlreadyKnockedOutValueEqualsRebate(
				new BarrierOption(
						MATURITY,
						STRIKE,
						SABR_UP_OUT_BARRIER,
						REBATE,
						CallOrPut.CALL,
						BarrierType.UP_OUT
				),
				setup.model,
				setup.sNodes,
				setup.yNodes,
				SABR_ALREADY_KNOCKED_OUT_SPOT,
				SABR_INITIAL_ALPHA
		);

		assertAlreadyKnockedOutValueEqualsRebate(
				new BarrierOption(
						MATURITY,
						STRIKE,
						SABR_UP_OUT_BARRIER,
						REBATE,
						CallOrPut.CALL,
						BarrierType.UP_OUT,
						new BermudanExercise(new double[] { 0.25, 0.50, 0.75, 1.00 })
				),
				setup.model,
				setup.sNodes,
				setup.yNodes,
				SABR_ALREADY_KNOCKED_OUT_SPOT,
				SABR_INITIAL_ALPHA
		);

		assertAlreadyKnockedOutValueEqualsRebate(
				new BarrierOption(
						MATURITY,
						STRIKE,
						SABR_UP_OUT_BARRIER,
						REBATE,
						CallOrPut.CALL,
						BarrierType.UP_OUT,
						new AmericanExercise(MATURITY)
				),
				setup.model,
				setup.sNodes,
				setup.yNodes,
				SABR_ALREADY_KNOCKED_OUT_SPOT,
				SABR_INITIAL_ALPHA
		);
	}

	private void assertAlreadyKnockedOutValueEqualsRebate(
			final BarrierOption option,
			final FiniteDifferenceEquityModel model,
			final double[] sNodes,
			final double[] yNodes,
			final double knockedOutSpot,
			final double secondStateValue) {

		final double value = interpolateAt(
				option.getValue(0.0, model),
				sNodes,
				yNodes,
				knockedOutSpot,
				secondStateValue
		);

		assertEquals(REBATE, value, REBATE_TOLERANCE);
	}

	private HestonSetup createHestonSetup(final double barrier, final BarrierType barrierType) {
		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", RISK_FREE_RATE);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", DIVIDEND_YIELD);

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				NUMBER_OF_TIME_STEPS,
				MATURITY / NUMBER_OF_TIME_STEPS
		);

		final Grid sGrid = createSpotGrid(barrier, barrierType);

		final double yMin = 0.0;
		final double yMax = Math.max(
				4.0 * HESTON_LONG_RUN_VARIANCE,
				HESTON_INITIAL_VARIANCE + 4.0 * HESTON_XI * Math.sqrt(MATURITY)
		);
		final Grid yGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_Y, yMin, yMax);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, yGrid },
				timeDiscretization,
				THETA,
				new double[] { SPOT, HESTON_INITIAL_VARIANCE }
		);

		final FDMHestonModel model = new FDMHestonModel(
				SPOT,
				HESTON_INITIAL_VARIANCE,
				riskFreeCurve,
				dividendCurve,
				HESTON_KAPPA,
				HESTON_LONG_RUN_VARIANCE,
				HESTON_XI,
				HESTON_RHO,
				spaceTime
		);

		return new HestonSetup(model, sGrid.getGrid(), yGrid.getGrid());
	}

	private SabrSetup createSabrSetup(final double barrier, final BarrierType barrierType) {
		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", RISK_FREE_RATE);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", DIVIDEND_YIELD);

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				NUMBER_OF_TIME_STEPS,
				MATURITY / NUMBER_OF_TIME_STEPS
		);

		final Grid sGrid = createSpotGrid(barrier, barrierType);

		final double yMin = 0.0;
		final double yMax = Math.max(1.0, 4.0 * SABR_INITIAL_ALPHA);
		final Grid yGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_Y, yMin, yMax);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, yGrid },
				timeDiscretization,
				THETA,
				new double[] { SPOT, SABR_INITIAL_ALPHA }
		);

		final FDMSabrModel model = new FDMSabrModel(
				SPOT,
				SABR_INITIAL_ALPHA,
				riskFreeCurve,
				dividendCurve,
				SABR_BETA,
				SABR_NU,
				SABR_RHO,
				spaceTime
		);

		return new SabrSetup(model, sGrid.getGrid(), yGrid.getGrid());
	}

	private Grid createSpotGrid(final double barrier, final BarrierType barrierType) {
		final double deltaS = Math.abs(barrier - SPOT) / STEPS_BETWEEN_BARRIER_AND_SPOT;

		final double sMin;
		final double sMax;

		if(barrierType == BarrierType.DOWN_IN || barrierType == BarrierType.DOWN_OUT) {
			sMin = Math.max(1E-8, barrier - 8.0 * deltaS);
			sMax = Math.max(3.0 * SPOT, SPOT + 12.0 * deltaS);
		}
		else {
			sMin = 0.0;
			sMax = barrier + 8.0 * deltaS;
		}

		final int numberOfSteps = Math.max(
				NUMBER_OF_SPACE_STEPS_S,
				(int)Math.round((sMax - sMin) / deltaS)
		);

		return BarrierAlignedSpotGridFactory.createBarrierAlignedUniformGrid(
				numberOfSteps,
				sMin,
				sMax,
				barrier
		);
	}

	private double interpolateAt(
			final double[] flattenedValues,
			final double[] sNodes,
			final double[] yNodes,
			final double spot,
			final double secondState) {

		final int nS = sNodes.length;
		final int nY = yNodes.length;

		final double[][] valueSurface = new double[nS][nY];
		for(int j = 0; j < nY; j++) {
			for(int i = 0; i < nS; i++) {
				valueSurface[i][j] = flattenedValues[flatten(i, j, nS)];
			}
		}

		return new BiLinearInterpolation(sNodes, yNodes, valueSurface).apply(spot, secondState);
	}

	private int flatten(final int iS, final int iY, final int numberOfSNodes) {
		return iS + iY * numberOfSNodes;
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

	private static final class HestonSetup {

		private final FDMHestonModel model;
		private final double[] sNodes;
		private final double[] yNodes;

		private HestonSetup(
				final FDMHestonModel model,
				final double[] sNodes,
				final double[] yNodes) {
			this.model = model;
			this.sNodes = sNodes;
			this.yNodes = yNodes;
		}
	}

	private static final class SabrSetup {

		private final FDMSabrModel model;
		private final double[] sNodes;
		private final double[] yNodes;

		private SabrSetup(
				final FDMSabrModel model,
				final double[] sNodes,
				final double[] yNodes) {
			this.model = model;
			this.sNodes = sNodes;
			this.yNodes = yNodes;
		}
	}
}