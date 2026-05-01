package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
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
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.MonitoringType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for 2D discretely monitored knock-in barrier options under
 * Heston and SABR.
 *
 * <p>
 * Current scope:
 * European exercise only.
 * </p>
 *
 * <p>
 * Structural properties under test:
 * </p>
 * <ul>
 *   <li>discrete knock-in values are monotone in monitoring frequency:
 *       coarse discrete &lt;= fine discrete,</li>
 *   <li>knock-in values are bounded above by the corresponding vanilla value,</li>
 *   <li>at a monitoring date, nodes already beyond the barrier are replaced by the
 *       vanilla continuation value.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionStochasticVolatilityDiscreteMonitoringKnockInTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double REBATE = 0.0;

	private static final double SPOT = 100.0;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.02;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 180;
	private static final int NUMBER_OF_SPACE_STEPS_S = 220;
	private static final int NUMBER_OF_SPACE_STEPS_Y = 120;
	private static final int STEPS_BETWEEN_BARRIER_AND_SPOT = 40;

	private static final double ORDERING_TOLERANCE = 7.5E-3;
	private static final double EVENT_TOLERANCE = 1E-8;
	private static final double BARRIER_TOLERANCE = 1E-8;

	private static final double[] COARSE_MONITORING_TIMES = new double[] { 0.50, 1.00 };
	private static final double[] FINE_MONITORING_TIMES = new double[] { 0.25, 0.50, 0.75, 1.00 };

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

	private static final double HESTON_DOWN_IN_BARRIER = 80.0;
	private static final double SABR_UP_IN_BARRIER = 120.0;

	@Test
	public void testHestonDownInPutDiscreteMonitoringOrderingAndVanillaUpperBound() {

		final HestonSetup setup = createHestonSetup(HESTON_DOWN_IN_BARRIER, BarrierType.DOWN_IN);

		final BarrierOption coarseDiscrete = new BarrierOption(
				MATURITY,
				STRIKE,
				HESTON_DOWN_IN_BARRIER,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final BarrierOption fineDiscrete = new BarrierOption(
				MATURITY,
				STRIKE,
				HESTON_DOWN_IN_BARRIER,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final EuropeanOption vanilla = new EuropeanOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.PUT
		);

		final double coarsePrice = interpolateAt(
				coarseDiscrete.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				HESTON_INITIAL_VARIANCE
		);

		final double finePrice = interpolateAt(
				fineDiscrete.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				HESTON_INITIAL_VARIANCE
		);

		final double vanillaPrice = interpolateAt(
				vanilla.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				HESTON_INITIAL_VARIANCE
		);

		assertTrue(Double.isFinite(coarsePrice));
		assertTrue(Double.isFinite(finePrice));
		assertTrue(Double.isFinite(vanillaPrice));

		assertTrue(coarsePrice <= finePrice + ORDERING_TOLERANCE);
		assertTrue(finePrice <= vanillaPrice + ORDERING_TOLERANCE);
		assertTrue(coarsePrice >= -ORDERING_TOLERANCE);
	}

	@Test
	public void testSabrUpInCallDiscreteMonitoringOrderingAgainstContinuousAndVanilla() {

		final SabrSetup setup = createSabrSetup(SABR_UP_IN_BARRIER, BarrierType.UP_IN);

		final BarrierOption coarseDiscrete = new BarrierOption(
				MATURITY,
				STRIKE,
				SABR_UP_IN_BARRIER,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final BarrierOption fineDiscrete = new BarrierOption(
				MATURITY,
				STRIKE,
				SABR_UP_IN_BARRIER,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final BarrierOption continuous = new BarrierOption(
				MATURITY,
				STRIKE,
				SABR_UP_IN_BARRIER,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.CONTINUOUS,
				null
		);

		final EuropeanOption vanilla = new EuropeanOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.CALL
		);

		final double coarsePrice = interpolateAt(
				coarseDiscrete.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				SABR_INITIAL_ALPHA
		);

		final double finePrice = interpolateAt(
				fineDiscrete.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				SABR_INITIAL_ALPHA
		);

		final double continuousPrice = interpolateAt(
				continuous.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				SABR_INITIAL_ALPHA
		);

		final double vanillaPrice = interpolateAt(
				vanilla.getValue(0.0, setup.model),
				setup.sNodes,
				setup.yNodes,
				SPOT,
				SABR_INITIAL_ALPHA
		);

		assertTrue(Double.isFinite(coarsePrice));
		assertTrue(Double.isFinite(finePrice));
		assertTrue(Double.isFinite(continuousPrice));
		assertTrue(Double.isFinite(vanillaPrice));

		assertTrue(coarsePrice <= finePrice + ORDERING_TOLERANCE);
		assertTrue(finePrice <= continuousPrice + ORDERING_TOLERANCE);
		assertTrue(continuousPrice <= vanillaPrice + ORDERING_TOLERANCE);
	}

	@Test
	public void testHestonDownInEventConditionReplacesByVanillaContinuationValue() {

		final double monitoringTime = 0.50;

		final HestonSetup setup = createHestonSetup(HESTON_DOWN_IN_BARRIER, BarrierType.DOWN_IN);

		final BarrierOption discreteKnockIn = new BarrierOption(
				MATURITY,
				STRIKE,
				HESTON_DOWN_IN_BARRIER,
				REBATE,
				CallOrPut.PUT,
				BarrierType.DOWN_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				new double[] { monitoringTime, MATURITY }
		);

		final EuropeanOption vanilla = new EuropeanOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.PUT
		);

		final double[][] knockInValues = discreteKnockIn.getValues(setup.model);
		final double[][] vanillaValues = vanilla.getValues(setup.model);

		final double tau = MATURITY - monitoringTime;
		final int timeIndex = setup.model.getSpaceTimeDiscretization().getTimeDiscretization().getTimeIndex(tau);

		final int nS = setup.sNodes.length;
		final int nY = setup.yNodes.length;

		for(int j = 0; j < nY; j++) {
			for(int i = 0; i < nS; i++) {
				if(setup.sNodes[i] <= HESTON_DOWN_IN_BARRIER + BARRIER_TOLERANCE) {
					final int k = flatten(i, j, nS);
					assertEquals(
							vanillaValues[k][timeIndex],
							knockInValues[k][timeIndex],
							EVENT_TOLERANCE
					);
				}
			}
		}
	}

	@Test
	public void testSabrUpInEventConditionReplacesByVanillaContinuationValue() {

		final double monitoringTime = 0.75;

		final SabrSetup setup = createSabrSetup(SABR_UP_IN_BARRIER, BarrierType.UP_IN);

		final BarrierOption discreteKnockIn = new BarrierOption(
				MATURITY,
				STRIKE,
				SABR_UP_IN_BARRIER,
				REBATE,
				CallOrPut.CALL,
				BarrierType.UP_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				new double[] { monitoringTime, MATURITY }
		);

		final EuropeanOption vanilla = new EuropeanOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.CALL
		);

		final double[][] knockInValues = discreteKnockIn.getValues(setup.model);
		final double[][] vanillaValues = vanilla.getValues(setup.model);

		final double tau = MATURITY - monitoringTime;
		final int timeIndex = setup.model.getSpaceTimeDiscretization().getTimeDiscretization().getTimeIndex(tau);

		final int nS = setup.sNodes.length;
		final int nY = setup.yNodes.length;

		for(int j = 0; j < nY; j++) {
			for(int i = 0; i < nS; i++) {
				if(setup.sNodes[i] >= SABR_UP_IN_BARRIER - BARRIER_TOLERANCE) {
					final int k = flatten(i, j, nS);
					assertEquals(
							vanillaValues[k][timeIndex],
							knockInValues[k][timeIndex],
							EVENT_TOLERANCE
					);
				}
			}
		}
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

	private static int flatten(final int iS, final int iY, final int numberOfSNodes) {
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