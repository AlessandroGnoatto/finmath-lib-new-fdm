package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.AmericanExercise;
import net.finmath.modelling.BermudanExercise;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.DoubleBarrierType;
import net.finmath.modelling.products.MonitoringType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for 2D discretely monitored double knock-in options
 * under Heston and SABR.
 */
public class DoubleBarrierOptionStochasticVolatilityDiscreteMonitoringKnockInTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double SPOT = 100.0;

	private static final double LOWER_BARRIER = 80.0;
	private static final double UPPER_BARRIER = 120.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 180;

	private static final double[] COARSE_MONITORING_TIMES = new double[] { 0.50, 1.00 };
	private static final double[] FINE_MONITORING_TIMES = new double[] { 0.25, 0.50, 0.75, 1.00 };
	private static final double[] BERMUDAN_EXERCISE_TIMES = new double[] { 0.25, 0.50, 0.75, 1.00 };

	private static final double ORDERING_TOLERANCE_2D = 5E-2;
	private static final double EVENT_TOLERANCE_2D = 5E-2;
	private static final double GRID_TOLERANCE = 1E-12;

	@Test
	public void testHestonEuropeanDiscreteMonitoringOrderingAgainstContinuousAndVanilla() {

		final FDMHestonModel model = createHestonModel();

		final DoubleBarrierOption coarseDiscrete = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final DoubleBarrierOption fineDiscrete = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final DoubleBarrierOption continuous = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_IN,
				new EuropeanExercise(MATURITY)
		);

		final EuropeanOption vanilla = new EuropeanOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.CALL
		);

		final double coarseValue = extractValueAtInitialState(coarseDiscrete.getValue(0.0, model), model);
		final double fineValue = extractValueAtInitialState(fineDiscrete.getValue(0.0, model), model);
		final double continuousValue = extractValueAtInitialState(continuous.getValue(0.0, model), model);
		final double vanillaValue = extractValueAtInitialState(vanilla.getValue(0.0, model), model);

		assertTrue(coarseValue >= -ORDERING_TOLERANCE_2D);
		assertTrue(coarseValue <= fineValue + ORDERING_TOLERANCE_2D);
		assertTrue(fineValue <= continuousValue + ORDERING_TOLERANCE_2D);
		assertTrue(continuousValue <= vanillaValue + ORDERING_TOLERANCE_2D);
	}

	@Test
	public void testSabrEuropeanDiscreteMonitoringOrderingAgainstContinuousAndVanilla() {

		final FDMSabrModel model = createSabrModel();

		final DoubleBarrierOption coarseDiscrete = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.PUT,
				DoubleBarrierType.KNOCK_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				COARSE_MONITORING_TIMES
		);

		final DoubleBarrierOption fineDiscrete = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.PUT,
				DoubleBarrierType.KNOCK_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final DoubleBarrierOption continuous = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.PUT,
				DoubleBarrierType.KNOCK_IN,
				new EuropeanExercise(MATURITY)
		);

		final EuropeanOption vanilla = new EuropeanOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.PUT
		);

		final double coarseValue = extractValueAtInitialState(coarseDiscrete.getValue(0.0, model), model);
		final double fineValue = extractValueAtInitialState(fineDiscrete.getValue(0.0, model), model);
		final double continuousValue = extractValueAtInitialState(continuous.getValue(0.0, model), model);
		final double vanillaValue = extractValueAtInitialState(vanilla.getValue(0.0, model), model);

		assertTrue(coarseValue >= -ORDERING_TOLERANCE_2D);
		assertTrue(coarseValue <= fineValue + ORDERING_TOLERANCE_2D);
		assertTrue(fineValue <= continuousValue + ORDERING_TOLERANCE_2D);
		assertTrue(continuousValue <= vanillaValue + ORDERING_TOLERANCE_2D);
	}

	@Test
	public void testHestonBermudanAndAmericanExerciseOrderingUnderDiscreteMonitoring() {

		final FDMHestonModel model = createHestonModel();

		final DoubleBarrierOption european = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final DoubleBarrierOption bermudan = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_IN,
				new BermudanExercise(BERMUDAN_EXERCISE_TIMES),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final DoubleBarrierOption american = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_IN,
				new AmericanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final BermudanOption vanillaBermudan = new BermudanOption(
				null,
				BERMUDAN_EXERCISE_TIMES,
				STRIKE,
				CallOrPut.CALL
		);

		final AmericanOption vanillaAmerican = new AmericanOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.CALL
		);

		final double europeanValue = extractValueAtInitialState(european.getValue(0.0, model), model);
		final double bermudanValue = extractValueAtInitialState(bermudan.getValue(0.0, model), model);
		final double americanValue = extractValueAtInitialState(american.getValue(0.0, model), model);

		final double vanillaBermudanValue = extractValueAtInitialState(vanillaBermudan.getValue(0.0, model), model);
		final double vanillaAmericanValue = extractValueAtInitialState(vanillaAmerican.getValue(0.0, model), model);

		assertTrue(bermudanValue + ORDERING_TOLERANCE_2D >= europeanValue);
		assertTrue(americanValue + ORDERING_TOLERANCE_2D >= bermudanValue);

		assertTrue(vanillaBermudanValue + ORDERING_TOLERANCE_2D >= bermudanValue);
		assertTrue(vanillaAmericanValue + ORDERING_TOLERANCE_2D >= americanValue);
	}

	@Test
	public void testSabrBermudanAndAmericanExerciseOrderingUnderDiscreteMonitoring() {

		final FDMSabrModel model = createSabrModel();

		final DoubleBarrierOption european = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.PUT,
				DoubleBarrierType.KNOCK_IN,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final DoubleBarrierOption bermudan = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.PUT,
				DoubleBarrierType.KNOCK_IN,
				new BermudanExercise(BERMUDAN_EXERCISE_TIMES),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final DoubleBarrierOption american = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.PUT,
				DoubleBarrierType.KNOCK_IN,
				new AmericanExercise(MATURITY),
				MonitoringType.DISCRETE,
				FINE_MONITORING_TIMES
		);

		final BermudanOption vanillaBermudan = new BermudanOption(
				null,
				BERMUDAN_EXERCISE_TIMES,
				STRIKE,
				CallOrPut.PUT
		);

		final AmericanOption vanillaAmerican = new AmericanOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.PUT
		);

		final double europeanValue = extractValueAtInitialState(european.getValue(0.0, model), model);
		final double bermudanValue = extractValueAtInitialState(bermudan.getValue(0.0, model), model);
		final double americanValue = extractValueAtInitialState(american.getValue(0.0, model), model);

		final double vanillaBermudanValue = extractValueAtInitialState(vanillaBermudan.getValue(0.0, model), model);
		final double vanillaAmericanValue = extractValueAtInitialState(vanillaAmerican.getValue(0.0, model), model);

		assertTrue(bermudanValue + ORDERING_TOLERANCE_2D >= europeanValue);
		assertTrue(americanValue + ORDERING_TOLERANCE_2D >= bermudanValue);

		assertTrue(vanillaBermudanValue + ORDERING_TOLERANCE_2D >= bermudanValue);
		assertTrue(vanillaAmericanValue + ORDERING_TOLERANCE_2D >= americanValue);
	}

	@Test
	public void testHestonBermudanEventConditionReplacesByActivatedContinuationValue() {

		final double monitoringTime = 0.50;
		final FDMHestonModel model = createHestonModel();

		final DoubleBarrierOption discreteKnockIn = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_IN,
				new BermudanExercise(BERMUDAN_EXERCISE_TIMES),
				MonitoringType.DISCRETE,
				new double[] { monitoringTime, MATURITY }
		);

		final BermudanOption activated = new BermudanOption(
				null,
				BERMUDAN_EXERCISE_TIMES,
				STRIKE,
				CallOrPut.CALL
		);

		final double[] knockInValuesAtMonitoring = discreteKnockIn.getValue(monitoringTime, model);
		final double[] activatedValuesAtMonitoring = activated.getValue(monitoringTime, model);

		final double[] x0 = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double[] x1 = model.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid();

		for(int j = 0; j < x1.length; j++) {
			for(int i = 0; i < x0.length; i++) {
				if(x0[i] <= LOWER_BARRIER + GRID_TOLERANCE || x0[i] >= UPPER_BARRIER - GRID_TOLERANCE) {
					final int k = flatten(i, j, x0.length);
					assertEquals(activatedValuesAtMonitoring[k], knockInValuesAtMonitoring[k], EVENT_TOLERANCE_2D);
				}
			}
		}
	}

	@Test
	public void testSabrAmericanEventConditionReplacesByActivatedContinuationValue() {

		final double monitoringTime = 0.75;
		final FDMSabrModel model = createSabrModel();

		final DoubleBarrierOption discreteKnockIn = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.PUT,
				DoubleBarrierType.KNOCK_IN,
				new AmericanExercise(MATURITY),
				MonitoringType.DISCRETE,
				new double[] { monitoringTime, MATURITY }
		);

		final AmericanOption activated = new AmericanOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.PUT
		);

		final double[] knockInValuesAtMonitoring = discreteKnockIn.getValue(monitoringTime, model);
		final double[] activatedValuesAtMonitoring = activated.getValue(monitoringTime, model);

		final double[] x0 = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double[] x1 = model.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid();

		for(int j = 0; j < x1.length; j++) {
			for(int i = 0; i < x0.length; i++) {
				if(x0[i] <= LOWER_BARRIER + GRID_TOLERANCE || x0[i] >= UPPER_BARRIER - GRID_TOLERANCE) {
					final int k = flatten(i, j, x0.length);
					assertEquals(activatedValuesAtMonitoring[k], knockInValuesAtMonitoring[k], EVENT_TOLERANCE_2D);
				}
			}
		}
	}

	private FDMHestonModel createHestonModel() {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", RISK_FREE_RATE);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", DIVIDEND_YIELD);

		final Grid spotGrid = new UniformGrid(160, 60.0, 140.0);
		final Grid varianceGrid = new UniformGrid(80, 0.0, 0.4);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { spotGrid, varianceGrid },
				timeDiscretization,
				THETA,
				new double[] { SPOT, 0.04 }
		);

		return new FDMHestonModel(
				SPOT,
				0.04,
				riskFreeCurve,
				dividendCurve,
				1.5,
				0.04,
				0.35,
				-0.30,
				spaceTime
		);
	}

	private FDMSabrModel createSabrModel() {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", RISK_FREE_RATE);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", DIVIDEND_YIELD);

		final Grid spotGrid = new UniformGrid(160, 60.0, 140.0);
		final Grid alphaGrid = new UniformGrid(120, 0.0, 0.6);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { spotGrid, alphaGrid },
				timeDiscretization,
				THETA,
				new double[] { SPOT, 0.2 }
		);

		return new FDMSabrModel(
				SPOT,
				0.2,
				riskFreeCurve,
				dividendCurve,
				0.5,
				0.4,
				-0.2,
				spaceTime
		);
	}

	private double extractValueAtInitialState(
			final double[] valuesOnGrid,
			final FiniteDifferenceEquityModel model) {

		final double[] x0 = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double[] x1 = model.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid();

		final int i0 = getGridIndex(x0, model.getInitialValue()[0]);
		final int i1 = getGridIndex(x1, model.getInitialValue()[1]);

		if(i0 < 0 || i1 < 0) {
			throw new IllegalArgumentException("Initial state is expected to be a grid node in this test.");
		}

		return valuesOnGrid[flatten(i0, i1, x0.length)];
	}

	private static int getGridIndex(final double[] grid, final double value) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - value) < GRID_TOLERANCE) {
				return i;
			}
		}
		return -1;
	}

	private static int flatten(final int i0, final int i1, final int n0) {
		return i0 + i1 * n0;
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
}