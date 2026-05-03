package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
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
 * Regression tests for 1D discretely monitored double knock-in options.
 */
public class DoubleBarrierOptionDiscreteMonitoringKnockInTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;
	private static final double SPOT = 100.0;

	private static final double LOWER_BARRIER = 80.0;
	private static final double UPPER_BARRIER = 120.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;
	private static final double VOLATILITY = 0.25;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 240;
	private static final int NUMBER_OF_SPACE_STEPS = 160;

	private static final double[] COARSE_MONITORING_TIMES = new double[] { 0.50, 1.00 };
	private static final double[] FINE_MONITORING_TIMES = new double[] { 0.25, 0.50, 0.75, 1.00 };
	private static final double[] BERMUDAN_EXERCISE_TIMES = new double[] { 0.25, 0.50, 0.75, 1.00 };

	private static final double ORDERING_TOLERANCE = 1E-2;
	private static final double EVENT_TOLERANCE = 1E-10;
	private static final double GRID_TOLERANCE = 1E-12;

	@Test
	public void testEuropeanCallDiscreteMonitoringOrderingAgainstContinuousAndVanilla() {

		final FDMBlackScholesModel model = createBlackScholesModel();

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

		final double coarseValue = extractValueAtSpot(coarseDiscrete.getValue(0.0, model), model, SPOT);
		final double fineValue = extractValueAtSpot(fineDiscrete.getValue(0.0, model), model, SPOT);
		final double continuousValue = extractValueAtSpot(continuous.getValue(0.0, model), model, SPOT);
		final double vanillaValue = extractValueAtSpot(vanilla.getValue(0.0, model), model, SPOT);

		assertTrue(coarseValue >= -ORDERING_TOLERANCE);
		assertTrue(coarseValue <= fineValue + ORDERING_TOLERANCE);
		assertTrue(fineValue <= continuousValue + ORDERING_TOLERANCE);
		assertTrue(continuousValue <= vanillaValue + ORDERING_TOLERANCE);
	}

	@Test
	public void testEuropeanPutDiscreteMonitoringOrderingAgainstContinuousAndVanilla() {

		final FDMBlackScholesModel model = createBlackScholesModel();

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

		final double coarseValue = extractValueAtSpot(coarseDiscrete.getValue(0.0, model), model, SPOT);
		final double fineValue = extractValueAtSpot(fineDiscrete.getValue(0.0, model), model, SPOT);
		final double continuousValue = extractValueAtSpot(continuous.getValue(0.0, model), model, SPOT);
		final double vanillaValue = extractValueAtSpot(vanilla.getValue(0.0, model), model, SPOT);

		assertTrue(coarseValue >= -ORDERING_TOLERANCE);
		assertTrue(coarseValue <= fineValue + ORDERING_TOLERANCE);
		assertTrue(fineValue <= vanillaValue + ORDERING_TOLERANCE);
		assertTrue(continuousValue <= vanillaValue + ORDERING_TOLERANCE);
	}

	@Test
	public void testBermudanAndAmericanExerciseOrderingUnderDiscreteMonitoring() {

		final FDMBlackScholesModel model = createBlackScholesModel();

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

		final double europeanValue = extractValueAtSpot(european.getValue(0.0, model), model, SPOT);
		final double bermudanValue = extractValueAtSpot(bermudan.getValue(0.0, model), model, SPOT);
		final double americanValue = extractValueAtSpot(american.getValue(0.0, model), model, SPOT);

		final double vanillaBermudanValue = extractValueAtSpot(vanillaBermudan.getValue(0.0, model), model, SPOT);
		final double vanillaAmericanValue = extractValueAtSpot(vanillaAmerican.getValue(0.0, model), model, SPOT);

		assertTrue(bermudanValue + ORDERING_TOLERANCE >= europeanValue);
		assertTrue(americanValue + ORDERING_TOLERANCE >= bermudanValue);

		assertTrue(vanillaBermudanValue + ORDERING_TOLERANCE >= bermudanValue);
		assertTrue(vanillaAmericanValue + ORDERING_TOLERANCE >= americanValue);
	}

	@Test
	public void testBermudanEventConditionReplacesByActivatedContinuationValue() {

		final double monitoringTime = 0.50;
		final FDMBlackScholesModel model = createBlackScholesModel();

		final DoubleBarrierOption discreteKnockIn = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.PUT,
				DoubleBarrierType.KNOCK_IN,
				new BermudanExercise(BERMUDAN_EXERCISE_TIMES),
				MonitoringType.DISCRETE,
				new double[] { monitoringTime, MATURITY }
		);

		final BermudanOption activated = new BermudanOption(
				null,
				BERMUDAN_EXERCISE_TIMES,
				STRIKE,
				CallOrPut.PUT
		);

		final double[] knockInValuesAtMonitoring = discreteKnockIn.getValue(monitoringTime, model);
		final double[] activatedValuesAtMonitoring = activated.getValue(monitoringTime, model);
		final double[] spotGrid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

		for(int i = 0; i < spotGrid.length; i++) {
			if(spotGrid[i] <= LOWER_BARRIER + GRID_TOLERANCE || spotGrid[i] >= UPPER_BARRIER - GRID_TOLERANCE) {
				assertEquals(activatedValuesAtMonitoring[i], knockInValuesAtMonitoring[i], EVENT_TOLERANCE);
			}
		}
	}

	@Test
	public void testAmericanEventConditionReplacesByActivatedContinuationValue() {

		final double monitoringTime = 0.75;
		final FDMBlackScholesModel model = createBlackScholesModel();

		final DoubleBarrierOption discreteKnockIn = new DoubleBarrierOption(
				null,
				MATURITY,
				STRIKE,
				LOWER_BARRIER,
				UPPER_BARRIER,
				CallOrPut.CALL,
				DoubleBarrierType.KNOCK_IN,
				new AmericanExercise(MATURITY),
				MonitoringType.DISCRETE,
				new double[] { monitoringTime, MATURITY }
		);

		final AmericanOption activated = new AmericanOption(
				null,
				MATURITY,
				STRIKE,
				CallOrPut.CALL
		);

		final double[] knockInValuesAtMonitoring = discreteKnockIn.getValue(monitoringTime, model);
		final double[] activatedValuesAtMonitoring = activated.getValue(monitoringTime, model);
		final double[] spotGrid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();

		for(int i = 0; i < spotGrid.length; i++) {
			if(spotGrid[i] <= LOWER_BARRIER + GRID_TOLERANCE || spotGrid[i] >= UPPER_BARRIER - GRID_TOLERANCE) {
				assertEquals(activatedValuesAtMonitoring[i], knockInValuesAtMonitoring[i], EVENT_TOLERANCE);
			}
		}
	}

	private FDMBlackScholesModel createBlackScholesModel() {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", RISK_FREE_RATE);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", DIVIDEND_YIELD);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final Grid spotGrid = new UniformGrid(
				NUMBER_OF_SPACE_STEPS,
				60.0,
				140.0
		);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				spotGrid,
				timeDiscretization,
				THETA,
				new double[] { SPOT }
		);

		return new FDMBlackScholesModel(
				SPOT,
				riskFreeCurve,
				dividendCurve,
				VOLATILITY,
				spaceTime
		);
	}

	private double extractValueAtSpot(
			final double[] valuesOnGrid,
			final FDMBlackScholesModel model,
			final double spot) {

		final double[] spotNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final int index = getGridIndex(spotNodes, spot);
		if(index < 0) {
			throw new IllegalArgumentException("Spot is expected to be a grid node in this test.");
		}
		return valuesOnGrid[index];
	}

	private static int getGridIndex(final double[] grid, final double value) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - value) < GRID_TOLERANCE) {
				return i;
			}
		}
		return -1;
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