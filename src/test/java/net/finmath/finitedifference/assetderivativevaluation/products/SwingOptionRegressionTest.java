package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.SwingQuantityMode;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for the 1D fixed-strike SwingOption under Black-Scholes.
 *
 * <p>
 * These are numerical regression tests, not closed-form swing tests.
 * Fine configurations are used as internal references.
 * </p>
 */
public class SwingOptionRegressionTest {

	private static final double SPOT = 100.0;
	private static final double STRIKE = 100.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;
	private static final double VOLATILITY = 0.25;
	private static final double THETA = 0.5;

	private static final double SPACE_MIN = 20.0;
	private static final double SPACE_MAX = 200.0;

	private static final double[] DECISION_TIMES = new double[] { 1.0 / 3.0, 2.0 / 3.0, 1.0 };
	private static final double MATURITY = DECISION_TIMES[DECISION_TIMES.length - 1];

	private static final int TIME_STEPS_COARSE = 70;
	private static final int SPACE_STEPS_COARSE = 100;

	private static final int TIME_STEPS_MEDIUM = 110;
	private static final int SPACE_STEPS_MEDIUM = 150;

	private static final int TIME_STEPS_FINE = 160;
	private static final int SPACE_STEPS_FINE = 220;

	private static final int TIME_STEPS_REFERENCE = 240;
	private static final int SPACE_STEPS_REFERENCE = 300;

	private static final double ZERO_TOL = 1E-12;
	private static final double ORDERING_TOL = 1E-10;
	private static final double DETERMINISTIC_TOL = 1.2E-1;

	private static final double COARSE_REF_TOL = 3.5E-1;
	private static final double MEDIUM_REF_TOL = 2.0E-1;
	private static final double FINE_REF_TOL = 1.2E-1;

	private static final double COARSE_QUANTITY_REF_TOL = 5.0E-1;
	private static final double MEDIUM_QUANTITY_REF_TOL = 2.5E-1;
	private static final double FINE_QUANTITY_REF_TOL = 1.2E-1;

	@Test
	public void testZeroQuantitiesGiveZeroValue() {
		final BlackScholesSetup setup = createModel(SPOT, MATURITY, TIME_STEPS_FINE, SPACE_STEPS_FINE);

		final SwingOption swing = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				0.0,
				0.0,
				0.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final double value = interpolate1DAtSpot(
				swing.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		assertEquals(0.0, value, ZERO_TOL);
	}

	@Test
	public void testDeterministicScheduleMatchesSumOfEuropeansCall() {
		/*
		 * Deterministic quantity schedule:
		 * - local min = local max = 1 at each decision time
		 * - global min = global max = 3
		 *
		 * Therefore the swing is just the sum of three European call values.
		 */
		final BlackScholesSetup swingSetup = createModel(SPOT, MATURITY, TIME_STEPS_FINE, SPACE_STEPS_FINE);

		final SwingOption swing = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				1.0,
				1.0,
				3.0,
				3.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final double swingValue = interpolate1DAtSpot(
				swing.getValue(0.0, swingSetup.model),
				swingSetup.sNodes,
				SPOT
		);

		double referenceValue = 0.0;
		for(final double decisionTime : DECISION_TIMES) {
			final BlackScholesSetup setupForEuropean = createModel(
					SPOT,
					decisionTime,
					getNumberOfTimeStepsForMaturity(decisionTime, TIME_STEPS_FINE),
					SPACE_STEPS_FINE
			);

			final EuropeanOption european = new EuropeanOption(
					decisionTime,
					STRIKE,
					CallOrPut.CALL
			);

			referenceValue += interpolate1DAtSpot(
					european.getValue(0.0, setupForEuropean.model),
					setupForEuropean.sNodes,
					SPOT
			);
		}

		assertEquals(referenceValue, swingValue, DETERMINISTIC_TOL);
	}

	@Test
	public void testDiscreteQuantityGridDominatesBangBangPut() {
		final BlackScholesSetup setup = createModel(SPOT, MATURITY, TIME_STEPS_FINE, SPACE_STEPS_FINE);

		final SwingOption bangBang = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				1.5,
				CallOrPut.PUT,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final SwingOption discreteGrid = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				1.5,
				CallOrPut.PUT,
				SwingQuantityMode.DISCRETE_QUANTITY_GRID,
				0.25
		);

		final double bangBangValue = interpolate1DAtSpot(
				bangBang.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		final double discreteGridValue = interpolate1DAtSpot(
				discreteGrid.getValue(0.0, setup.model),
				setup.sNodes,
				SPOT
		);

		assertTrue(discreteGridValue + ORDERING_TOL >= bangBangValue);
	}

	@Test
	public void testCallSwingStabilizesWithTimeAndSpaceRefinement() {
		final double referenceValue = priceCallSwing(
				SPOT,
				TIME_STEPS_REFERENCE,
				SPACE_STEPS_REFERENCE,
				0.25
		);

		final double coarseValue = priceCallSwing(
				SPOT,
				TIME_STEPS_COARSE,
				SPACE_STEPS_COARSE,
				0.25
		);

		final double mediumValue = priceCallSwing(
				SPOT,
				TIME_STEPS_MEDIUM,
				SPACE_STEPS_MEDIUM,
				0.25
		);

		final double fineValue = priceCallSwing(
				SPOT,
				TIME_STEPS_FINE,
				SPACE_STEPS_FINE,
				0.25
		);

		final double coarseError = Math.abs(coarseValue - referenceValue);
		final double mediumError = Math.abs(mediumValue - referenceValue);
		final double fineError = Math.abs(fineValue - referenceValue);

		assertTrue(
				"Coarse swing error too large versus reference: " + coarseError,
				coarseError < COARSE_REF_TOL
		);

		assertTrue(
				"Medium swing error too large versus reference: " + mediumError,
				mediumError < MEDIUM_REF_TOL
		);

		assertTrue(
				"Fine swing error too large versus reference: " + fineError,
				fineError < FINE_REF_TOL
		);

		assertTrue(
				"Medium refinement should not be materially worse than coarse refinement.",
				mediumError <= coarseError + 5.0E-2
		);

		assertTrue(
				"Fine refinement should not be materially worse than medium refinement.",
				fineError <= mediumError + 5.0E-2
		);
	}

	@Test
	public void testCallSwingStabilizesWithQuantityGridRefinement() {
		final double referenceValue = priceCallSwing(
				SPOT,
				TIME_STEPS_FINE,
				SPACE_STEPS_FINE,
				0.0625
		);

		final double coarseQuantityValue = priceCallSwing(
				SPOT,
				TIME_STEPS_FINE,
				SPACE_STEPS_FINE,
				0.50
		);

		final double mediumQuantityValue = priceCallSwing(
				SPOT,
				TIME_STEPS_FINE,
				SPACE_STEPS_FINE,
				0.25
		);

		final double fineQuantityValue = priceCallSwing(
				SPOT,
				TIME_STEPS_FINE,
				SPACE_STEPS_FINE,
				0.125
		);

		final double coarseError = Math.abs(coarseQuantityValue - referenceValue);
		final double mediumError = Math.abs(mediumQuantityValue - referenceValue);
		final double fineError = Math.abs(fineQuantityValue - referenceValue);

		assertTrue(
				"Coarse quantity-grid swing error too large versus reference: " + coarseError,
				coarseError < COARSE_QUANTITY_REF_TOL
		);

		assertTrue(
				"Medium quantity-grid swing error too large versus reference: " + mediumError,
				mediumError < MEDIUM_QUANTITY_REF_TOL
		);

		assertTrue(
				"Fine quantity-grid swing error too large versus reference: " + fineError,
				fineError < FINE_QUANTITY_REF_TOL
		);

		assertTrue(
				"Medium quantity-grid refinement should not be materially worse than coarse.",
				mediumError <= coarseError + 5.0E-2
		);

		assertTrue(
				"Fine quantity-grid refinement should not be materially worse than medium.",
				fineError <= mediumError + 5.0E-2
		);
	}

	private double priceCallSwing(
			final double spot,
			final int timeSteps,
			final int spaceSteps,
			final double quantityGridStep) {

		final BlackScholesSetup setup = createModel(spot, MATURITY, timeSteps, spaceSteps);

		final SwingOption swing = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				2.0,
				CallOrPut.CALL,
				SwingQuantityMode.DISCRETE_QUANTITY_GRID,
				quantityGridStep
		);

		return interpolate1DAtSpot(
				swing.getValue(0.0, setup.model),
				setup.sNodes,
				spot
		);
	}

	private BlackScholesSetup createModel(
			final double initialSpot,
			final double maturity,
			final int numberOfTimeSteps,
			final int numberOfSpaceSteps) {

		final Grid sGrid = new UniformGrid(numberOfSpaceSteps, SPACE_MIN, SPACE_MAX);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						numberOfTimeSteps,
						maturity / numberOfTimeSteps
				);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { initialSpot }
		);

		final FDMBlackScholesModel model = new FDMBlackScholesModel(
				initialSpot,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				VOLATILITY,
				spaceTime
		);

		return new BlackScholesSetup(model, sGrid.getGrid());
	}

	private int getNumberOfTimeStepsForMaturity(final double maturity, final int baseStepsAtFinalMaturity) {
		return Math.max(20, (int)Math.round(baseStepsAtFinalMaturity * maturity / MATURITY));
	}

	private double interpolate1DAtSpot(
			final double[] values,
			final double[] sNodes,
			final double spot) {

		if(spot <= sNodes[0]) {
			return values[0];
		}
		if(spot >= sNodes[sNodes.length - 1]) {
			return values[sNodes.length - 1];
		}

		int upperIndex = 1;
		while(upperIndex < sNodes.length && sNodes[upperIndex] < spot) {
			upperIndex++;
		}
		final int lowerIndex = upperIndex - 1;

		final double xL = sNodes[lowerIndex];
		final double xU = sNodes[upperIndex];
		final double yL = values[lowerIndex];
		final double yU = values[upperIndex];

		if(Math.abs(xU - xL) < 1E-14) {
			return yL;
		}

		final double w = (spot - xL) / (xU - xL);
		return (1.0 - w) * yL + w * yU;
	}

	private static final class BlackScholesSetup {

		private final FDMBlackScholesModel model;
		private final double[] sNodes;

		private BlackScholesSetup(
				final FDMBlackScholesModel model,
				final double[] sNodes) {
			this.model = model;
			this.sNodes = sNodes;
		}
	}
}