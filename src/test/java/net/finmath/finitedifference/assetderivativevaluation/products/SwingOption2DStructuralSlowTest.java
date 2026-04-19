package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.modelling.products.SwingQuantityMode;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Slower structural tests for the 2D fixed-strike SwingOption recursion.
 *
 * <p>
 * Focus:
 * </p>
 * <ul>
 *   <li>deeper multi-date recursion,</li>
 *   <li>relaxing the global minimum cannot reduce value for non-negative call payoffs,</li>
 *   <li>relaxing the global maximum cannot reduce value,</li>
 *   <li>a discretized quantity grid dominates bang-bang control over three decision dates.</li>
 * </ul>
 */
public class SwingOption2DStructuralSlowTest {

	private enum ModelType {
		HESTON,
		SABR
	}

	private static final double SPOT = 100.0;
	private static final double STRIKE = 100.0;
	private static final double[] DECISION_TIMES = new double[] { 1.0 / 3.0, 2.0 / 3.0, 1.0 };

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;

	private static final double HESTON_VOLATILITY = 0.25;
	private static final double HESTON_INITIAL_VARIANCE = HESTON_VOLATILITY * HESTON_VOLATILITY;
	private static final double HESTON_KAPPA = 1.5;
	private static final double HESTON_THETA_V = HESTON_INITIAL_VARIANCE;
	private static final double HESTON_XI = 0.30;
	private static final double HESTON_RHO = -0.70;

	private static final double SABR_INITIAL_ALPHA = 0.20;
	private static final double SABR_BETA = 1.0;
	private static final double SABR_NU = 0.30;
	private static final double SABR_RHO = -0.50;

	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 45;
	private static final int NUMBER_OF_SPACE_STEPS_S = 80;
	private static final int NUMBER_OF_SPACE_STEPS_SECOND = 30;

	private static final double SPACE_MIN = 20.0;
	private static final double SPACE_MAX = 200.0;

	private static final double ORDERING_TOL_2D = 2.5E-1;

	@Test
	public void testHestonRelaxingGlobalMinimumCannotReduceValue() {
		runRelaxingGlobalMinimumCannotReduceValueTest(ModelType.HESTON);
	}

	@Test
	public void testSabrRelaxingGlobalMinimumCannotReduceValue() {
		runRelaxingGlobalMinimumCannotReduceValueTest(ModelType.SABR);
	}

	@Test
	public void testHestonRelaxingGlobalMaximumCannotReduceValue() {
		runRelaxingGlobalMaximumCannotReduceValueTest(ModelType.HESTON);
	}

	@Test
	public void testSabrRelaxingGlobalMaximumCannotReduceValue() {
		runRelaxingGlobalMaximumCannotReduceValueTest(ModelType.SABR);
	}

	@Test
	public void testHestonThreeDateDiscreteQuantityGridDominatesBangBang() {
		runThreeDateDiscreteQuantityGridDominatesBangBangTest(ModelType.HESTON);
	}

	@Test
	public void testSabrThreeDateDiscreteQuantityGridDominatesBangBang() {
		runThreeDateDiscreteQuantityGridDominatesBangBangTest(ModelType.SABR);
	}

	private void runRelaxingGlobalMinimumCannotReduceValueTest(final ModelType modelType) {
		final TwoDimensionalSetup setup = createTwoDimensionalSetup(modelType, SPOT);

		final SwingOption stricterGlobalMin = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				1.0,
				3.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final SwingOption looserGlobalMin = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				3.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final double stricterValue = interpolate2DAtInitialState(
				stricterGlobalMin.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT,
				setup.initialSecondState
		);

		final double looserValue = interpolate2DAtInitialState(
				looserGlobalMin.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT,
				setup.initialSecondState
		);

		assertTrue(looserValue + ORDERING_TOL_2D >= stricterValue);
	}

	private void runRelaxingGlobalMaximumCannotReduceValueTest(final ModelType modelType) {
		final TwoDimensionalSetup setup = createTwoDimensionalSetup(modelType, SPOT);

		final SwingOption tighterGlobalMax = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				1.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final SwingOption looserGlobalMax = new SwingOption(
				DECISION_TIMES,
				STRIKE,
				0.0,
				1.0,
				0.0,
				3.0,
				CallOrPut.CALL,
				SwingQuantityMode.BANG_BANG,
				0.25
		);

		final double tighterValue = interpolate2DAtInitialState(
				tighterGlobalMax.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT,
				setup.initialSecondState
		);

		final double looserValue = interpolate2DAtInitialState(
				looserGlobalMax.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT,
				setup.initialSecondState
		);

		assertTrue(looserValue + ORDERING_TOL_2D >= tighterValue);
	}

	private void runThreeDateDiscreteQuantityGridDominatesBangBangTest(final ModelType modelType) {
		final TwoDimensionalSetup setup = createTwoDimensionalSetup(modelType, SPOT);

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

		final double bangBangValue = interpolate2DAtInitialState(
				bangBang.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT,
				setup.initialSecondState
		);

		final double discreteGridValue = interpolate2DAtInitialState(
				discreteGrid.getValue(0.0, setup.model),
				setup.sNodes,
				setup.secondNodes,
				SPOT,
				setup.initialSecondState
		);

		assertTrue(discreteGridValue + ORDERING_TOL_2D >= bangBangValue);
	}

	private TwoDimensionalSetup createTwoDimensionalSetup(
			final ModelType modelType,
			final double initialSpot) {

		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_S, SPACE_MIN, SPACE_MAX);
		final Grid secondGrid = createSecondGrid(modelType);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						DECISION_TIMES[DECISION_TIMES.length - 1] / NUMBER_OF_TIME_STEPS
				);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, secondGrid },
				timeDiscretization,
				THETA,
				new double[] { initialSpot, getInitialSecondState(modelType) }
		);

		return new TwoDimensionalSetup(
				createTwoDimensionalModel(modelType, initialSpot, spaceTime),
				sGrid.getGrid(),
				secondGrid.getGrid(),
				getInitialSecondState(modelType)
		);
	}

	private Grid createSecondGrid(final ModelType modelType) {
		if(modelType == ModelType.HESTON) {
			final double vMax = Math.max(
					4.0 * HESTON_THETA_V,
					HESTON_INITIAL_VARIANCE + 4.0 * HESTON_XI * Math.sqrt(DECISION_TIMES[DECISION_TIMES.length - 1])
			);
			return new UniformGrid(NUMBER_OF_SPACE_STEPS_SECOND, 0.0, vMax);
		}
		else if(modelType == ModelType.SABR) {
			final double alphaMax = Math.max(4.0 * SABR_INITIAL_ALPHA, 1.0);
			return new UniformGrid(NUMBER_OF_SPACE_STEPS_SECOND, 0.0, alphaMax);
		}
		else {
			throw new IllegalArgumentException("Unsupported model type: " + modelType);
		}
	}

	private FiniteDifferenceEquityModel createTwoDimensionalModel(
			final ModelType modelType,
			final double initialSpot,
			final SpaceTimeDiscretization spaceTime) {

		if(modelType == ModelType.HESTON) {
			return new FDMHestonModel(
					initialSpot,
					HESTON_INITIAL_VARIANCE,
					RISK_FREE_RATE,
					DIVIDEND_YIELD,
					HESTON_KAPPA,
					HESTON_THETA_V,
					HESTON_XI,
					HESTON_RHO,
					spaceTime
			);
		}
		else if(modelType == ModelType.SABR) {
			return new FDMSabrModel(
					initialSpot,
					SABR_INITIAL_ALPHA,
					RISK_FREE_RATE,
					DIVIDEND_YIELD,
					SABR_BETA,
					SABR_NU,
					SABR_RHO,
					spaceTime
			);
		}
		else {
			throw new IllegalArgumentException("Unsupported model type: " + modelType);
		}
	}

	private double getInitialSecondState(final ModelType modelType) {
		if(modelType == ModelType.HESTON) {
			return HESTON_INITIAL_VARIANCE;
		}
		if(modelType == ModelType.SABR) {
			return SABR_INITIAL_ALPHA;
		}
		throw new IllegalArgumentException("Unsupported model type: " + modelType);
	}

	private double interpolate2DAtInitialState(
			final double[] flattenedValues,
			final double[] sNodes,
			final double[] secondNodes,
			final double spot,
			final double secondState) {

		final int nS = sNodes.length;

		final int i0 = getLowerBracketIndex(sNodes, spot);
		final int i1 = Math.min(i0 + 1, sNodes.length - 1);

		final int j0 = getLowerBracketIndex(secondNodes, secondState);
		final int j1 = Math.min(j0 + 1, secondNodes.length - 1);

		final double x0 = sNodes[i0];
		final double x1 = sNodes[i1];
		final double y0 = secondNodes[j0];
		final double y1 = secondNodes[j1];

		final double f00 = flattenedValues[flatten(i0, j0, nS)];
		final double f10 = flattenedValues[flatten(i1, j0, nS)];
		final double f01 = flattenedValues[flatten(i0, j1, nS)];
		final double f11 = flattenedValues[flatten(i1, j1, nS)];

		final double wx = Math.abs(x1 - x0) < 1E-14 ? 0.0 : (spot - x0) / (x1 - x0);
		final double wy = Math.abs(y1 - y0) < 1E-14 ? 0.0 : (secondState - y0) / (y1 - y0);

		return (1.0 - wx) * (1.0 - wy) * f00
				+ wx * (1.0 - wy) * f10
				+ (1.0 - wx) * wy * f01
				+ wx * wy * f11;
	}

	private int getLowerBracketIndex(final double[] grid, final double x) {
		if(x <= grid[0]) {
			return 0;
		}
		if(x >= grid[grid.length - 1]) {
			return grid.length - 2;
		}

		int upperIndex = 1;
		while(upperIndex < grid.length && grid[upperIndex] < x) {
			upperIndex++;
		}
		return upperIndex - 1;
	}

	private int flatten(final int iS, final int i2, final int numberOfSNodes) {
		return iS + i2 * numberOfSNodes;
	}

	private static final class TwoDimensionalSetup {

		private final FiniteDifferenceEquityModel model;
		private final double[] sNodes;
		private final double[] secondNodes;
		private final double initialSecondState;

		private TwoDimensionalSetup(
				final FiniteDifferenceEquityModel model,
				final double[] sNodes,
				final double[] secondNodes,
				final double initialSecondState) {
			this.model = model;
			this.sNodes = sNodes;
			this.secondNodes = secondNodes;
			this.initialSecondState = initialSecondState;
		}
	}
}