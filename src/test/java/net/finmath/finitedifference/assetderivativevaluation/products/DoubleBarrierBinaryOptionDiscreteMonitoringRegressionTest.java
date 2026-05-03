package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBachelierModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMCevModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.DoubleBarrierType;
import net.finmath.modelling.products.MonitoringType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for discretely monitored DoubleBarrierBinaryOption.
 *
 * <p>
 * These tests focus on event-condition consistency across all supported model
 * families: Black-Scholes, CEV, Bachelier, Heston, and SABR.
 * </p>
 */
public class DoubleBarrierBinaryOptionDiscreteMonitoringRegressionTest {

	private enum ModelType {
		BLACK_SCHOLES,
		CEV,
		BACHELIER,
		HESTON,
		SABR
	}

	private static final double MATURITY = 1.0;
	private static final double CASH_PAYOFF = 10.0;

	private static final double SPOT = 100.0;
	private static final double LOWER_BARRIER = 80.0;
	private static final double UPPER_BARRIER = 120.0;

	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD = 0.00;

	private static final double BS_VOLATILITY = 0.25;

	private static final double CEV_SIGMA = 0.25;
	private static final double CEV_BETA = 1.0;

	private static final double BACHELIER_VOLATILITY = 25.0;

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

	private static final double SPACE_MIN = 0.0;
	private static final double SPACE_MAX = 200.0;

	private static final int TIME_STEPS_1D = 120;
	private static final int SPACE_STEPS_1D = 160;

	private static final int TIME_STEPS_2D = 80;
	private static final int SPACE_STEPS_S_2D = 120;
	private static final int SPACE_STEPS_SECOND_2D = 60;

	private static final double[] MONITORING_TIMES = new double[] {
			0.25,
			0.50,
			0.75,
			1.00
	};

	private static final double PARITY_TOL_1D = 1.0E-1;
	private static final double PARITY_TOL_2D = 5.0E-1;

	private static final double KIKO_KOKI_TOL_1D = 1.0E-1;
	private static final double KIKO_KOKI_TOL_2D = 7.5E-1;

	private static final double VALUE_BOUND_TOL_1D = 1.0E-8;
	private static final double VALUE_BOUND_TOL_2D = 1.0E-1;

	private static final double MONOTONICITY_TOL = 1.0E-8;

	@Test
	public void testDiscreteKnockInPlusKnockOutMatchesDiscountedCashAcrossAllModels() {

		for(final ModelType modelType : ModelType.values()) {

			final FiniteDifferenceEquityModel model = createModel(modelType);

			final DoubleBarrierBinaryOption knockIn = createDiscreteProduct(DoubleBarrierType.KNOCK_IN);
			final DoubleBarrierBinaryOption knockOut = createDiscreteProduct(DoubleBarrierType.KNOCK_OUT);

			final double knockInValue = priceAtInitialState(knockIn, model);
			final double knockOutValue = priceAtInitialState(knockOut, model);

			final double expectedDiscountedCash =
					CASH_PAYOFF * model.getRiskFreeCurve().getDiscountFactor(MATURITY);

			assertEquals(
					modelType + " discrete KI + KO should match discounted cash.",
					expectedDiscountedCash,
					knockInValue + knockOutValue,
					getParityTolerance(modelType)
			);
		}
	}

	@Test
	public void testDiscreteKikoPlusKokiMatchesKnockInAcrossAllModels() {

		for(final ModelType modelType : ModelType.values()) {

			final FiniteDifferenceEquityModel model = createModel(modelType);

			final DoubleBarrierBinaryOption knockIn = createDiscreteProduct(DoubleBarrierType.KNOCK_IN);
			final DoubleBarrierBinaryOption kiko = createDiscreteProduct(DoubleBarrierType.KIKO);
			final DoubleBarrierBinaryOption koki = createDiscreteProduct(DoubleBarrierType.KOKI);

			final double knockInValue = priceAtInitialState(knockIn, model);
			final double kikoValue = priceAtInitialState(kiko, model);
			final double kokiValue = priceAtInitialState(koki, model);

			assertEquals(
					modelType + " discrete KIKO + KOKI should match KNOCK_IN.",
					knockInValue,
					kikoValue + kokiValue,
					getKikoKokiTolerance(modelType)
			);
		}
	}

	@Test
	public void testDiscreteValuesAreBoundedAcrossAllModels() {

		for(final ModelType modelType : ModelType.values()) {
			for(final DoubleBarrierType doubleBarrierType : DoubleBarrierType.values()) {

				final FiniteDifferenceEquityModel model = createModel(modelType);
				final DoubleBarrierBinaryOption product = createDiscreteProduct(doubleBarrierType);

				final double value = priceAtInitialState(product, model);
				final double tolerance = getValueBoundTolerance(modelType);

				assertTrue(
						modelType + " " + doubleBarrierType + " value should be non-negative.",
						value >= -tolerance
				);

				assertTrue(
						modelType + " " + doubleBarrierType + " value should not exceed cash payoff.",
						value <= CASH_PAYOFF + tolerance
				);
			}
		}
	}

	@Test
	public void testDiscreteMonitoringOrderingAgainstContinuousBlackScholes() {

		final FiniteDifferenceEquityModel model = createModel(ModelType.BLACK_SCHOLES);

		final DoubleBarrierBinaryOption discreteKnockOut =
				createDiscreteProduct(DoubleBarrierType.KNOCK_OUT);
		final DoubleBarrierBinaryOption continuousKnockOut =
				createContinuousProduct(DoubleBarrierType.KNOCK_OUT);

		final DoubleBarrierBinaryOption discreteKnockIn =
				createDiscreteProduct(DoubleBarrierType.KNOCK_IN);
		final DoubleBarrierBinaryOption continuousKnockIn =
				createContinuousProduct(DoubleBarrierType.KNOCK_IN);

		final double discreteKnockOutValue = priceAtInitialState(discreteKnockOut, model);
		final double continuousKnockOutValue = priceAtInitialState(continuousKnockOut, model);

		final double discreteKnockInValue = priceAtInitialState(discreteKnockIn, model);
		final double continuousKnockInValue = priceAtInitialState(continuousKnockIn, model);

		assertTrue(
				"Discrete KO should be at least as valuable as continuous KO.",
				discreteKnockOutValue + MONOTONICITY_TOL >= continuousKnockOutValue
		);

		assertTrue(
				"Discrete KI should be at most as valuable as continuous KI.",
				discreteKnockInValue <= continuousKnockInValue + MONOTONICITY_TOL
		);
	}

	private DoubleBarrierBinaryOption createDiscreteProduct(final DoubleBarrierType doubleBarrierType) {
		return new DoubleBarrierBinaryOption(
				MATURITY,
				CASH_PAYOFF,
				LOWER_BARRIER,
				UPPER_BARRIER,
				doubleBarrierType,
				new EuropeanExercise(MATURITY),
				MonitoringType.DISCRETE,
				MONITORING_TIMES
		);
	}

	private DoubleBarrierBinaryOption createContinuousProduct(final DoubleBarrierType doubleBarrierType) {
		return new DoubleBarrierBinaryOption(
				MATURITY,
				CASH_PAYOFF,
				LOWER_BARRIER,
				UPPER_BARRIER,
				doubleBarrierType,
				new EuropeanExercise(MATURITY)
		);
	}

	private double priceAtInitialState(
			final DoubleBarrierBinaryOption product,
			final FiniteDifferenceEquityModel model) {

		final double[] values = product.getValue(0.0, model);
		final SpaceTimeDiscretization discretization = model.getSpaceTimeDiscretization();

		if(discretization.getNumberOfSpaceGrids() == 1) {
			return interpolate1DAtSpot(
					values,
					discretization.getSpaceGrid(0).getGrid(),
					SPOT
			);
		}

		return interpolate2DAtInitialState(
				values,
				discretization.getSpaceGrid(0).getGrid(),
				discretization.getSpaceGrid(1).getGrid(),
				SPOT,
				model.getInitialValue()[1]
		);
	}

	private FiniteDifferenceEquityModel createModel(final ModelType modelType) {

		if(modelType == ModelType.BLACK_SCHOLES) {
			return createBlackScholesModel();
		}
		if(modelType == ModelType.CEV) {
			return createCevModel();
		}
		if(modelType == ModelType.BACHELIER) {
			return createBachelierModel();
		}
		if(modelType == ModelType.HESTON) {
			return createHestonModel();
		}
		if(modelType == ModelType.SABR) {
			return createSabrModel();
		}

		throw new IllegalArgumentException("Unsupported model type: " + modelType);
	}

	private FDMBlackScholesModel createBlackScholesModel() {
		return new FDMBlackScholesModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				BS_VOLATILITY,
				createOneDimensionalSpaceTimeDiscretization()
		);
	}

	private FDMCevModel createCevModel() {
		return new FDMCevModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				CEV_SIGMA,
				CEV_BETA,
				createOneDimensionalSpaceTimeDiscretization()
		);
	}

	private FDMBachelierModel createBachelierModel() {
		return new FDMBachelierModel(
				SPOT,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				BACHELIER_VOLATILITY,
				createOneDimensionalSpaceTimeDiscretization()
		);
	}

	private FDMHestonModel createHestonModel() {

		final SpaceTimeDiscretization spaceTime = createTwoDimensionalSpaceTimeDiscretization(
				createHestonVarianceGrid(),
				HESTON_INITIAL_VARIANCE
		);

		return new FDMHestonModel(
				SPOT,
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

	private FDMSabrModel createSabrModel() {

		final SpaceTimeDiscretization spaceTime = createTwoDimensionalSpaceTimeDiscretization(
				createSabrAlphaGrid(),
				SABR_INITIAL_ALPHA
		);

		return new FDMSabrModel(
				SPOT,
				SABR_INITIAL_ALPHA,
				RISK_FREE_RATE,
				DIVIDEND_YIELD,
				SABR_BETA,
				SABR_NU,
				SABR_RHO,
				spaceTime
		);
	}

	private SpaceTimeDiscretization createOneDimensionalSpaceTimeDiscretization() {

		final Grid sGrid = new UniformGrid(SPACE_STEPS_1D, SPACE_MIN, SPACE_MAX);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						TIME_STEPS_1D,
						MATURITY / TIME_STEPS_1D
				);

		return new SpaceTimeDiscretization(
				sGrid,
				timeDiscretization,
				THETA,
				new double[] { SPOT }
		);
	}

	private SpaceTimeDiscretization createTwoDimensionalSpaceTimeDiscretization(
			final Grid secondGrid,
			final double initialSecondState) {

		final Grid sGrid = new UniformGrid(SPACE_STEPS_S_2D, SPACE_MIN, SPACE_MAX);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						TIME_STEPS_2D,
						MATURITY / TIME_STEPS_2D
				);

		return new SpaceTimeDiscretization(
				new Grid[] { sGrid, secondGrid },
				timeDiscretization,
				THETA,
				new double[] { SPOT, initialSecondState }
		);
	}

	private Grid createHestonVarianceGrid() {
		final double varianceMax = Math.max(
				4.0 * HESTON_THETA_V,
				HESTON_INITIAL_VARIANCE + 4.0 * HESTON_XI * Math.sqrt(MATURITY)
		);

		return new UniformGrid(
				SPACE_STEPS_SECOND_2D,
				0.0,
				varianceMax
		);
	}

	private Grid createSabrAlphaGrid() {
		final double alphaMax = Math.max(4.0 * SABR_INITIAL_ALPHA, 1.0);

		return new UniformGrid(
				SPACE_STEPS_SECOND_2D,
				0.0,
				alphaMax
		);
	}

	private double getParityTolerance(final ModelType modelType) {
		return isOneDimensionalModel(modelType) ? PARITY_TOL_1D : PARITY_TOL_2D;
	}

	private double getKikoKokiTolerance(final ModelType modelType) {
		return isOneDimensionalModel(modelType) ? KIKO_KOKI_TOL_1D : KIKO_KOKI_TOL_2D;
	}

	private double getValueBoundTolerance(final ModelType modelType) {
		return isOneDimensionalModel(modelType) ? VALUE_BOUND_TOL_1D : VALUE_BOUND_TOL_2D;
	}

	private boolean isOneDimensionalModel(final ModelType modelType) {
		return modelType == ModelType.BLACK_SCHOLES
				|| modelType == ModelType.CEV
				|| modelType == ModelType.BACHELIER;
	}

	private double interpolate1DAtSpot(
			final double[] values,
			final double[] sNodes,
			final double spot) {

		final int exactIndex = findExactIndex(sNodes, spot);
		if(exactIndex >= 0) {
			return values[exactIndex];
		}

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

		final double weight = (spot - xL) / (xU - xL);

		return (1.0 - weight) * yL + weight * yU;
	}

	private double interpolate2DAtInitialState(
			final double[] flattenedValues,
			final double[] sNodes,
			final double[] secondNodes,
			final double spot,
			final double secondState) {

		final int numberOfSNodes = sNodes.length;

		final int i0 = getLowerBracketIndex(sNodes, spot);
		final int i1 = Math.min(i0 + 1, sNodes.length - 1);

		final int j0 = getLowerBracketIndex(secondNodes, secondState);
		final int j1 = Math.min(j0 + 1, secondNodes.length - 1);

		final double x0 = sNodes[i0];
		final double x1 = sNodes[i1];
		final double y0 = secondNodes[j0];
		final double y1 = secondNodes[j1];

		final double f00 = flattenedValues[flatten(i0, j0, numberOfSNodes)];
		final double f10 = flattenedValues[flatten(i1, j0, numberOfSNodes)];
		final double f01 = flattenedValues[flatten(i0, j1, numberOfSNodes)];
		final double f11 = flattenedValues[flatten(i1, j1, numberOfSNodes)];

		final double wx = Math.abs(x1 - x0) < 1E-14 ? 0.0 : (spot - x0) / (x1 - x0);
		final double wy = Math.abs(y1 - y0) < 1E-14 ? 0.0 : (secondState - y0) / (y1 - y0);

		return (1.0 - wx) * (1.0 - wy) * f00
				+ wx * (1.0 - wy) * f10
				+ (1.0 - wx) * wy * f01
				+ wx * wy * f11;
	}

	private int findExactIndex(final double[] grid, final double x) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - x) < 1E-12) {
				return i;
			}
		}

		return -1;
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
}