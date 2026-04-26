package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.exception.CalculationException;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMMultiAssetBlackScholesModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.GridWithMandatoryPoint;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloMultiAssetBlackScholesModel;
import net.finmath.stochastic.RandomVariable;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Tests for {@link WorstOfOption}.
 *
 * <p>
 * Since finmath does not yet provide a dedicated Monte Carlo worst-of product,
 * the Monte Carlo payoff is constructed inline from the simulated asset values.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class WorstOfOptionTest {

	private static final double MATURITY = 1.0;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD_1 = 0.0;
	private static final double DIVIDEND_YIELD_2 = 0.0;
	private static final double VOLATILITY_1 = 0.20;
	private static final double VOLATILITY_2 = 0.25;
	private static final double CORRELATION = 0.30;

	private static final double SPOT_1 = 100.0;
	private static final double SPOT_2 = 95.0;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS_PDE = 100;

	private static final int NUMBER_OF_TIME_STEPS_MC = 100;
	private static final int NUMBER_OF_PATHS_MC = 200000;
	private static final int RANDOM_SEED = 3141;

	private static final double SYMMETRY_TOLERANCE = 1.0E-3;
	private static final double PDE_DISCRETIZATION_ALLOWANCE = 2.5E-1;
	private static final double MONTE_CARLO_SIGMA_MULTIPLIER = 4.0;

	@Test
	public void testSymmetryUnderAssetSwap() {
		final double strike = 100.0;

		final WorstOfOption originalOption = new WorstOfOption(
				"Asset1",
				"Asset2",
				MATURITY,
				strike,
				CallOrPut.CALL
		);

		final FDMMultiAssetBlackScholesModel originalModel = createTwoAssetPdeModel(
				SPOT_1,
				SPOT_2,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_1,
				DIVIDEND_YIELD_2,
				VOLATILITY_1,
				VOLATILITY_2,
				CORRELATION
		);

		final double originalPrice = interpolateAtInitialState(
				originalOption.getValue(0.0, originalModel),
				originalModel
		);

		final WorstOfOption swappedOption = new WorstOfOption(
				"Asset2",
				"Asset1",
				MATURITY,
				strike,
				CallOrPut.CALL
		);

		final FDMMultiAssetBlackScholesModel swappedModel = createTwoAssetPdeModel(
				SPOT_2,
				SPOT_1,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_2,
				DIVIDEND_YIELD_1,
				VOLATILITY_2,
				VOLATILITY_1,
				CORRELATION
		);

		final double swappedPrice = interpolateAtInitialState(
				swappedOption.getValue(0.0, swappedModel),
				swappedModel
		);

		assertEquals(originalPrice, swappedPrice, SYMMETRY_TOLERANCE);
	}

	@Test
	public void testWorstOfCallAgainstMonteCarlo() throws CalculationException {
		final double strike = 100.0;

		final WorstOfOption pdeProduct = new WorstOfOption(
				"Asset1",
				"Asset2",
				MATURITY,
				strike,
				CallOrPut.CALL
		);

		final FDMMultiAssetBlackScholesModel pdeModel = createTwoAssetPdeModel(
				SPOT_1,
				SPOT_2,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_1,
				DIVIDEND_YIELD_2,
				VOLATILITY_1,
				VOLATILITY_2,
				CORRELATION
		);

		final double pdePrice = interpolateAtInitialState(
				pdeProduct.getValue(0.0, pdeModel),
				pdeModel
		);

		final MonteCarloMultiAssetBlackScholesModel monteCarloModel = createMonteCarloModel(
				SPOT_1,
				SPOT_2,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_1,
				DIVIDEND_YIELD_2,
				VOLATILITY_1,
				VOLATILITY_2,
				CORRELATION
		);

		final MonteCarloEstimate monteCarloEstimate = getMonteCarloWorstOfValue(
				monteCarloModel,
				MATURITY,
				strike,
				CallOrPut.CALL
		);

		final double tolerance =
				PDE_DISCRETIZATION_ALLOWANCE
				+ MONTE_CARLO_SIGMA_MULTIPLIER * monteCarloEstimate.getStandardError();

		assertTrue(
				"PDE price " + pdePrice + " differs from Monte Carlo price "
						+ monteCarloEstimate.getValue() + " by more than tolerance " + tolerance,
				Math.abs(pdePrice - monteCarloEstimate.getValue()) <= tolerance
		);
	}

	@Test
	public void testWorstOfPutAgainstMonteCarlo() throws CalculationException {
		final double strike = 100.0;

		final WorstOfOption pdeProduct = new WorstOfOption(
				"Asset1",
				"Asset2",
				MATURITY,
				strike,
				CallOrPut.PUT
		);

		final FDMMultiAssetBlackScholesModel pdeModel = createTwoAssetPdeModel(
				SPOT_1,
				SPOT_2,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_1,
				DIVIDEND_YIELD_2,
				VOLATILITY_1,
				VOLATILITY_2,
				CORRELATION
		);

		final double pdePrice = interpolateAtInitialState(
				pdeProduct.getValue(0.0, pdeModel),
				pdeModel
		);

		final MonteCarloMultiAssetBlackScholesModel monteCarloModel = createMonteCarloModel(
				SPOT_1,
				SPOT_2,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_1,
				DIVIDEND_YIELD_2,
				VOLATILITY_1,
				VOLATILITY_2,
				CORRELATION
		);

		final MonteCarloEstimate monteCarloEstimate = getMonteCarloWorstOfValue(
				monteCarloModel,
				MATURITY,
				strike,
				CallOrPut.PUT
		);

		final double tolerance =
				PDE_DISCRETIZATION_ALLOWANCE
				+ MONTE_CARLO_SIGMA_MULTIPLIER * monteCarloEstimate.getStandardError();

		assertTrue(
				"PDE price " + pdePrice + " differs from Monte Carlo price "
						+ monteCarloEstimate.getValue() + " by more than tolerance " + tolerance,
				Math.abs(pdePrice - monteCarloEstimate.getValue()) <= tolerance
		);
	}

	private static MonteCarloEstimate getMonteCarloWorstOfValue(
			final MonteCarloMultiAssetBlackScholesModel model,
			final double maturity,
			final double strike,
			final CallOrPut callOrPut) throws CalculationException {

		final RandomVariable asset1 = model.getAssetValue(maturity, 0);
		final RandomVariable asset2 = model.getAssetValue(maturity, 1);

		/*
		 * If asset1 - asset2 >= 0, then worstOf = asset2, otherwise worstOf = asset1.
		 */
		final RandomVariable worstOf = asset1.sub(asset2).choose(asset2, asset1);
		final RandomVariable payoff = worstOf.sub(strike).mult(callOrPut.toInteger()).floor(0.0);
		final RandomVariable discountedPayoff = payoff.mult(Math.exp(-RISK_FREE_RATE * maturity));

		return new MonteCarloEstimate(
				discountedPayoff.getAverage(),
				discountedPayoff.getStandardError()
		);
	}

	private static FDMMultiAssetBlackScholesModel createTwoAssetPdeModel(
			final double spot1,
			final double spot2,
			final double riskFreeRate,
			final double dividendYield1,
			final double dividendYield2,
			final double volatility1,
			final double volatility2,
			final double correlation) {

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				NUMBER_OF_TIME_STEPS_PDE,
				MATURITY / NUMBER_OF_TIME_STEPS_PDE
		);

		final Grid firstAssetGrid = new GridWithMandatoryPoint(80, 0.0, 200.0, spot1);
		final Grid secondAssetGrid = new GridWithMandatoryPoint(76, 0.0, 190.0, spot2);

		final SpaceTimeDiscretization spaceTimeDiscretization = new SpaceTimeDiscretization(
				new Grid[] { firstAssetGrid, secondAssetGrid },
				timeDiscretization,
				THETA,
				new double[] { spot1, spot2 }
		);

		return new FDMMultiAssetBlackScholesModel(
				new double[] { spot1, spot2 },
				createFlatDiscountCurve("riskFreeCurve", riskFreeRate),
				new DiscountCurve[] {
						createFlatDiscountCurve("dividendCurve1", dividendYield1),
						createFlatDiscountCurve("dividendCurve2", dividendYield2)
				},
				new double[] { volatility1, volatility2 },
				new double[][] {
						{ 1.0, correlation },
						{ correlation, 1.0 }
				},
				spaceTimeDiscretization
		);
	}

	private static MonteCarloMultiAssetBlackScholesModel createMonteCarloModel(
			final double spot1,
			final double spot2,
			final double riskFreeRate,
			final double dividendYield1,
			final double dividendYield2,
			final double volatility1,
			final double volatility2,
			final double correlation) {

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				NUMBER_OF_TIME_STEPS_MC,
				MATURITY / NUMBER_OF_TIME_STEPS_MC
		);

		final BrownianMotion brownianMotion = new BrownianMotionFromMersenneRandomNumbers(
				timeDiscretization,
				2,
				NUMBER_OF_PATHS_MC,
				RANDOM_SEED
		);

		return new MonteCarloMultiAssetBlackScholesModel(
				timeDiscretization,
				NUMBER_OF_PATHS_MC,
				new double[] { spot1, spot2 },
				riskFreeRate,
				new double[] { volatility1, volatility2 },
				new double[][] {
						{ 1.0, correlation },
						{ correlation, 1.0 }
				}
		);
	}

	private static double interpolateAtInitialState(
			final double[] flattenedValues,
			final FDMMultiAssetBlackScholesModel model) {

		final double[] firstGrid = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double[] secondGrid = model.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid();

		final double initialValueFirstAsset = model.getInitialValue()[0];
		final double initialValueSecondAsset = model.getInitialValue()[1];

		if(isGridNode(firstGrid, initialValueFirstAsset) && isGridNode(secondGrid, initialValueSecondAsset)) {
			final int firstIndex = getGridIndex(firstGrid, initialValueFirstAsset);
			final int secondIndex = getGridIndex(secondGrid, initialValueSecondAsset);
			return flattenedValues[firstIndex + secondIndex * firstGrid.length];
		}

		return bilinearInterpolation(
				flattenedValues,
				firstGrid,
				secondGrid,
				initialValueFirstAsset,
				initialValueSecondAsset
		);
	}

	private static double bilinearInterpolation(
			final double[] flattenedValues,
			final double[] firstGrid,
			final double[] secondGrid,
			final double x,
			final double y) {

		final int i0 = getLowerBracketIndex(firstGrid, x);
		final int i1 = Math.min(i0 + 1, firstGrid.length - 1);

		final int j0 = getLowerBracketIndex(secondGrid, y);
		final int j1 = Math.min(j0 + 1, secondGrid.length - 1);

		final double x0 = firstGrid[i0];
		final double x1 = firstGrid[i1];
		final double y0 = secondGrid[j0];
		final double y1 = secondGrid[j1];

		final double f00 = flattenedValues[i0 + j0 * firstGrid.length];
		final double f10 = flattenedValues[i1 + j0 * firstGrid.length];
		final double f01 = flattenedValues[i0 + j1 * firstGrid.length];
		final double f11 = flattenedValues[i1 + j1 * firstGrid.length];

		final double wx = (i0 == i1 || Math.abs(x1 - x0) < 1E-14) ? 0.0 : (x - x0) / (x1 - x0);
		final double wy = (j0 == j1 || Math.abs(y1 - y0) < 1E-14) ? 0.0 : (y - y0) / (y1 - y0);

		return (1.0 - wx) * (1.0 - wy) * f00
				+ wx * (1.0 - wy) * f10
				+ (1.0 - wx) * wy * f01
				+ wx * wy * f11;
	}

	private static int getLowerBracketIndex(final double[] grid, final double x) {
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

	private static boolean isGridNode(final double[] grid, final double x) {
		for(final double node : grid) {
			if(Math.abs(node - x) < 1E-12) {
				return true;
			}
		}
		return false;
	}

	private static int getGridIndex(final double[] grid, final double x) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - x) < 1E-12) {
				return i;
			}
		}
		throw new IllegalArgumentException("Point is not a grid node.");
	}

	private static DiscountCurve createFlatDiscountCurve(final String name, final double rate) {
		final double[] times = new double[] { 0.0, 1.0, 2.0, 5.0, 10.0 };
		final double[] zeroRates = new double[] { rate, rate, rate, rate, rate };

		return DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
				name,
				null,
				times,
				zeroRates,
				InterpolationMethod.LINEAR,
				ExtrapolationMethod.CONSTANT,
				InterpolationEntity.VALUE
		);
	}

	private static final class MonteCarloEstimate {

		private final double value;
		private final double standardError;

		private MonteCarloEstimate(final double value, final double standardError) {
			this.value = value;
			this.standardError = standardError;
		}

		private double getValue() {
			return value;
		}

		private double getStandardError() {
			return standardError;
		}
	}
}