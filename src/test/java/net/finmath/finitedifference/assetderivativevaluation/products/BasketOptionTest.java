package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
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
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Unit tests for {@link BasketOption} under {@link FDMMultiAssetBlackScholesModel}.
 *
 * <p>
 * The test suite focuses on structural properties that should hold independently of
 * the Monte Carlo benchmark layer:
 * </p>
 * <ul>
 *   <li>the two-asset basket reduces to a one-asset vanilla option when one weight is zero,</li>
 *   <li>a strike-zero basket call reduces to the discounted basket forward value,</li>
 *   <li>the price is symmetric under swapping the two assets together with all associated parameters.</li>
 * </ul>
 *
 * @author Alessandro Gnoatto
 */
public class BasketOptionTest {

	private static final double MATURITY = 1.0;
	private static final double RISK_FREE_RATE = 0.05;
	private static final double DIVIDEND_YIELD_1 = 0.01;
	private static final double DIVIDEND_YIELD_2 = 0.02;
	private static final double VOLATILITY_1 = 0.20;
	private static final double VOLATILITY_2 = 0.25;
	private static final double CORRELATION = 0.30;

	private static final double SPOT_1 = 100.0;
	private static final double SPOT_2 = 95.0;

	private static final double THETA = 0.5;
	private static final int NUMBER_OF_TIME_STEPS = 100;

	private static final double REDUCTION_TOLERANCE = 1.5E-1;
	private static final double STRIKE_ZERO_TOLERANCE = 1.5E-1;
	private static final double SYMMETRY_TOLERANCE = 1.0E-4;

	@Test
	public void testSmokeCallEvaluation() {
		final BasketOption basketOption = new BasketOption(
				new String[] { "Asset1", "Asset2" },
				MATURITY,
				new double[] { 0.6, 0.4 },
				100.0,
				CallOrPut.CALL
		);

		final FDMMultiAssetBlackScholesModel model = createTwoAssetModel(
				SPOT_1,
				SPOT_2,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_1,
				DIVIDEND_YIELD_2,
				VOLATILITY_1,
				VOLATILITY_2,
				CORRELATION
		);

		final double price = interpolateAtInitialState(basketOption.getValue(0.0, model), model);

		assertTrue(price >= -1E-12);
	}

	@Test
	public void testReductionToVanillaOnFirstAsset() {
		final double strike = 100.0;

		final BasketOption basketOption = new BasketOption(
				new String[] { "Asset1", "Asset2" },
				MATURITY,
				new double[] { 1.0, 0.0 },
				strike,
				CallOrPut.CALL
		);

		final FDMMultiAssetBlackScholesModel twoAssetModel = createTwoAssetModel(
				SPOT_1,
				SPOT_2,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_1,
				DIVIDEND_YIELD_2,
				VOLATILITY_1,
				VOLATILITY_2,
				CORRELATION
		);

		final double basketPrice = interpolateAtInitialState(
				basketOption.getValue(0.0, twoAssetModel),
				twoAssetModel
		);

		final FDMBlackScholesModel oneAssetModel = createOneAssetModel(
				SPOT_1,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_1,
				VOLATILITY_1
		);

		final EuropeanOption europeanOption = new EuropeanOption(
				"Asset1",
				MATURITY,
				strike,
				CallOrPut.CALL
		);

		final double vanillaPrice = interpolateAtSpot(
				europeanOption.getValue(0.0, oneAssetModel),
				oneAssetModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
				SPOT_1
		);

		assertEquals(vanillaPrice, basketPrice, REDUCTION_TOLERANCE);
	}

	@Test
	public void testReductionToVanillaOnSecondAsset() {
		final double strike = 100.0;

		final BasketOption basketOption = new BasketOption(
				new String[] { "Asset1", "Asset2" },
				MATURITY,
				new double[] { 0.0, 1.0 },
				strike,
				CallOrPut.CALL
		);

		final FDMMultiAssetBlackScholesModel twoAssetModel = createTwoAssetModel(
				SPOT_1,
				SPOT_2,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_1,
				DIVIDEND_YIELD_2,
				VOLATILITY_1,
				VOLATILITY_2,
				CORRELATION
		);

		final double basketPrice = interpolateAtInitialState(
				basketOption.getValue(0.0, twoAssetModel),
				twoAssetModel
		);

		final FDMBlackScholesModel oneAssetModel = createOneAssetModel(
				SPOT_2,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_2,
				VOLATILITY_2
		);

		final EuropeanOption europeanOption = new EuropeanOption(
				"Asset2",
				MATURITY,
				strike,
				CallOrPut.CALL
		);

		final double vanillaPrice = interpolateAtSpot(
				europeanOption.getValue(0.0, oneAssetModel),
				oneAssetModel.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid(),
				SPOT_2
		);

		assertEquals(vanillaPrice, basketPrice, REDUCTION_TOLERANCE);
	}

	@Test
	public void testStrikeZeroBasketCallEqualsDiscountedForwardBasket() {
		final BasketOption basketOption = new BasketOption(
				new String[] { "Asset1", "Asset2" },
				MATURITY,
				new double[] { 0.6, 0.4 },
				0.0,
				CallOrPut.CALL
		);

		final FDMMultiAssetBlackScholesModel model = createTwoAssetModel(
				SPOT_1,
				SPOT_2,
				RISK_FREE_RATE,
				DIVIDEND_YIELD_1,
				DIVIDEND_YIELD_2,
				VOLATILITY_1,
				VOLATILITY_2,
				CORRELATION
		);

		final double basketPrice = interpolateAtInitialState(
				basketOption.getValue(0.0, model),
				model
		);

		final double expectedValue =
				0.6 * SPOT_1 * Math.exp(-DIVIDEND_YIELD_1 * MATURITY)
				+ 0.4 * SPOT_2 * Math.exp(-DIVIDEND_YIELD_2 * MATURITY);

		assertEquals(expectedValue, basketPrice, STRIKE_ZERO_TOLERANCE);
	}

	@Test
	public void testSymmetryUnderAssetSwap() {
		final double[] weights = new double[] { 0.55, 0.45 };
		final double strike = 100.0;

		final BasketOption originalOption = new BasketOption(
				new String[] { "Asset1", "Asset2" },
				MATURITY,
				weights,
				strike,
				CallOrPut.CALL
		);

		final FDMMultiAssetBlackScholesModel originalModel = createTwoAssetModel(
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

		final BasketOption swappedOption = new BasketOption(
				new String[] { "Asset2", "Asset1" },
				MATURITY,
				new double[] { weights[1], weights[0] },
				strike,
				CallOrPut.CALL
		);

		final FDMMultiAssetBlackScholesModel swappedModel = createTwoAssetModel(
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

	private static FDMMultiAssetBlackScholesModel createTwoAssetModel(
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
				NUMBER_OF_TIME_STEPS,
				MATURITY / NUMBER_OF_TIME_STEPS
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

	private static FDMBlackScholesModel createOneAssetModel(
			final double spot,
			final double riskFreeRate,
			final double dividendYield,
			final double volatility) {

		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				NUMBER_OF_TIME_STEPS,
				MATURITY / NUMBER_OF_TIME_STEPS
		);

		final Grid assetGrid = new GridWithMandatoryPoint(160, 0.0, 2.0 * spot, spot);

		final SpaceTimeDiscretization spaceTimeDiscretization = new SpaceTimeDiscretization(
				assetGrid,
				timeDiscretization,
				THETA,
				new double[] { spot }
		);

		return new FDMBlackScholesModel(
				spot,
				riskFreeRate,
				dividendYield,
				volatility,
				spaceTimeDiscretization
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

	private static double interpolateAtSpot(
			final double[] values,
			final double[] grid,
			final double spot) {

		if(isGridNode(grid, spot)) {
			return values[getGridIndex(grid, spot)];
		}

		int upperIndex = 1;
		while(upperIndex < grid.length && grid[upperIndex] < spot) {
			upperIndex++;
		}

		final int lowerIndex = upperIndex - 1;
		final double x0 = grid[lowerIndex];
		final double x1 = grid[upperIndex];
		final double y0 = values[lowerIndex];
		final double y1 = values[upperIndex];

		final double weight = (spot - x0) / (x1 - x0);
		return (1.0 - weight) * y0 + weight * y1;
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
}