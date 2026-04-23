package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMMultiAssetBlackScholesModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.functions2.AnalyticFormulas2;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Tests for {@link ExchangeOption}.
 *
 * <p>
 * The tests validate the two-dimensional finite-difference price of the exchange
 * option against the Margrabe formula under the multi-asset Black-Scholes model.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class ExchangeOptionTest {

	private static final double TOLERANCE_CALL = 7E-2;
	private static final double TOLERANCE_PUT = 7E-2;
	private static final double TOLERANCE_CORRELATION_SWEEP = 9E-2;

	@Test
	public void testExchangeOptionCallAgainstMargrabe() {
		final double maturity = 1.0;

		final double initialValueFirstAsset = 100.0;
		final double initialValueSecondAsset = 95.0;

		final double dividendYieldFirstAsset = 0.01;
		final double dividendYieldSecondAsset = 0.02;

		final double volatilityFirstAsset = 0.20;
		final double volatilitySecondAsset = 0.25;
		final double correlation = 0.30;

		final FDMMultiAssetBlackScholesModel model = createModel(
				initialValueFirstAsset,
				initialValueSecondAsset,
				0.05,
				dividendYieldFirstAsset,
				dividendYieldSecondAsset,
				volatilityFirstAsset,
				volatilitySecondAsset,
				correlation,
				maturity
		);

		final ExchangeOption product = new ExchangeOption(
				"Asset1",
				"Asset2",
				maturity,
				CallOrPut.CALL
		);

		final double pdeValue = getValueAtInitialState(product, model);
		final double analyticValue = AnalyticFormulas2.margrabeExchangeOptionValue(
				initialValueFirstAsset,
				initialValueSecondAsset,
				dividendYieldFirstAsset,
				dividendYieldSecondAsset,
				volatilityFirstAsset,
				volatilitySecondAsset,
				correlation,
				maturity
		);

		assertEquals(analyticValue, pdeValue, TOLERANCE_CALL);
	}

	@Test
	public void testExchangeOptionPutAgainstMargrabeBySymmetry() {
		final double maturity = 1.0;

		final double initialValueFirstAsset = 100.0;
		final double initialValueSecondAsset = 95.0;

		final double dividendYieldFirstAsset = 0.01;
		final double dividendYieldSecondAsset = 0.02;

		final double volatilityFirstAsset = 0.20;
		final double volatilitySecondAsset = 0.25;
		final double correlation = 0.30;

		final FDMMultiAssetBlackScholesModel model = createModel(
				initialValueFirstAsset,
				initialValueSecondAsset,
				0.05,
				dividendYieldFirstAsset,
				dividendYieldSecondAsset,
				volatilityFirstAsset,
				volatilitySecondAsset,
				correlation,
				maturity
		);

		final ExchangeOption product = new ExchangeOption(
				"Asset1",
				"Asset2",
				maturity,
				CallOrPut.PUT
		);

		final double pdeValue = getValueAtInitialState(product, model);

		final double analyticValue = AnalyticFormulas2.margrabeExchangeOptionValue(
				initialValueSecondAsset,
				initialValueFirstAsset,
				dividendYieldSecondAsset,
				dividendYieldFirstAsset,
				volatilitySecondAsset,
				volatilityFirstAsset,
				correlation,
				maturity
		);

		assertEquals(analyticValue, pdeValue, TOLERANCE_PUT);
	}

	@Test
	public void testExchangeOptionCallCorrelationSweepAgainstMargrabe() {
		final double maturity = 1.0;

		final double initialValueFirstAsset = 100.0;
		final double initialValueSecondAsset = 95.0;

		final double dividendYieldFirstAsset = 0.01;
		final double dividendYieldSecondAsset = 0.02;

		final double volatilityFirstAsset = 0.20;
		final double volatilitySecondAsset = 0.25;

		final double[] correlations = new double[] { -0.9, 0.0, 0.9 };

		for(final double correlation : correlations) {
			final FDMMultiAssetBlackScholesModel model = createModel(
					initialValueFirstAsset,
					initialValueSecondAsset,
					0.05,
					dividendYieldFirstAsset,
					dividendYieldSecondAsset,
					volatilityFirstAsset,
					volatilitySecondAsset,
					correlation,
					maturity
			);

			final ExchangeOption product = new ExchangeOption(
					"Asset1",
					"Asset2",
					maturity,
					CallOrPut.CALL
			);

			final double pdeValue = getValueAtInitialState(product, model);
			final double analyticValue = AnalyticFormulas2.margrabeExchangeOptionValue(
					initialValueFirstAsset,
					initialValueSecondAsset,
					dividendYieldFirstAsset,
					dividendYieldSecondAsset,
					volatilityFirstAsset,
					volatilitySecondAsset,
					correlation,
					maturity
			);

			assertEquals(
					"Mismatch for correlation = " + correlation,
					analyticValue,
					pdeValue,
					TOLERANCE_CORRELATION_SWEEP
			);
		}
	}

	private static FDMMultiAssetBlackScholesModel createModel(
			final double initialValueFirstAsset,
			final double initialValueSecondAsset,
			final double riskFreeRate,
			final double dividendYieldFirstAsset,
			final double dividendYieldSecondAsset,
			final double volatilityFirstAsset,
			final double volatilitySecondAsset,
			final double correlation,
			final double maturity) {

		final int numberOfTimeSteps = 100;
		final TimeDiscretization timeDiscretization = new TimeDiscretizationFromArray(
				0.0,
				numberOfTimeSteps,
				maturity / numberOfTimeSteps
		);

		/*
		 * Choose symmetric spot grids and place the initial values exactly on the grid.
		 */
		final Grid firstAssetGrid = new UniformGrid(80, 0.0, 200.0);
		final Grid secondAssetGrid = new UniformGrid(76, 0.0, 190.0);

		final SpaceTimeDiscretization spaceTimeDiscretization = new SpaceTimeDiscretization(
				new Grid[] { firstAssetGrid, secondAssetGrid },
				timeDiscretization,
				0.5,
				new double[] { initialValueFirstAsset, initialValueSecondAsset }
		);

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("riskFreeCurve", riskFreeRate);
		final DiscountCurve dividendCurveFirstAsset = createFlatDiscountCurve("dividendCurve1", dividendYieldFirstAsset);
		final DiscountCurve dividendCurveSecondAsset = createFlatDiscountCurve("dividendCurve2", dividendYieldSecondAsset);

		return new FDMMultiAssetBlackScholesModel(
				new double[] { initialValueFirstAsset, initialValueSecondAsset },
				riskFreeCurve,
				new DiscountCurve[] { dividendCurveFirstAsset, dividendCurveSecondAsset },
				new double[] { volatilityFirstAsset, volatilitySecondAsset },
				new double[][] {
					{ 1.0, correlation },
					{ correlation, 1.0 }
				},
				spaceTimeDiscretization
		);
	}

	private static double getValueAtInitialState(
			final ExchangeOption product,
			final FDMMultiAssetBlackScholesModel model) {

		final double[] values = product.getValue(0.0, model);

		final SpaceTimeDiscretization discretization = model.getSpaceTimeDiscretization();
		final double[] firstGrid = discretization.getSpaceGrid(0).getGrid();
		final double[] secondGrid = discretization.getSpaceGrid(1).getGrid();

		final double initialValueFirstAsset = model.getInitialValue()[0];
		final double initialValueSecondAsset = model.getInitialValue()[1];

		final int firstIndex = getExactGridIndex(firstGrid, initialValueFirstAsset);
		final int secondIndex = getExactGridIndex(secondGrid, initialValueSecondAsset);

		final int flattenedIndex = firstIndex + secondIndex * firstGrid.length;
		return values[flattenedIndex];
	}

	private static int getExactGridIndex(final double[] grid, final double value) {
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - value) < 1E-12) {
				return i;
			}
		}

		throw new IllegalArgumentException("Value " + value + " is not a grid point.");
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