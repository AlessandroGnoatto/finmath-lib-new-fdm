package net.finmath.finitedifference.assetderivativevaluation.models;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloAssetModel;
import net.finmath.montecarlo.process.EulerSchemeFromProcessModel;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Compares European SABR call prices across:
 * - Hagan implied Black volatility approximation,
 * - finite differences under {@link FDMSabrModel},
 * - Monte Carlo under {@link net.finmath.montecarlo.assetderivativevaluation.mymodels.SabrModel}.
 *
 * <p>
 * The finite-difference solver returns prices on the full (S, alpha) grid and
 * the test interpolates the value at (S0, alpha0).
 * </p>
 */
public class EuropeanCallSabrFdmVsMonteCarloVsHaganTest {

	private static final double MATURITY = 1.0;
	private static final double[] STRIKES = new double[] {
			60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0
	};

	private static final double S0 = 100.0;
	private static final double ALPHA0 = 0.20;
	private static final double BETA = 0.50;
	private static final double NU = 0.40;
	private static final double RHO = -0.30;

	private static final double R = 0.02;
	private static final double Q = 0.01;

	private static final int FD_NUMBER_OF_TIME_STEPS = 100;
	private static final int FD_NUMBER_OF_SPACE_STEPS_S = 200;
	private static final int FD_NUMBER_OF_SPACE_STEPS_ALPHA = 100;
	private static final double FD_THETA = 0.5;

	private static final int MC_NUMBER_OF_TIME_STEPS = 250;
	private static final int MC_NUMBER_OF_PATHS = 50_000;
	private static final int MC_SEED = 31415;

	/*
	 * These tolerances are intentionally looser than a pure analytic-vs-analytic test:
	 * - Hagan is an approximation
	 * - FDM has grid error
	 * - MC has sampling error
	 */
	private static final double TOLERANCE_FDM_VS_HAGAN = 1.00;
	private static final double TOLERANCE_MC_VS_HAGAN = 1.00;
	private static final double TOLERANCE_FDM_VS_MC = 1.00;

	@Test
	public void testEuropeanCallsSabrFdmVsMonteCarloVsHaganForMultipleStrikes() throws Exception {

		final TestSetup setup = createSetup();

		final double discountFactorRiskFree = setup.riskFreeCurve.getDiscountFactor(MATURITY);
		final double discountFactorDividend = setup.dividendCurve.getDiscountFactor(MATURITY);
		final double forward = S0 * discountFactorDividend / discountFactorRiskFree;

		System.out.println("Strike\tHagan+Black\tFDM\tMC\t|FDM-Hagan|\t|MC-Hagan|\t|FDM-MC|");

		for(final double strike : STRIKES) {

			final EuropeanOption fdmProduct = new EuropeanOption(MATURITY, strike, CallOrPut.CALL);

			final double[] fdColumn = fdmProduct.getValue(0.0, setup.fdmModel);
			final double fdmPrice = interpolateFdmValue(
					fdColumn,
					setup.sNodes,
					setup.alphaNodes,
					S0,
					ALPHA0
			);

			final net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption mcProduct =
					new net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption(MATURITY, strike);

			final double mcPrice = mcProduct.getValue(setup.mcModel);

			final double haganBlackVol =
					AnalyticFormulas.sabrHaganLognormalBlackVolatilityApproximation(
							ALPHA0,
							BETA,
							RHO,
							NU,
							forward,
							strike,
							MATURITY
					);

			final double haganBlackPrice =
					AnalyticFormulas.blackScholesGeneralizedOptionValue(
							forward,
							haganBlackVol,
							MATURITY,
							strike,
							discountFactorRiskFree
					);

			final double diffFdmVsHagan = Math.abs(fdmPrice - haganBlackPrice);
			final double diffMcVsHagan = Math.abs(mcPrice - haganBlackPrice);
			final double diffFdmVsMc = Math.abs(fdmPrice - mcPrice);

			System.out.println(
					String.format(
							"%.2f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f",
							strike,
							haganBlackPrice,
							fdmPrice,
							mcPrice,
							diffFdmVsHagan,
							diffMcVsHagan,
							diffFdmVsMc
					)
			);

			assertTrue("Hagan price must be non-negative at strike " + strike, haganBlackPrice >= 0.0);
			assertTrue("FDM price must be non-negative at strike " + strike, fdmPrice >= 0.0);
			assertTrue("MC price must be non-negative at strike " + strike, mcPrice >= 0.0);

			assertEquals("FDM vs Hagan mismatch at strike " + strike, haganBlackPrice, fdmPrice, TOLERANCE_FDM_VS_HAGAN);
			assertEquals("MC vs Hagan mismatch at strike " + strike, haganBlackPrice, mcPrice, TOLERANCE_MC_VS_HAGAN);
			assertEquals("FDM vs MC mismatch at strike " + strike, mcPrice, fdmPrice, TOLERANCE_FDM_VS_MC);
		}
	}

	private TestSetup createSetup() {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization fdTimes =
				new TimeDiscretizationFromArray(
						0.0,
						FD_NUMBER_OF_TIME_STEPS,
						MATURITY / FD_NUMBER_OF_TIME_STEPS
				);

		final Grid sGrid = new UniformGrid(FD_NUMBER_OF_SPACE_STEPS_S - 1, 0.0, 2.0 * S0);
		final Grid alphaGrid = new UniformGrid(FD_NUMBER_OF_SPACE_STEPS_ALPHA - 1, 0.0, 1.00);

		final SpaceTimeDiscretization fdSpaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, alphaGrid },
				fdTimes,
				FD_THETA,
				new double[] { S0, ALPHA0 }
		);

		final FDMSabrModel fdmModel = new FDMSabrModel(
				S0,
				ALPHA0,
				riskFreeCurve,
				dividendCurve,
				BETA,
				NU,
				RHO,
				fdSpaceTime
		);

		final TimeDiscretization mcTimes =
				new TimeDiscretizationFromArray(
						0.0,
						MC_NUMBER_OF_TIME_STEPS,
						MATURITY / MC_NUMBER_OF_TIME_STEPS
				);

		final BrownianMotionFromMersenneRandomNumbers brownianMotion =
				new BrownianMotionFromMersenneRandomNumbers(
						mcTimes,
						2,
						MC_NUMBER_OF_PATHS,
						MC_SEED
				);

		final net.finmath.montecarlo.assetderivativevaluation.mymodels.SabrModel mcSabrModel =
				new net.finmath.montecarlo.assetderivativevaluation.mymodels.SabrModel(
						S0,
						R,
						Q,
						ALPHA0,
						BETA,
						NU,
						RHO
				);

		
		final EulerSchemeFromProcessModel mcProcess =
				new EulerSchemeFromProcessModel(mcSabrModel, brownianMotion);

		final MonteCarloAssetModel mcModel = new MonteCarloAssetModel(mcProcess);

		return new TestSetup(
				fdmModel,
				mcModel,
				riskFreeCurve,
				dividendCurve,
				sGrid.getGrid(),
				alphaGrid.getGrid()
		);
	}

	private double interpolateFdmValue(
			final double[] fdmValueVector,
			final double[] sNodes,
			final double[] alphaNodes,
			final double spot,
			final double alpha) {

		final int nS = sNodes.length;
		final int nAlpha = alphaNodes.length;

		final double[][] valueSurface = new double[nS][nAlpha];
		for(int j = 0; j < nAlpha; j++) {
			for(int i = 0; i < nS; i++) {
				valueSurface[i][j] = fdmValueVector[i + j * nS];
			}
		}

		final BiLinearInterpolation interpolation =
				new BiLinearInterpolation(sNodes, alphaNodes, valueSurface);

		return interpolation.apply(spot, alpha);
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

	private static class TestSetup {

		private final FDMSabrModel fdmModel;
		private final MonteCarloAssetModel mcModel;
		private final DiscountCurve riskFreeCurve;
		private final DiscountCurve dividendCurve;
		private final double[] sNodes;
		private final double[] alphaNodes;

		private TestSetup(
				final FDMSabrModel fdmModel,
				final MonteCarloAssetModel mcModel,
				final DiscountCurve riskFreeCurve,
				final DiscountCurve dividendCurve,
				final double[] sNodes,
				final double[] alphaNodes) {
			this.fdmModel = fdmModel;
			this.mcModel = mcModel;
			this.riskFreeCurve = riskFreeCurve;
			this.dividendCurve = dividendCurve;
			this.sNodes = sNodes;
			this.alphaNodes = alphaNodes;
		}
	}
}