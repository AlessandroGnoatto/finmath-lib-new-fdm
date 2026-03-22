package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.AmericanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.BermudanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.interpolation.BiLinearInterpolation;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression tests for early exercise under Heston FD/ADI.
 *
 * These tests verify the basic ordering:
 *
 * European put <= Bermudan put <= American put
 *
 * and in particular:
 *
 * American put >= European put
 */
public class HestonEarlyExerciseOrderingTest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;

	private static final double S0 = 100.0;
	private static final double V0 = 0.04;

	private static final double R = 0.02;
	private static final double Q = 0.00;

	private static final double KAPPA = 1.5;
	private static final double THETA_V = 0.04;
	private static final double XI = 0.30;
	private static final double RHO = -0.70;

	/*
	 * Small numerical slack for grid/interpolation noise.
	 */
	private static final double EPS = 1E-8;

	@Test
	public void testAmericanHestonPutIsAtLeastEuropeanPut() {

		final FDMHestonModel model = createHestonModel();

		final EuropeanOption europeanPut = new EuropeanOption(MATURITY, STRIKE, CallOrPut.PUT);
		final AmericanOption americanPut = new AmericanOption(MATURITY, STRIKE, CallOrPut.PUT);

		final double europeanPrice = extractInterpolatedPrice(europeanPut.getValue(0.0, model), model, S0, V0);
		final double americanPrice = extractInterpolatedPrice(americanPut.getValue(0.0, model), model, S0, V0);

		System.out.println("European Heston put = " + europeanPrice);
		System.out.println("American Heston put = " + americanPrice);

		assertTrue("European put should be positive.", europeanPrice > 0.0);
		assertTrue("American put should be positive.", americanPrice > 0.0);
		assertTrue(
				"American Heston put should be at least as valuable as the European put.",
				americanPrice + EPS >= europeanPrice
		);
	}

	@Test
	public void testBermudanHestonPutLiesBetweenEuropeanAndAmerican() {

		final FDMHestonModel model = createHestonModel();

		final EuropeanOption europeanPut = new EuropeanOption(MATURITY, STRIKE, CallOrPut.PUT);
		final BermudanOption bermudanPut = new BermudanOption(
				new double[] { 0.25, 0.50, 0.75, 1.00 },
				STRIKE,
				CallOrPut.PUT
		);
		final AmericanOption americanPut = new AmericanOption(MATURITY, STRIKE, CallOrPut.PUT);

		final double europeanPrice = extractInterpolatedPrice(europeanPut.getValue(0.0, model), model, S0, V0);
		final double bermudanPrice = extractInterpolatedPrice(bermudanPut.getValue(0.0, model), model, S0, V0);
		final double americanPrice = extractInterpolatedPrice(americanPut.getValue(0.0, model), model, S0, V0);

		System.out.println("European Heston put = " + europeanPrice);
		System.out.println("Bermudan Heston put = " + bermudanPrice);
		System.out.println("American Heston put = " + americanPrice);

		assertTrue("European put should be positive.", europeanPrice > 0.0);
		assertTrue("Bermudan put should be positive.", bermudanPrice > 0.0);
		assertTrue("American put should be positive.", americanPrice > 0.0);

		assertTrue(
				"Bermudan Heston put should be at least as valuable as the European put.",
				bermudanPrice + EPS >= europeanPrice
		);

		assertTrue(
				"Bermudan Heston put should be no more valuable than the American put.",
				americanPrice + EPS >= bermudanPrice
		);
	}

	private static FDMHestonModel createHestonModel() {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final int nTimeSteps = 40;
		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, nTimeSteps, MATURITY / nTimeSteps);

		/*
		 * Moderately wide/practical Heston grid.
		 * This is intentionally a bit stronger than a toy grid, to make the ordering
		 * tests more stable.
		 */
		final int nS = 40;
		final int nV = 40;
		final Grid sGrid = new UniformGrid(nS - 1, 0.0, 4.0 * S0);
		final Grid vGrid = new UniformGrid(nV - 1, 0.0, 0.50);

		final double theta = 0.5;
		final double[] center = new double[] { S0, V0 };

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, vGrid },
				timeDiscretization,
				theta,
				center
		);

		return new FDMHestonModel(
				S0,
				V0,
				riskFreeCurve,
				dividendCurve,
				KAPPA,
				THETA_V,
				XI,
				RHO,
				spaceTime
		);
	}

	private static double extractInterpolatedPrice(
			final double[] flattenedValues,
			final FDMHestonModel model,
			final double spot,
			final double variance) {

		final double[] sNodes = model.getSpaceTimeDiscretization().getSpaceGrid(0).getGrid();
		final double[] vNodes = model.getSpaceTimeDiscretization().getSpaceGrid(1).getGrid();

		final int nS = sNodes.length;
		final int nV = vNodes.length;

		final double[][] valueSurface = new double[nS][nV];
		for(int j = 0; j < nV; j++) {
			for(int i = 0; i < nS; i++) {
				final int k = i + j * nS;
				valueSurface[i][j] = flattenedValues[k];
			}
		}

		final BiLinearInterpolation interpolator = new BiLinearInterpolation(sNodes, vNodes, valueSurface);
		return interpolator.apply(spot, variance);
	}

	private static DiscountCurve createFlatDiscountCurve(final String name, final double rate) {
		final double[] times = new double[] { 0.0, 1.0 };
		final double[] zeroRates = new double[] { rate, rate };
		final InterpolationMethod interpolationMethod = InterpolationMethod.LINEAR;
		final InterpolationEntity interpolationEntity = InterpolationEntity.VALUE;
		final ExtrapolationMethod extrapolationMethod = ExtrapolationMethod.CONSTANT;

		return DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
				name,
				LocalDate.of(2010, 8, 1),
				times,
				zeroRates,
				interpolationMethod,
				extrapolationMethod,
				interpolationEntity
		);
	}
}