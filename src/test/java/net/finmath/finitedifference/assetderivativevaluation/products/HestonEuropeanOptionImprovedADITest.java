package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.finitedifference.solvers.adi.FDMHestonADI2DImproved;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Regression test for the improved Heston ADI solver based on the new
 * semidiscrete split hierarchy.
 *
 * <p>
 * This test compares vanilla European option prices from
 * {@link FDMHestonADI2DImproved} against finmath's Fourier Heston pricer.
 * The purpose is to validate the new operator split on the simplest Heston
 * products before using it for more complicated payoffs.
 * </p>
 */
public class HestonEuropeanOptionImprovedADITest {

	private static final double MATURITY = 1.0;
	private static final double STRIKE = 100.0;

	private static final double S0 = 100.0;
	private static final double R = 0.05;
	private static final double Q = 0.00;

	private static final double VOLATILITY = 0.25;
	private static final double VOLATILITY_SQUARED = VOLATILITY * VOLATILITY;
	private static final double KAPPA = 1.5;
	private static final double THETA_H = VOLATILITY_SQUARED;
	private static final double XI = 0.30;
	private static final double RHO = -0.70;

	/*
	 * Douglas control setting for the first validation pass.
	 */
	private static final double THETA = 0.5;

	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS_S = 200;
	private static final int NUMBER_OF_SPACE_STEPS_V = 100;

	private static final double SPOT_MIN = 0.0;
	private static final double SPOT_MAX = 250.0;

	/*
	 * Start with a moderate tolerance. Tightening can come later once
	 * the improved hierarchy is stable and calibrated.
	 */
	private static final double RELATIVE_TOLERANCE = 0.03;

	@Test
	public void testEuropeanCallAgainstFourier() throws Exception {
		runEuropeanTest(CallOrPut.CALL);
	}

	@Test
	public void testEuropeanPutAgainstFourier() throws Exception {
		runEuropeanTest(CallOrPut.PUT);
	}

	private void runEuropeanTest(final CallOrPut callOrPut) throws Exception {

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", R);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", Q);

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(
						0.0,
						NUMBER_OF_TIME_STEPS,
						MATURITY / NUMBER_OF_TIME_STEPS
				);

		final Grid sGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_S, SPOT_MIN, SPOT_MAX);

		final double vMin = 0.0;
		final double vMax =
				Math.max(4.0 * THETA_H, VOLATILITY_SQUARED + 4.0 * XI * Math.sqrt(MATURITY));
		final Grid vGrid = new UniformGrid(NUMBER_OF_SPACE_STEPS_V, vMin, vMax);

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, vGrid },
				timeDiscretization,
				THETA,
				new double[] { S0, VOLATILITY_SQUARED }
		);

		final FDMHestonModel fdmModel = new FDMHestonModel(
				S0,
				VOLATILITY_SQUARED,
				riskFreeCurve,
				dividendCurve,
				KAPPA,
				THETA_H,
				XI,
				RHO,
				spaceTime
		);

		final EuropeanOption product = new EuropeanOption(
				null,
				MATURITY,
				STRIKE,
				callOrPut
		);

		final FDMHestonADI2DImproved solver = new FDMHestonADI2DImproved(
				fdmModel,
				product,
				spaceTime,
				new EuropeanExercise(MATURITY)
		);

		final double[] valuesOnGrid = solver.getValue(
				0.0,
				MATURITY,
				(s, v) -> callOrPut == CallOrPut.CALL
						? Math.max(s - STRIKE, 0.0)
						: Math.max(STRIKE - s, 0.0)
		);

		final double fdPrice = interpolatePriceAtInitialState(valuesOnGrid, fdmModel);
		final double fourierPrice = getFourierVanillaPrice(callOrPut);

		final double relativeError =
				Math.abs(fdPrice - fourierPrice) / Math.max(Math.abs(fourierPrice), 1E-8);

		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();
		final double[] vNodes = spaceTime.getSpaceGrid(1).getGrid();

		System.out.println("HESTON EUROPEAN IMPROVED ADI TEST");
		System.out.println("Call/Put        = " + callOrPut);
		System.out.println("Theta           = " + THETA);
		System.out.println("Time steps      = " + NUMBER_OF_TIME_STEPS);
		System.out.println("Space steps S   = " + NUMBER_OF_SPACE_STEPS_S);
		System.out.println("Space steps V   = " + NUMBER_OF_SPACE_STEPS_V);
		System.out.println("Spot grid min   = " + sNodes[0]);
		System.out.println("Spot grid max   = " + sNodes[sNodes.length - 1]);
		System.out.println("Variance grid min = " + vNodes[0]);
		System.out.println("Variance grid max = " + vNodes[vNodes.length - 1]);
		System.out.println("S0 on grid      = " + (getGridIndex(sNodes, S0) >= 0));
		System.out.println("v0 on grid      = " + (getGridIndex(vNodes, VOLATILITY_SQUARED) >= 0));
		System.out.println("FD price        = " + fdPrice);
		System.out.println("Fourier price   = " + fourierPrice);
		System.out.println("Relative error  = " + relativeError);

		assertTrue(Double.isFinite(fdPrice));
		assertTrue(Double.isFinite(fourierPrice));
		assertTrue(fdPrice >= -1E-10);
		assertTrue(fourierPrice >= -1E-10);

		assertEquals(
				"Improved Heston ADI price differs too much from Fourier reference for " + callOrPut,
				0.0,
				relativeError,
				RELATIVE_TOLERANCE
		);
	}

	private double getFourierVanillaPrice(final CallOrPut callOrPut) throws Exception {

		final net.finmath.fouriermethod.models.HestonModel fourierModel =
				new net.finmath.fouriermethod.models.HestonModel(
						S0,
						R - Q,
						Math.sqrt(VOLATILITY_SQUARED),
						R,
						THETA_H,
						KAPPA,
						XI,
						RHO
				);

		final net.finmath.fouriermethod.products.EuropeanOption fourierCall =
				new net.finmath.fouriermethod.products.EuropeanOption(MATURITY, STRIKE);

		final double callPrice = fourierCall.getValue(fourierModel);

		if(callOrPut == CallOrPut.CALL) {
			return callPrice;
		}

		final double discountFactor = Math.exp(-R * MATURITY);
		final double dividendDiscountFactor = Math.exp(-Q * MATURITY);

		return callPrice - S0 * dividendDiscountFactor + STRIKE * discountFactor;
	}

	private double interpolatePriceAtInitialState(
			final double[] valuesOnGrid,
			final FDMHestonModel model) {

		final SpaceTimeDiscretization spaceTime = model.getSpaceTimeDiscretization();
		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();
		final double[] vNodes = spaceTime.getSpaceGrid(1).getGrid();

		final int nS = sNodes.length;
		final int v0Index = getNearestGridIndex(vNodes, VOLATILITY_SQUARED);

		final double[] sliceAtInitialVariance = new double[nS];
		for(int i = 0; i < nS; i++) {
			final int flatIndex = i + v0Index * nS;
			sliceAtInitialVariance[i] = valuesOnGrid[flatIndex];
		}

		final int s0Index = getGridIndex(sNodes, S0);
		if(s0Index >= 0) {
			return sliceAtInitialVariance[s0Index];
		}

		final PolynomialSplineFunction interpolation =
				new LinearInterpolator().interpolate(sNodes, sliceAtInitialVariance);
		return interpolation.value(S0);
	}

	private static int getGridIndex(final double[] grid, final double value) {
		final double tolerance = 1E-12;
		for(int i = 0; i < grid.length; i++) {
			if(Math.abs(grid[i] - value) < tolerance) {
				return i;
			}
		}
		return -1;
	}

	private static int getNearestGridIndex(final double[] grid, final double value) {
		int bestIndex = 0;
		double bestDistance = Math.abs(grid[0] - value);

		for(int i = 1; i < grid.length; i++) {
			final double distance = Math.abs(grid[i] - value);
			if(distance < bestDistance) {
				bestDistance = distance;
				bestIndex = i;
			}
		}

		return bestIndex;
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