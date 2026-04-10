package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.apache.commons.math3.analysis.interpolation.LinearInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.finitedifference.solvers.adi.AbstractSplitADI2D;
import net.finmath.finitedifference.solvers.adi.FDMHestonADI2D;
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
 * Compares legacy and improved Heston ADI solvers against the Fourier Heston reference
 * on vanilla European options.
 *
 * <p>
 * This test runs:
 * </p>
 * <ul>
 *   <li>legacy Heston ADI solver,</li>
 *   <li>improved Heston ADI solver with Douglas,</li>
 *   <li>improved Heston ADI solver with MCS.</li>
 * </ul>
 */
public class HestonEuropeanOptionImprovedADIComparisonTest {

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

	private static final int NUMBER_OF_TIME_STEPS = 100;
	private static final int NUMBER_OF_SPACE_STEPS_S = 200;
	private static final int NUMBER_OF_SPACE_STEPS_V = 100;

	private static final double SPOT_MIN = 0.0;
	private static final double SPOT_MAX = 250.0;

	private static final double THETA_DOUGLAS = 0.5;
	private static final double THETA_MCS = 1.0 / 3.0;

	@Test
	public void testEuropeanCallComparison() throws Exception {
		runComparisonTest(CallOrPut.CALL);
	}

	@Test
	public void testEuropeanPutComparison() throws Exception {
		runComparisonTest(CallOrPut.PUT);
	}

	private void runComparisonTest(final CallOrPut callOrPut) throws Exception {

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

		final SpaceTimeDiscretization legacySpaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, vGrid },
				timeDiscretization,
				THETA_DOUGLAS,
				new double[] { S0, VOLATILITY_SQUARED }
		);

		final SpaceTimeDiscretization improvedDouglasSpaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, vGrid },
				timeDiscretization,
				THETA_DOUGLAS,
				new double[] { S0, VOLATILITY_SQUARED }
		);

		final SpaceTimeDiscretization improvedMcsSpaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, vGrid },
				timeDiscretization,
				THETA_MCS,
				new double[] { S0, VOLATILITY_SQUARED }
		);

		final FDMHestonModel legacyModel = new FDMHestonModel(
				S0,
				VOLATILITY_SQUARED,
				riskFreeCurve,
				dividendCurve,
				KAPPA,
				THETA_H,
				XI,
				RHO,
				legacySpaceTime
		);

		final FDMHestonModel improvedDouglasModel = new FDMHestonModel(
				S0,
				VOLATILITY_SQUARED,
				riskFreeCurve,
				dividendCurve,
				KAPPA,
				THETA_H,
				XI,
				RHO,
				improvedDouglasSpaceTime
		);

		final FDMHestonModel improvedMcsModel = new FDMHestonModel(
				S0,
				VOLATILITY_SQUARED,
				riskFreeCurve,
				dividendCurve,
				KAPPA,
				THETA_H,
				XI,
				RHO,
				improvedMcsSpaceTime
		);

		final EuropeanOption product = new EuropeanOption(
				null,
				MATURITY,
				STRIKE,
				callOrPut
		);

		final FDMHestonADI2D legacySolver = new FDMHestonADI2D(
				legacyModel,
				product,
				legacySpaceTime,
				new EuropeanExercise(MATURITY)
		);

		final FDMHestonADI2DImproved improvedDouglasSolver = new FDMHestonADI2DImproved(
				improvedDouglasModel,
				product,
				improvedDouglasSpaceTime,
				new EuropeanExercise(MATURITY),
				AbstractSplitADI2D.ADIScheme.DOUGLAS
		);

		final FDMHestonADI2DImproved improvedMcsSolver = new FDMHestonADI2DImproved(
				improvedMcsModel,
				product,
				improvedMcsSpaceTime,
				new EuropeanExercise(MATURITY),
				AbstractSplitADI2D.ADIScheme.MCS
		);

		final double[] legacyValuesOnGrid = legacySolver.getValue(
				0.0,
				MATURITY,
				(s, v) -> callOrPut == CallOrPut.CALL
						? Math.max(s - STRIKE, 0.0)
						: Math.max(STRIKE - s, 0.0)
		);

		final double[] improvedDouglasValuesOnGrid = improvedDouglasSolver.getValue(
				0.0,
				MATURITY,
				(s, v) -> callOrPut == CallOrPut.CALL
						? Math.max(s - STRIKE, 0.0)
						: Math.max(STRIKE - s, 0.0)
		);

		final double[] improvedMcsValuesOnGrid = improvedMcsSolver.getValue(
				0.0,
				MATURITY,
				(s, v) -> callOrPut == CallOrPut.CALL
						? Math.max(s - STRIKE, 0.0)
						: Math.max(STRIKE - s, 0.0)
		);

		final double legacyPrice = interpolatePriceAtInitialState(legacyValuesOnGrid, legacyModel);
		final double improvedDouglasPrice = interpolatePriceAtInitialState(improvedDouglasValuesOnGrid, improvedDouglasModel);
		final double improvedMcsPrice = interpolatePriceAtInitialState(improvedMcsValuesOnGrid, improvedMcsModel);
		final double fourierPrice = getFourierVanillaPrice(callOrPut);

		final double legacyRelativeError =
				Math.abs(legacyPrice - fourierPrice) / Math.max(Math.abs(fourierPrice), 1E-8);

		final double improvedDouglasRelativeError =
				Math.abs(improvedDouglasPrice - fourierPrice) / Math.max(Math.abs(fourierPrice), 1E-8);

		final double improvedMcsRelativeError =
				Math.abs(improvedMcsPrice - fourierPrice) / Math.max(Math.abs(fourierPrice), 1E-8);

		final double relativeDifferenceDouglasVsLegacy =
				Math.abs(improvedDouglasPrice - legacyPrice) / Math.max(Math.abs(fourierPrice), 1E-8);

		final double relativeDifferenceMcsVsLegacy =
				Math.abs(improvedMcsPrice - legacyPrice) / Math.max(Math.abs(fourierPrice), 1E-8);

		System.out.println("HESTON EUROPEAN ADI COMPARISON TEST");
		System.out.println("Call/Put                        = " + callOrPut);
		System.out.println("Legacy theta                    = " + THETA_DOUGLAS);
		System.out.println("Improved Douglas theta          = " + THETA_DOUGLAS);
		System.out.println("Improved MCS theta              = " + THETA_MCS);
		System.out.println("Legacy FD price                 = " + legacyPrice);
		System.out.println("Improved Douglas FD price       = " + improvedDouglasPrice);
		System.out.println("Improved MCS FD price           = " + improvedMcsPrice);
		System.out.println("Fourier price                   = " + fourierPrice);
		System.out.println("Legacy relative error           = " + legacyRelativeError);
		System.out.println("Improved Douglas relative error = " + improvedDouglasRelativeError);
		System.out.println("Improved MCS relative error     = " + improvedMcsRelativeError);
		System.out.println("Rel diff Douglas vs legacy      = " + relativeDifferenceDouglasVsLegacy);
		System.out.println("Rel diff MCS vs legacy          = " + relativeDifferenceMcsVsLegacy);

		assertTrue(Double.isFinite(legacyPrice));
		assertTrue(Double.isFinite(improvedDouglasPrice));
		assertTrue(Double.isFinite(improvedMcsPrice));
		assertTrue(Double.isFinite(fourierPrice));

		assertTrue(legacyPrice >= -1E-10);
		assertTrue(improvedDouglasPrice >= -1E-10);
		assertTrue(improvedMcsPrice >= -1E-10);
		assertTrue(fourierPrice >= -1E-10);
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