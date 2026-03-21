package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.finitedifference.solvers.adi.FDMHestonADI2D;
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
 * Vanilla Heston regression test for the ADI solver.
 *
 * <p>
 * Compares:
 * </p>
 * <ul>
 *   <li>ADI finite differences,</li>
 *   <li>finmath Monte Carlo,</li>
 *   <li>finmath Fourier pricer.</li>
 * </ul>
 *
 * <p>
 * This is intended as the base stability test before enabling barrier / two-state Heston ADI pricing.
 * </p>
 */
public class HestonEuropeanOptionAdiComparisonTest {

	@Test
	public void testEuropeanCallHestonAdiVsMonteCarloVsFourier() throws Exception {

		final double maturity = 1.0;
		final double strike = 100.0;
		final CallOrPut callOrPut = CallOrPut.CALL;

		final double s0 = 100.0;
		final double v0 = 0.04;
		final double r = 0.02;
		final double q = 0.00;

		final double kappa = 1.5;
		final double thetaV = 0.04;
		final double xi = 0.30;
		final double rho = -0.70;

		final DiscountCurve riskFreeCurve = createFlatDiscountCurve("r", r);
		final DiscountCurve dividendCurve = createFlatDiscountCurve("q", q);
		
		final int nTimeSteps = 100;
		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, nTimeSteps, maturity / nTimeSteps);

		final int nS = 100;
		final int nV = 100;
		final Grid sGrid = new UniformGrid(nS - 1, 0.0, 4.0 * s0);
		final Grid vGrid = new UniformGrid(nV - 1, 0.0, 0.50);

		final double theta = 0.5;
		final double[] center = new double[] { s0, v0 };

		final SpaceTimeDiscretization spaceTime = new SpaceTimeDiscretization(
				new Grid[] { sGrid, vGrid },
				timeDiscretization,
				theta,
				center
		);

		final FDMHestonModel fdmModel = new FDMHestonModel(
				s0,
				v0,
				riskFreeCurve,
				dividendCurve,
				kappa,
				thetaV,
				xi,
				rho,
				spaceTime
		);

		final EuropeanOption fdProduct = new EuropeanOption(maturity, strike, callOrPut);

		final FDMHestonADI2D adiSolver = new FDMHestonADI2D(fdmModel,fdProduct,spaceTime,fdProduct.getExercise());

		final double[] fdColumn = adiSolver.getValue(
				0.0,
				maturity,
				(stock, variance) -> Math.max(stock - strike, 0.0)
		);

		final double[] sNodes = spaceTime.getSpaceGrid(0).getGrid();
		final double[] vNodes = spaceTime.getSpaceGrid(1).getGrid();
		final int ns = sNodes.length;
		final int nv = vNodes.length;

		final double[][] valueSurface = new double[ns][nv];
		for(int j = 0; j < nv; j++) {
			for(int i = 0; i < ns; i++) {
				final int k = i + j * ns;
				valueSurface[i][j] = fdColumn[k];
			}
		}

		final BiLinearInterpolation interpolator = new BiLinearInterpolation(sNodes, vNodes, valueSurface);
		final double fdPrice = interpolator.apply(s0, v0);

		final net.finmath.fouriermethod.models.HestonModel fourierModel =
				new net.finmath.fouriermethod.models.HestonModel(
						s0,
						r - q,
						Math.sqrt(v0),
						r,
						thetaV,
						kappa,
						xi,
						rho
				);

		final net.finmath.fouriermethod.products.EuropeanOption fourierProduct =
				new net.finmath.fouriermethod.products.EuropeanOption(maturity, strike);

		final double fourierPrice = fourierProduct.getValue(fourierModel);

		final int numberOfPaths = 20000;
		final int seed = 31415;

		final TimeDiscretization mcTimes =
				new TimeDiscretizationFromArray(0.0, 100, maturity / 100.0);

		final net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers brownianMotion =
				new net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers(mcTimes, 2, numberOfPaths, seed);

		final net.finmath.montecarlo.assetderivativevaluation.models.HestonModel mcHestonModel =
				new net.finmath.montecarlo.assetderivativevaluation.models.HestonModel(
						s0,
						r - q,
						Math.sqrt(v0),
						r,
						thetaV,
						kappa,
						xi,
						rho,
						net.finmath.montecarlo.assetderivativevaluation.models.HestonModel.Scheme.FULL_TRUNCATION
				);

		final net.finmath.montecarlo.process.EulerSchemeFromProcessModel process =
				new net.finmath.montecarlo.process.EulerSchemeFromProcessModel(mcHestonModel, brownianMotion);

		final net.finmath.montecarlo.assetderivativevaluation.MonteCarloAssetModel mcModel =
				new net.finmath.montecarlo.assetderivativevaluation.MonteCarloAssetModel(process);

		final net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption mcProduct =
				new net.finmath.montecarlo.assetderivativevaluation.products.EuropeanOption(maturity, strike);

		final double mcPrice = mcProduct.getValue(mcModel);

		System.out.println("ADI price      = " + fdPrice);
		System.out.println("Fourier price  = " + fourierPrice);
		System.out.println("MC price       = " + mcPrice);

		assertTrue(Double.isFinite(fdPrice));
		assertTrue(Double.isFinite(fourierPrice));
		assertTrue(Double.isFinite(mcPrice));

		assertTrue(fdPrice > 0.0);
		assertTrue(fourierPrice > 0.0);
		assertTrue(mcPrice > 0.0);

		final double tolADIvsFourier = 0.75;
		final double tolMCvsFourier = 0.70;

		assertEquals("ADI vs Fourier", fourierPrice, fdPrice, tolADIvsFourier);
		assertEquals("MC vs Fourier", fourierPrice, mcPrice, tolMCvsFourier);
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
