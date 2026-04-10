package net.finmath.finitedifference.assetderivativevaluation.models;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.finitedifference.solvers.adi.FDMSabrADI2D;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.modelling.EuropeanExercise;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Test of European option pricing under SABR via ADI finite differences.
 *
 * <p>
 * The PDE price is compared against a SABR-Hagan implied Black volatility
 * approximation combined with a Black pricing formula from finmath for a range
 * of strikes.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class EuropeanCallAgainstHaganBlackApproximationTest {

	@Test
	public void testEuropeanCallsAgainstHaganBlackApproximationForMultipleStrikes() {

		/*
		 * SABR parameters
		 */
		final double initialSpot = 100.0;
		final double initialAlpha = 0.20;
		final double beta = 0.50;
		final double rho = -0.30;
		final double nu = 0.40;

		/*
		 * Market data
		 */
		final double riskFreeRate = 0.02;
		final double dividendYieldRate = 0.01;

		/*
		 * Option data
		 */
		final double maturity = 1.0;
		final double[] strikes = new double[] {60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0};

		/*
		 * PDE discretization
		 */
		final int numberOfTimeSteps = 100;

		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, numberOfTimeSteps, maturity / numberOfTimeSteps);

		final int nS = 200;
		final int nV = 100;
		final Grid sGrid = new UniformGrid(nS - 1, 0.0, 2.0 * initialSpot);
		final Grid vGrid = new UniformGrid(nV - 1, 0.0, 1.00);

		final double theta = 0.5;
		final double[] center = new double[] {initialSpot, initialAlpha};

		final SpaceTimeDiscretization discretization = new SpaceTimeDiscretization(
				new Grid[] {sGrid, vGrid},
				timeDiscretization,
				theta,
				center
		);

		final FDMSabrModel model =
				new FDMSabrModel(
						initialSpot,
						initialAlpha,
						riskFreeRate,
						dividendYieldRate,
						beta,
						nu,
						rho,
						discretization
				);

		final int spotIndex = getNearestIndex(discretization.getSpaceGrid(0).getGrid(), initialSpot);
		final int alphaIndex = getNearestIndex(discretization.getSpaceGrid(1).getGrid(), initialAlpha);
		final int nSpot = discretization.getSpaceGrid(0).getGrid().length;

		final DiscountCurve riskFreeCurve = model.getRiskFreeCurve();
		final DiscountCurve dividendCurve = model.getDividendYieldCurve();

		final double discountFactorRiskFree = riskFreeCurve.getDiscountFactor(maturity);
		final double discountFactorDividend = dividendCurve.getDiscountFactor(maturity);
		final double forward = initialSpot * discountFactorDividend / discountFactorRiskFree;

		final double tolerance = 1E1;
		System.out.println("Strike " + "\t" + "FDM price " + "\t" + "Hagan + BS price ");
		for(final double strike : strikes) {

			final EuropeanOption product = new EuropeanOption(maturity, strike, net.finmath.modelling.products.CallOrPut.CALL);

			final FDMSabrADI2D solver =
					new FDMSabrADI2D(
							model,
							product,
							discretization,
							new EuropeanExercise(maturity)
					);

			final double[] values =
					solver.getValue(
							0.0,
							maturity,
							s -> Math.max(s - strike, 0.0)
					);

			final double pdePrice = values[spotIndex + alphaIndex * nSpot];

			final double haganBlackVol =
					AnalyticFormulas.sabrHaganLognormalBlackVolatilityApproximation(
							initialAlpha,
							beta,
							rho,
							nu,
							forward,
							strike,
							maturity
					);

			final double blackPrice =
					AnalyticFormulas.blackScholesGeneralizedOptionValue(
							forward,
							haganBlackVol,
							maturity,
							strike,
							discountFactorRiskFree
					);
			System.out.println(strike + "\t" + pdePrice + "\t" + blackPrice);

			assertEquals("Mismatch at strike " + strike, blackPrice, pdePrice, tolerance);
		}
	}

	private static int getNearestIndex(final double[] grid, final double value) {
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
}