package net.finmath.finitedifference.assetderivativevaluation.products;

import static org.junit.Assert.assertTrue;

import java.util.Arrays;

import org.junit.Test;

import net.finmath.exception.CalculationException;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

public class BermudanConvergenceTest {

	@Test
	public void testPutBermudanConvergesToAmericanFDPrice() throws CalculationException {

		final double initialValue = 50.0;
		final double riskFreeRate = 0.06;
		final double dividendYield = 0.20;
		final double volatility = 0.40;
		final double maturity = 1.0;
		final double strike = 50.0;

		final int numTimesteps = 400;
		final int numSpacesteps = 160;
		final int numStandardDeviations = 5;
		final double theta = 0.5;

		final double forwardValue =
				initialValue * Math.exp((riskFreeRate - dividendYield) * maturity);
		final double variance =
				initialValue * initialValue
				* Math.exp(2.0 * (riskFreeRate - dividendYield) * maturity)
				* (Math.exp(volatility * volatility * maturity) - 1.0);

		final double maximumStockPrice =
				forwardValue + numStandardDeviations * Math.sqrt(variance);
		final double minimumStockPrice =
				Math.max(forwardValue - numStandardDeviations * Math.sqrt(variance), 0.0);

		final Grid spaceGrid = new UniformGrid(numSpacesteps, minimumStockPrice, maximumStockPrice);
		final TimeDiscretization timeDiscretization =
				new TimeDiscretizationFromArray(0.0, numTimesteps, maturity / numTimesteps);

		final SpaceTimeDiscretization spaceTimeDiscretization =
				new SpaceTimeDiscretization(
						spaceGrid,
						timeDiscretization,
						theta,
						new double[] { initialValue }
				);

		final FiniteDifferenceEquityModel model =
				new FDMBlackScholesModel(
						initialValue,
						riskFreeRate,
						dividendYield,
						volatility,
						spaceTimeDiscretization
				);

		final FiniteDifferenceProduct americanOption =
				new AmericanOption(maturity, strike, CallOrPut.PUT);

		final double[] americanValues = americanOption.getValue(0.0, model);

		final double[] stockGrid = spaceGrid.getGrid();
		int spotIndex = 0;
		double bestDistance = Double.POSITIVE_INFINITY;

		for(int i = 0; i < stockGrid.length; i++) {
			final double distance = Math.abs(stockGrid[i] - initialValue);
			if(distance < bestDistance) {
				bestDistance = distance;
				spotIndex = i;
			}
		}

		final double americanPrice = americanValues[spotIndex];

		final int[] numberOfExerciseDates = new int[] { 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 200 };
		final double[] bermudanPrices = new double[numberOfExerciseDates.length];
		final double[] errors = new double[numberOfExerciseDates.length];

		for(int k = 0; k < numberOfExerciseDates.length; k++) {
			final int n = numberOfExerciseDates[k];
			final double[] exerciseTimes = new double[n];

			for(int i = 0; i < n; i++) {
				exerciseTimes[i] = maturity * (i + 1.0) / n;
			}

			final FiniteDifferenceProduct bermudanOption =
					new BermudanOption(exerciseTimes, strike, CallOrPut.PUT);

			final double[] bermudanValues = bermudanOption.getValue(0.0, model);
			bermudanPrices[k] = bermudanValues[spotIndex];
			errors[k] = Math.abs(bermudanPrices[k] - americanPrice);
		}

		System.out.println("American FD        = " + americanPrice);
		System.out.println("FD Bermudan prices = " + Arrays.toString(bermudanPrices));
		System.out.println("FD errors          = " + Arrays.toString(errors));

		for(int k = 1; k < errors.length; k++) {
			assertTrue(
					"FD Bermudan error should not increase materially as exercise dates become denser.",
					errors[k] <= errors[k - 1] + 1E-2
			);
		}

		assertTrue(
				"Fine-mesh FD Bermudan should be very close to FD American.",
				errors[errors.length - 1] < 1E-4
		);
	}
}