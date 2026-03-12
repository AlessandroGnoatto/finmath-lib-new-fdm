package com.finitediffernce.curvestest;

import java.time.LocalDate;
import java.util.Arrays;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.ExponentialGrid;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;

public class CurvesTest {

	private static DiscountCurve getDiscountCurve(final String name, final LocalDate referenceDate, final double riskFreeRate) {
		final double[] times = new double[] { 1.0 };
		final double[] givenAnnualizedZeroRates = new double[] { riskFreeRate };
		final InterpolationMethod interpolationMethod = InterpolationMethod.LINEAR;
		final InterpolationEntity interpolationEntity = InterpolationEntity.LOG_OF_VALUE_PER_TIME;
		final ExtrapolationMethod extrapolationMethod = ExtrapolationMethod.CONSTANT;
		return DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
				name, referenceDate, times, givenAnnualizedZeroRates,
				interpolationMethod, extrapolationMethod, interpolationEntity
		);
	}

	@Test
	public void testEuropeanCallOption() throws AssertionError {
		final LocalDate referenceDate = LocalDate.of(2010, 8, 1);
		final double riskFreeRate = 0.06;
		final DiscountCurve riskFreeCurve = getDiscountCurve("riskFreeCurve", referenceDate, riskFreeRate);
		final double dividendYieldRate = 0.0;
		final DiscountCurve dividendYieldCurve = getDiscountCurve("dividendCurve", referenceDate, dividendYieldRate);

		final double initialValue = 50;
		final double volatility = 0.4;
		final int numTimesteps = 35;
		final int numSpacesteps = 120;
		final int numStandardDeviations = 5;
		final double optionStrike = 50; 
		final double theta = 0.5;
		final double optionMaturity = 1;

		final double forwardValue = initialValue * Math.exp((riskFreeRate-dividendYieldRate) * optionMaturity);
		final double varianceStock = Math.pow(initialValue, 2) * Math.exp(2 * (riskFreeRate-dividendYieldRate) * optionMaturity)
				* (Math.exp(Math.pow(volatility, 2) * optionMaturity) - 1);
		final double maximumStockPrice = forwardValue + numStandardDeviations * Math.sqrt(varianceStock);
		final double minimumStockPrice = Math.max((forwardValue - numStandardDeviations * Math.sqrt(varianceStock)), 0);

		final Grid spaceGridUnif = new UniformGrid(numSpacesteps, minimumStockPrice, maximumStockPrice);

		TimeDiscretization uniformTimeDiscretization = new TimeDiscretizationFromArray(0.0,numTimesteps,1.0/numTimesteps);
		
		SpaceTimeDiscretization spaceTimeDiscretization = new SpaceTimeDiscretization(spaceGridUnif, uniformTimeDiscretization, theta, new double[] {initialValue});
			
	    final FiniteDifferenceEquityModel modelUnif = new FDMBlackScholesModel(
	        		initialValue, riskFreeCurve, dividendYieldCurve, volatility,
	        		spaceTimeDiscretization);

		final FiniteDifferenceProduct callOptionUnif = new EuropeanOption(optionMaturity, optionStrike);

		final double[] valueCallFDMUnif = callOptionUnif.getValue(0.0, modelUnif);

		final double[] initialStockPriceForCall = spaceGridUnif.getInteriorGrid();
		final double[] callOptionValueUnif = java.util.Arrays.copyOfRange(valueCallFDMUnif, 1, valueCallFDMUnif.length-1); 
		

		final double[] analyticalCallOptionValue = new double[callOptionValueUnif.length];
		for (int i = 0; i < analyticalCallOptionValue.length; i++) {
			analyticalCallOptionValue[i] = AnalyticFormulas.blackScholesOptionValue(
					initialStockPriceForCall[i], riskFreeRate, volatility, optionMaturity, optionStrike, true);
		}

		System.out.println("Call values uniform gird:");
		System.out.println(Arrays.toString(callOptionValueUnif));
		//Assert.assertArrayEquals(callOptionValueUnif, analyticalCallOptionValue, 1e-2);
		
		final Grid spaceGridExp = new ExponentialGrid(numSpacesteps, minimumStockPrice, maximumStockPrice, 0.9);
		
		SpaceTimeDiscretization spaceTimeDiscretizationExp = new SpaceTimeDiscretization(spaceGridExp, uniformTimeDiscretization, theta, new double[] {initialValue});
			
	    final FiniteDifferenceEquityModel modelExp = new FDMBlackScholesModel(
	        		initialValue, riskFreeCurve, dividendYieldCurve, volatility,
	        		spaceTimeDiscretizationExp);


		final FiniteDifferenceProduct callOptionExp = new EuropeanOption(optionMaturity, optionStrike);

		final double[] valueCallFDMExp = callOptionExp.getValue(0.0, modelExp);

		final double[] callOptionValueExp = java.util.Arrays.copyOfRange(valueCallFDMExp, 1, valueCallFDMExp.length-1);
		
		System.out.println("Call values exponential gird:");
		System.out.println(Arrays.toString(callOptionValueExp));
		
	}
}
