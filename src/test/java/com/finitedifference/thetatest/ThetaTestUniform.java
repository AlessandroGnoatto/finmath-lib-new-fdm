package com.finitedifference.thetatest;
import java.util.Arrays;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Assert;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

public class ThetaTestUniform {

	@Test
	public void testEuropeanCallOption() throws AssertionError {
		final double initialValue = 50;
		final double riskFreeRate = 0.06;
		final double volatility = 0.4;
		final int numTimesteps = 35;
		final int numSpacesteps = 120;
		final int numStandardDeviations = 5;
		final double optionStrike = 50; //center
		final double theta = 0.5;

		//final double dividendYield = 0.03;
		
		final double optionMaturity = 1;

		final double forwardValue = initialValue*Math.exp(riskFreeRate * optionMaturity);
		final double varianceStock = Math.pow(initialValue, 2) * Math.exp(2 * riskFreeRate * optionMaturity)
				* (Math.exp(Math.pow(volatility, 2) * optionMaturity) - 1);
		
//		final double forwardValue = initialValue*Math.exp((riskFreeRate-dividendYield) * optionMaturity);
//		final double varianceStock = Math.pow(initialValue, 2) * Math.exp(2 * (riskFreeRate-dividendYield) * optionMaturity)
//				* (Math.exp(Math.pow(volatility, 2) * optionMaturity) - 1);
		
		final double maximumStockPrice = forwardValue + numStandardDeviations*Math.sqrt(varianceStock);
		final double minimumStockPrice = Math.max((forwardValue-numStandardDeviations*Math.sqrt(varianceStock)), 0);

		final Grid spaceGrid = new UniformGrid(numSpacesteps, minimumStockPrice, maximumStockPrice);

		TimeDiscretization uniformTimeDiscretization = new TimeDiscretizationFromArray(0.0,numTimesteps,1.0/numTimesteps);
		
		SpaceTimeDiscretization spaceTimeDiscretization = new SpaceTimeDiscretization(spaceGrid, uniformTimeDiscretization, theta, new double[] {initialValue});
			
	    final FiniteDifferenceEquityModel model = new FDMBlackScholesModel(
	        		initialValue, riskFreeRate, volatility,
	        		spaceTimeDiscretization);

	    final FiniteDifferenceProduct callOption = new EuropeanOption(optionMaturity, optionStrike,CallOrPut.CALL);

		final double[] valueCallFDM = callOption.getValue(0.0, model);

		final double[] initialStockPriceForCall = spaceGrid.getInteriorGrid();//[0];
		final double[] callOptionValue = java.util.Arrays.copyOfRange(valueCallFDM, 1, valueCallFDM.length-1);

		final double[] analyticalCallOptionValue = new double[callOptionValue.length];
		for (int i = 0; i < analyticalCallOptionValue.length; i++) {
			analyticalCallOptionValue[i] = AnalyticFormulas.blackScholesOptionValue(initialStockPriceForCall[i], riskFreeRate,
					volatility, optionMaturity, optionStrike, true);
		}

		System.out.print("Call values:");
		System.out.println(Arrays.toString(callOptionValue));
		Assert.assertArrayEquals(callOptionValue, analyticalCallOptionValue, 1e1);
	}


	@Test 
	public void testEuropeanPutOption() throws AssertionError { 
		final double initialValue = 50; 
		final double riskFreeRate = 0.06; 
		final double volatility = 0.4; 
		final int numTimesteps = 35; 
		final int numSpacesteps = 120;
		final int numStandardDeviations = 5; 
		final double optionStrike = 50; //center
		final double theta = 0.5;

		final double optionMaturity = 1;


		final double forwardValue = initialValue*Math.exp(riskFreeRate * optionMaturity); 
		final double varianceStock = Math.pow(initialValue, 2) * Math.exp(2 * riskFreeRate * optionMaturity)
				* (Math.exp(Math.pow(volatility, 2) * optionMaturity) - 1); 
		final double maximumStockPrice = forwardValue + numStandardDeviations*Math.sqrt(varianceStock); 
		final double minimumStockPrice = Math.max((forwardValue-numStandardDeviations*Math.sqrt(varianceStock)), 0);

		final Grid spaceGrid = new UniformGrid(numSpacesteps, minimumStockPrice, maximumStockPrice);

		TimeDiscretization uniformTimeDiscretization = new TimeDiscretizationFromArray(0.0,numTimesteps,1.0/numTimesteps);
		
		SpaceTimeDiscretization spaceTimeDiscretization = new SpaceTimeDiscretization(spaceGrid, uniformTimeDiscretization, theta, new double[] {initialValue});
			
	    final FiniteDifferenceEquityModel model = new FDMBlackScholesModel(
	        		initialValue, riskFreeRate, volatility,
	        		spaceTimeDiscretization);
	    final FiniteDifferenceProduct putOption = new EuropeanOption(optionMaturity, optionStrike,CallOrPut.PUT); 

		final double[] valuePutFDM = putOption.getValue(0.0, model); 

		final double[] initialStockPriceForPut = spaceGrid.getInteriorGrid();//[0]; 
		final double[] putOptionValue = java.util.Arrays.copyOfRange(valuePutFDM, 1, valuePutFDM.length-1); 
		 
		final double[] analyticalPutOptionValue = new double[putOptionValue.length];
		for (int i = 0; i < analyticalPutOptionValue.length; i++) { 
			analyticalPutOptionValue[i] = AnalyticFormulas.blackScholesOptionValue(initialStockPriceForPut[i], riskFreeRate,
					volatility, optionMaturity, optionStrike, false); 
		}

		System.out.print("Put values:");
		System.out.println(Arrays.toString(putOptionValue));
		Assert.assertArrayEquals(putOptionValue, analyticalPutOptionValue, 1e1); } 
}
