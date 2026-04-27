package com.finitedifference.americantest;

import java.util.Arrays;
import org.junit.Test;

import net.finmath.exception.CalculationException;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.AmericanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceEquityProduct;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.functions.AnalyticFormulas;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.montecarlo.BrownianMotion;
import net.finmath.montecarlo.BrownianMotionFromMersenneRandomNumbers;
import net.finmath.montecarlo.assetderivativevaluation.MonteCarloBlackScholesModel;
import net.finmath.montecarlo.assetderivativevaluation.products.AbstractAssetMonteCarloProduct;
import net.finmath.montecarlo.assetderivativevaluation.products.BermudanOption;
import net.finmath.montecarlo.assetderivativevaluation.products.BermudanOption.ExerciseMethod;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

public class AmericanTest {


	@Test
	public void testAmericanOption() throws CalculationException {

		final double initialValue = 50;
		final double riskFreeRate = 0.06;
		final double volatility = 0.4;
		final int numTimesteps = 35;
		final int numSpacesteps = 120;
		final int numStandardDeviations = 5;
		final double optionStrike = 50; //center
		final double theta = 0.5;

		final double dividendYield = 0.2;

		final double optionMaturity = 1.0;

		//griglia con tasso di dividendo
		final double forwardValueDividendYeld= initialValue*Math.exp((riskFreeRate-dividendYield) * optionMaturity);
		final double varianceStockDividendYeld = Math.pow(initialValue, 2) * Math.exp(2 * (riskFreeRate-dividendYield) * optionMaturity)
				* (Math.exp(Math.pow(volatility, 2) * optionMaturity) - 1);

		final double maximumStockPriceDividendYeld = forwardValueDividendYeld + numStandardDeviations*Math.sqrt(varianceStockDividendYeld);
		final double minimumStockPriceDividendYeld = Math.max((forwardValueDividendYeld-numStandardDeviations*Math.sqrt(varianceStockDividendYeld)), 0);

		final Grid spaceGridDividendYeld = new UniformGrid(numSpacesteps, minimumStockPriceDividendYeld, maximumStockPriceDividendYeld);

		//griglia senza tasso di dividendo
		final double forwardValue = initialValue*Math.exp(riskFreeRate * optionMaturity);
		final double varianceStock = Math.pow(initialValue, 2) * Math.exp(2 * riskFreeRate * optionMaturity)
				* (Math.exp(Math.pow(volatility, 2) * optionMaturity) - 1);
		final double maximumStockPrice = forwardValue + numStandardDeviations*Math.sqrt(varianceStock);
		final double minimumStockPrice = Math.max((forwardValue-numStandardDeviations*Math.sqrt(varianceStock)), 0);

		final Grid spaceGrid = new UniformGrid(numSpacesteps, minimumStockPrice, maximumStockPrice);


		//modello per opzioni americane
		//con tasso di dividendo
		TimeDiscretization uniformTimeDiscretization = new TimeDiscretizationFromArray(0.0,numTimesteps,1.0/numTimesteps);
		
		SpaceTimeDiscretization spaceTimeDiscretizationUniform = new SpaceTimeDiscretization(spaceGrid, uniformTimeDiscretization, theta, new double[] {initialValue});
		
        final FiniteDifferenceEquityModel modelAmericanYield = new FDMBlackScholesModel(
        		initialValue, riskFreeRate, dividendYield, volatility,
        		spaceTimeDiscretizationUniform);
		
		
		final FiniteDifferenceEquityProduct americanCallOptionYield = new AmericanOption(optionMaturity, optionStrike, CallOrPut.CALL);
		final FiniteDifferenceEquityProduct americanPutOptionYield = new AmericanOption(optionMaturity, optionStrike, CallOrPut.PUT);

		final double[] valueAmericanCallYield = americanCallOptionYield.getValue(0.0, modelAmericanYield);
		final double[] valueAmericanPutYield = americanPutOptionYield.getValue(0.0, modelAmericanYield);
		final double[] americanCallPricesYield = java.util.Arrays.copyOfRange(valueAmericanCallYield, 1, valueAmericanCallYield.length-1);
		final double[] americanPutPricesYield = java.util.Arrays.copyOfRange(valueAmericanPutYield, 1, valueAmericanPutYield.length-1);

		//modello per opzioni americane
		//senza tasso di dividendo
		final FiniteDifferenceEquityModel modelAmerican = new FDMBlackScholesModel(
	        		initialValue, riskFreeRate, volatility,
	        		spaceTimeDiscretizationUniform);
		
		final FiniteDifferenceEquityProduct americanCallOption = new AmericanOption(optionMaturity, optionStrike, CallOrPut.CALL);
		final FiniteDifferenceEquityProduct americanPutOption = new AmericanOption(optionMaturity, optionStrike, CallOrPut.PUT);

		final double[] valueAmericanCall = americanCallOption.getValue(0.0, modelAmerican);
		final double[] valueAmericanPut = americanPutOption.getValue(0.0, modelAmerican);
		
		final double[] americanCallPrices = java.util.Arrays.copyOfRange(valueAmericanCall, 1, valueAmericanCall.length-1);
		final double[] americanPutPrices = java.util.Arrays.copyOfRange(valueAmericanPut, 1, valueAmericanPut.length-1);


		//modello per opzioni europee
		//senza tasso di dividendo
		final FiniteDifferenceEquityModel modelEuropean = new FDMBlackScholesModel(
        		initialValue, riskFreeRate, volatility,
        		spaceTimeDiscretizationUniform);
		final FiniteDifferenceEquityProduct europeanCallOption = new EuropeanOption(optionMaturity, optionStrike, CallOrPut.CALL);
		final FiniteDifferenceEquityProduct europeanPutOption = new EuropeanOption(optionMaturity, optionStrike, CallOrPut.PUT);

		final double[] valueeuropeanCall = europeanCallOption.getValue(0.0, modelEuropean);
		final double[] valueeuropeanPut = europeanPutOption.getValue(0.0, modelEuropean);

		final double[] europeanCallPrices =  java.util.Arrays.copyOfRange(valueeuropeanCall, 1, valueeuropeanCall.length-1);
		final double[] europeanPutPrices = java.util.Arrays.copyOfRange(valueeuropeanPut, 1, valueeuropeanPut.length-1);

		final double[] stock = spaceGrid.getInteriorGrid();

		//valore analitico
		final double[] analyticalCallOptionValue = new double[europeanCallPrices.length];

		for (int i = 0; i < analyticalCallOptionValue.length; i++) {
			analyticalCallOptionValue[i] = AnalyticFormulas.blackScholesOptionValue(stock[i], riskFreeRate,
					volatility, optionMaturity, optionStrike, true);
		}

		//benchmark opzione americana
		TimeDiscretization times = new TimeDiscretizationFromArray(0.0, numTimesteps, optionMaturity/numTimesteps);
		double[] exerciseTimesAmerican = times.getAsDoubleArray();
		
		double[] notionals = new double[exerciseTimesAmerican.length];
		Arrays.fill(notionals, 1.0);
		
		double[] strikes = new double[exerciseTimesAmerican.length];
		Arrays.fill(strikes, optionStrike);
		
		//parametri della simulazione
		int numberOfPaths = 30000;
		int seed = 1897;
				
		BrownianMotion ourDriver = new BrownianMotionFromMersenneRandomNumbers(times, 1, numberOfPaths, seed);
		
		final AbstractAssetMonteCarloProduct bermudanOption = new BermudanOption(exerciseTimesAmerican, notionals, strikes, ExerciseMethod.UPPER_BOUND_METHOD);
		
		//considero alcuni valori dello stock iniziale per cui calcolare e confrontare i prezzi 
		//dell'opzione americana e bermudiana
		int[] indices = {stock.length / 4, stock.length / 2, stock.length-1};

		for(int index : indices) {

		    double S0 = stock[index];
		    //processo di Black&Sholes
		    MonteCarloBlackScholesModel blackScholesProcess = new MonteCarloBlackScholesModel(S0, riskFreeRate, volatility, ourDriver);
		
		    double bermudanValue = bermudanOption.getValue(blackScholesProcess);

		    double americanFDMValue = americanCallPrices[index];
		    
		    double americanDividendFDMValue = americanCallPricesYield[index];

		    double europeanFDMValue = europeanCallPrices[index];
		    
		    double analyticalCallValue = analyticalCallOptionValue[index];
		    
		    System.out.println("S0 = " + S0);
		    System.out.println("FDM American Call = " + americanFDMValue);
		    System.out.println("FDM American Call Dividend = " + americanDividendFDMValue);
		    System.out.println("Bermudan Call = " + bermudanValue);
		    System.out.println("European Call = " + europeanFDMValue);
		    System.out.println("Analytical Call = " + analyticalCallValue);
		    System.out.println();
		}
		 
		System.out.print("Stock values:");
		System.out.println(Arrays.toString(stock));
		System.out.println();

		System.out.print("American call option values (q = " + (dividendYield * 100) + "%)");
		System.out.println(Arrays.toString(americanCallPricesYield));
		System.out.println();

		System.out.print("American put option values (q = " + (dividendYield * 100) + "%)");
		System.out.println(Arrays.toString(americanPutPricesYield));
		System.out.println();
		
		System.out.print("American call option values ");
		System.out.println(Arrays.toString(americanCallPrices));
		System.out.println();

		System.out.print("American put option values ");
		System.out.println(Arrays.toString(americanPutPrices));
		System.out.println();
		
		System.out.print("European call option values ");
		System.out.println(Arrays.toString(europeanCallPrices));
		System.out.println();

		System.out.print("European put option values ");
		System.out.println(Arrays.toString(europeanPutPrices));
		System.out.println();
		
		System.out.print("Analytical call values: ");
		System.out.println(Arrays.toString(analyticalCallOptionValue));
	}
}
