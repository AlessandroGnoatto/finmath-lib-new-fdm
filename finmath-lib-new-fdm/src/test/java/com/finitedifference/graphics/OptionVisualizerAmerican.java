package com.finitedifference.graphics;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.GridLayout;

import javax.swing.JFrame;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.AmericanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.ExponentialGrid;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class OptionVisualizerAmerican {

	public static void main(String[] args) {

		//Parametri di input
		final double initialValue = 50;
		final double riskFreeRate = 0.06;
		final double volatility = 0.4;
		final int numTimesteps = 35;
		final int numSpacesteps = 120;
		final int numStandardDeviations = 5;
		final double optionStrike = 50;
		final double theta = 0.5;
		final double dividendYield = 0.2;
		final double optionMaturity = 1.0;
		
		//Costruzione intervalli griglia
		final double forwardValue = initialValue * Math.exp(riskFreeRate * optionMaturity);
		final double varianceStock = Math.pow(initialValue, 2) * Math.exp(2 * riskFreeRate * optionMaturity)
				* (Math.exp(Math.pow(volatility, 2) * optionMaturity) - 1);
		
		final double maximumStockPrice = forwardValue + numStandardDeviations * Math.sqrt(varianceStock);
		final double minimumStockPrice = Math.max(forwardValue - numStandardDeviations * Math.sqrt(varianceStock), 0);

		//Costruzione girglie
		final Grid spaceExponentialGrid = new ExponentialGrid(numSpacesteps, minimumStockPrice, maximumStockPrice, 0.9);
		final Grid spaceUniformGrid = new UniformGrid(numSpacesteps, minimumStockPrice, maximumStockPrice);

		TimeDiscretization uniformTimeDiscretization = new TimeDiscretizationFromArray(0.0,numTimesteps,1.0/numTimesteps);
		
		SpaceTimeDiscretization spaceTimeDiscretizationUniform = new SpaceTimeDiscretization(spaceUniformGrid, uniformTimeDiscretization, theta, new double[] {initialValue});
		SpaceTimeDiscretization spaceTimeDiscretizationExponential = new SpaceTimeDiscretization(spaceExponentialGrid, uniformTimeDiscretization, theta, new double[] {initialValue});
		
		//1) American con dividendo
		//griglia esponenziale
		FiniteDifferenceEquityModel modelAmericanYieldExp = new FDMBlackScholesModel(initialValue, riskFreeRate, dividendYield, volatility, spaceTimeDiscretizationExponential);

		FiniteDifferenceProduct americanCallOptionYield = new AmericanOption(optionMaturity, optionStrike, CallOrPut.CALL);
		FiniteDifferenceProduct americanPutOptionYield = new AmericanOption(optionMaturity, optionStrike,CallOrPut.PUT);

		double[] valueAmericanCallYieldExp = americanCallOptionYield.getValue(0.0, modelAmericanYieldExp);
		double[] valueAmericanPutYieldExp = americanPutOptionYield.getValue(0.0, modelAmericanYieldExp);

		double[] stockExp = spaceExponentialGrid.getInteriorGrid();
		double[] americanCallPricesYieldExp = java.util.Arrays.copyOfRange(valueAmericanCallYieldExp, 1, valueAmericanCallYieldExp.length-1);
		double[] americanPutPricesYieldExp = java.util.Arrays.copyOfRange(valueAmericanPutYieldExp, 1, valueAmericanPutYieldExp.length-1);

		//griglia uniforme
		FiniteDifferenceEquityModel modelAmericanYieldUnif = new FDMBlackScholesModel(initialValue, riskFreeRate, dividendYield, volatility, spaceTimeDiscretizationUniform);
		double[] valueAmericanCallYieldUnif = americanCallOptionYield.getValue(0.0, modelAmericanYieldUnif);
		double[] valueAmericanPutYieldUnif = americanPutOptionYield.getValue(0.0, modelAmericanYieldUnif);

		double[] stockUnif = spaceUniformGrid.getInteriorGrid();
		double[] americanCallPricesYieldUnif = java.util.Arrays.copyOfRange(valueAmericanCallYieldUnif, 1, valueAmericanCallYieldUnif.length-1);
		double[] americanPutPricesYieldUnif = java.util.Arrays.copyOfRange(valueAmericanPutYieldUnif, 1, valueAmericanPutYieldUnif.length-1);


		//2) American senza dividendo
		//solo griglia uniforme
		FiniteDifferenceEquityModel modelAmerican = new FDMBlackScholesModel(initialValue, riskFreeRate, volatility, spaceTimeDiscretizationUniform);
		FiniteDifferenceProduct americanCallOption = new AmericanOption(optionMaturity, optionStrike, CallOrPut.CALL);
		FiniteDifferenceProduct americanPutOption = new AmericanOption(optionMaturity, optionStrike,CallOrPut.PUT);

		double[] valueAmericanCall = americanCallOption.getValue(0.0, modelAmerican);
		double[] valueAmericanPut = americanPutOption.getValue(0.0, modelAmerican);

		double[] stockNoYield = spaceUniformGrid.getInteriorGrid();
		double[] americanCallPrices = java.util.Arrays.copyOfRange(valueAmericanCall, 1, valueAmericanCall.length-1);
		double[] americanPutPrices = java.util.Arrays.copyOfRange(valueAmericanPut, 1, valueAmericanPut.length-1);


		//Dataset separati per Call e Put con dividendo
		//griglia esponenziale
		XYSeries seriesAmericanCallYieldExp = new XYSeries("American Call (con dividendo) - exp");
		for (int i = 0; i < stockExp.length; i++) {
			seriesAmericanCallYieldExp.add(stockExp[i], americanCallPricesYieldExp[i]);
		}
		XYSeriesCollection datasetCallYieldExp = new XYSeriesCollection();
		datasetCallYieldExp.addSeries(seriesAmericanCallYieldExp);

		
		XYSeries seriesAmericanPutYieldExp = new XYSeries("American Put (con dividendo) - exp");
		for (int i = 0; i < stockExp.length; i++) {
			seriesAmericanPutYieldExp.add(stockExp[i], americanPutPricesYieldExp[i]);
		}
		XYSeriesCollection datasetPutYieldExp = new XYSeriesCollection();
		datasetPutYieldExp.addSeries(seriesAmericanPutYieldExp);

		//griglia uniforme
		XYSeries seriesAmericanCallYieldUnif = new XYSeries("American Call (con dividendo) - unif");
		for (int i = 0; i < stockUnif.length; i++) {
			seriesAmericanCallYieldUnif.add(stockUnif[i], americanCallPricesYieldUnif[i]);
		}
		XYSeriesCollection datasetCallYieldUnif = new XYSeriesCollection();
		datasetCallYieldUnif.addSeries(seriesAmericanCallYieldUnif);

		XYSeries seriesAmericanPutYieldUnif = new XYSeries("American Put (con dividendo) - unif");
		for (int i = 0; i < stockUnif.length; i++) {
			seriesAmericanPutYieldUnif.add(stockUnif[i], americanPutPricesYieldUnif[i]);
		}
		XYSeriesCollection datasetPutYieldUnif = new XYSeriesCollection();
		datasetPutYieldUnif.addSeries(seriesAmericanPutYieldUnif);

		// Dataset separati per Call e Put senza dividendo
		XYSeries seriesAmericanCall = new XYSeries("American Call (no dividendo)");
		for (int i = 0; i < stockNoYield.length; i++) {
			seriesAmericanCall.add(stockNoYield[i], americanCallPrices[i]);
		}
		XYSeriesCollection datasetCall = new XYSeriesCollection();
		datasetCall.addSeries(seriesAmericanCall);

		XYSeries seriesAmericanPut = new XYSeries("American Put (no dividendo)");
		for (int i = 0; i < stockNoYield.length; i++) {
			seriesAmericanPut.add(stockNoYield[i], americanPutPrices[i]);
		}
		XYSeriesCollection datasetPut = new XYSeriesCollection();
		datasetPut.addSeries(seriesAmericanPut);

		
		//Creazione grafici separati
		JFreeChart chartCallYieldExp = ChartFactory.createXYLineChart(
				"American Call con dividendo e griglia esponenziale",
				"Prezzo Sottostante",
				"Valore Opzione",
				datasetCallYieldExp,
				PlotOrientation.VERTICAL,
				true, true, false);

		JFreeChart chartCallYieldUnif = ChartFactory.createXYLineChart(
				"American Call con dividendo e griglia uniforme",
				"Prezzo Sottostante",
				"Valore Opzione",
				datasetCallYieldUnif,
				PlotOrientation.VERTICAL,
				true, true, false);


		JFreeChart chartPutYieldExp = ChartFactory.createXYLineChart(
				"American Put con dividendo e griglia esponenziale",
				"Prezzo Sottostante",
				"Valore Opzione",
				datasetPutYieldExp,
				PlotOrientation.VERTICAL,
				true, true, false);

		JFreeChart chartPutYieldUnif = ChartFactory.createXYLineChart(
				"American Put con dividendo e griglia uniforme",
				"Prezzo Sottostante",
				"Valore Opzione",
				datasetPutYieldUnif,
				PlotOrientation.VERTICAL,
				true, true, false);


		JFreeChart chartCallNoYield = ChartFactory.createXYLineChart(
				"American Call (no dividendo)",
				"Prezzo Sottostante",
				"Valore Opzione",
				datasetCall,
				PlotOrientation.VERTICAL,
				true, true, false);

		JFreeChart chartPutNoYield = ChartFactory.createXYLineChart(
				"American Put (no dividendo)",
				"Prezzo Sottostante",
				"Valore Opzione",
				datasetPut,
				PlotOrientation.VERTICAL,
				true, true, false);

		//opzioni personalizzazione
		//Call
		XYLineAndShapeRenderer rendererCallValoreExp = new XYLineAndShapeRenderer();
		rendererCallValoreExp.setSeriesPaint(0, Color.GREEN);
		rendererCallValoreExp.setSeriesPaint(1, Color.RED);
		rendererCallValoreExp.setSeriesStroke(0, new BasicStroke(2.5f));
		rendererCallValoreExp.setSeriesStroke(1, new BasicStroke(2.5f));
		chartCallYieldExp.getXYPlot().setRenderer(rendererCallValoreExp);
		chartCallYieldExp.setBackgroundPaint(Color.WHITE);
		chartCallYieldExp.getPlot().setBackgroundPaint(Color.WHITE);
		chartCallYieldExp.getPlot().setOutlinePaint(Color.DARK_GRAY);

		XYLineAndShapeRenderer rendererCallValoreUnif = new XYLineAndShapeRenderer();
		rendererCallValoreUnif.setSeriesPaint(0, Color.PINK);
		rendererCallValoreUnif.setSeriesPaint(1, Color.RED);
		rendererCallValoreUnif.setSeriesStroke(0, new BasicStroke(2.5f));
		rendererCallValoreUnif.setSeriesStroke(1, new BasicStroke(2.5f));
		chartCallYieldUnif.getXYPlot().setRenderer(rendererCallValoreUnif);
		chartCallYieldUnif.setBackgroundPaint(Color.WHITE);
		chartCallYieldUnif.getPlot().setBackgroundPaint(Color.WHITE);
		chartCallYieldUnif.getPlot().setOutlinePaint(Color.DARK_GRAY);


		//Put
		XYLineAndShapeRenderer rendererPutValoreExp = new XYLineAndShapeRenderer();
		rendererPutValoreExp.setSeriesPaint(0, Color.BLUE);
		rendererPutValoreExp.setSeriesStroke(0, new BasicStroke(2.5f));
		chartPutYieldExp.getXYPlot().setRenderer(rendererPutValoreExp);
		chartPutYieldExp.setBackgroundPaint(Color.WHITE);
		chartPutYieldExp.getPlot().setBackgroundPaint(Color.WHITE);
		chartPutYieldExp.getPlot().setOutlinePaint(Color.DARK_GRAY);

		XYLineAndShapeRenderer rendererPutValoreUnif = new XYLineAndShapeRenderer();
		rendererPutValoreUnif.setSeriesPaint(0, Color.GRAY);
		rendererPutValoreUnif.setSeriesStroke(0, new BasicStroke(2.5f));
		chartPutYieldUnif.getXYPlot().setRenderer(rendererPutValoreUnif);
		chartPutYieldUnif.setBackgroundPaint(Color.WHITE);
		chartPutYieldUnif.getPlot().setBackgroundPaint(Color.WHITE);
		chartPutYieldUnif.getPlot().setOutlinePaint(Color.DARK_GRAY);


		//Call
		XYLineAndShapeRenderer rendererPutValore = new XYLineAndShapeRenderer();
		rendererPutValore.setSeriesPaint(0, Color.ORANGE);
		rendererPutValore.setSeriesStroke(0, new BasicStroke(2.5f));
		chartCallNoYield.getXYPlot().setRenderer(rendererPutValore);
		chartCallNoYield.setBackgroundPaint(Color.WHITE);
		chartCallNoYield.getPlot().setBackgroundPaint(Color.WHITE);
		chartCallNoYield.getPlot().setOutlinePaint(Color.DARK_GRAY);


		
		//Configurazione finestre JFrame ---
		//finestra con 4 grafici
		JFrame frame = new JFrame("Option Visualizer American Options");
		frame.setLayout(new GridLayout(2, 2));
		frame.add(new ChartPanel(chartCallYieldExp));
		frame.add(new ChartPanel(chartPutYieldExp));
		frame.add(new ChartPanel(chartCallYieldUnif));
		frame.add(new ChartPanel(chartPutYieldUnif));
		frame.pack();
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);


		JFrame frameYieldCall = new JFrame("American Call con dividendo");
		frameYieldCall.setLayout(new GridLayout(2, 1)); // 2 righe, 1 colonna
		frameYieldCall.add(new ChartPanel(chartCallYieldExp));  // call sopra
		frameYieldCall.add(new ChartPanel(chartCallYieldUnif));   // put sotto
		frameYieldCall.pack();
		frameYieldCall.setLocation(100, 100);
		frameYieldCall.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frameYieldCall.setVisible(true);
		
		JFrame frameYieldPut = new JFrame("American Put con dividendo");
		frameYieldPut.setLayout(new GridLayout(2, 1)); // 2 righe, 1 colonna
		frameYieldPut.add(new ChartPanel(chartPutYieldExp));  // call sopra
		frameYieldPut.add(new ChartPanel(chartPutYieldUnif));   // put sotto
		frameYieldPut.pack();
		frameYieldPut.setLocation(100, 100);
		frameYieldPut.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frameYieldPut.setVisible(true);

		JFrame frameNoYield = new JFrame("American Options senza dividendo");
		frameNoYield.setLayout(new GridLayout(2, 1)); // modifica anche qui: 2 righe, 1 colonna
		frameNoYield.add(new ChartPanel(chartCallNoYield)); // call sopra
		frameNoYield.add(new ChartPanel(chartPutNoYield));  // put sotto
		frameNoYield.pack();
		frameNoYield.setLocation(frameYieldCall.getX() + frameYieldCall.getWidth() + 20, 100);
		frameNoYield.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frameNoYield.setVisible(true);
	}
}