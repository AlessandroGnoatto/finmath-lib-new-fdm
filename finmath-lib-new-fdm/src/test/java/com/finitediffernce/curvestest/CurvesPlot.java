package com.finitediffernce.curvestest;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.GridLayout;
import java.time.LocalDate;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
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

public class CurvesPlot {

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

	public static void main(String[] args) {
		SwingUtilities.invokeLater(() -> {
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
			final double minimumStockPrice = Math.max(forwardValue - numStandardDeviations * Math.sqrt(varianceStock), 0);

			final Grid spaceGrid = new UniformGrid(numSpacesteps, minimumStockPrice, maximumStockPrice);
			TimeDiscretization uniformTimeDiscretization = new TimeDiscretizationFromArray(0.0,numTimesteps,1.0/numTimesteps);
			
			SpaceTimeDiscretization spaceTimeDiscretization = new SpaceTimeDiscretization(spaceGrid, uniformTimeDiscretization, theta, new double[] {initialValue});
				
		    final FiniteDifferenceEquityModel model = new FDMBlackScholesModel(
		        		initialValue, riskFreeCurve, dividendYieldCurve, volatility,
		        		spaceTimeDiscretization);

			final FiniteDifferenceProduct callOption = new EuropeanOption(optionMaturity, optionStrike);

			final double[] valueCallFDM = callOption.getValue(0.0, model);
			final double[] initialStockPriceForCall = spaceGrid.getInteriorGrid();
			final double[] callOptionValue = java.util.Arrays.copyOfRange(valueCallFDM, 1, valueCallFDM.length-1); 
			

			XYSeries seriesCallFDM = new XYSeries("Call FDM");
			XYSeries seriesCallAnalytic = new XYSeries("Call Analitica");
			XYSeries seriesCallError = new XYSeries("Errore Call");

			for (int i = 0; i < initialStockPriceForCall.length; i++) {
				double valAnalytic = AnalyticFormulas.blackScholesOptionValue(
						initialStockPriceForCall[i], riskFreeRate, volatility, optionMaturity, optionStrike, true);
				seriesCallFDM.add(initialStockPriceForCall[i], callOptionValue[i]);
				seriesCallAnalytic.add(initialStockPriceForCall[i], valAnalytic);
				seriesCallError.add(initialStockPriceForCall[i], Math.abs(callOptionValue[i] - valAnalytic));
			}

			XYSeriesCollection datasetCallValore = new XYSeriesCollection();
			datasetCallValore.addSeries(seriesCallFDM);
			datasetCallValore.addSeries(seriesCallAnalytic);

			JFreeChart chartCallValore = ChartFactory.createXYLineChart(
					"Call Option Uniform Grid: FDM vs Analitica",
					"Prezzo Iniziale",
					"Valore Opzione",
					datasetCallValore,
					PlotOrientation.VERTICAL,
					true, true, false);

			XYSeriesCollection datasetCallErrore = new XYSeriesCollection();
			datasetCallErrore.addSeries(seriesCallError);

			JFreeChart chartCallErrore = ChartFactory.createXYLineChart(
					"Errore Assoluto Call Uniform Grid",
					"Prezzo Iniziale",
					"Errore",
					datasetCallErrore,
					PlotOrientation.VERTICAL,
					false, true, false);

			XYLineAndShapeRenderer rendererCallValore = new XYLineAndShapeRenderer();
			rendererCallValore.setSeriesPaint(0, Color.ORANGE);
			rendererCallValore.setSeriesPaint(1, Color.RED);
			rendererCallValore.setSeriesStroke(0, new BasicStroke(0.5f));
			rendererCallValore.setSeriesStroke(1, new BasicStroke(6.5f));
			chartCallValore.getXYPlot().setRenderer(rendererCallValore);
			chartCallValore.setBackgroundPaint(Color.WHITE);
			chartCallValore.getPlot().setBackgroundPaint(Color.WHITE);
			chartCallValore.getPlot().setOutlinePaint(Color.DARK_GRAY);

			
			XYLineAndShapeRenderer rendererCallErrore = new XYLineAndShapeRenderer();
			rendererCallErrore.setSeriesPaint(0, Color.PINK);
			rendererCallErrore.setSeriesStroke(0, new BasicStroke(2.5f));
			chartCallErrore.getXYPlot().setRenderer(rendererCallErrore);
			chartCallErrore.setBackgroundPaint(Color.WHITE);
			chartCallErrore.getPlot().setBackgroundPaint(Color.WHITE);
			chartCallErrore.getPlot().setOutlinePaint(Color.DARK_GRAY);


			JFrame frameCall = new JFrame("Call Option Uniform Grid: Valore + Errore");
			frameCall.setLayout(new GridLayout(2, 1));
			frameCall.add(new ChartPanel(chartCallValore));
			frameCall.add(new ChartPanel(chartCallErrore));
			frameCall.pack();
			frameCall.setLocation(100, 100);
			frameCall.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
			frameCall.setVisible(true);
		});
	}
}
