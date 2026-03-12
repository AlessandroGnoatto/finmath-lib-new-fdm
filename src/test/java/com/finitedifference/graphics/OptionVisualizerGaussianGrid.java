package com.finitedifference.graphics;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.GaussianGrid;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.functions.AnalyticFormulas;
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

import javax.swing.JFrame;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.GridLayout;

public class OptionVisualizerGaussianGrid {

    public static void main(String[] args) {

        final double initialValue = 50;
        final double riskFreeRate = 0.06;
        final double volatility = 0.4;
        final int numTimesteps = 35;
        final int numSpacesteps = 520;
        final int numStandardDeviations = 5;
        final double optionStrike = 50;
        final double theta = 0.5;
        
        final double optionMaturity = 1;

        final double forwardValue = initialValue * Math.exp(riskFreeRate * optionMaturity);
        final double varianceStock = Math.pow(initialValue, 2) * Math.exp(2 * riskFreeRate * optionMaturity)
                * (Math.exp(Math.pow(volatility, 2) * optionMaturity) - 1);
        final double maximumStockPrice = forwardValue + numStandardDeviations * Math.sqrt(varianceStock);
        final double minimumStockPrice = Math.max((forwardValue - numStandardDeviations * Math.sqrt(varianceStock)), 0);

        final Grid spaceGrid = new GaussianGrid(numSpacesteps, minimumStockPrice, maximumStockPrice, 50.0, 150.0);
        
        TimeDiscretization uniformTimeDiscretization = new TimeDiscretizationFromArray(0.0,numTimesteps,1.0/numTimesteps);
		
		SpaceTimeDiscretization spaceTimeDiscretizationGaussian = new SpaceTimeDiscretization(spaceGrid, uniformTimeDiscretization, theta, new double[] {initialValue});
		
        final FiniteDifferenceEquityModel model = new FDMBlackScholesModel(
        		initialValue, riskFreeRate, volatility,
        		spaceTimeDiscretizationGaussian);

        final FiniteDifferenceProduct callOption = new EuropeanOption(optionMaturity, optionStrike);
       
        final double[] valueCallFDM = callOption.getValue(0.0, model);
        final double[] xCall = spaceGrid.getInteriorGrid();//valueCallFDM[0]; //initialStockPriceForCall
		final double[] yCallFDM = java.util.Arrays.copyOfRange(valueCallFDM, 1, valueCallFDM.length-1);

        
        XYSeries seriesCallFDM = new XYSeries("Call FDM");
        XYSeries seriesCallAnalytic = new XYSeries("Call Analitica");
        XYSeries seriesCallError = new XYSeries("Errore Call");

        for (int i = 0; i < xCall.length; i++) {
            double valAnalytic = AnalyticFormulas.blackScholesOptionValue(
                    xCall[i], riskFreeRate, volatility, optionMaturity, optionStrike, true);
            seriesCallFDM.add(xCall[i], yCallFDM[i]);
            seriesCallAnalytic.add(xCall[i], valAnalytic);
            seriesCallError.add(xCall[i], Math.abs(yCallFDM[i] - valAnalytic));
        }

        
        final FiniteDifferenceProduct putOption = new EuropeanOption(optionMaturity, optionStrike, CallOrPut.PUT);
        
        final double[] valuePutFDM = putOption.getValue(0.0, model);
        final double[] xPut = spaceGrid.getInteriorGrid();//valuePutFDM[0]; //initialStockPriceForPut
		final double[] yPutFDM =  java.util.Arrays.copyOfRange(valuePutFDM, 1, valuePutFDM.length-1);

        
        XYSeries seriesPutFDM = new XYSeries("Put FDM");
        XYSeries seriesPutAnalytic = new XYSeries("Put Analitica");
        XYSeries seriesPutError = new XYSeries("Errore Put");

        for (int i = 0; i < xPut.length; i++) {
            double valAnalytic = AnalyticFormulas.blackScholesOptionValue(
                    xPut[i], riskFreeRate, volatility, optionMaturity, optionStrike, false);
            seriesPutFDM.add(xPut[i], yPutFDM[i]);
            seriesPutAnalytic.add(xPut[i], valAnalytic);
            seriesPutError.add(xPut[i], Math.abs(yPutFDM[i] - valAnalytic));
        }

        
        //grafico Call
        XYSeriesCollection datasetCallValore = new XYSeriesCollection();
        datasetCallValore.addSeries(seriesCallFDM);
        datasetCallValore.addSeries(seriesCallAnalytic);

        JFreeChart chartCallValore = ChartFactory.createXYLineChart(
                "Call Option Gaussian Grid: FDM vs Analitica",
                "Prezzo Iniziale",
                "Valore Opzione",
                datasetCallValore,
                PlotOrientation.VERTICAL,
                true, true, false);

        //grafico errore Call
        XYSeriesCollection datasetCallErrore = new XYSeriesCollection();
        datasetCallErrore.addSeries(seriesCallError);

        JFreeChart chartCallErrore = ChartFactory.createXYLineChart(
                "Errore Assoluto Call Gaussian Grid",
                "Prezzo Iniziale",
                "Errore",
                datasetCallErrore,
                PlotOrientation.VERTICAL,
                false, true, false);

        //grafico Put
        XYSeriesCollection datasetPutValore = new XYSeriesCollection();
        datasetPutValore.addSeries(seriesPutFDM);
        datasetPutValore.addSeries(seriesPutAnalytic);

        JFreeChart chartPutValore = ChartFactory.createXYLineChart(
                "Put Option Gaussian Grid: FDM vs Analitica",
                "Prezzo Iniziale",
                "Valore Opzione",
                datasetPutValore,
                PlotOrientation.VERTICAL,
                true, true, false);

        //grafico errore Put
        XYSeriesCollection datasetPutErrore = new XYSeriesCollection();
        datasetPutErrore.addSeries(seriesPutError);

        JFreeChart chartPutErrore = ChartFactory.createXYLineChart(
                "Errore Assoluto Put Gaussian Grid",
                "Prezzo Iniziale",
                "Errore",
                datasetPutErrore,
                PlotOrientation.VERTICAL,
                false, true, false);

        //opzioni personalizzazione

        //Call
        XYLineAndShapeRenderer rendererCallValore = new XYLineAndShapeRenderer();
        rendererCallValore.setSeriesPaint(0, Color.GREEN);
        rendererCallValore.setSeriesPaint(1, Color.RED);
        rendererCallValore.setSeriesStroke(0, new BasicStroke(2.5f));
        rendererCallValore.setSeriesStroke(1, new BasicStroke(2.5f));
        chartCallValore.getXYPlot().setRenderer(rendererCallValore);
        chartCallValore.setBackgroundPaint(Color.WHITE);
        chartCallValore.getPlot().setBackgroundPaint(Color.WHITE);
        chartCallValore.getPlot().setOutlinePaint(Color.DARK_GRAY);

        //Call errore
        XYLineAndShapeRenderer rendererCallErrore = new XYLineAndShapeRenderer();
        rendererCallErrore.setSeriesPaint(0, Color.PINK);
        rendererCallErrore.setSeriesStroke(0, new BasicStroke(2.5f));
        chartCallErrore.getXYPlot().setRenderer(rendererCallErrore);
        chartCallErrore.setBackgroundPaint(Color.WHITE);
        chartCallErrore.getPlot().setBackgroundPaint(Color.WHITE);
        chartCallErrore.getPlot().setOutlinePaint(Color.DARK_GRAY);

        //Put
        XYLineAndShapeRenderer rendererPutValore = new XYLineAndShapeRenderer();
        rendererPutValore.setSeriesPaint(0, Color.ORANGE);
        rendererPutValore.setSeriesPaint(1, Color.RED);
        rendererPutValore.setSeriesStroke(0, new BasicStroke(2.5f));
        rendererPutValore.setSeriesStroke(1, new BasicStroke(2.5f));
        chartPutValore.getXYPlot().setRenderer(rendererPutValore);
        chartPutValore.setBackgroundPaint(Color.WHITE);
        chartPutValore.getPlot().setBackgroundPaint(Color.WHITE);
        chartPutValore.getPlot().setOutlinePaint(Color.DARK_GRAY);

        //Put
        XYLineAndShapeRenderer rendererPutErrore = new XYLineAndShapeRenderer();
        rendererPutErrore.setSeriesPaint(0, Color.MAGENTA);
        rendererPutErrore.setSeriesStroke(0, new BasicStroke(2.5f));
        chartPutErrore.getXYPlot().setRenderer(rendererPutErrore);
        chartPutErrore.setBackgroundPaint(Color.WHITE);
        chartPutErrore.getPlot().setBackgroundPaint(Color.WHITE);
        chartPutErrore.getPlot().setOutlinePaint(Color.DARK_GRAY);

        //finestra con 4 grafici
        JFrame frame = new JFrame("Option Visualizer Gaussian Grid: Call & Put con errori");
        frame.setLayout(new GridLayout(2, 2));

        frame.add(new ChartPanel(chartCallValore));
        frame.add(new ChartPanel(chartPutValore));
        frame.add(new ChartPanel(chartCallErrore));
        frame.add(new ChartPanel(chartPutErrore));

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
        
        //finestra Call e errore Call
        JFrame frameCall = new JFrame("Call Option Gaussian Grid: Valore + Errore");
        frameCall.setLayout(new GridLayout(2, 1));
        frameCall.add(new ChartPanel(chartCallValore));
        frameCall.add(new ChartPanel(chartCallErrore));
        frameCall.pack();
        frameCall.setLocation(100, 100); // sposta un po' la finestra
        frameCall.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frameCall.setVisible(true);

        //finestra Put e errore Put
        JFrame framePut = new JFrame("Put Option Gaussian Grid: Valore + Errore");
        framePut.setLayout(new GridLayout(2, 1));
        framePut.add(new ChartPanel(chartPutValore));
        framePut.add(new ChartPanel(chartPutErrore));
        framePut.pack();
        framePut.setLocation(700, 100);
        framePut.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        framePut.setVisible(true);
    }
}
