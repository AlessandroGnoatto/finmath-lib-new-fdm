package com.finitedifference.gridtest;


import net.finmath.finitedifference.grids.ExponentialGrid;
import net.finmath.finitedifference.grids.GaussianGrid;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.HyperbolicSineGrid;
import net.finmath.finitedifference.grids.UniformGrid;


public class GridTest {

	public static void main(String[] strings) throws Exception {


		//uniform grid test 
		Grid grid = new UniformGrid(20, 0.0, 45.0); 
		double[] valuesGrid = grid.getGrid(); 
		double[] deltaGrid = grid.getDelta(valuesGrid);

		System.out.println("La lunghezza della griglia UNIFORME è " + valuesGrid.length); 
		for(int i = 0; i < valuesGrid.length; i++) {
			System.out.print(String.format("%.4f",valuesGrid[i]) + " "); 
		}
		System.out.println();

		System.out.println("La lunghezza del delta è " + deltaGrid.length); 
		for(int i = 0; i < deltaGrid.length; i++) { 
			System.out.print(String.format("%.4f",
					deltaGrid[i]) + " "); 
		} 
		System.out.println(); 
		System.out.println();

		//exponential grid test
		Grid grid4 = new ExponentialGrid(20, 0, 45, 2.0); 
		double[] valuesGrid4 = grid4.getGrid(); 
		double[] deltaGrid4 = grid4.getDelta(valuesGrid4);

		System.out.println("La lunghezza della griglia ESPONENZIALE è " + valuesGrid4.length); 
		for(int i = 0; i < valuesGrid4.length; i++) {
			System.out.print(String.format("%.4f", valuesGrid4[i]) + " "); 
		}
		System.out.println();
		System.out.println("La lunghezza del delta è " + deltaGrid4.length); 
		for(int i = 0; i < deltaGrid4.length; i++) { 
			System.out.print(String.format("%.4f", deltaGrid4[i]) + " "); 
		}
		System.out.println(); 
		System.out.println(); 

		//hyperbolic sine grid test
		Grid grid3 = new HyperbolicSineGrid(20, 0, 45, 30, 0.5);
		double[] valuesGrid3 = grid3.getGrid();
		double[] deltaGrid3 = grid3.getDelta(valuesGrid3);

		System.out.println("La lunghezza della griglia SENO-IPERBOLICA è " + valuesGrid3.length);
		for(int i = 0; i < valuesGrid3.length; i++) {
			System.out.print(String.format("%.4f", valuesGrid3[i]) + " ");
		}
		System.out.println();

		System.out.println("La lunghezza del delta è " + deltaGrid3.length);
		for(int i = 0; i < deltaGrid3.length; i++) {
			System.out.print(String.format("%.4f", deltaGrid3[i]) + " ");
		}
		System.out.println();
		System.out.println();

		//gaussian grid test 
		Grid grid2 = new GaussianGrid(20, 0, 45, 30, 150); 
		double[] valuesGrid2 = grid2.getGrid(); 
		double[] deltaGrid2 = grid2.getDelta(valuesGrid2);

		System.out.println("La lunghezza della griglia GAUSSIANA è " + valuesGrid2.length);
		for(int i = 0; i < valuesGrid2.length; i++) {
			System.out.print(String.format("%.4f", valuesGrid2[i]) + " "); 
		}
		System.out.println();

		System.out.println("La lunghezza del delta è " + deltaGrid2.length); 
		for(int i = 0; i < deltaGrid2.length; i++) { 
			System.out.print(String.format("%.4f", deltaGrid2[i]) + " "); 
		} 
		System.out.println(); 
		System.out.println(); 

	}
}