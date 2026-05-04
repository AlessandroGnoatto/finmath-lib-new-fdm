package net.finmath.finitedifference.utilities;

import java.util.function.DoubleUnaryOperator;

/**
 * Dependency-free plot descriptor for a one-dimensional finite-difference curve.
 *
 * <p>
 * This class intentionally depends only on standard Java functional interfaces.
 * Plotting libraries may consume the returned {@link DoubleUnaryOperator}
 * without creating a compile-time dependency from the finite-difference module
 * to a plotting module.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public final class FiniteDifferencePlotData1D {

	private final double xMin;
	private final double xMax;
	private final int numberOfPoints;
	private final DoubleUnaryOperator function;
	private final String title;
	private final String xAxisLabel;
	private final String yAxisLabel;

	/**
	 * Creates a one-dimensional plot descriptor.
	 *
	 * @param xMin The lower x-axis bound.
	 * @param xMax The upper x-axis bound.
	 * @param numberOfPoints The number of plot points.
	 * @param function The function to plot.
	 * @param title The plot title.
	 * @param xAxisLabel The x-axis label.
	 * @param yAxisLabel The y-axis label.
	 */
	public FiniteDifferencePlotData1D(
			final double xMin,
			final double xMax,
			final int numberOfPoints,
			final DoubleUnaryOperator function,
			final String title,
			final String xAxisLabel,
			final String yAxisLabel) {

		if(numberOfPoints <= 0) {
			throw new IllegalArgumentException("numberOfPoints must be positive.");
		}
		if(function == null) {
			throw new IllegalArgumentException("function must not be null.");
		}

		this.xMin = xMin;
		this.xMax = xMax;
		this.numberOfPoints = numberOfPoints;
		this.function = function;
		this.title = title;
		this.xAxisLabel = xAxisLabel;
		this.yAxisLabel = yAxisLabel;
	}

	public double getXMin() {
		return xMin;
	}

	public double getXMax() {
		return xMax;
	}

	public int getNumberOfPoints() {
		return numberOfPoints;
	}

	public DoubleUnaryOperator getFunction() {
		return function;
	}

	public String getTitle() {
		return title;
	}

	public String getXAxisLabel() {
		return xAxisLabel;
	}

	public String getYAxisLabel() {
		return yAxisLabel;
	}
}