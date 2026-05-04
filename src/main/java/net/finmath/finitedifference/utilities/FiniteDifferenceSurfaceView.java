package net.finmath.finitedifference.utilities;

import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import net.finmath.finitedifference.grids.SpaceTimeDiscretization;

/**
 * Lightweight view on a finite-difference value surface.
 *
 * <p>
 * The view couples a raw finite-difference value surface with its
 * {@link SpaceTimeDiscretization}. It provides indexed access, spatial
 * interpolation, and dependency-free function adapters for plotting.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public final class FiniteDifferenceSurfaceView {

	private static final int DEFAULT_NUMBER_OF_PLOT_POINTS = 100;

	private final SpaceTimeDiscretization discretization;
	private final double[][] values;
	private final FiniteDifferenceGridLayout layout;

	/**
	 * Creates a surface view.
	 *
	 * @param discretization The space-time discretization.
	 * @param values The value surface indexed by flattened spatial index and time index.
	 */
	public FiniteDifferenceSurfaceView(
			final SpaceTimeDiscretization discretization,
			final double[][] values) {

		this.layout = FiniteDifferenceGridLayout.of(discretization);
		layout.validateSurface(values);

		this.discretization = discretization;
		this.values = values;
	}

	public static FiniteDifferenceSurfaceView of(
			final SpaceTimeDiscretization discretization,
			final double[][] values) {
		return new FiniteDifferenceSurfaceView(discretization, values);
	}

	public double getValue(final int timeIndex, final int... indices) {
		validateTimeIndex(timeIndex);
		return values[layout.flatten(indices)][timeIndex];
	}

	public double interpolate(final int timeIndex, final double... coordinates) {
		validateTimeIndex(timeIndex);
		return FiniteDifferenceValueInterpolator.interpolateTimeIndex(
				values,
				discretization,
				timeIndex,
				coordinates
		);
	}

	public double interpolate(
			final double evaluationTime,
			final double maturity,
			final double... coordinates) {
		return FiniteDifferenceValueInterpolator.interpolateSurface(
				values,
				discretization,
				evaluationTime,
				maturity,
				coordinates
		);
	}

	public double[] getTimeSlice(final int timeIndex) {
		validateTimeIndex(timeIndex);
		return FiniteDifferenceValueInterpolator.getTimeSlice(
				values,
				discretization,
				timeIndex
		);
	}

	public DoubleUnaryOperator asFunction1D(final int timeIndex) {
		if(layout.getDimension() != 1) {
			throw new IllegalArgumentException("asFunction1D requires a one-dimensional surface.");
		}

		validateTimeIndex(timeIndex);

		return x -> interpolate(timeIndex, x);
	}

	public DoubleBinaryOperator asFunction2D(
			final int timeIndex,
			final int xDimension,
			final int yDimension,
			final double... fixedCoordinates) {

		validateTimeIndex(timeIndex);
		validatePlotDimensions(xDimension, yDimension);

		final double[] baseCoordinates = createBaseCoordinates(fixedCoordinates);

		return (x, y) -> {
			final double[] coordinates = baseCoordinates.clone();
			coordinates[xDimension] = x;
			coordinates[yDimension] = y;
			return interpolate(timeIndex, coordinates);
		};
	}

	public FiniteDifferencePlotData1D asPlotData1D(
			final int timeIndex,
			final int numberOfPoints,
			final String title,
			final String xAxisLabel,
			final String yAxisLabel) {

		if(layout.getDimension() != 1) {
			throw new IllegalArgumentException("asPlotData1D requires a one-dimensional surface.");
		}

		final double[] grid = discretization.getSpaceGrid(0).getGrid();

		return new FiniteDifferencePlotData1D(
				grid[0],
				grid[grid.length - 1],
				numberOfPoints,
				asFunction1D(timeIndex),
				title,
				xAxisLabel,
				yAxisLabel
		);
	}

	public FiniteDifferencePlotData1D asPlotData1D(final int timeIndex) {
		return asPlotData1D(
				timeIndex,
				DEFAULT_NUMBER_OF_PLOT_POINTS,
				"",
				"x",
				"value"
		);
	}

	public FiniteDifferencePlotData2D asPlotData2D(
			final int timeIndex,
			final int xDimension,
			final int yDimension,
			final int numberOfPointsX,
			final int numberOfPointsY,
			final String title,
			final String xAxisLabel,
			final String yAxisLabel,
			final String zAxisLabel,
			final double... fixedCoordinates) {

		validateTimeIndex(timeIndex);
		validatePlotDimensions(xDimension, yDimension);

		final double[] xGrid = discretization.getSpaceGrid(xDimension).getGrid();
		final double[] yGrid = discretization.getSpaceGrid(yDimension).getGrid();

		return new FiniteDifferencePlotData2D(
				xGrid[0],
				xGrid[xGrid.length - 1],
				yGrid[0],
				yGrid[yGrid.length - 1],
				numberOfPointsX,
				numberOfPointsY,
				asFunction2D(timeIndex, xDimension, yDimension, fixedCoordinates),
				title,
				xAxisLabel,
				yAxisLabel,
				zAxisLabel
		);
	}

	public FiniteDifferencePlotData2D asPlotData2D(
			final int timeIndex,
			final int xDimension,
			final int yDimension) {

		return asPlotData2D(
				timeIndex,
				xDimension,
				yDimension,
				DEFAULT_NUMBER_OF_PLOT_POINTS,
				DEFAULT_NUMBER_OF_PLOT_POINTS,
				"",
				"x",
				"y",
				"value"
		);
	}

	public SpaceTimeDiscretization getDiscretization() {
		return discretization;
	}

	public double[][] getValues() {
		return values;
	}

	public FiniteDifferenceGridLayout getLayout() {
		return layout;
	}

	private double[] createBaseCoordinates(final double... fixedCoordinates) {
		final double[] baseCoordinates = new double[layout.getDimension()];

		for(int i = 0; i < baseCoordinates.length; i++) {
			baseCoordinates[i] = discretization.getCenter(i);
		}

		if(fixedCoordinates != null && fixedCoordinates.length > 0) {
			if(fixedCoordinates.length != layout.getDimension()) {
				throw new IllegalArgumentException(
						"fixedCoordinates must either be empty or have one entry per spatial dimension."
				);
			}

			for(int i = 0; i < fixedCoordinates.length; i++) {
				baseCoordinates[i] = fixedCoordinates[i];
			}
		}

		return baseCoordinates;
	}

	private void validatePlotDimensions(final int xDimension, final int yDimension) {
		if(xDimension == yDimension) {
			throw new IllegalArgumentException("xDimension and yDimension must be different.");
		}
		if(xDimension < 0 || xDimension >= layout.getDimension()) {
			throw new IllegalArgumentException("xDimension out of range.");
		}
		if(yDimension < 0 || yDimension >= layout.getDimension()) {
			throw new IllegalArgumentException("yDimension out of range.");
		}
	}

	private void validateTimeIndex(final int timeIndex) {
		if(values.length == 0 || values[0].length == 0) {
			throw new IllegalArgumentException("The surface contains no time indices.");
		}
		if(timeIndex < 0 || timeIndex >= values[0].length) {
			throw new IllegalArgumentException("timeIndex out of range.");
		}
	}
}