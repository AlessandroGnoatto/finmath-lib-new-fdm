package net.finmath.finitedifference.assetderivativevaluation.products.internal;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.interpolation.RationalFunctionInterpolation;
import net.finmath.interpolation.RationalFunctionInterpolation.ExtrapolationMethod;
import net.finmath.interpolation.RationalFunctionInterpolation.InterpolationMethod;

/**
 * Helper methods for discretely monitored knock-in products which
 * replace pre-hit values by an activated continuation surface on event dates.
 */
public final class DiscreteKnockInActivationSupport {

	private DiscreteKnockInActivationSupport() {
	}

	public static boolean hasSameGrid(
			final double[] first,
			final double[] second,
			final double tolerance) {

		if(first.length != second.length) {
			return false;
		}

		for(int i = 0; i < first.length; i++) {
			if(Math.abs(first[i] - second[i]) > tolerance) {
				return false;
			}
		}

		return true;
	}

	public static double[] buildZeroTerminalValues(final SpaceTimeDiscretization discretization) {
		final int dims = discretization.getNumberOfSpaceGrids();

		if(dims == 1) {
			final double[] grid = discretization.getSpaceGrid(0).getGrid();
			return new double[grid.length];
		}

		if(dims == 2) {
			final double[] grid0 = discretization.getSpaceGrid(0).getGrid();
			final double[] grid1 = discretization.getSpaceGrid(1).getGrid();
			return new double[grid0.length * grid1.length];
		}

		throw new IllegalArgumentException("Only 1D and 2D grids are supported.");
	}

	public static double[] getActivatedVectorAt(
			final double time,
			final double maturity,
			final double[][] activatedSurface,
			final SpaceTimeDiscretization activatedDiscretization,
			final FiniteDifferenceEquityModel targetModel,
			final double gridTolerance) {

		final double tau = maturity - time;
		int timeIndex = activatedDiscretization.getTimeDiscretization().getTimeIndex(tau);

		if(timeIndex < 0) {
			timeIndex = activatedDiscretization.getTimeDiscretization()
					.getTimeIndexNearestLessOrEqual(tau);
		}

		final SpaceTimeDiscretization targetDiscretization = targetModel.getSpaceTimeDiscretization();
		final int dims = targetDiscretization.getNumberOfSpaceGrids();

		if(dims == 1) {
			return getActivatedVectorAt1D(
					timeIndex,
					activatedSurface,
					activatedDiscretization,
					targetDiscretization,
					gridTolerance
			);
		}

		if(dims == 2) {
			return getActivatedVectorAt2D(
					timeIndex,
					activatedSurface,
					activatedDiscretization,
					targetDiscretization,
					gridTolerance
			);
		}

		throw new IllegalArgumentException("Only 1D and 2D grids are supported.");
	}

	private static double[] getActivatedVectorAt1D(
			final int timeIndex,
			final double[][] activatedSurface,
			final SpaceTimeDiscretization sourceDiscretization,
			final SpaceTimeDiscretization targetDiscretization,
			final double gridTolerance) {

		final double[] targetGrid = targetDiscretization.getSpaceGrid(0).getGrid();
		final double[] sourceGrid = sourceDiscretization.getSpaceGrid(0).getGrid();

		final double[] activatedVector = new double[targetGrid.length];

		if(hasSameGrid(sourceGrid, targetGrid, gridTolerance)) {
			for(int i = 0; i < targetGrid.length; i++) {
				activatedVector[i] = activatedSurface[i][timeIndex];
			}
			return activatedVector;
		}

		final RationalFunctionInterpolation interpolator = new RationalFunctionInterpolation(
				sourceGrid,
				getColumn(activatedSurface, timeIndex),
				InterpolationMethod.LINEAR,
				ExtrapolationMethod.CONSTANT
		);

		for(int i = 0; i < targetGrid.length; i++) {
			activatedVector[i] = interpolator.getValue(targetGrid[i]);
		}

		return activatedVector;
	}

	private static double[] getActivatedVectorAt2D(
			final int timeIndex,
			final double[][] activatedSurface,
			final SpaceTimeDiscretization sourceDiscretization,
			final SpaceTimeDiscretization targetDiscretization,
			final double gridTolerance) {

		final double[] targetX0 = targetDiscretization.getSpaceGrid(0).getGrid();
		final double[] targetX1 = targetDiscretization.getSpaceGrid(1).getGrid();
		final double[] sourceX0 = sourceDiscretization.getSpaceGrid(0).getGrid();
		final double[] sourceX1 = sourceDiscretization.getSpaceGrid(1).getGrid();

		if(!hasSameGrid(sourceX0, targetX0, gridTolerance)
				|| !hasSameGrid(sourceX1, targetX1, gridTolerance)) {
			throw new IllegalArgumentException(
					"Discrete 2D knock-in currently requires activated and target grids to match."
			);
		}

		final int n0 = targetX0.length;
		final int n1 = targetX1.length;
		final double[] activatedVector = new double[n0 * n1];

		for(int j = 0; j < n1; j++) {
			for(int i = 0; i < n0; i++) {
				final int k = flatten(i, j, n0);
				activatedVector[k] = activatedSurface[k][timeIndex];
			}
		}

		return activatedVector;
	}

	public static int flatten(final int i0, final int i1, final int n0) {
		return i0 + i1 * n0;
	}

	public static double[] getColumn(final double[][] matrix, final int columnIndex) {
		final double[] column = new double[matrix.length];
		for(int i = 0; i < matrix.length; i++) {
			column[i] = matrix[i][columnIndex];
		}
		return column;
	}
}