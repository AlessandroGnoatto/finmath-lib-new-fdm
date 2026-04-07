package net.finmath.finitedifference.solvers.adi;

import java.util.Arrays;

/**
 * Immutable container for the activated barrier trace used in direct 2D knock-in pricing.
 *
 * <p>
 * The trace stores the value of the activated post-hit contract evaluated on the
 * barrier, across:
 * </p>
 * <ul>
 *   <li>the second state variable index (variance for Heston, alpha for SABR),</li>
 *   <li>time index.</li>
 * </ul>
 *
 * <p>
 * The array layout is
 * </p>
 *
 * <pre>
 * values[secondStateIndex][timeIndex]
 * </pre>
 *
 * <p>
 * This class is immutable and thread-safe.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public final class ActivatedBarrierTrace2D {

	private final double barrierLevel;
	private final double[] secondStateGrid;
	private final double[][] values;

	public ActivatedBarrierTrace2D(
			final double barrierLevel,
			final double[] secondStateGrid,
			final double[][] values) {

		if(secondStateGrid == null) {
			throw new IllegalArgumentException("secondStateGrid must not be null.");
		}
		if(values == null) {
			throw new IllegalArgumentException("values must not be null.");
		}
		if(values.length != secondStateGrid.length) {
			throw new IllegalArgumentException(
					"values.length must equal secondStateGrid.length."
			);
		}
		if(values.length == 0) {
			throw new IllegalArgumentException("values must contain at least one row.");
		}
		if(values[0] == null || values[0].length == 0) {
			throw new IllegalArgumentException("values must contain at least one time column.");
		}

		final int numberOfTimePoints = values[0].length;
		for(int j = 0; j < values.length; j++) {
			if(values[j] == null) {
				throw new IllegalArgumentException("values row " + j + " must not be null.");
			}
			if(values[j].length != numberOfTimePoints) {
				throw new IllegalArgumentException(
						"All rows of values must have the same length."
				);
			}
		}

		this.barrierLevel = barrierLevel;
		this.secondStateGrid = Arrays.copyOf(secondStateGrid, secondStateGrid.length);
		this.values = deepCopy(values);
	}

	public double getBarrierLevel() {
		return barrierLevel;
	}

	public double[] getSecondStateGrid() {
		return Arrays.copyOf(secondStateGrid, secondStateGrid.length);
	}

	public double[][] getValues() {
		return deepCopy(values);
	}

	public int getNumberOfSecondStatePoints() {
		return values.length;
	}

	public int getNumberOfTimePoints() {
		return values[0].length;
	}

	public double getValue(final int secondStateIndex, final int timeIndex) {
		return values[secondStateIndex][timeIndex];
	}

	private static double[][] deepCopy(final double[][] data) {
		final double[][] copy = new double[data.length][];
		for(int i = 0; i < data.length; i++) {
			copy[i] = Arrays.copyOf(data[i], data[i].length);
		}
		return copy;
	}
}