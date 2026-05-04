package net.finmath.finitedifference.assetderivativevaluation.products.internal;

import java.util.Map;

/**
 * Event-time keyed activated value vectors.
 * 
 * @author Alessandro Gnoatto
 */
public final class ActivatedVectorEventState {

	private final Map<Double, double[]> activatedVectorsAtEventTimes;
	private final double timeTolerance;

	public ActivatedVectorEventState(
			final Map<Double, double[]> activatedVectorsAtEventTimes,
			final double timeTolerance) {

		if(activatedVectorsAtEventTimes == null) {
			throw new IllegalArgumentException("activatedVectorsAtEventTimes must not be null.");
		}
		if(timeTolerance < 0.0) {
			throw new IllegalArgumentException("timeTolerance must be non-negative.");
		}

		this.activatedVectorsAtEventTimes = activatedVectorsAtEventTimes;
		this.timeTolerance = timeTolerance;
	}

	public double[] getActivatedVector(final double eventTime) {

		for(final Map.Entry<Double, double[]> entry : activatedVectorsAtEventTimes.entrySet()) {
			if(Math.abs(entry.getKey() - eventTime) <= timeTolerance) {
				return entry.getValue();
			}
		}

		throw new IllegalArgumentException(
				"No activated vector found for event time " + eventTime + "."
		);
	}
}