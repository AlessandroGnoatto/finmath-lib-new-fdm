package net.finmath.finitedifference.assetderivativevaluation.products.internal;

import java.util.Set;
import java.util.TreeSet;

import net.finmath.modelling.products.MonitoringType;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

/**
 * Utility methods for products with continuous / discrete monitoring.
 */
public final class DiscreteMonitoringSupport {

	public static final double DEFAULT_MONITORING_TIME_TOLERANCE = 1E-12;

	private DiscreteMonitoringSupport() {
	}

	public static boolean usesDiscreteMonitoring(final MonitoringType monitoringType) {
		return monitoringType == MonitoringType.DISCRETE;
	}

	public static boolean isMonitoringTime(
			final double time,
			final double[] monitoringTimes,
			final double tolerance) {

		if(monitoringTimes == null) {
			return false;
		}

		for(final double monitoringTime : monitoringTimes) {
			if(Math.abs(monitoringTime - time) <= tolerance) {
				return true;
			}
		}

		return false;
	}

	public static void validateMonitoringSpecification(
			final MonitoringType monitoringType,
			final double[] monitoringTimes,
			final double maturity,
			final double tolerance) {

		if(monitoringType == null) {
			throw new IllegalArgumentException("monitoringType must not be null.");
		}

		if(monitoringType == MonitoringType.CONTINUOUS) {
			if(monitoringTimes != null && monitoringTimes.length > 0) {
				throw new IllegalArgumentException(
						"Continuous monitoring must not specify monitoringTimes."
				);
			}
			return;
		}

		if(monitoringTimes == null || monitoringTimes.length == 0) {
			throw new IllegalArgumentException(
					"Discrete monitoring requires a non-empty monitoringTimes array."
			);
		}

		double previousTime = -Double.MAX_VALUE;
		for(final double monitoringTime : monitoringTimes) {
			if(monitoringTime < -tolerance || monitoringTime > maturity + tolerance) {
				throw new IllegalArgumentException(
						"Monitoring times must lie in [0,maturity]."
				);
			}

			if(monitoringTime <= previousTime + tolerance) {
				throw new IllegalArgumentException(
						"Monitoring times must be strictly increasing."
				);
			}

			previousTime = monitoringTime;
		}
	}

	public static TimeDiscretization refineTimeDiscretizationWithMonitoring(
			final TimeDiscretization baseTimeDiscretization,
			final double maturity,
			final double[] monitoringTimes) {

		if(monitoringTimes == null || monitoringTimes.length == 0) {
			return baseTimeDiscretization;
		}

		final Set<Double> mergedTauTimes = new TreeSet<>();

		for(int i = 0; i < baseTimeDiscretization.getNumberOfTimes(); i++) {
			mergedTauTimes.add(baseTimeDiscretization.getTime(i));
		}

		for(final double monitoringTime : monitoringTimes) {
			mergedTauTimes.add(maturity - monitoringTime);
		}

		final double[] refinedTimes = new double[mergedTauTimes.size()];
		int index = 0;
		for(final Double time : mergedTauTimes) {
			refinedTimes[index++] = time;
		}

		return new TimeDiscretizationFromArray(refinedTimes);
	}
}