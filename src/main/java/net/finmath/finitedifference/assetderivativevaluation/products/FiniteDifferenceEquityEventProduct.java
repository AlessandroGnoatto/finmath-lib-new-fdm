package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;

/**
 * Equity finite-difference products with vector-level event conditions.
 *
 * <p>
 * Event times are running times t at which the backward induction first
 * computes the continuation value V(t+,x) and then applies a jump/event rule
 * to obtain V(t-,x).
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public interface FiniteDifferenceEquityEventProduct extends FiniteDifferenceEquityProduct {

	default double[] getEventTimes() {
		return new double[0];
	}

	default double[] applyEventCondition(
			final double time,
			final double[] valuesAfterEvent,
			final FiniteDifferenceEquityModel model) {
		return valuesAfterEvent;
	}
}