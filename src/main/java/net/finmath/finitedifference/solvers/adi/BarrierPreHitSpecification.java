package net.finmath.finitedifference.solvers.adi;

import net.finmath.modelling.products.BarrierType;

/**
 * Immutable specification for a direct 2D pre-hit knock-in barrier PDE solve.
 *
 * <p>
 * This object describes the pre-hit problem solved in {@link BarrierPDEMode#IN_PRE_HIT}.
 * It contains:
 * </p>
 * <ul>
 *   <li>the barrier type (currently DOWN_IN or UP_IN),</li>
 *   <li>the activated barrier trace providing the Dirichlet boundary data on the barrier.</li>
 * </ul>
 *
 * <p>
 * This class is immutable and thread-safe.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public final class BarrierPreHitSpecification {

	private final BarrierType barrierType;
	private final ActivatedBarrierTrace2D activatedBarrierTrace;

	public BarrierPreHitSpecification(
			final BarrierType barrierType,
			final ActivatedBarrierTrace2D activatedBarrierTrace) {

		if(barrierType == null) {
			throw new IllegalArgumentException("barrierType must not be null.");
		}
		if(barrierType != BarrierType.DOWN_IN && barrierType != BarrierType.UP_IN) {
			throw new IllegalArgumentException(
					"BarrierPreHitSpecification requires DOWN_IN or UP_IN."
			);
		}
		if(activatedBarrierTrace == null) {
			throw new IllegalArgumentException("activatedBarrierTrace must not be null.");
		}

		this.barrierType = barrierType;
		this.activatedBarrierTrace = activatedBarrierTrace;
	}

	public BarrierType getBarrierType() {
		return barrierType;
	}

	public ActivatedBarrierTrace2D getActivatedBarrierTrace() {
		return activatedBarrierTrace;
	}

	public boolean isDownIn() {
		return barrierType == BarrierType.DOWN_IN;
	}

	public boolean isUpIn() {
		return barrierType == BarrierType.UP_IN;
	}
}