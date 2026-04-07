package net.finmath.finitedifference.solvers.adi;

/**
 * Pricing mode for direct barrier PDE solves.
 *
 * OUT:
 *   direct knock-out pricing on the alive region.
 *
 * IN_PRE_HIT:
 *   direct pre-hit knock-in pricing on the not-yet-activated region,
 *   using a barrier Dirichlet trace from the corresponding activated vanilla.
 */
public enum BarrierPDEMode {
	OUT,
	IN_PRE_HIT
}