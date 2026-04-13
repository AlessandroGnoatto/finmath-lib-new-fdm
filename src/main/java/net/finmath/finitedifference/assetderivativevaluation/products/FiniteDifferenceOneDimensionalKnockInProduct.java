package net.finmath.finitedifference.assetderivativevaluation.products;

import net.finmath.modelling.products.BarrierType;

public interface FiniteDifferenceOneDimensionalKnockInProduct {

    double getMaturity();

    double getBarrierValue();

    BarrierType getBarrierType();

    /**
     * Value used on the inactive regime in the no-hit region at maturity.
     * For vanilla barrier options this is the rebate.
     * For digital barrier options this is 0.0.
     */
    double getInactiveValueAtMaturity();
}