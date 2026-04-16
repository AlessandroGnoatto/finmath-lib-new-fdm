package net.finmath.finitedifference.solvers;

/**
 * Governs how barrier activation couples the inactive and active regimes
 * in the direct 1D two-state knock-in solver.
 * 
 * @author Alessandro Gnoatto
 */
public interface TwoStateActivationPolicy {

    /**
     * Inactive-regime value on the already-hit region at maturity.
     *
     * @param stateVariable State variable.
     * @param activePayoffValue Terminal payoff used for the active regime.
     * @param inactiveNoHitValue Terminal value if the barrier has never been hit.
     * @return Inactive-regime terminal value on the already-hit region.
     */
    double getAlreadyHitValueAtMaturity(
            double stateVariable,
            double activePayoffValue,
            double inactiveNoHitValue
    );

    /**
     * Inactive-regime value on the already-hit region during backward stepping.
     *
     * @param currentTime Current model time.
     * @param stateVariable State variable.
     * @param activeValue Value of the active regime at the same grid point.
     * @return Inactive-regime value on the already-hit region.
     */
    double getAlreadyHitValue(
            double currentTime,
            double stateVariable,
            double activeValue
    );

    /**
     * Dirichlet value seen by the continuation-side inactive PDE at the barrier interface.
     *
     * @param currentTime Current model time.
     * @param barrierStateVariable State variable at the barrier node.
     * @param activeValueAtBarrier Active-regime value at the barrier node.
     * @return Interface value for the inactive continuation-side solve.
     */
    double getInterfaceValue(
            double currentTime,
            double barrierStateVariable,
            double activeValueAtBarrier
    );
}