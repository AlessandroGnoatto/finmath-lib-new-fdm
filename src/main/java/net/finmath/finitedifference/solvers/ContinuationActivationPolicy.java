package net.finmath.finitedifference.solvers;

public class ContinuationActivationPolicy implements TwoStateActivationPolicy {

    @Override
    public double getAlreadyHitValueAtMaturity(
            final double stateVariable,
            final double activePayoffValue,
            final double inactiveNoHitValue) {
        return activePayoffValue;
    }

    @Override
    public double getAlreadyHitValue(
            final double currentTime,
            final double stateVariable,
            final double activeValue) {
        return activeValue;
    }

    @Override
    public double getInterfaceValue(
            final double currentTime,
            final double barrierStateVariable,
            final double activeValueAtBarrier) {
        return activeValueAtBarrier;
    }
}