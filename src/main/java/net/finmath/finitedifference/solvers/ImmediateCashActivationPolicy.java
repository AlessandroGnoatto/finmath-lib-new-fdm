package net.finmath.finitedifference.solvers;

public class ImmediateCashActivationPolicy implements TwoStateActivationPolicy {

    private final double payoffAmount;

    public ImmediateCashActivationPolicy(final double payoffAmount) {
        this.payoffAmount = payoffAmount;
    }

    @Override
    public double getAlreadyHitValueAtMaturity(
            final double stateVariable,
            final double activePayoffValue,
            final double inactiveNoHitValue) {
        return payoffAmount;
    }

    @Override
    public double getAlreadyHitValue(
            final double currentTime,
            final double stateVariable,
            final double activeValue) {
        return payoffAmount;
    }

    @Override
    public double getInterfaceValue(
            final double currentTime,
            final double barrierStateVariable,
            final double activeValueAtBarrier) {
        return payoffAmount;
    }
}