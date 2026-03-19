package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.BarrierOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.boundaries.StandardBoundaryCondition;
import net.finmath.modelling.products.BarrierType;
import net.finmath.modelling.products.CallOrPut;

/**
 * Boundary conditions for {@link BarrierOption} under the {@link FDMHestonModel}.
 *
 * <p>
 * State variables are assumed to be (S, v), where S is the asset level and v the variance.
 * For knock-out options, Dirichlet conditions are imposed on the barrier side in the asset direction.
 * Variance-direction boundaries are left untouched via {@link StandardBoundaryCondition#none()}.
 * </p>
 *
 * <p>
 * The barrier is assumed to coincide with the lower or upper boundary of the asset grid.
 * In-options are currently handled by parity in the product class.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class BarrierOptionHestonModelBoundary
		implements FiniteDifferenceBoundary, FiniteDifferenceBoundaryConditions {

	private static final double EPSILON = 1E-6;

	private final FDMHestonModel model;

	public BarrierOptionHestonModelBoundary(final FDMHestonModel model) {
		this.model = model;
	}

	@Override
	public BoundaryCondition[] getBoundaryConditionsAtLowerBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... stateVariables) {

		final BarrierOption option = (BarrierOption) product;
		final BarrierType barrierType = option.getBarrierType();
		final CallOrPut sign = option.getCallOrPut();

		time = Math.max(time, EPSILON);

		final BoundaryCondition[] result = new BoundaryCondition[2];

		// S -> lower boundary
		if(barrierType == BarrierType.DOWN_OUT) {
			result[0] = StandardBoundaryCondition.dirichlet(option.getRebate());
		}
		else if(sign == CallOrPut.CALL) {
			result[0] = StandardBoundaryCondition.dirichlet(0.0);
		}
		else {
			final double discountFactorRiskFree = model.getRiskFreeCurve().getDiscountFactor(time);
			final double riskFreeRate = -Math.log(discountFactorRiskFree) / time;

			final double strike = option.getStrike();
			final double maturity = option.getMaturity();

			result[0] = StandardBoundaryCondition.dirichlet(
					strike * Math.exp(-riskFreeRate * (maturity - time))
			);
		}

		// v -> lower boundary
		result[1] = StandardBoundaryCondition.none();

		return result;
	}

	@Override
	public BoundaryCondition[] getBoundaryConditionsAtUpperBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... stateVariables) {

		final BarrierOption option = (BarrierOption) product;
		final BarrierType barrierType = option.getBarrierType();
		final CallOrPut sign = option.getCallOrPut();

		time = Math.max(time, EPSILON);

		final BoundaryCondition[] result = new BoundaryCondition[2];

		// S -> upper boundary
		if(barrierType == BarrierType.UP_OUT) {
			result[0] = StandardBoundaryCondition.dirichlet(option.getRebate());
		}
		else if(sign == CallOrPut.CALL) {
			final double discountFactorRiskFree = model.getRiskFreeCurve().getDiscountFactor(time);
			final double riskFreeRate = -Math.log(discountFactorRiskFree) / time;

			final double discountFactorDividend = model.getDividendYieldCurve().getDiscountFactor(time);
			final double dividendYieldRate = -Math.log(discountFactorDividend) / time;

			final double strike = option.getStrike();
			final double maturity = option.getMaturity();
			final double S = stateVariables.length > 0 ? stateVariables[0] : 0.0;

			final double dividendAdjustedStockPrice =
					S * Math.exp(-dividendYieldRate * (maturity - time));

			result[0] = StandardBoundaryCondition.dirichlet(
					dividendAdjustedStockPrice
					- strike * Math.exp(-riskFreeRate * (maturity - time))
			);
		}
		else {
			result[0] = StandardBoundaryCondition.dirichlet(0.0);
		}

		// v -> upper boundary
		result[1] = StandardBoundaryCondition.none();

		return result;
	}

	@Override
	public double[] getValueAtLowerBoundary(
			final FiniteDifferenceProduct product,
			final double time,
			final double... stateVariables) {
		return toLegacyArray(getBoundaryConditionsAtLowerBoundary(product, time, stateVariables));
	}

	@Override
	public double[] getValueAtUpperBoundary(
			final FiniteDifferenceProduct product,
			final double time,
			final double... stateVariables) {
		return toLegacyArray(getBoundaryConditionsAtUpperBoundary(product, time, stateVariables));
	}

	private static double[] toLegacyArray(final BoundaryCondition[] boundaryConditions) {
		final double[] values = new double[boundaryConditions.length];
		for(int i = 0; i < boundaryConditions.length; i++) {
			values[i] = boundaryConditions[i].isDirichlet()
					? boundaryConditions[i].getValue()
					: Double.NaN;
		}
		return values;
	}
}