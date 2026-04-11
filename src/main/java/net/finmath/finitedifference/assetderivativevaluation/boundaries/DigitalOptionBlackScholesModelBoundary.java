package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.products.DigitalOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.boundaries.BoundaryCondition;
import net.finmath.finitedifference.boundaries.StandardBoundaryCondition;
import net.finmath.modelling.products.CallOrPut;

/**
 * Boundary conditions for {@link DigitalOption} under the
 * {@link FDMBlackScholesModel}.
 *
 * <p>
 * This class supports the explicit boundary-condition API returning
 * {@link BoundaryCondition} objects.
 * </p>
 *
 * <p>
 * Conventions:
 * </p>
 * <ul>
 *   <li>For a cash-or-nothing call, the lower boundary is zero and the upper
 *       boundary is the discounted cash payoff.</li>
 *   <li>For a cash-or-nothing put, the lower boundary is the discounted cash
 *       payoff and the upper boundary is zero.</li>
 *   <li>For an asset-or-nothing call, the lower boundary is zero and the upper
 *       boundary is the dividend-adjusted stock value.</li>
 *   <li>For an asset-or-nothing put, the lower boundary is the
 *       dividend-adjusted stock value and the upper boundary is zero.</li>
 * </ul>
 *
 * <p>
 * The implementation assumes European exercise.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class DigitalOptionBlackScholesModelBoundary
		implements FiniteDifferenceBoundary {

	private static final double EPSILON = 1E-6;

	private final FDMBlackScholesModel model;

	/**
	 * Creates the boundary condition associated with a given
	 * {@link FDMBlackScholesModel}.
	 *
	 * @param model The Black-Scholes model used to determine
	 *              risk-free and dividend discount factors.
	 */
	public DigitalOptionBlackScholesModelBoundary(final FDMBlackScholesModel model) {
		this.model = model;
	}

	@Override
	public BoundaryCondition[] getBoundaryConditionsAtLowerBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... stateVariables) {

		final DigitalOption option = (DigitalOption) product;
		final CallOrPut sign = option.getCallOrPut();
		final DigitalOption.DigitalPayoffType payoffType = option.getDigitalPayoffType();

		time = Math.max(time, EPSILON);

		final double maturity = option.getMaturity();
		final double discountFactorRiskFree = model.getRiskFreeCurve().getDiscountFactor(time);
		final double riskFreeRate = -Math.log(discountFactorRiskFree) / time;

		final double discountFactorDividend = model.getDividendYieldCurve().getDiscountFactor(time);
		final double dividendYieldRate = -Math.log(discountFactorDividend) / time;

		if(payoffType == DigitalOption.DigitalPayoffType.CASH_OR_NOTHING) {
			if(sign == CallOrPut.CALL) {
				return new BoundaryCondition[] {
						StandardBoundaryCondition.dirichlet(0.0)
				};
			}
			else {
				final double value =
						option.getCashPayoff() * Math.exp(-riskFreeRate * (maturity - time));

				return new BoundaryCondition[] {
						StandardBoundaryCondition.dirichlet(value)
				};
			}
		}
		else if(payoffType == DigitalOption.DigitalPayoffType.ASSET_OR_NOTHING) {
			if(sign == CallOrPut.CALL) {
				return new BoundaryCondition[] {
						StandardBoundaryCondition.dirichlet(0.0)
				};
			}
			else {
				final double stateVariable = stateVariables[0];
				final double value =
						stateVariable * Math.exp(-dividendYieldRate * (maturity - time));

				return new BoundaryCondition[] {
						StandardBoundaryCondition.dirichlet(value)
				};
			}
		}
		else {
			throw new IllegalArgumentException("Unsupported digital payoff type.");
		}
	}

	@Override
	public BoundaryCondition[] getBoundaryConditionsAtUpperBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... stateVariables) {

		final DigitalOption option = (DigitalOption) product;
		final CallOrPut sign = option.getCallOrPut();
		final DigitalOption.DigitalPayoffType payoffType = option.getDigitalPayoffType();

		time = Math.max(time, EPSILON);

		final double maturity = option.getMaturity();
		final double discountFactorRiskFree = model.getRiskFreeCurve().getDiscountFactor(time);
		final double riskFreeRate = -Math.log(discountFactorRiskFree) / time;

		final double discountFactorDividend = model.getDividendYieldCurve().getDiscountFactor(time);
		final double dividendYieldRate = -Math.log(discountFactorDividend) / time;

		if(payoffType == DigitalOption.DigitalPayoffType.CASH_OR_NOTHING) {
			if(sign == CallOrPut.CALL) {
				final double value =
						option.getCashPayoff() * Math.exp(-riskFreeRate * (maturity - time));

				return new BoundaryCondition[] {
						StandardBoundaryCondition.dirichlet(value)
				};
			}
			else {
				return new BoundaryCondition[] {
						StandardBoundaryCondition.dirichlet(0.0)
				};
			}
		}
		else if(payoffType == DigitalOption.DigitalPayoffType.ASSET_OR_NOTHING) {
			if(sign == CallOrPut.CALL) {
				final double stateVariable = stateVariables[0];
				final double value =
						stateVariable * Math.exp(-dividendYieldRate * (maturity - time));

				return new BoundaryCondition[] {
						StandardBoundaryCondition.dirichlet(value)
				};
			}
			else {
				return new BoundaryCondition[] {
						StandardBoundaryCondition.dirichlet(0.0)
				};
			}
		}
		else {
			throw new IllegalArgumentException("Unsupported digital payoff type.");
		}
	}
}