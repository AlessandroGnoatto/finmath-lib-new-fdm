package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.products.AmericanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.modelling.products.CallOrPut;

/**
 * Boundary condition for an {@link AmericanOption} under the
 * {@link FDMBlackScholesModel}.
 * <p>
 * This class provides Dirichlet boundary values for the finite difference
 * approximation of the Black–Scholes PDE.
 * </p>
 *
 * @author Andrea Mazzon
 */
public class AmericanOptionBlackScholesModelBoundary implements FiniteDifferenceBoundary {

	private static final double EPSILON = 1E-6;

	private final FDMBlackScholesModel model;

	/**
	 * Creates the boundary condition associated with a given
	 * {@link FDMBlackScholesModel}.
	 *
	 * @param model The Black–Scholes model used to determine
	 *              risk-free and dividend discount factors.
	 */
	public AmericanOptionBlackScholesModelBoundary(final FDMBlackScholesModel model) {
		this.model = model;
	}

	@Override
	public double[] getValueAtUpperBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[1];
		final AmericanOption option = (AmericanOption) product;
		final CallOrPut sign = option.getCallOrPut();

		if(sign == CallOrPut.CALL) {

			time = Math.max(time, EPSILON);

			final double discountFactorRiskFree =
					model.getRiskFreeCurve().getDiscountFactor(time);
			final double riskFreeRate =
					-Math.log(discountFactorRiskFree) / time;

			final double discountFactorDividend =
					model.getDividendYieldCurve().getDiscountFactor(time);
			final double dividendYieldRate =
					-Math.log(discountFactorDividend) / time;

			final double strike = option.getStrike();
			final double maturity = option.getMaturity();

			final double dividendAdjustedStockPrice =
					riskFactors[0] * Math.exp(
							-dividendYieldRate * (maturity - time));

			result[0] =
					dividendAdjustedStockPrice
					- strike * Math.exp(
							-riskFreeRate * (maturity - time));

		}
		else {
			result[0] = 0.0;
		}

		return result;
	}

	@Override
	public double[] getValueAtLowerBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[1];
		final AmericanOption option = (AmericanOption) product;
		final CallOrPut sign = option.getCallOrPut();

		if(sign == CallOrPut.CALL) {
			result[0] = 0.0;
		}
		else {

			time = Math.max(time, EPSILON);

			final double discountFactorRiskFree =
					model.getRiskFreeCurve().getDiscountFactor(time);
			final double riskFreeRate =
					-Math.log(discountFactorRiskFree) / time;

			final double strike = option.getStrike();
			final double maturity = option.getMaturity();

			result[0] =
					strike * Math.exp(
							-riskFreeRate * (maturity - time));
		}

		return result;
	}
}