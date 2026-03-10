package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.modelling.products.CallOrPut;

/**
 * Boundary conditions for {@link EuropeanOption} under the
 * {@link FDMBlackScholesModel}.
 *
 * <p>
 * This class provides Dirichlet-type boundary values for the finite
 * difference approximation of the Black–Scholes PDE.
 * </p>
 *
 * @author Andrea Mazzon
 */
public class EuropeanOptionBlackScholesModelBoundary implements FiniteDifferenceBoundary {

	private FDMBlackScholesModel model;

	/**
	 * Creates the boundary condition associated with a given
	 * Black–Scholes model.
	 *
	 * @param model The Black–Scholes model providing risk-free and dividend curves.
	 */
	public EuropeanOptionBlackScholesModelBoundary(final FDMBlackScholesModel model) {
		this.model = model;
	}

	@Override
	public double[] getValueAtLowerBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[1];
		final CallOrPut sign =
				((EuropeanOption) product).getCallOrPut();

		if(sign == CallOrPut.CALL) {
			result[0] = 0.0;
			return result;
		}
		else { // Put option
			if(time == 0) {
				time = 0.000001;
			}

			final double rF =
					model.getRiskFreeCurve().getDiscountFactor(time);
			final double riskFreeRate =
					-Math.log(rF) / 1.0;

			final double strike =
					((EuropeanOption) product).getStrike();
			final double maturity =
					((EuropeanOption) product).getMaturity();

			result[0] =
					strike * Math.exp(
							-riskFreeRate * (maturity - time));
		}

		return result;
	}

	@Override
	public double[] getValueAtUpperBoundary(
			final FiniteDifferenceProduct product,
			double time,
			final double... riskFactors) {

		final double[] result = new double[1];
		final CallOrPut sign =
				((EuropeanOption) product).getCallOrPut();

		if(sign == CallOrPut.CALL) {

			if(time == 0) {
				time = 0.000001;
			}

			final double rF =
					model.getRiskFreeCurve().getDiscountFactor(time);
			final double riskFreeRate =
					-Math.log(rF) / time;

			final double dY =
					model.getDividendYieldCurve().getDiscountFactor(time);
			final double dividendYieldRate =
					-Math.log(dY) / time;

			final double strike =
					((EuropeanOption) product).getStrike();
			final double maturity =
					((EuropeanOption) product).getMaturity();

			final double dividendAdjustedStockPrice =
					riskFactors[0]
							* Math.exp(
									-dividendYieldRate
									* (maturity - time));

			result[0] =
					dividendAdjustedStockPrice
					- strike * Math.exp(
							-riskFreeRate * (maturity - time));

			return result;
		}
		else { // Put option
			result[0] = 0.0;
			return result;
		}
	}
}